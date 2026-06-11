"""LiveKit WakeWord Wyoming TCP server — drop-in replacement for openWakeWord.

Mirrors openWakeWord's handler pattern: processes each audio chunk inline
and sends Detection events immediately when the wake word is detected.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np

from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.wake import Detect, Detection, NotDetected

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class LiveKitWakeHandler(AsyncEventHandler):
    """Wyoming event handler — processes audio inline like openWakeWord."""

    def __init__(self, model, threshold: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._threshold = threshold
        self._converter = AudioChunkConverter(rate=SAMPLE_RATE, width=2, channels=1)
        self._detecting = False
        self._detected = False
        self._names: list[str] = []
        self._cooldown_until: float = 0.0
        self._audio_timestamp: int = 0
        # Rolling buffer for last 2s of audio (100 chunks at 20ms)
        self._chunk_buffer: deque[np.ndarray] = deque(maxlen=100)
        self._chunks_since_predict: int = 0

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            model_names = list(self._model.models.keys()) if hasattr(self._model, 'models') else ["hey_jarvis"]
            info = Info(
                wake=[WakeProgram(
                    name="livekit-wakeword",
                    description="LiveKit WakeWord Detection",
                    attribution=Attribution(name="LiveKit", url="https://github.com/livekit/livekit-wakeword"),
                    installed=True,
                    models=[WakeModel(name=n, description=f"Wake word: {n}",
                                      attribution=Attribution(name="custom", url=""),
                                      installed=True, languages=["en"])
                            for n in model_names],
                )],
            )
            await self.write_event(info.event())
            return True

        if Detect.is_type(event.type):
            detect = Detect.from_event(event)
            self._detecting = True
            self._detected = False
            self._names = detect.names or []
            self._chunk_buffer.clear()
            self._chunks_since_predict = 0
            return True

        if AudioStart.is_type(event.type):
            self._audio_timestamp = 0
            self._chunk_buffer.clear()
            self._chunks_since_predict = 0
            return True

        if AudioChunk.is_type(event.type):
            if not self._detecting:
                return True

            chunk = self._converter.convert(AudioChunk.from_event(event))
            audio = np.frombuffer(chunk.audio, dtype=np.int16)
            self._chunk_buffer.append(audio)
            self._audio_timestamp += chunk.milliseconds
            self._chunks_since_predict += 1

            # Run prediction every 10 chunks (~200ms) once we have 2s buffered
            if self._chunks_since_predict >= 10 and len(self._chunk_buffer) >= 50:
                self._chunks_since_predict = 0

                # Cooldown check
                if time.monotonic() < self._cooldown_until:
                    return True

                # Concatenate buffer into window
                window = np.concatenate(list(self._chunk_buffer))

                # Run prediction
                scores = self._model.predict(window)

                for name, score in scores.items():
                    if score >= self._threshold:
                        if self._names and name not in self._names:
                            continue
                        log.info("wake detected: %s (score=%.3f)", name, score)
                        self._detecting = False
                        self._detected = True
                        self._cooldown_until = time.monotonic() + 2.0
                        try:
                            await self.write_event(
                                Detection(name=name, timestamp=self._audio_timestamp).event()
                            )
                        except (ConnectionResetError, BrokenPipeError, OSError):
                            log.debug("client disconnected before detection sent")
                        return True

            return True

        if AudioStop.is_type(event.type):
            if self._detecting and not self._detected:
                try:
                    await self.write_event(NotDetected().event())
                except (ConnectionResetError, BrokenPipeError, OSError):
                    pass
            self._detecting = False
            return True

        return True


async def main() -> None:
    parser = argparse.ArgumentParser(description="LiveKit WakeWord Wyoming Server")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--threshold", type=float, default=0.10, help="Detection threshold")
    parser.add_argument("--port", type=int, default=10401, help="Wyoming TCP port")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from livekit.wakeword import WakeWordModel
    log.info("Loading model: %s", args.model)
    model = WakeWordModel(models=[args.model])
    model_names = list(model.models.keys()) if hasattr(model, 'models') else [Path(args.model).stem]
    log.info("Model loaded: %s (threshold=%.2f)", model_names, args.threshold)

    server = AsyncServer.from_uri(f"tcp://{args.host}:{args.port}")
    log.info("Listening on %s:%d", args.host, args.port)

    await server.run(partial(LiveKitWakeHandler, model, args.threshold))


if __name__ == "__main__":
    asyncio.run(main())
