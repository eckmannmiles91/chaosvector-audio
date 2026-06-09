"""LiveKit WakeWord Wyoming TCP server — drop-in replacement for openWakeWord.

Runs the LiveKit WakeWord ONNX model and exposes it via the Wyoming protocol
so existing Wyoming clients (including our wake.py) can connect unchanged.

Usage:
    python -m chaosvector_audio.wake_livekit --model /path/to/hey_jarvis.onnx --port 10400

Or as a systemd service replacing wyoming-openwakeword.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.wake import Detect, Detection, NotDetected

log = logging.getLogger(__name__)

# LiveKit model expects 16kHz mono, processes ~2s windows
SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SAMPLES = SAMPLE_RATE * 2  # 2 second sliding window


class LiveKitWakeHandler(AsyncEventHandler):
    """Wyoming event handler for LiveKit WakeWord detection."""

    def __init__(
        self,
        model,
        threshold: float,
        *args, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._threshold = threshold
        self._audio_buffer = bytearray()
        self._detecting = False
        self._names: list[str] = []
        self._cooldown_until: float = 0.0

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            info = Info(
                wake=[
                    WakeProgram(
                        name="livekit-wakeword",
                        description="LiveKit WakeWord Detection",
                        attribution=Attribution(
                            name="LiveKit", url="https://github.com/livekit/livekit-wakeword",
                        ),
                        installed=True,
                        models=[
                            WakeModel(
                                name=name,
                                description=f"Wake word: {name}",
                                attribution=Attribution(name="custom", url=""),
                                installed=True,
                                languages=["en"],
                            )
                            for name in (list(self._model.models.keys()) if hasattr(self._model, 'models') else ["hey_jarvis"])
                        ],
                    )
                ],
            )
            await self.write_event(info.event())
            return True

        if Detect.is_type(event.type):
            detect = Detect.from_event(event)
            self._detecting = True
            self._names = detect.names or []
            self._audio_buffer.clear()
            return True

        if AudioStart.is_type(event.type):
            return True

        if AudioChunk.is_type(event.type):
            if not self._detecting:
                return True

            chunk = AudioChunk.from_event(event)
            self._audio_buffer.extend(chunk.audio)

            # Process when we have enough audio (~2s window)
            if len(self._audio_buffer) >= WINDOW_SAMPLES * 2:  # 2 bytes per sample
                # Cooldown check
                if time.monotonic() < self._cooldown_until:
                    # Trim buffer but don't process
                    self._audio_buffer = self._audio_buffer[-(WINDOW_SAMPLES * 2):]
                    return True

                # Convert to numpy
                audio = np.frombuffer(bytes(self._audio_buffer[-(WINDOW_SAMPLES * 2):]),
                                      dtype=np.int16)

                # Run prediction
                scores = self._model.predict(audio)

                # Check each model
                for name, score in scores.items():
                    if score >= self._threshold:
                        if self._names and name not in self._names:
                            continue
                        log.info("wake detected: %s (score=%.3f)", name, score)
                        self._detecting = False
                        self._cooldown_until = time.monotonic() + 2.0
                        await self.write_event(
                            Detection(name=name, timestamp=int(time.time() * 1000)).event()
                        )
                        return True

                # Keep only the last window
                self._audio_buffer = self._audio_buffer[-(WINDOW_SAMPLES * 2):]
            return True

        if AudioStop.is_type(event.type):
            if self._detecting:
                self._detecting = False
                await self.write_event(NotDetected().event())
            return True

        return True


async def main() -> None:
    parser = argparse.ArgumentParser(description="LiveKit WakeWord Wyoming Server")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--port", type=int, default=10400, help="Wyoming TCP port")
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
    log.info("Model loaded: %s", model_names)

    server = AsyncServer.from_uri(f"tcp://{args.host}:{args.port}")
    log.info("Listening on %s:%d", args.host, args.port)

    await server.run(partial(LiveKitWakeHandler, model, args.threshold))


if __name__ == "__main__":
    asyncio.run(main())
