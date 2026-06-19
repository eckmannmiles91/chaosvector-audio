"""Streaming STT client — sends audio chunks in real-time during speech.

Instead of buffering the entire utterance then sending, this opens the
STT connection at the start of listening and streams chunks as they
arrive from the mic. When VAD detects end-of-speech, AudioStop is sent
and the transcript comes back almost instantly since the server already
has all the audio.

Savings: eliminates the buffer-then-send delay (typically 200-500ms
for a 2s utterance on the Wyoming TCP protocol).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from wyoming.audio import AudioChunk as WyAudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.client import AsyncTcpClient

from chaosvector_audio.capture import AudioChunk

log = logging.getLogger(__name__)


@dataclass
class StreamingSTTConfig:
    host: str = "10.1.1.240"
    port: int = 10301
    language: str = "en"
    timeout: float = 10.0


class StreamingSTTSession:
    """Manages a streaming STT session — connect once, send chunks, get transcript."""

    def __init__(self, config: StreamingSTTConfig | None = None):
        self.config = config or StreamingSTTConfig()
        self._client: AsyncTcpClient | None = None
        self._connected = False
        self._t_start: float = 0
        self._chunks_sent: int = 0

    async def start(self) -> bool:
        """Open connection and send handshake. Call this before sending chunks."""
        self._t_start = time.monotonic()
        self._chunks_sent = 0
        self._client = AsyncTcpClient(self.config.host, self.config.port)
        try:
            await asyncio.wait_for(self._client.connect(), timeout=3.0)
            await self._client.write_event(
                Transcribe(language=self.config.language).event()
            )
            await self._client.write_event(
                AudioStart(rate=16000, width=2, channels=1).event()
            )
            self._connected = True
            log.debug("Streaming STT: connected to %s:%d", self.config.host, self.config.port)
            return True
        except (OSError, asyncio.TimeoutError) as e:
            log.error("Streaming STT connect failed: %s", e)
            self._connected = False
            return False

    async def send_chunk(self, chunk: AudioChunk) -> None:
        """Send a single audio chunk. Call this for each chunk during listening."""
        if not self._connected or self._client is None:
            return
        try:
            raw = chunk.samples.astype("<i2").tobytes()
            await self._client.write_event(
                WyAudioChunk(rate=16000, width=2, channels=1, audio=raw).event()
            )
            self._chunks_sent += 1
        except (ConnectionError, OSError) as e:
            log.warning("Streaming STT: send failed: %s", e)
            self._connected = False

    async def finish(self) -> str | None:
        """Send AudioStop and wait for transcript. Call after VAD end-of-speech."""
        if not self._connected or self._client is None:
            return None

        try:
            t_stop = time.monotonic()
            await self._client.write_event(AudioStop().event())

            async with asyncio.timeout(self.config.timeout):
                while True:
                    event = await self._client.read_event()
                    if event is None:
                        log.warning("Streaming STT: connection closed before transcript")
                        return None
                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        t_done = time.monotonic()
                        inference_ms = (t_done - t_stop) * 1000
                        total_ms = (t_done - self._t_start) * 1000
                        audio_ms = self._chunks_sent * 20  # 20ms per chunk
                        log.info(
                            "Streaming STT: %d chunks (%.0fms audio), "
                            "inference=%.0fms after-stop, total=%.0fms → \"%s\"",
                            self._chunks_sent, audio_ms,
                            inference_ms, total_ms,
                            transcript.text or "",
                        )
                        return transcript.text or None

        except asyncio.TimeoutError:
            log.warning("Streaming STT: timed out after %.1fs", self.config.timeout)
            return None
        except (ConnectionError, OSError) as e:
            log.warning("Streaming STT: error during finish: %s", e)
            return None
        finally:
            self._connected = False
            try:
                if self._client:
                    await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    async def abort(self) -> None:
        """Cancel the session without waiting for a transcript."""
        self._connected = False
        try:
            if self._client:
                await self._client.disconnect()
        except Exception:
            pass
        self._client = None
