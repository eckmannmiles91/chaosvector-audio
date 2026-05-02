"""STT client — Wyoming TCP to ChaosVector STT.

Simple per-request connection: connect, send audio, get transcript, disconnect.
No persistent connections, no stale socket bugs.
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
class STTConfig:
    host: str = "10.1.1.240"
    port: int = 10301
    language: str = "en"
    timeout: float = 10.0  # max time to wait for transcript


async def transcribe(chunks: list[AudioChunk], config: STTConfig | None = None) -> str | None:
    """Send buffered audio chunks to ChaosVector STT, return transcript.

    Opens a fresh TCP connection per request. This eliminates stale connection
    bugs — the connection lives only as long as the transcription.
    """
    config = config or STTConfig()
    t_start = time.monotonic()

    total_samples = sum(len(c.samples) for c in chunks)
    audio_duration_ms = total_samples / 16000 * 1000
    log.info("STT: sending %d chunks (%.0fms audio)", len(chunks), audio_duration_ms)

    client = AsyncTcpClient(config.host, config.port)
    try:
        await asyncio.wait_for(client.connect(), timeout=5.0)
    except (OSError, asyncio.TimeoutError) as e:
        log.error("STT connect failed: %s", e)
        return None

    try:
        # Handshake
        await client.write_event(Transcribe(language=config.language).event())
        await client.write_event(
            AudioStart(rate=16000, width=2, channels=1).event()
        )

        # Stream audio
        for chunk in chunks:
            raw = chunk.samples.astype("<i2").tobytes()  # int16 little-endian
            await client.write_event(
                WyAudioChunk(rate=16000, width=2, channels=1, audio=raw).event()
            )

        t_sent = time.monotonic()
        await client.write_event(AudioStop().event())

        # Wait for transcript
        async with asyncio.timeout(config.timeout):
            while True:
                event = await client.read_event()
                if event is None:
                    log.warning("STT connection closed before transcript")
                    return None
                if Transcript.is_type(event.type):
                    transcript = Transcript.from_event(event)
                    t_done = time.monotonic()
                    log.info(
                        "STT: inference=%.0fms total=%.0fms → \"%s\"",
                        (t_done - t_sent) * 1000,
                        (t_done - t_start) * 1000,
                        transcript.text or "",
                    )
                    return transcript.text or None

    except asyncio.TimeoutError:
        log.warning("STT timed out after %.1fs", config.timeout)
        return None
    except (ConnectionError, OSError) as e:
        log.warning("STT connection error: %s", e)
        return None
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
