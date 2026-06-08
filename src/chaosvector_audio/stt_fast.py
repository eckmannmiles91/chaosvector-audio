"""Fast STT via Speech-to-Phrase — constrained recognition for device commands.

Uses Wyoming protocol to Speech-to-Phrase add-on in HA. Only recognizes
known device/area names and command patterns. Returns None if the utterance
doesn't match any known command (fall back to full STT).
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
class FastSTTConfig:
    host: str = "10.1.1.53"
    port: int = 10302
    timeout: float = 5.0
    enabled: bool = True


async def transcribe_fast(
    chunks: list[AudioChunk], config: FastSTTConfig
) -> str | None:
    """Try constrained recognition via Speech-to-Phrase.

    Returns transcript if matched, None if no match or error
    (caller should fall back to full STT).
    """
    if not config.enabled:
        return None

    t_start = time.monotonic()

    client = AsyncTcpClient(config.host, config.port)
    try:
        await asyncio.wait_for(client.connect(), timeout=3.0)
    except (OSError, asyncio.TimeoutError) as e:
        log.debug("fast STT connect failed: %s", e)
        return None

    try:
        await client.write_event(Transcribe(language="en").event())
        await client.write_event(
            AudioStart(rate=16000, width=2, channels=1).event()
        )

        for chunk in chunks:
            raw = chunk.samples.astype("<i2").tobytes()
            await client.write_event(
                WyAudioChunk(rate=16000, width=2, channels=1, audio=raw).event()
            )

        await client.write_event(AudioStop().event())

        async with asyncio.timeout(config.timeout):
            while True:
                event = await client.read_event()
                if event is None:
                    return None
                if Transcript.is_type(event.type):
                    transcript = Transcript.from_event(event)
                    text = (transcript.text or "").strip()
                    elapsed = (time.monotonic() - t_start) * 1000
                    if text:
                        log.info("fast STT: %.0fms → \"%s\"", elapsed, text)
                        return text
                    else:
                        log.debug("fast STT: %.0fms → no match", elapsed)
                        return None

    except asyncio.TimeoutError:
        log.debug("fast STT timed out")
        return None
    except (ConnectionError, OSError) as e:
        log.debug("fast STT error: %s", e)
        return None
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
