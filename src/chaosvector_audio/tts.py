"""TTS client — Wyoming TCP to ChaosVector TTS.

Simple per-request connection: connect, send text, receive audio, disconnect.
Returns raw int16 numpy array ready for PlaybackManager.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np
from wyoming.audio import AudioChunk as WyAudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize, SynthesizeVoice

log = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    host: str = "10.1.1.240"
    port: int = 10210
    voice: str = "af_heart"
    timeout: float = 10.0  # max time to wait for synthesis


@dataclass
class TTSResult:
    audio: np.ndarray   # int16
    sample_rate: int
    channels: int
    duration_ms: float
    synthesis_ms: float


async def synthesize(text: str, config: TTSConfig | None = None) -> TTSResult | None:
    """Send text to ChaosVector TTS, return audio as int16 numpy array.

    Opens a fresh TCP connection per request.
    """
    config = config or TTSConfig()
    t_start = time.monotonic()

    log.info("TTS: synthesizing \"%s\" (voice=%s)", text[:80], config.voice)

    client = AsyncTcpClient(config.host, config.port)
    try:
        await asyncio.wait_for(client.connect(), timeout=5.0)
    except (OSError, asyncio.TimeoutError) as e:
        log.error("TTS connect failed: %s", e)
        return None

    try:
        # Send synthesis request
        await client.write_event(
            Synthesize(text=text, voice=SynthesizeVoice(name=config.voice)).event()
        )

        # Receive audio
        audio_data = bytearray()
        sample_rate = 22050
        channels = 1

        async with asyncio.timeout(config.timeout):
            while True:
                event = await client.read_event()
                if event is None:
                    log.warning("TTS connection closed before audio complete")
                    break
                if AudioStart.is_type(event.type):
                    audio_start = AudioStart.from_event(event)
                    sample_rate = audio_start.rate
                    channels = audio_start.channels
                elif WyAudioChunk.is_type(event.type):
                    chunk = WyAudioChunk.from_event(event)
                    audio_data.extend(chunk.audio)
                elif AudioStop.is_type(event.type):
                    break

        if not audio_data:
            log.warning("TTS returned no audio")
            return None

        audio = np.frombuffer(bytes(audio_data), dtype=np.int16)
        t_done = time.monotonic()
        synthesis_ms = (t_done - t_start) * 1000
        duration_ms = len(audio) / sample_rate * 1000

        log.info(
            "TTS: %.0fms synthesis, %.0fms audio, %d samples @ %dHz",
            synthesis_ms, duration_ms, len(audio), sample_rate,
        )

        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            duration_ms=duration_ms,
            synthesis_ms=synthesis_ms,
        )

    except asyncio.TimeoutError:
        log.warning("TTS timed out after %.1fs", config.timeout)
        return None
    except (ConnectionError, OSError) as e:
        log.warning("TTS connection error: %s", e)
        return None
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
