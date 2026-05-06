"""TTS client — ChaosVector TTS (remote) with local Piper fallback.

Waterfall: remote Wyoming TCP → local Piper TTS.
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
    timeout: float = 10.0
    # Local Piper fallback
    piper_model: str = "/home/chaos/piper-models/en_US-lessac-medium.onnx"
    piper_enabled: bool = True


@dataclass
class TTSResult:
    audio: np.ndarray   # int16
    sample_rate: int
    channels: int
    duration_ms: float
    synthesis_ms: float
    source: str = "remote"  # "remote" or "local"


# Cache the Piper model (expensive to load, ~7s on Pi 5)
_piper_voice = None
_piper_load_attempted = False


def _get_piper_voice(model_path: str):
    """Lazy-load Piper voice model (cached after first load)."""
    global _piper_voice, _piper_load_attempted
    if _piper_load_attempted:
        return _piper_voice
    _piper_load_attempted = True
    try:
        from piper import PiperVoice
        _piper_voice = PiperVoice.load(model_path)
        log.info("local Piper loaded: %s (rate=%d)",
                 model_path, _piper_voice.config.sample_rate)
    except Exception as e:
        log.warning("local Piper unavailable: %s", e)
        _piper_voice = None
    return _piper_voice


async def synthesize(text: str, config: TTSConfig | None = None) -> TTSResult | None:
    """TTS waterfall: try remote ChaosVector TTS, fall back to local Piper."""
    config = config or TTSConfig()

    # Try remote first
    result = await _synthesize_remote(text, config)
    if result is not None:
        return result

    # Fallback to local Piper
    if config.piper_enabled:
        log.info("TTS: remote failed, trying local Piper")
        return await _synthesize_local(text, config)

    return None


async def _synthesize_remote(text: str, config: TTSConfig) -> TTSResult | None:
    """Send text to ChaosVector TTS via Wyoming TCP."""
    t_start = time.monotonic()

    log.info("TTS: synthesizing \"%s\" (voice=%s)", text[:80], config.voice)

    client = AsyncTcpClient(config.host, config.port)
    try:
        await asyncio.wait_for(client.connect(), timeout=5.0)
    except (OSError, asyncio.TimeoutError) as e:
        log.warning("TTS remote connect failed: %s", e)
        return None

    try:
        await client.write_event(
            Synthesize(text=text, voice=SynthesizeVoice(name=config.voice)).event()
        )

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
            log.warning("TTS remote returned no audio")
            return None

        audio = np.frombuffer(bytes(audio_data), dtype=np.int16)
        t_done = time.monotonic()
        synthesis_ms = (t_done - t_start) * 1000
        duration_ms = len(audio) / sample_rate * 1000

        log.info(
            "TTS: %.0fms synthesis, %.0fms audio, %d samples @ %dHz (remote)",
            synthesis_ms, duration_ms, len(audio), sample_rate,
        )

        return TTSResult(
            audio=audio, sample_rate=sample_rate, channels=channels,
            duration_ms=duration_ms, synthesis_ms=synthesis_ms, source="remote",
        )

    except asyncio.TimeoutError:
        log.warning("TTS remote timed out after %.1fs", config.timeout)
        return None
    except (ConnectionError, OSError) as e:
        log.warning("TTS remote connection error: %s", e)
        return None
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass


async def _synthesize_local(text: str, config: TTSConfig) -> TTSResult | None:
    """Synthesize using local Piper TTS on the Pi 5."""
    t_start = time.monotonic()

    loop = asyncio.get_running_loop()
    voice = await loop.run_in_executor(None, _get_piper_voice, config.piper_model)
    if voice is None:
        return None

    try:
        import io
        import wave

        # Run synthesis in executor (CPU-bound)
        def _do_synth():
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(text, wf)
            buf.seek(44)  # skip WAV header
            return np.frombuffer(buf.read(), dtype=np.int16), voice.config.sample_rate

        audio, sample_rate = await loop.run_in_executor(None, _do_synth)

        t_done = time.monotonic()
        synthesis_ms = (t_done - t_start) * 1000
        duration_ms = len(audio) / sample_rate * 1000

        log.info(
            "TTS: %.0fms synthesis, %.0fms audio, %d samples @ %dHz (local Piper)",
            synthesis_ms, duration_ms, len(audio), sample_rate,
        )

        return TTSResult(
            audio=audio, sample_rate=sample_rate, channels=1,
            duration_ms=duration_ms, synthesis_ms=synthesis_ms, source="local",
        )

    except Exception as e:
        log.error("TTS local synthesis failed: %s", e)
        return None
