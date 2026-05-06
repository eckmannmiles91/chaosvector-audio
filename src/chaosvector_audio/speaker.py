"""Speaker verification — HTTP POST to Resemblyzer service.

Identifies who is speaking from a short audio sample.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SpeakerConfig:
    url: str = "http://10.1.1.228:8500"
    enabled: bool = True
    timeout: float = 5.0


async def identify_speaker(
    audio: np.ndarray,
    config: SpeakerConfig | None = None,
) -> str | None:
    """Send audio to speaker verification service, return speaker name or None.

    Args:
        audio: int16 PCM audio (16kHz mono, ~3s recommended)
        config: Service configuration
    """
    config = config or SpeakerConfig()
    if not config.enabled:
        return None

    audio_bytes = audio.astype("<i2").tobytes()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config.url}/verify",
                data=audio_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=aiohttp.ClientTimeout(total=config.timeout),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    speaker = data.get("speaker")
                    confidence = data.get("confidence", 0.0)
                    if speaker:
                        log.info("speaker identified: %s (confidence=%.2f)", speaker, confidence)
                    return speaker
                else:
                    log.warning("speaker verify HTTP %d", resp.status)
                    return None
    except Exception as e:
        log.debug("speaker verify unavailable: %s", e)
        return None
