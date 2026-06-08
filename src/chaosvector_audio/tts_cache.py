"""TTS cache — pre-synthesize common phrases for instant playback.

Caches synthesis results in memory (LRU) and optionally pre-warms
with time strings and frequent responses on startup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CachedAudio:
    audio: np.ndarray
    sample_rate: int
    channels: int
    duration_ms: float


class TTSCache:
    """LRU cache for TTS synthesis results."""

    def __init__(self, max_size: int = 200) -> None:
        self._cache: OrderedDict[str, CachedAudio] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> CachedAudio | None:
        key = text.strip().lower()
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, audio: np.ndarray, sample_rate: int,
            channels: int, duration_ms: float) -> None:
        key = text.strip().lower()
        self._cache[key] = CachedAudio(
            audio=audio, sample_rate=sample_rate,
            channels=channels, duration_ms=duration_ms,
        )
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# Common responses that should be pre-cached
_PRECACHE_PHRASES = [
    # Device confirmations
    "Done.", "Got it.", "On it.", "All set.", "You got it.",
    "Turned on the light", "Turned off the light",
    "Turned on the lights", "Turned off the lights",
    # Error/fallback
    "I'm not sure about that. Try asking differently.",
    "I heard you, but I can't reach the language model right now.",
    "Sorry, I didn't get a response.",
    "I can't do that yet.",
    # Common context answers
    "No upcoming events on the calendar.",
]


def generate_time_phrases() -> list[str]:
    """Generate time phrases for the next 10 minutes."""
    phrases = []
    now = datetime.now()
    for delta_min in range(10):
        t = now + timedelta(minutes=delta_min)
        hour = t.hour % 12 or 12
        minute = t.minute
        ampm = "AM" if t.hour < 12 else "PM"
        if minute == 0:
            phrases.append(f"It's {hour} {ampm}.")
        elif minute < 10:
            phrases.append(f"It's {hour} oh {minute} {ampm}.")
        else:
            phrases.append(f"It's {hour} {minute} {ampm}.")
    return phrases


async def prewarm_cache(cache: TTSCache, synthesize_fn, config) -> int:
    """Pre-synthesize common phrases into the cache.

    Args:
        cache: TTSCache instance
        synthesize_fn: async function(text, config) -> TTSResult
        config: TTSConfig for synthesis

    Returns number of phrases cached.
    """
    phrases = _PRECACHE_PHRASES + generate_time_phrases()
    cached = 0

    for phrase in phrases:
        if cache.get(phrase) is not None:
            continue
        try:
            result = await synthesize_fn(phrase, config)
            if result is not None:
                cache.put(phrase, result.audio, result.sample_rate,
                          result.channels, result.duration_ms)
                cached += 1
        except Exception as e:
            log.debug("precache failed for '%s': %s", phrase[:30], e)

    log.info("TTS cache prewarmed: %d phrases (total %d)", cached, cache.size)
    return cached
