"""TTS cache — pre-synthesize common phrases for instant playback.

Caches synthesis results in memory (LRU) and optionally pre-warms
with time strings and frequent responses on startup.
Persists to disk so cache survives daemon restarts.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_CACHE_FILE = Path("/tmp/chaosvector_tts_cache.pkl")


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

    def save_to_disk(self) -> None:
        """Persist cache to disk for restart survival."""
        try:
            _CACHE_FILE.write_bytes(pickle.dumps(dict(self._cache)))
            log.info("TTS cache saved to disk: %d entries", len(self._cache))
        except Exception as e:
            log.debug("TTS cache save failed: %s", e)

    def load_from_disk(self) -> int:
        """Load cache from disk. Returns number of entries loaded."""
        try:
            if not _CACHE_FILE.exists():
                return 0
            data = pickle.loads(_CACHE_FILE.read_bytes())
            loaded = 0
            for key, entry in data.items():
                if isinstance(entry, CachedAudio):
                    self._cache[key] = entry
                    loaded += 1
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            log.info("TTS cache loaded from disk: %d entries", loaded)
            return loaded
        except Exception as e:
            log.debug("TTS cache load failed: %s", e)
            return 0


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


def generate_time_phrases(hours_ahead: int = 2) -> list[str]:
    """Generate time phrases for the next N hours.

    Covers the rolling window so time queries always hit the cache.
    Called on startup and periodically by the precache loop.
    Default 2 hours = 120 phrases, fits well within the 200-entry LRU.
    """
    phrases = []
    now = datetime.now()
    for delta_min in range(hours_ahead * 60):
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
    # Load disk cache first (instant, no TTS calls needed)
    disk_loaded = cache.load_from_disk()

    # Only synthesize phrases not already in cache (from disk or prior)
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

    # Save to disk after prewarm so next restart is fast
    if cached > 0:
        cache.save_to_disk()

    log.info("TTS cache prewarmed: %d from disk, %d synthesized (total %d)",
             disk_loaded, cached, cache.size)
    return cached
