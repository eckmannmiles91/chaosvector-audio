"""Sound effects — thinking indicator and notification sounds.

Loads WAV files from pi-fi sounds directory.
"""

from __future__ import annotations

import asyncio
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_sound_cache: dict[str, tuple[np.ndarray, int]] = {}


def load_sound(name: str, sounds_dir: str) -> tuple[np.ndarray, int] | None:
    """Load a WAV file, return (int16 audio, sample_rate). Cached."""
    if name in _sound_cache:
        return _sound_cache[name]

    path = Path(sounds_dir) / f"{name}.wav"
    try:
        with wave.open(str(path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            rate = wf.getframerate()
            _sound_cache[name] = (audio, rate)
            return audio, rate
    except Exception as e:
        log.debug("sound '%s' not found: %s", name, e)
        return None


class ThinkingIndicator:
    """Plays thinking.wav after a 500ms delay (skips fast paths).

    Usage:
        thinking = ThinkingIndicator(playback, sounds_dir)
        await thinking.start()
        ... do processing ...
        await thinking.stop()  # cancels if still waiting
    """

    def __init__(self, playback, sounds_dir: str) -> None:
        self._playback = playback
        self._sounds_dir = sounds_dir
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is not None:
            return

        async def _delayed():
            try:
                await asyncio.sleep(0.5)
                sound = load_sound("thinking", self._sounds_dir)
                if sound is not None:
                    audio, rate = sound
                    from chaosvector_audio.playback import PlaybackPriority
                    await self._playback.enqueue(
                        audio, sample_rate=rate,
                        priority=PlaybackPriority.NOTIFICATION,
                        label="thinking",
                    )
            except asyncio.CancelledError:
                pass

        self._task = asyncio.create_task(_delayed())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
