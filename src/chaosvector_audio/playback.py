"""Audio playback manager — direct PipeWire sink output with priority queue."""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

class PlaybackPriority(enum.IntEnum):
    """Lower value = higher priority."""
    WAKE_BEEP = 0
    TTS = 10
    MUSIC = 20
    NOTIFICATION = 5


@dataclass(order=True)
class PlaybackItem:
    priority: int
    audio: np.ndarray = field(compare=False)       # int16
    sample_rate: int = field(compare=False, default=16000)
    channels: int = field(compare=False, default=1)
    label: str = field(compare=False, default="")


# ---------------------------------------------------------------------------
# Playback manager
# ---------------------------------------------------------------------------

@dataclass
class PlaybackConfig:
    device: str | None = None
    sample_rate: int = 22050
    channels: int = 1
    volume: float = 1.0           # 0.0 .. 1.0
    duck_volume: float = 0.3      # volume while ducking


class PlaybackManager:
    """Plays audio through PipeWire, handles priority queue and barge-in."""

    def __init__(self, config: PlaybackConfig | None = None) -> None:
        self.config = config or PlaybackConfig()
        self._queue: asyncio.PriorityQueue[PlaybackItem] = asyncio.PriorityQueue()
        self._current_task: asyncio.Task | None = None
        self._playing = asyncio.Event()
        self._cancelled = asyncio.Event()
        self._running = False
        self._volume = self.config.volume
        self._reference_cb: Callable[[np.ndarray], None] | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._cancelled.clear()
        self._current_task = asyncio.create_task(self._playback_loop())
        log.info("playback manager started")

    async def stop(self) -> None:
        self._running = False
        self._cancelled.set()
        if self._current_task is not None:
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        log.info("playback manager stopped")

    # -- public API ----------------------------------------------------------

    async def enqueue(
        self,
        audio: np.ndarray,
        sample_rate: int = 22050,
        channels: int = 1,
        priority: PlaybackPriority = PlaybackPriority.TTS,
        label: str = "",
    ) -> None:
        item = PlaybackItem(
            priority=int(priority),
            audio=audio,
            sample_rate=sample_rate,
            channels=channels,
            label=label,
        )
        await self._queue.put(item)
        log.debug("enqueued playback: %s (pri=%d)", label, priority)

    def barge_in(self) -> None:
        """Interrupt current playback immediately (wake word detected)."""
        if self._playing.is_set():
            log.info("barge-in: stopping current playback")
            self._cancelled.set()

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))

    def duck(self) -> None:
        """Lower volume for listening period."""
        self._volume = self.config.duck_volume

    def unduck(self) -> None:
        self._volume = self.config.volume

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def set_reference_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Register callback that receives a copy of every played audio block.

        Used by AEC to capture the reference signal.
        """
        self._reference_cb = cb

    # -- internal ------------------------------------------------------------

    async def _playback_loop(self) -> None:
        import sounddevice as sd  # deferred

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            self._cancelled.clear()
            self._playing.set()
            log.info("playing: %s (pri=%d)", item.label, item.priority)

            try:
                await self._play_audio(sd, item)
            except asyncio.CancelledError:
                log.debug("playback cancelled")
            finally:
                self._playing.clear()

    async def _play_audio(self, sd, item: PlaybackItem) -> None:
        """Stream audio to the output device in blocks."""
        block_size = 1024
        audio = item.audio.astype(np.float64) / 32768.0
        audio = audio * self._volume

        loop = asyncio.get_running_loop()
        done = asyncio.Event()

        def _finished_cb() -> None:
            loop.call_soon_threadsafe(done.set)

        stream = sd.OutputStream(
            device=self.config.device,
            samplerate=item.sample_rate,
            channels=item.channels,
            dtype="float32",
            blocksize=block_size,
            finished_callback=_finished_cb,
        )
        offset = 0

        with stream:
            while offset < len(audio) and not self._cancelled.is_set():
                end = min(offset + block_size, len(audio))
                block = audio[offset:end].astype(np.float32)

                # Feed reference signal to AEC
                if self._reference_cb is not None:
                    self._reference_cb(block)

                stream.write(block.reshape(-1, item.channels))
                offset = end
                # Yield control so barge-in can fire
                await asyncio.sleep(0)

        if self._cancelled.is_set():
            log.debug("playback interrupted at %.1f%%", offset / len(audio) * 100)
