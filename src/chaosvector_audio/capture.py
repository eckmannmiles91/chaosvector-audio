"""Audio capture manager — ALSA/PipeWire mic capture with ring buffer."""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioChunk:
    """A timestamped chunk of captured audio."""

    samples: np.ndarray          # int16, shape (frames,) or (frames, channels)
    sample_rate: int
    channels: int
    timestamp: float             # monotonic time at capture start
    rms: float                   # RMS energy of this chunk

    @property
    def duration_ms(self) -> float:
        return len(self.samples) / self.sample_rate * 1000


# ---------------------------------------------------------------------------
# Ring buffer for pre-roll
# ---------------------------------------------------------------------------

class PreRollBuffer:
    """Fixed-duration ring buffer that retains the last N ms of audio."""

    def __init__(self, duration_ms: int, sample_rate: int, channels: int) -> None:
        self._max_frames = int(sample_rate * duration_ms / 1000)
        self._sample_rate = sample_rate
        self._channels = channels
        self._buf: collections.deque[AudioChunk] = collections.deque()
        self._total_frames = 0

    def push(self, chunk: AudioChunk) -> None:
        self._buf.append(chunk)
        self._total_frames += len(chunk.samples)
        while self._total_frames > self._max_frames and len(self._buf) > 1:
            evicted = self._buf.popleft()
            self._total_frames -= len(evicted.samples)

    def drain(self) -> list[AudioChunk]:
        """Return all buffered chunks and clear the buffer."""
        chunks = list(self._buf)
        self._buf.clear()
        self._total_frames = 0
        return chunks


# ---------------------------------------------------------------------------
# Capture manager
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    device: str | None = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30       # ~30 ms frames, good for VAD
    pre_roll_ms: int = 500
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.int16))


class CaptureManager:
    """Manages audio capture from ALSA/PipeWire and exposes an async stream."""

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self.config = config or CaptureConfig()
        self._stream = None                     # sounddevice.InputStream (lazy)
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=200)
        self._pre_roll = PreRollBuffer(
            self.config.pre_roll_ms,
            self.config.sample_rate,
            self.config.channels,
        )
        self._running = False

    # -- lifecycle -----------------------------------------------------------

    async def open(self) -> None:
        """Open the capture device and begin reading audio."""
        import sounddevice as sd  # deferred so import errors surface clearly

        frames_per_chunk = int(
            self.config.sample_rate * self.config.chunk_duration_ms / 1000
        )
        self._running = True

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                log.warning("capture status: %s", status)
            samples = indata[:, 0].copy() if self.config.channels == 1 else indata.copy()
            rms = _compute_rms(samples)
            chunk = AudioChunk(
                samples=samples.astype(np.int16),
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                timestamp=time.monotonic(),
                rms=rms,
            )
            self._pre_roll.push(chunk)
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                log.warning("capture queue full, dropping chunk")

        self._stream = sd.InputStream(
            device=self.config.device,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            blocksize=frames_per_chunk,
            callback=_callback,
        )
        self._stream.start()
        log.info(
            "capture open: device=%s rate=%d ch=%d chunk=%dms",
            self.config.device,
            self.config.sample_rate,
            self.config.channels,
            self.config.chunk_duration_ms,
        )

    async def close(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Unblock any consumer waiting on the queue
        await self._queue.put(None)

    # -- stream interface ----------------------------------------------------

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks as they arrive from the capture device."""
        while self._running:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk

    def drain_pre_roll(self) -> list[AudioChunk]:
        """Return buffered pre-roll audio (call after wake word fires)."""
        return self._pre_roll.drain()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rms(samples: np.ndarray) -> float:
    """Return RMS energy of int16 samples, normalised to 0.0-1.0."""
    if len(samples) == 0:
        return 0.0
    floats = samples.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(floats ** 2)))
