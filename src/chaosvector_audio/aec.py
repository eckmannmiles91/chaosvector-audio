"""Echo cancellation — reference signal capture and echo gating."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AECConfig:
    # Duration (ms) after playback ends during which mic input is suppressed
    echo_tail_ms: int = 200
    # Use hardware AEC passthrough (e.g. XVF3800) instead of software
    hardware_aec: bool = False
    # WebRTC AEC3 filter length in ms
    filter_length_ms: int = 128


# ---------------------------------------------------------------------------
# Echo canceller
# ---------------------------------------------------------------------------

class EchoCanceller:
    """Manages echo cancellation between playback and capture paths.

    Two modes:
      1. Hardware AEC passthrough (XVF3800) — mic already has AEC applied,
         this module only provides the echo gate.
      2. Software AEC via WebRTC AEC3 — captures reference signal from
         playback and applies cancellation to mic frames.
    """

    def __init__(self, config: AECConfig | None = None) -> None:
        self.config = config or AECConfig()
        self._last_playback_end: float = 0.0
        self._reference_buf: list[np.ndarray] = []
        self._aec_processor = None  # lazy init for WebRTC AEC3

    # -- reference signal (called from playback path) -----------------------

    def feed_reference(self, block: np.ndarray) -> None:
        """Called by PlaybackManager for each played audio block."""
        self._reference_buf.append(block.copy())
        self._last_playback_end = time.monotonic()

    def notify_playback_stopped(self) -> None:
        self._last_playback_end = time.monotonic()
        self._reference_buf.clear()

    # -- echo gate -----------------------------------------------------------

    @property
    def echo_active(self) -> bool:
        """True if we are within the echo tail window after playback."""
        if self._last_playback_end == 0.0:
            return False
        elapsed_ms = (time.monotonic() - self._last_playback_end) * 1000
        return elapsed_ms < self.config.echo_tail_ms

    def should_suppress_stt(self) -> bool:
        """Return True if STT input should be suppressed due to echo."""
        return self.echo_active

    # -- frame processing ----------------------------------------------------

    def process_capture_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a capture frame through AEC.

        For hardware AEC mode, returns the frame unchanged.
        For software AEC, applies WebRTC AEC3 cancellation.
        """
        if self.config.hardware_aec:
            return frame

        return self._software_aec(frame)

    def _software_aec(self, frame: np.ndarray) -> np.ndarray:
        """Apply software echo cancellation.

        TODO: Integrate webrtc-audio-processing Python bindings.
        For now, applies simple echo gate (mute during echo tail).
        """
        if self.echo_active:
            log.debug("echo gate active — zeroing capture frame")
            return np.zeros_like(frame)
        return frame

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        self._reference_buf.clear()
        self._last_playback_end = 0.0
