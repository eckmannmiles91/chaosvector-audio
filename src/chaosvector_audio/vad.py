"""Voice activity detection — WebRTC VAD wrapper with end-of-speech."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

log = logging.getLogger(__name__)


class SpeechState(Enum):
    SILENCE = auto()
    SPEECH = auto()


@dataclass
class VADConfig:
    aggressiveness: int = 2          # 0 (least) to 3 (most aggressive filtering)
    sample_rate: int = 16000
    frame_duration_ms: int = 30      # must be 10, 20, or 30 for WebRTC VAD
    # End-of-speech: require this many consecutive silent frames
    silence_frames_threshold: int = 15   # ~450 ms at 30 ms frames
    # Minimum speech frames before we consider it real speech
    min_speech_frames: int = 3           # ~90 ms at 30 ms frames


class VoiceActivityDetector:
    """WebRTC VAD wrapper with end-of-speech detection via frame counting."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self._vad = None  # lazy import
        self._state = SpeechState.SILENCE
        self._silent_frames = 0
        self._speech_frames = 0

    def _ensure_vad(self) -> None:
        if self._vad is not None:
            return
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.config.aggressiveness)
            log.info("WebRTC VAD initialised (aggressiveness=%d)", self.config.aggressiveness)
        except ImportError:
            log.warning("webrtcvad not installed — using energy-based fallback")
            self._vad = None

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return True if the given int16 audio frame contains speech."""
        self._ensure_vad()

        if self._vad is not None:
            raw = frame.astype(np.int16).tobytes()
            return self._vad.is_speech(raw, self.config.sample_rate)

        # Fallback: energy-based detection
        rms = _rms(frame)
        return rms > 0.02  # empirical threshold

    def process_frame(self, frame: np.ndarray) -> tuple[SpeechState, bool]:
        """Process one frame and return (current_state, end_of_speech).

        end_of_speech is True exactly once when speech transitions to silence
        after enough consecutive silent frames.
        """
        speech = self.is_speech(frame)
        end_of_speech = False

        if speech:
            self._speech_frames += 1
            self._silent_frames = 0
            if (
                self._state == SpeechState.SILENCE
                and self._speech_frames >= self.config.min_speech_frames
            ):
                self._state = SpeechState.SPEECH
                log.debug("VAD: speech started")
        else:
            self._silent_frames += 1
            if self._state == SpeechState.SPEECH:
                if self._silent_frames >= self.config.silence_frames_threshold:
                    end_of_speech = True
                    self._state = SpeechState.SILENCE
                    self._speech_frames = 0
                    log.debug("VAD: end of speech")

        return self._state, end_of_speech

    def reset(self) -> None:
        self._state = SpeechState.SILENCE
        self._silent_frames = 0
        self._speech_frames = 0


def _rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    floats = samples.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(floats ** 2)))
