"""Shadow wake word detector — runs ChaosVector Wake ONNX model in parallel
with openWakeWord for comparison. Logs its decisions but doesn't control
the pipeline.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import math
import time
import array as _array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CLIP_DURATION = 2.0
N_MELS = 40
HOP_LENGTH = 160
N_FFT = 400
TARGET_FRAMES = int(SAMPLE_RATE * CLIP_DURATION / HOP_LENGTH)  # 200 frames


@dataclass
class ShadowWakeConfig:
    model_path: str = "/home/chaos/chaosvector-audio/model/chaosvector-wake.onnx"
    threshold: float = 0.7        # sigmoid probability threshold
    trigger_level: int = 2        # consecutive positive detections required
    energy_threshold: float = 350.0
    chunk_ms: int = 20


class ShadowWakeDetector:
    """Runs the ChaosVector Wake ONNX model as a shadow alongside openWakeWord."""

    def __init__(self, config: ShadowWakeConfig | None = None):
        self.config = config or ShadowWakeConfig()
        self._session = None
        self._running = False
        self._consecutive = 0
        self._muted = False  # Set True during TTS playback to prevent self-hearing
        self._mute_until: float = 0.0  # monotonic timestamp — muted until this time
        # Rolling audio buffer — last 2 seconds of audio
        self._audio_buf = collections.deque(maxlen=int(SAMPLE_RATE * CLIP_DURATION))
        self._wake_event = asyncio.Event()
        self._wake_rms: float = 0.0

    def mute(self, duration: float = 2.0) -> None:
        """Mute detection for duration seconds (call before TTS playback)."""
        import time
        self._mute_until = time.monotonic() + duration
        self._consecutive = 0
        self._audio_buf.clear()

    @property
    def is_muted(self) -> bool:
        import time
        return time.monotonic() < self._mute_until

    def load(self) -> bool:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                self.config.model_path, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            log.info("shadow wake model loaded: %s", self.config.model_path)
            return True
        except Exception as e:
            log.warning("shadow wake model load failed: %s", e)
            return False

    async def start(self, audio_queue: asyncio.Queue) -> None:
        """Run shadow detection loop. Consumes audio chunks from the queue."""
        if self._session is None:
            if not self.load():
                return
        self._running = True
        self._consecutive = 0
        chunk_count = 0

        while self._running:
            try:
                chunk_bytes = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if chunk_bytes is None:
                break

            # Accumulate audio
            samples = _array.array("h")
            samples.frombytes(chunk_bytes)
            self._audio_buf.extend(samples)
            chunk_count += 1

            # Run inference every 500ms (25 chunks at 20ms)
            if chunk_count % 25 != 0:
                continue

            # Echo gate — skip during/after TTS playback
            if self.is_muted:
                self._consecutive = 0
                continue

            # RMS energy check
            buf_array = np.array(self._audio_buf, dtype=np.float32)
            rms = math.sqrt(np.mean(buf_array ** 2))
            if rms < self.config.energy_threshold:
                self._consecutive = 0
                continue

            # Compute mel spectrogram
            mel = self._compute_mel(buf_array / 32768.0)
            if mel is None:
                continue

            # Run ONNX inference
            t0 = time.perf_counter()
            logits = self._session.run(None, {"mel_spectrogram": mel})[0]
            prob = 1.0 / (1.0 + np.exp(-logits[0]))  # sigmoid
            infer_ms = (time.perf_counter() - t0) * 1000

            if prob >= self.config.threshold:
                self._consecutive += 1
                if self._consecutive >= self.config.trigger_level:
                    log.info("SHADOW WAKE: detected (prob=%.3f, rms=%.0f, infer=%.1fms)",
                             prob, rms, infer_ms)
                    self._wake_rms = rms
                    self._wake_event.set()
                    self._consecutive = 0
            else:
                if self._consecutive > 0:
                    log.debug("shadow wake: reset (prob=%.3f < %.2f)", prob, self.config.threshold)
                self._consecutive = 0

    async def wait_for_wake(self) -> tuple[str, float]:
        """Block until wake word is detected. Returns (name, rms)."""
        self._wake_event.clear()
        await self._wake_event.wait()
        return "hey_jarvis", self._wake_rms

    def has_pending_wake(self) -> bool:
        """Non-blocking check if a wake event fired (for barge-in detection)."""
        if self._wake_event.is_set():
            self._wake_event.clear()
            return True
        return False

    def stop(self):
        self._running = False

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray | None:
        """Compute log mel spectrogram matching the training pipeline."""
        try:
            # Pad/trim to target length
            target_len = int(SAMPLE_RATE * CLIP_DURATION)
            if len(audio) > target_len:
                audio = audio[-target_len:]  # take last 2s
            elif len(audio) < target_len:
                audio = np.pad(audio, (target_len - len(audio), 0))

            # STFT
            n_fft = N_FFT
            hop = HOP_LENGTH
            window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

            frames = []
            for start in range(0, len(audio) - n_fft + 1, hop):
                frame = audio[start:start + n_fft] * window
                spectrum = np.abs(np.fft.rfft(frame)) ** 2
                frames.append(spectrum)

            if not frames:
                return None

            power = np.array(frames).T  # (n_fft//2+1, time)

            # Mel filterbank
            mel_filters = self._mel_filterbank(SAMPLE_RATE, n_fft, N_MELS)
            mel = mel_filters @ power  # (n_mels, time)
            mel = np.log(mel + 1e-9)

            # Pad/trim time dimension to match training
            if mel.shape[1] > TARGET_FRAMES:
                mel = mel[:, :TARGET_FRAMES]
            elif mel.shape[1] < TARGET_FRAMES:
                mel = np.pad(mel, ((0, 0), (0, TARGET_FRAMES - mel.shape[1])))

            # Shape: (1, 1, n_mels, time)
            return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)

        except Exception as e:
            log.debug("mel computation failed: %s", e)
            return None

    @staticmethod
    def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Create a mel filterbank matrix."""
        low_freq = 0.0
        high_freq = sr / 2.0
        low_mel = 2595.0 * np.log10(1.0 + low_freq / 700.0)
        high_mel = 2595.0 * np.log10(1.0 + high_freq / 700.0)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_mels):
            for j in range(bins[i], bins[i + 1]):
                filters[i, j] = (j - bins[i]) / max(bins[i + 1] - bins[i], 1)
            for j in range(bins[i + 1], bins[i + 2]):
                filters[i, j] = (bins[i + 2] - j) / max(bins[i + 2] - bins[i + 1], 1)

        return filters
