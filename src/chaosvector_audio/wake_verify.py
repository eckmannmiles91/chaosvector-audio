"""Wake word verifier — speaker-specific filter on top of openWakeWord.

Uses openWakeWord's embedding model to extract features from the audio
that triggered the wake word, then runs a trained LogisticRegression
verifier to confirm it's actually the target speaker saying the wake word.

This dramatically reduces false positives from TV, conversations, etc.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class WakeVerifier:
    """Verifies wake word detections using speaker-specific embeddings."""

    def __init__(self, verifier_path: str, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._clf = None
        self._features = None

        try:
            with open(verifier_path, "rb") as f:
                self._clf = pickle.load(f)
            log.info("wake verifier loaded: %s (threshold=%.2f)", verifier_path, threshold)
        except Exception as e:
            log.warning("wake verifier unavailable: %s", e)

        # Load openWakeWord AudioFeatures for embedding extraction
        try:
            from openwakeword.utils import AudioFeatures
            self._features = AudioFeatures()
            log.info("wake verifier embedding model loaded")
        except Exception as e:
            log.warning("openwakeword AudioFeatures unavailable: %s", e)

    @property
    def is_available(self) -> bool:
        return self._clf is not None and self._features is not None

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        """Verify a wake word detection.

        Args:
            audio: int16 PCM audio (16kHz mono, ~2-3s of the wake word audio)

        Returns:
            (accepted, score) — True if verified, plus the confidence score
        """
        if not self.is_available:
            return True, 1.0  # pass through if verifier unavailable

        try:
            # Pad to at least 2s
            if len(audio) < 32000:
                audio = np.pad(audio, (0, 32000 - len(audio)))

            # Extract embedding
            emb = self._features.embed_clips(audio.reshape(1, -1))
            features = emb.flatten()

            # Trim to match training dimension
            expected_dim = self._clf.n_features_in_
            if len(features) > expected_dim:
                features = features[:expected_dim]
            elif len(features) < expected_dim:
                features = np.pad(features, (0, expected_dim - len(features)))

            # Predict
            score = self._clf.predict_proba(features.reshape(1, -1))[0, 1]
            accepted = score >= self._threshold

            if accepted:
                log.info("wake verified: score=%.3f (threshold=%.2f)", score, self._threshold)
            else:
                log.info("wake REJECTED: score=%.3f (threshold=%.2f)", score, self._threshold)

            return accepted, score

        except Exception as e:
            log.warning("wake verify error: %s", e)
            return True, 1.0  # pass through on error
