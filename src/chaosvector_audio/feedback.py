"""Feedback JSONL logger — logs every interaction for eval pipeline.

Writes one JSON record per line to daily rotating files.
Schema matches satellite.py's FeedbackLogger for downstream compatibility.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class FeedbackLogger:
    """Append-only JSONL logger for voice interactions."""

    def __init__(self, log_dir: str = "/var/lib/pi-fi/feedback") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _today_path(self) -> Path:
        return self._log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    def log(self, record: dict) -> None:
        """Append a record to today's JSONL file (synchronous)."""
        if "interaction_id" not in record:
            record["interaction_id"] = str(uuid.uuid4())
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()

        try:
            line = json.dumps(record, ensure_ascii=False)
            if len(line) > 65536:
                line = line[:65536]
            with open(self._today_path(), "a") as f:
                f.write(line + "\n")
        except Exception as e:
            log.warning("feedback log write failed: %s", e)

    def log_interaction(
        self,
        *,
        transcript: str,
        intent_type: str,
        response_text: str,
        speaker: str | None = None,
        route: str = "",
        model: str = "",
        wake_rms: float = 0.0,
        stt_ms: float = 0.0,
        intent_ms: float = 0.0,
        tts_ms: float = 0.0,
        total_ms: float = 0.0,
        context_query: str | None = None,
    ) -> None:
        """Log a complete voice interaction."""
        self.log({
            "transcript": transcript,
            "intent": intent_type,
            "response": response_text[:500],
            "speaker": speaker or "unknown",
            "route": route,
            "model": model,
            "wake_rms": round(wake_rms, 1),
            "timing": {
                "stt_ms": round(stt_ms),
                "intent_ms": round(intent_ms),
                "tts_ms": round(tts_ms),
                "total_ms": round(total_ms),
            },
            "context_query": context_query,
        })
