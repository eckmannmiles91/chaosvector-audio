"""STT post-processing — name corrections and hallucination filtering.

Ported from satellite.py's _correct_stt and _is_stt_garbage.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whisper hallucination patterns
# ---------------------------------------------------------------------------

_WHISPER_HALLUCINATIONS = re.compile(
    r"(?:thank(?:s|\s+you)\s+for\s+(?:watching|listening|the\s+music)"
    r"|please\s+subscribe"
    r"|subscribe\s+to"
    r"|like\s+and\s+subscribe"
    r"|see\s+you\s+(?:next|in\s+the)"
    r"|(?:you|bye|\.){3,})",
    re.IGNORECASE,
)

_WHISPER_SHORT_GARBAGE = frozenset({
    "you know", "yeah", "yes", "no", "okay", "ok",
    "that's me", "so", "um", "uh", "hmm", "huh",
    "i'm sorry", "thank you", "oh", "right", "sure",
    "please", "b", "the", "a", "i", "it", "and",
    "what", "but", "is", "was", "that", "this",
    "bye", "hey", "hi", "well", "now", "then",
    "music", "thanks", "good", "great", "nice",
})


def is_stt_garbage(text: str) -> bool:
    """Return True if the transcript is a hallucination or garbage."""
    if not text:
        return True
    clean = text.strip().rstrip(".!?,").lower()
    if clean in _WHISPER_SHORT_GARBAGE:
        return True
    if _WHISPER_HALLUCINATIONS.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Name corrections
# ---------------------------------------------------------------------------

_STT_NAME_MAP: dict[str, str] = {
    # Kinzleigh
    "kimsley": "Kinzleigh", "kinsley": "Kinzleigh", "kensley": "Kinzleigh",
    "kinzley": "Kinzleigh", "kinslee": "Kinzleigh", "kinslieh": "Kinzleigh",
    "kinslie": "Kinzleigh", "kinsley's": "Kinzleigh's", "kimsley's": "Kinzleigh's",
    # Zoey
    "sewy": "Zoey", "sewing": "Zoey", "zo": "Zoey", "zoe": "Zoey",
    "zoey's": "Zoey's", "zoe's": "Zoey's",
    # Jennie
    "genny": "Jennie", "genie": "Jennie", "jenny": "Jennie", "ginny": "Jennie",
    "genny's": "Jennie's", "jenny's": "Jennie's",
    # Lexi
    "lexia": "Lexi", "lex": "Lexi", "lexy": "Lexi",
    "lexia's": "Lexi's", "lexi's": "Lexi's",
    # Eli
    "ellie": "Eli", "ally": "Eli", "eli's": "Eli's",
    # Miles
    "mile's": "Miles's",
    # Product names
    "clicks": "Plex", "plecks": "Plex", "flex": "Plex",
}

_STT_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_STT_NAME_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Context-aware fixes
_MUSIC_WORDS = re.compile(r"\b(?:music|song|playlist|album|artist|track|play)\b", re.I)
_PLEASE_PLAY = re.compile(r"^please\s+", re.I)

# General word fixes
_WORD_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\ba\s+marrow\b", re.I), "tomorrow"),
]


def correct_stt(text: str) -> str:
    """Apply name corrections and context-aware STT fixes."""
    original = text

    # 1. Word-level name normalization
    def _name_replace(m: re.Match) -> str:
        return _STT_NAME_MAP[m.group(0).lower()]

    corrected = _STT_NAME_PATTERN.sub(_name_replace, text)
    if corrected != text:
        log.info("STT name fix: \"%s\" → \"%s\"", text, corrected)
        text = corrected

    # 2. General word fixes
    for pattern, replacement in _WORD_FIXES:
        text = pattern.sub(replacement, text)

    # 3. Context-aware: "please music" → "play music"
    if _PLEASE_PLAY.match(text) and _MUSIC_WORDS.search(text):
        text = _PLEASE_PLAY.sub("play ", text)

    if text != original:
        log.info("STT corrected: \"%s\" → \"%s\"", original, text)

    return text
