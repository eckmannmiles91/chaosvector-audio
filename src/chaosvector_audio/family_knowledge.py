"""Static family knowledge for direct answers without LLM.

These are facts from the Jarvis system prompt that never change.
Answering them locally saves 3-9 seconds of LLM inference per query.
"""

from __future__ import annotations

import re

# Family data — keep in sync with the Phase 1B system prompt
FAMILY = {
    "miles": {"role": "Dad", "age": 34, "interests": "tech, cars, and alternative rock", "birthday": "September 9"},
    "jennie": {"role": "Mom", "age": 37, "interests": "reading and movies", "birthday": "March 12"},
    "sam": {"role": "child", "age": 16, "interests": "video games", "birthday": "July 24", "order": "oldest"},
    "zoey": {"role": "child", "age": 14, "interests": "crochet, bass, and cooking", "birthday": "October 31"},
    "kinzleigh": {"role": "child", "age": 13, "interests": "games and fashion", "birthday": "November 12", "nicknames": ["kinz"]},
    "lexi": {"role": "child", "age": 13, "interests": "singing and music", "birthday": "January 22"},
    "eli": {"role": "child", "age": 10, "interests": "soccer, games, and basketball", "birthday": "September 21", "order": "youngest", "full_name": "Elias"},
}

CHILDREN = ["Sam", "Zoey", "Kinzleigh", "Lexi", "Eli"]
PET = {"name": "Honey", "type": "dog"}

# Patterns for family queries
_OLDEST_RE = re.compile(r"\b(?:oldest|eldest|first.born|first\s+born)\b", re.I)
_YOUNGEST_RE = re.compile(r"\b(?:youngest|baby|littlest|last.born)\b", re.I)
_HOW_OLD_RE = re.compile(r"\b(?:how\s+old|age|what\s+age)\s+(?:is\s+)?(\w+)", re.I)
_BIRTHDAY_RE = re.compile(r"\b(?:when\s+is|what\s+is|what's)\s+(\w+?)(?:'?s)?\s+birthday", re.I)
_INTERESTS_RE = re.compile(r"\b(?:what\s+does|what\s+do)\s+(\w+)\s+(?:like|enjoy|love|do\s+for\s+fun)", re.I)
_INTERESTS_RE2 = re.compile(r"\b(?:what\s+(?:are|is))\s+(\w+?)(?:'?s)?\s+(?:interests?|hobbies?|favorites?)", re.I)
_HOW_MANY_KIDS_RE = re.compile(r"\b(?:how\s+many\s+(?:kids?|children|siblings?))", re.I)
_WHO_IS_RE = re.compile(r"\b(?:who\s+is|who's)\s+(\w+)", re.I)
_KIDS_RE = re.compile(r"\b(?:(?:name|list)\s+(?:the\s+)?(?:kids?|children)|who\s+are\s+the\s+kids)", re.I)
_PET_RE = re.compile(r"\b(?:pet|dog|cat|animal)\b", re.I)
_DAD_RE = re.compile(r"\b(?:who\s+is\s+(?:the\s+)?dad|who's\s+(?:the\s+)?dad|who\s+is\s+(?:my|the)\s+father)", re.I)
_MOM_RE = re.compile(r"\b(?:who\s+is\s+(?:the\s+)?mom|who's\s+(?:the\s+)?mom|who\s+is\s+(?:my|the)\s+mother)", re.I)


def _find_person(name: str) -> dict | None:
    """Find a family member by name (case-insensitive, supports nicknames)."""
    name_lower = name.lower()
    if name_lower in FAMILY:
        return FAMILY[name_lower]
    # Check nicknames
    for key, data in FAMILY.items():
        if name_lower in [n.lower() for n in data.get("nicknames", [])]:
            return data
    # Check "dad" / "mom"
    if name_lower in ("dad", "father", "daddy"):
        return FAMILY["miles"]
    if name_lower in ("mom", "mother", "mommy"):
        return FAMILY["jennie"]
    return None


def answer_family_question(text: str) -> str | None:
    """Try to answer a family knowledge question from static data.

    Returns a voice-friendly answer string, or None if the question
    isn't about family knowledge.
    """
    text_lower = text.lower()

    # "Who is the oldest/youngest kid?"
    if _OLDEST_RE.search(text_lower):
        return "Sam is the oldest at 16."
    if _YOUNGEST_RE.search(text_lower):
        return "Eli is the youngest at 10."

    # "How many kids?"
    if _HOW_MANY_KIDS_RE.search(text_lower):
        return f"There are five kids: {', '.join(CHILDREN[:-1])}, and {CHILDREN[-1]}."

    # "Name the kids" / "Who are the kids?"
    if _KIDS_RE.search(text_lower):
        return f"The kids are {', '.join(CHILDREN[:-1])}, and {CHILDREN[-1]}."

    # "Who is the dad/mom?"
    if _DAD_RE.search(text_lower):
        return f"Miles is the dad. He's {FAMILY['miles']['age']} and into {FAMILY['miles']['interests']}."
    if _MOM_RE.search(text_lower):
        return f"Jennie is the mom. She's {FAMILY['jennie']['age']} and into {FAMILY['jennie']['interests']}."

    # "What's the pet?" / "Do we have a dog?"
    if _PET_RE.search(text_lower):
        return f"The family dog is {PET['name']}."

    # "How old is [name]?"
    m = _HOW_OLD_RE.search(text)
    if m:
        person = _find_person(m.group(1))
        if person:
            return f"{m.group(1).title()} is {person['age']}."

    # "When is [name]'s birthday?"
    m = _BIRTHDAY_RE.search(text)
    if m:
        person = _find_person(m.group(1))
        if person:
            return f"{m.group(1).title()}'s birthday is {person['birthday']}."

    # "What does [name] like?"
    m = _INTERESTS_RE.search(text) or _INTERESTS_RE2.search(text)
    if m:
        person = _find_person(m.group(1))
        if person:
            return f"{m.group(1).title()} is into {person['interests']}."

    # "Who is [name]?"
    m = _WHO_IS_RE.search(text)
    if m:
        name = m.group(1)
        person = _find_person(name)
        if person:
            role = person["role"]
            if role == "child":
                return f"{name.title()} is {person['age']} and into {person['interests']}."
            return f"{name.title()} is {role}, {person['age']}, and into {person['interests']}."

    return None
