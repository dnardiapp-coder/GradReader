"""Input validation and sanitisation helpers for the Streamlit app."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

import streamlit as st

_ALLOWED_LEVEL_SCHEMAS = {
    "CEFR": ["A1", "A2", "B1", "B2", "C1", "C2"],
    "HSK": ["1", "2", "3", "4", "5", "6"],
    "General": ["Beginner", "Elementary", "Intermediate", "Advanced"],
}

# A curated list of common BCP-47 tags for demo purposes.
_ALLOWED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi",
]


@st.cache_data(show_spinner=False)
def allowed_levels(schema: str) -> List[str]:
    """Return the available levels for the provided schema."""

    return _ALLOWED_LEVEL_SCHEMAS.get(schema, _ALLOWED_LEVEL_SCHEMAS["CEFR"])


def sanitize_topics(raw: str, *, max_topics: int = 5) -> List[str]:
    """Split a raw comma-separated string into a list of clean topic labels."""

    if not raw:
        return []
    topics: List[str] = []
    for chunk in raw.split(","):
        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if not cleaned:
            continue
        if len(cleaned) > 40:
            cleaned = cleaned[:40].rstrip() + "…"
        topics.append(cleaned)
        if len(topics) >= max_topics:
            break
    return topics


def validate_params(params: Dict[str, Any]) -> Tuple[bool, str | None]:
    """Validate user-provided parameters before generation."""

    count = int(params.get("story_count", 1))
    if count < 1 or count > 10:
        return False, "Please choose between 1 and 10 stories."

    learning_language = params.get("learning_language")
    native_language = params.get("native_language")
    if learning_language not in _ALLOWED_LANGUAGES:
        return False, "Choose a supported learning language."
    if native_language not in _ALLOWED_LANGUAGES:
        return False, "Choose a supported native language."
    if learning_language == native_language:
        return False, "Learning and native languages should differ for best results."

    schema = params.get("level_schema", "CEFR")
    if schema not in _ALLOWED_LEVEL_SCHEMAS:
        return False, "Select a valid proficiency framework."

    level = params.get("level")
    if level not in _ALLOWED_LEVEL_SCHEMAS[schema]:
        return False, "Select a valid proficiency level."

    topics: Iterable[str] = params.get("topics", [])
    if isinstance(topics, list) and len(topics) > 5:
        return False, "Limit topics to five short phrases."

    prompt_seed = params.get("prompt_seed", "")
    if prompt_seed and len(prompt_seed) > 300:
        return False, "Additional instructions are too long. Shorten to under 300 characters."

    return True, None


__all__ = ["allowed_levels", "sanitize_topics", "validate_params"]
