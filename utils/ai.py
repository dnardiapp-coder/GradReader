"""Utilities for interacting with OpenAI models for story generation."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI
from openai import OpenAIError


class MissingAPIKeyError(RuntimeError):
    """Raised when the OpenAI API key cannot be located."""


@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client using the configured API key.

    Raises
    ------
    MissingAPIKeyError
        If ``OPENAI_API_KEY`` cannot be located in environment variables or
        Streamlit secrets.
    """

    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not configured. Set it as an environment variable "
            "or add it to Streamlit secrets."
        )
    return OpenAI(api_key=api_key)


def build_story_prompt(params: Dict[str, Any]) -> str:
    """Construct a grading-aware prompt for the model.

    Parameters
    ----------
    params:
        Dictionary containing the story generation parameters.

    Returns
    -------
    str
        The prompt string to send to the language model.
    """

    topics: List[str] = params.get("topics", [])
    topic_text = "\n- ".join(topics) if topics else "None specified"

    glossary_instruction = (
        "Include a short glossary of 6-10 important words translated into the "
        f"reader's native language ({params.get('native_language')})."
        if params.get("include_glossary")
        else "Do not include a glossary section."
    )

    length_instruction = {
        "short": "around 280-380 words",
        "medium": "around 600-800 words (aim for a full A4 page when typeset)",
        "long": "around 800-1000 words",
    }.get(params.get("story_length", "medium"), "around 600-800 words")

    paragraph_instruction = {
        "short": "6-7",
        "medium": "8-9",
        "long": "9-11",
    }.get(params.get("story_length", "medium"), "8-9")

    phonetics_instruction = (
        "Use the standard phonetic guide for the learning language (e.g. Pinyin for Chinese, "
        "Romaji for Japanese, Revised Romanization for Korean). If the language already uses "
        "the Latin alphabet, provide syllable-level chunking with stress hints instead."
    )

    storyline_goal = params.get("story_goal", "Engage the reader with a positive tone.")

    prompt = f"""
You are an expert language teacher creating graded readers for learners.
Write a story in {params.get('learning_language')} that matches the following constraints:
- Level schema: {params.get('level_schema')} level {params.get('level')}
- Reader's native language: {params.get('native_language')}
- Story length: {length_instruction}
- Topics: {topic_text}
- Number of story paragraphs: {paragraph_instruction} with clear transitions.
- Maintain cultural neutrality and avoid idioms or slang unless it is level-appropriate.
- Provide a concise and descriptive title.
- Ensure the language strictly uses {params.get('learning_language')} without switching languages.
- {storyline_goal}
- Keep paragraphs short (max 4 sentences) and add line breaks between paragraphs.
- Provide paragraph-by-paragraph support materials described below.
- {glossary_instruction}
- Add 2-3 grammar or usage notes that highlight level-appropriate structures from the story.
- Add 2-3 suggested follow-up practice activities such as comprehension prompts or extension tasks.

Respond ONLY in valid JSON with the following structure:
{{
  "title": "...",
  "summary": "1-2 sentence overview in {params.get('native_language')} describing the plot and learning focus.",
  "reading_sections": [
    {{
      "original": "Paragraph of the story in {params.get('learning_language')}",
      "phonetics": "Matching paragraph rendered in phonetics ({phonetics_instruction})",
      "translation": "Paragraph translated to {params.get('native_language')}"
    }}
  ],
  "glossary": [
    {{"term": "", "definition": "translation in {params.get('native_language')}"}}
  ],
  "grammar_notes": ["Brief bullet explaining a grammar point with examples"],
  "practice_ideas": ["Suggestion for further practice or reflection question"],
  "culture_or_strategy_notes": ["Optional learning strategies or cultural insights that support the graded reader"]
}}
"""

    return prompt.strip()


def _repair_json_payload(payload: str) -> Dict[str, Any]:
    """Attempt to coerce an LLM payload into valid JSON.

    The OpenAI ``response_format`` safeguard generally keeps responses valid,
    but occasionally the model can still prepend/appended stray text (for
    example when it emits warnings). We take a forgiving approach by slicing
    out the innermost JSON object and retrying the parse before raising an
    error back to the caller. The original payload is preserved for debugging
    via the chained exception message.
    """

    text = (payload or "").strip()

    # Fast path – most responses are already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to salvage the first JSON object if the model added narration
    # such as "Here is the story" before the structured payload.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Model response was not valid JSON.")


def _parse_story_payload(payload: str) -> Dict[str, Any]:
    """Parse a JSON payload returned by the model into a dictionary."""

    try:
        data = _repair_json_payload(payload)
    except ValueError as exc:
        # Surface a shortened preview of the problematic payload to help the
        # user troubleshoot without overwhelming the UI.
        preview = (payload or "").strip().splitlines()
        preview_text = " ".join(preview)[:280]
        raise ValueError(
            f"Model response was not valid JSON. Received: {preview_text}"
        ) from exc

    title = data.get("title", "Untitled Story").strip()
    summary = str(data.get("summary", "")).strip()

    sections_raw = data.get("reading_sections") or []
    reading_sections: List[Dict[str, str]] = []
    body_paragraphs: List[str] = []
    translation_paragraphs: List[str] = []

    if isinstance(sections_raw, list):
        for section in sections_raw:
            if not isinstance(section, dict):
                continue
            original = str(section.get("original", "")).strip()
            phonetics = str(section.get("phonetics", "")).strip()
            translation = str(section.get("translation", "")).strip()
            if not original:
                continue
            reading_sections.append(
                {
                    "original": original,
                    "phonetics": phonetics,
                    "translation": translation,
                }
            )
            body_paragraphs.append(original)
            if translation:
                translation_paragraphs.append(translation)

    glossary = data.get("glossary") or []

    # Ensure glossary is a list of dictionaries with required keys
    cleaned_glossary: List[Dict[str, str]] = []
    if isinstance(glossary, list):
        for entry in glossary:
            if not isinstance(entry, dict):
                continue
            term = str(entry.get("term", "")).strip()
            definition = str(entry.get("definition", "")).strip()
            if term and definition:
                cleaned_glossary.append({"term": term, "definition": definition})

    grammar_notes_raw = data.get("grammar_notes") or []
    grammar_notes: List[str] = []
    if isinstance(grammar_notes_raw, list):
        for note in grammar_notes_raw:
            note_text = str(note).strip()
            if note_text:
                grammar_notes.append(note_text)

    practice_raw = data.get("practice_ideas") or []
    practice_ideas: List[str] = []
    if isinstance(practice_raw, list):
        for idea in practice_raw:
            idea_text = str(idea).strip()
            if idea_text:
                practice_ideas.append(idea_text)

    extras_raw = data.get("culture_or_strategy_notes") or []
    extra_notes: List[str] = []
    if isinstance(extras_raw, list):
        for entry in extras_raw:
            entry_text = str(entry).strip()
            if entry_text:
                extra_notes.append(entry_text)

    body = "\n\n".join(body_paragraphs).strip()
    full_translation = "\n\n".join(translation_paragraphs).strip()

    return {
        "title": title,
        "summary": summary,
        "body": body,
        "reading_sections": reading_sections,
        "translation": full_translation,
        "glossary": cleaned_glossary,
        "grammar_notes": grammar_notes,
        "practice_ideas": practice_ideas,
        "extra_notes": extra_notes,
    }


def generate_story(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a graded reader story using the OpenAI API.

    Parameters
    ----------
    params:
        Story generation parameters that include user preferences and model
        configuration.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ``title``, ``body``, and an optional ``glossary``.

    Raises
    ------
    OpenAIError
        When the API request fails for any reason.
    ValueError
        When the returned payload cannot be parsed as JSON.
    """

    client = get_openai_client()
    prompt = build_story_prompt(params)

    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 0.95))

    def _suggested_token_budget() -> int:
        story_length = str(params.get("story_length", "medium")).lower()
        base_budget = {
            # Readers now include phonetics, translations, and scaffolding for
            # every paragraph, which triples the footprint compared to the
            # original story text. Start with more generous budgets so the
            # model has room to complete the JSON payload.
            "short": 3200,
            "medium": 4800,
            "long": 6000,
        }.get(story_length, 4800)

        # Longer responses are needed when auxiliary study aids are included.
        extras = 600 if params.get("include_glossary") else 300
        # Grammar notes, practice ideas, and cultural strategies are always requested.
        extras += 650

        # Cap the budget to stay within the model's context window while ensuring
        # we request enough space for well-formed JSON.
        return max(2000, min(base_budget + extras, 7200))

    max_tokens = int(params.get("max_tokens") or _suggested_token_budget())

    attempt = 0
    max_attempts = 5
    last_error: Optional[Exception] = None
    while attempt < max_attempts:
        response = client.chat.completions.create(
            model=params.get("model_text", "gpt-4o-mini"),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a supportive language tutor and expert storyteller "
                        "who writes engaging graded readers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", "stop")
        if finish_reason and finish_reason != "stop":
            if finish_reason == "length" and attempt + 1 < max_attempts:
                max_tokens = min(max_tokens + 1400, 7200)
                attempt += 1
                last_error = ValueError(
                    "Model stopped early because the response hit the token limit; retrying with a larger budget."
                )
                continue

            raise ValueError(
                "Story generation halted early. Increase the max tokens setting or simplify the prompt."
            )

        content = choice.message.content if choice.message else None
        if not content:
            raise ValueError("No content returned by the model.")

        try:
            story = _parse_story_payload(content)
            return story
        except ValueError as exc:
            last_error = exc

            # If the payload is obviously truncated (e.g. missing a closing brace),
            # treat it like a length issue and substantially expand the token
            # budget before retrying.
            stripped = (content or "").strip()
            truncated = bool(stripped) and (
                not stripped.endswith("}")
                or stripped.count("{") > stripped.count("}")
                or stripped.endswith("\"")
            )

            if attempt + 1 < max_attempts:
                if truncated:
                    max_tokens = min(max_tokens + 1600, 7200)
                else:
                    max_tokens = min(int(max_tokens * 1.4), 7200)
                attempt += 1
                continue
            raise

    if last_error:
        raise last_error

    raise ValueError("Story generation failed for an unknown reason.")


__all__ = [
    "MissingAPIKeyError",
    "build_story_prompt",
    "generate_story",
    "get_openai_client",
]
