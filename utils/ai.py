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
        "Include a short glossary of 5-8 important words translated into the "
        f"reader's native language ({params.get('native_language')})."
        if params.get("include_glossary")
        else "Do not include a glossary."
    )

    length_instruction = {
        "short": "about 150-250 words",
        "medium": "about 300-450 words",
        "long": "about 500-700 words",
    }.get(params.get("story_length", "medium"), "about 300-450 words")

    storyline_goal = params.get("story_goal", "Engage the reader with a positive tone.")

    prompt = f"""
You are an expert language teacher creating graded readers for learners.
Write a story in {params.get('learning_language')} that matches the following constraints:
- Level schema: {params.get('level_schema')} level {params.get('level')}
- Reader's native language: {params.get('native_language')}
- Story length: {length_instruction}
- Topics: {topic_text}
- Number of paragraphs: 4-6 with short sentences for lower levels.
- Maintain cultural neutrality and avoid idioms or slang unless it is level-appropriate.
- Provide a concise and descriptive title.
- Ensure the language strictly uses {params.get('learning_language')} without switching languages.
- {storyline_goal}
- Keep paragraphs short (max 4 sentences) and add line breaks between paragraphs.
{glossary_instruction}

Respond ONLY in valid JSON with the following structure:
{{
  "title": "...",
  "story": "Story body in {params.get('learning_language')} with paragraph breaks",
  "glossary": [
    {{"term": "", "definition": "translation in {params.get('native_language')}"}}
  ]  // optional, omit or use [] if not requested
}}
"""

    return prompt.strip()


def _parse_story_payload(payload: str) -> Dict[str, Any]:
    """Parse a JSON payload returned by the model into a dictionary."""

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response was not valid JSON.") from exc

    title = data.get("title", "Untitled Story").strip()
    body = data.get("story", "").strip()
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

    return {"title": title, "body": body, "glossary": cleaned_glossary}


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

    response = client.chat.completions.create(
        model=params.get("model_text", "gpt-4o-mini"),
        temperature=temperature,
        top_p=top_p,
        max_tokens=int(params.get("max_tokens", 1200)),
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

    content = response.choices[0].message.content
    if not content:
        raise ValueError("No content returned by the model.")

    story = _parse_story_payload(content)
    return story


__all__ = [
    "MissingAPIKeyError",
    "build_story_prompt",
    "generate_story",
    "get_openai_client",
]
