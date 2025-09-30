"""Audio synthesis helpers using the OpenAI text-to-speech API."""
from __future__ import annotations

from typing import Optional

from openai import OpenAIError

from utils.ai import get_openai_client


class AudioSynthesisError(RuntimeError):
    """Raised when audio synthesis fails."""


def synthesize_audio(text: str, voice: str, model_tts: str) -> bytes:
    """Synthesize speech audio for the provided text using OpenAI's API.

    Parameters
    ----------
    text:
        The text to convert into speech.
    voice:
        The voice identifier supported by the chosen TTS model.
    model_tts:
        The OpenAI TTS model name.

    Returns
    -------
    bytes
        Raw MP3 audio bytes.

    Raises
    ------
    AudioSynthesisError
        If the TTS request fails or returns no audio.
    """

    if not text.strip():
        raise AudioSynthesisError("Cannot synthesize audio for empty text.")

    client = get_openai_client()

    try:
        response = client.audio.speech.create(
            model=model_tts,
            voice=voice,
            input=text,
        )
    except OpenAIError as exc:
        raise AudioSynthesisError("OpenAI audio synthesis failed.") from exc

    audio_bytes: Optional[bytes] = None

    # Try the most common response attributes from the OpenAI SDK. The
    # response objects differ slightly between versions, so we defensively
    # check a handful of possibilities before raising an error.
    if hasattr(response, "content") and response.content:
        audio_bytes = response.content  # type: ignore[assignment]
    elif hasattr(response, "audio") and response.audio:
        audio_bytes = response.audio  # type: ignore[assignment]
    elif hasattr(response, "read") and callable(response.read):
        audio_bytes = response.read()
    elif hasattr(response, "to_bytes") and callable(response.to_bytes):
        audio_bytes = response.to_bytes()
    elif hasattr(response, "data") and response.data:
        first = response.data[0]
        audio_bytes = getattr(first, "audio", None)

    if not audio_bytes:
        raise AudioSynthesisError("Audio synthesis returned no data.")

    return audio_bytes


__all__ = ["AudioSynthesisError", "synthesize_audio"]
