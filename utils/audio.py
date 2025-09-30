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
            format="mp3",
        )
    except OpenAIError as exc:
        raise AudioSynthesisError("OpenAI audio synthesis failed.") from exc

    audio_bytes: Optional[bytes] = getattr(response, "read", None)
    if callable(audio_bytes):
        audio_bytes = response.read()
    else:
        audio_bytes = getattr(response, "audio", None)
    if not audio_bytes:
        raise AudioSynthesisError("Audio synthesis returned no data.")

    return audio_bytes


__all__ = ["AudioSynthesisError", "synthesize_audio"]
