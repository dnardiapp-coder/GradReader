"""Streamlit app entry point for the GradReader story generator."""
from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Tuple
import zipfile

import streamlit as st
from openai import OpenAIError

from utils.ai import MissingAPIKeyError, generate_story
from utils.audio import AudioSynthesisError, synthesize_audio
from utils.pdf import stories_to_single_pdf, story_to_pdf
from utils.validators import allowed_levels, sanitize_topics, validate_params


LANGUAGE_OPTIONS: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
}


def _initialise_state() -> None:
    """Ensure required keys exist in Streamlit session state."""

    st.session_state.setdefault("stories", [])
    st.session_state.setdefault("audio", {})
    st.session_state.setdefault("pdfs", {})
    st.session_state.setdefault("combined_pdf", None)
    st.session_state.setdefault("zip_bundle", None)
    st.session_state.setdefault("last_params", None)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "-", value.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-") or "story"


@st.cache_data(show_spinner=False)
def build_zip_bundle(files: Tuple[Tuple[str, bytes], ...], manifest: str) -> bytes:
    """Create a zip archive from provided files and manifest JSON string."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in files:
            zip_file.writestr(filename, data)
        zip_file.writestr("manifest.json", manifest)
    buffer.seek(0)
    return buffer.read()


def _pdf_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "level_schema": base_params.get("level_schema"),
        "level": base_params.get("level"),
        "learning_language": LANGUAGE_OPTIONS.get(base_params.get("learning_language"), ""),
        "native_language": LANGUAGE_OPTIONS.get(base_params.get("native_language"), ""),
        "topics": base_params.get("topics", []),
        "include_glossary": base_params.get("include_glossary", False),
    }


def main() -> None:
    st.set_page_config(page_title="GradReader", layout="wide")
    _initialise_state()

    st.title("📚 GradReader")
    st.markdown(
        "Create custom graded reading stories with optional audio narration and downloadable PDFs."
    )

    with st.sidebar:
        st.header("Story Settings")
        native_language_label = st.selectbox(
            "Native language",
            options=list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda value: LANGUAGE_OPTIONS[value],
            index=list(LANGUAGE_OPTIONS.keys()).index("en"),
        )
        learning_language_label = st.selectbox(
            "Learning language",
            options=list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda value: LANGUAGE_OPTIONS[value],
            index=list(LANGUAGE_OPTIONS.keys()).index("es"),
        )

        schema = st.selectbox("Proficiency framework", options=["CEFR", "HSK", "General"], index=0)
        level = st.selectbox("Level", options=allowed_levels(schema))
        story_length = st.selectbox(
            "Story length",
            options=["short", "medium", "long"],
            help="Approximate word count for each story.",
        )
        topics_raw = st.text_input(
            "Topics", placeholder="food, travel, culture", help="Comma-separated list of topics."
        )
        include_glossary = st.checkbox("Include glossary", value=True)
        include_audio = st.checkbox("Include audio narration", value=True)
        voice = st.selectbox(
            "Narration voice",
            options=["alloy", "verse", "sol"],
            help="Voice availability depends on the chosen TTS model.",
        )
        story_count = st.slider("Number of stories", min_value=1, max_value=10, value=2)
        pdf_mode = st.radio(
            "PDF output",
            options=["combined", "split"],
            index=0,
            format_func=lambda value: "Single combined PDF" if value == "combined" else "One PDF per story",
        )

        st.header("Models")
        model_text = st.text_input("Text model", value="gpt-4o-mini")
        model_tts = st.text_input("TTS model", value="gpt-4o-mini-tts")
        temperature = st.slider("Creativity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

        generate_button = st.button("Generate stories", type="primary")

    topics = sanitize_topics(topics_raw)

    params = {
        "native_language": native_language_label,
        "learning_language": learning_language_label,
        "level_schema": schema,
        "level": level,
        "story_length": story_length,
        "topics": topics,
        "include_glossary": include_glossary,
        "include_audio": include_audio,
        "voice": voice,
        "story_count": story_count,
        "pdf_mode": pdf_mode,
        "model_text": model_text.strip(),
        "model_tts": model_tts.strip(),
        "temperature": temperature,
    }

    if generate_button:
        valid, message = validate_params(params)
        if not valid:
            st.warning(message)
        else:
            try:
                stories: List[Dict[str, Any]] = []
                audio_map: Dict[int, bytes] = {}
                pdf_map: Dict[int, bytes] = {}

                progress = st.progress(0)
                status = st.empty()

                for idx in range(story_count):
                    status.info(f"Generating story {idx + 1} of {story_count}…")
                    try:
                        story = generate_story({**params, "story_index": idx + 1})
                    except MissingAPIKeyError as exc:
                        st.error(str(exc))
                        break
                    except OpenAIError as exc:
                        st.error(f"OpenAI request failed: {exc}")
                        break
                    except ValueError as exc:
                        st.error(str(exc))
                        break

                    stories.append(story)

                    if include_audio:
                        try:
                            audio_bytes = synthesize_audio(
                                text=story["body"], voice=voice, model_tts=params["model_tts"]
                            )
                            audio_map[idx] = audio_bytes
                        except AudioSynthesisError as exc:
                            st.warning(
                                f"Audio generation for story {idx + 1} failed: {exc}. "
                                "PDF download is still available."
                            )

                    if pdf_mode == "split":
                        pdf_map[idx] = story_to_pdf(story, _pdf_params(params))

                    progress.progress((idx + 1) / story_count)

                status.empty()
                progress.empty()

                if len(stories) != story_count:
                    st.warning("Story generation halted early. Adjust settings and try again.")
                else:
                    combined_pdf = None
                    if pdf_mode == "combined":
                        combined_pdf = stories_to_single_pdf(stories, _pdf_params(params))

                    if pdf_mode == "split":
                        for idx, story in enumerate(stories):
                            if idx not in pdf_map:
                                pdf_map[idx] = story_to_pdf(story, _pdf_params(params))
                    else:
                        combined_pdf = combined_pdf or stories_to_single_pdf(
                            stories, _pdf_params(params)
                        )

                    manifest = {
                        "native_language": LANGUAGE_OPTIONS.get(params["native_language"], params["native_language"]),
                        "learning_language": LANGUAGE_OPTIONS.get(params["learning_language"], params["learning_language"]),
                        "level_schema": params["level_schema"],
                        "level": params["level"],
                        "story_length": params["story_length"],
                        "include_glossary": include_glossary,
                        "include_audio": include_audio,
                        "stories": [
                            {
                                "index": idx + 1,
                                "title": story["title"],
                                "has_audio": idx in audio_map,
                            }
                            for idx, story in enumerate(stories)
                        ],
                    }

                    files: List[Tuple[str, bytes]] = []
                    if combined_pdf:
                        files.append(("graded_reader_collection.pdf", combined_pdf))
                    if pdf_mode == "split":
                        for idx, story in enumerate(stories):
                            safe_title = _slugify(story["title"])
                            files.append((f"story_{idx + 1}_{safe_title}.pdf", pdf_map[idx]))
                    for idx, story in enumerate(stories):
                        if idx in audio_map:
                            safe_title = _slugify(story["title"])
                            files.append((f"story_{idx + 1}_{safe_title}.mp3", audio_map[idx]))

                    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
                    zip_bytes = build_zip_bundle(tuple(files), manifest_json)

                    st.session_state["stories"] = stories
                    st.session_state["audio"] = audio_map
                    st.session_state["pdfs"] = pdf_map
                    st.session_state["combined_pdf"] = combined_pdf
                    st.session_state["zip_bundle"] = zip_bytes
                    st.session_state["last_params"] = params

                    st.success("Stories generated successfully! Scroll down to review and download.")

            except MissingAPIKeyError as exc:
                st.error(str(exc))

    if st.session_state["stories"]:
        st.divider()
        st.subheader("Generated stories")

        combined_pdf_bytes = st.session_state.get("combined_pdf")
        if combined_pdf_bytes:
            st.download_button(
                "Download combined PDF",
                data=combined_pdf_bytes,
                file_name="graded_readers.pdf",
                mime="application/pdf",
                key="download_combined_pdf",
            )

        zip_bundle = st.session_state.get("zip_bundle")
        if zip_bundle:
            st.download_button(
                "Download ZIP (PDFs + audio + manifest)",
                data=zip_bundle,
                file_name="graded_reader_bundle.zip",
                mime="application/zip",
                key="download_zip_bundle",
            )

        params_state = st.session_state.get("last_params") or {}

        for idx, story in enumerate(st.session_state["stories"]):
            with st.expander(f"{idx + 1}. {story['title']}"):
                native_label = LANGUAGE_OPTIONS.get(params_state.get("native_language", ""), "")
                learning_label = LANGUAGE_OPTIONS.get(params_state.get("learning_language", ""), "")
                level_schema = params_state.get("level_schema", "")
                level_value = params_state.get("level", "")
                topic_display = ", ".join(params_state.get("topics", []))

                caption_parts = [
                    part
                    for part in [
                        f"Level: {level_schema} {level_value}".strip(),
                        f"{native_label} → {learning_label}".strip(" →"),
                        f"Topics: {topic_display}" if topic_display else "",
                    ]
                    if part
                ]
                if caption_parts:
                    st.caption(" • ".join(caption_parts))

                st.write(story["body"])
                glossary = story.get("glossary") or []
                if params_state.get("include_glossary") and glossary:
                    st.markdown("**Glossary**")
                    for entry in glossary:
                        st.write(f"- {entry['term']}: {entry['definition']}")

                audio_bytes = st.session_state["audio"].get(idx)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

                pdf_bytes = None
                if params_state.get("pdf_mode") == "split":
                    pdf_bytes = st.session_state["pdfs"].get(idx)
                elif st.session_state["combined_pdf"]:
                    pdf_bytes = None

                if pdf_bytes:
                    st.download_button(
                        "Download this story as PDF",
                        data=pdf_bytes,
                        file_name=f"story_{idx + 1}_{_slugify(story['title'])}.pdf",
                        mime="application/pdf",
                        key=f"download_story_pdf_{idx}",
                    )


if __name__ == "__main__":
    main()
