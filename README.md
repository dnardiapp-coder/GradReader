# GradReader

GradReader is a Streamlit application that generates graded reading stories, audio narration, and ready-to-share PDFs for language learners. The app wraps OpenAI's text generation and text-to-speech APIs to build mini story packs tailored to a learner's native language, target language, proficiency level, and favourite topics.

## Features

- Generate 1–10 stories in the target language with level-aware prompts.
- Produce individual PDFs per story or a single combined PDF with a table of contents.
- Download a ZIP archive containing PDFs, MP3 audio files, and a manifest.
- Optional bilingual glossary aligned to the reader's native language.
- Paragraph-by-paragraph reading support with phonetic guides, translations, and grammar notes.
- Unicode-friendly typography with automatic discovery (and optional on-demand download) of Noto/Source Han fonts for global language support.
- Streamlit session state keeps generated artefacts available across reruns.

## Getting Started

### Prerequisites

- Python 3.10 or later
- An [OpenAI API key](https://platform.openai.com/) with access to a chat-completions capable model and a text-to-speech model

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you use environment files, or export the variable directly:

```bash
export OPENAI_API_KEY="sk-..."
```

### Running the App Locally

```bash
streamlit run streamlit_app.py
```

The app will launch in your browser at `http://localhost:8501`.

### Deployment on Streamlit Cloud

1. Fork or clone this repository.
2. Push it to your GitHub account.
3. In Streamlit Cloud, create a new app and point it at `streamlit_app.py`.
4. Set the `OPENAI_API_KEY` secret in the app's **Advanced settings → Secrets** panel.
5. Deploy — Streamlit will install dependencies from `requirements.txt` automatically.

### Model Selection and Cost

The sidebar lets you pick separate models for text generation (`model_text`) and speech synthesis (`model_tts`). Lower-cost models and shorter story lengths reduce API usage. Audio generation can be disabled to save credits.

### Fonts and Licensing

When Unicode-capable fonts are missing, the app now attempts to download Noto Sans/Serif CJK fonts into `assets/fonts` automatically (internet access required). You can also place your own fonts in that directory or install them system-wide, and GradReader will discover them on the next run. This ensures PDFs render Chinese, Japanese, Korean, and other non-Latin scripts without fallback artifacts. All suggested fonts are distributed under the [SIL Open Font License](https://scripts.sil.org/OFL).

### Troubleshooting

- **Missing API key**: The app displays a friendly error if `OPENAI_API_KEY` is absent.
- **TTS unavailable**: Audio generation failures surface as warnings, while PDF downloads remain available.
- **Token usage**: Keep the story count and length modest to stay within token limits.

## License

This project is released under the MIT License. See `LICENSE` for details.
