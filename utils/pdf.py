"""PDF generation helpers for graded reader stories."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from urllib.request import urlopen

import streamlit as st
from fpdf import FPDF
from fpdf.errors import FPDFException


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

FONT_SEARCH_PATHS = [
    ASSETS_DIR,
    Path.home() / "fonts",
    Path.home() / ".fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("C:/Windows/Fonts"),
]

FONT_CANDIDATES = {
    "sans": [
        "NotoSansCJKsc-Regular.otf",
        "NotoSansCJK-Regular.ttc",
        "NotoSansSC-Regular.otf",
        "SourceHanSansSC-Regular.otf",
        "NotoSans-Regular.ttf",
        "DejaVuSans.ttf",
    ],
    "serif": [
        "NotoSerifCJKsc-Regular.otf",
        "SourceHanSerifSC-Regular.otf",
        "NotoSerif-Regular.ttf",
        "DejaVuSerif.ttf",
    ],
}

FONT_DOWNLOADS: Dict[str, Iterable[str]] = {
    "sans": (
        # Simplified Chinese Sans
        "https://github.com/googlefonts/noto-cjk/raw/refs/heads/main/Sans/OTF/"
        "SimplifiedChinese/NotoSansSC-Regular.otf",
        # General-purpose Unicode sans as a fallback
        "https://github.com/googlefonts/noto-fonts/raw/refs/heads/main/unhinted/"
        "otf/NotoSans/NotoSans-Regular.otf",
    ),
    "serif": (
        # Simplified Chinese Serif
        "https://github.com/googlefonts/noto-cjk/raw/refs/heads/main/Serif/OTF/"
        "SimplifiedChinese/NotoSerifSC-Regular.otf",
        # General-purpose Unicode serif fallback
        "https://github.com/googlefonts/noto-fonts/raw/refs/heads/main/unhinted/"
        "otf/NotoSerif/NotoSerif-Regular.otf",
    ),
}


class StoryPDF(FPDF):
    """FPDF subclass that renders footers with page numbers."""

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.alias_nb_pages()
        # Default to built-in fonts so the PDF can still be generated if
        # bundled fonts are unavailable (e.g. when running without Git LFS).
        self.sans_family = "Helvetica"
        self.serif_family = "Times"
        # Track whether we were able to register Unicode-capable fonts so we
        # can gracefully degrade text rendering when we fall back to the
        # built-in Latin-1 fonts.
        self.uses_unicode_fonts = False

    def _prepare_text(self, text: Any) -> str:
        """Return text that is safe for the currently selected fonts."""

        if text is None:
            return ""
        if isinstance(text, bytes):
            text_str = text.decode("latin-1", "replace")
        else:
            text_str = str(text)
        if self.uses_unicode_fonts:
            return text_str
        # Built-in fonts only support Latin-1, so replace unsupported
        # characters to avoid FPDF exceptions when measuring string width.
        return text_str.encode("latin-1", "replace").decode("latin-1")

    # FPDF wraps ``cell``/``multi_cell`` in decorators, so make sure to keep
    # the same signature when overriding them.
    def cell(
        self,
        w: float = 0,
        h: float = 0,
        txt: str | bytes = "",
        border: int | str = 0,
        ln: int = 0,
        align: str = "",
        fill: bool = False,
        link: str | int | None = "",
    ) -> None:
        super().cell(
            w,
            h,
            self._prepare_text(txt),
            border=border,
            ln=ln,
            align=align,
            fill=fill,
            link=link,
        )

    def multi_cell(
        self,
        w: float,
        h: float,
        txt: str | bytes,
        border: int | str = 0,
        align: str = "J",
        fill: bool = False,
    ) -> None:
        prepared = self._prepare_text(txt)
        try:
            super().multi_cell(
                w,
                h,
                prepared,
                border=border,
                align=align,
                fill=fill,
            )
        except FPDFException:
            # When falling back to the built-in Latin-1 fonts, FPDF cannot
            # automatically break long strings that do not contain whitespace
            # (for example, Chinese or Japanese text). Insert manual breakpoints
            # between characters so the PDF generation can continue instead of
            # raising an exception. This gracefully degrades the output when
            # Unicode-capable fonts are unavailable.
            if self.uses_unicode_fonts:
                raise

            effective_width = w
            if not effective_width:
                # ``multi_cell`` treats ``w=0`` as "use the remaining width on the
                # current line". When the cursor is already at the right margin,
                # the calculated width can be zero which triggers another
                # ``FPDFException``. Fall back to the full printable page width so
                # we always have enough space for at least one character.
                effective_width = self.w - self.l_margin - self.r_margin
            if effective_width <= 0:
                effective_width = max(self.w - self.l_margin - self.r_margin, 1)

            lines = prepared.split("\n")
            for idx, line in enumerate(lines):
                spaced = " ".join(line) if line else ""
                # Ensure each line starts from the left margin before drawing to
                # avoid the "no horizontal space" error when the cursor is close
                # to the right margin.
                self.set_x(self.l_margin)
                super().multi_cell(
                    effective_width,
                    h,
                    spaced,
                    border=border,
                    align=align,
                    fill=fill,
                )
                # ``multi_cell`` automatically moves to the next line, so only
                # add an explicit line break between the original lines.
                if idx != len(lines) - 1:
                    self.ln(h)

    def footer(self) -> None:  # type: ignore[override]
        self.set_y(-15)
        self.set_font(self.sans_family, size=10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"{self.page_no()} / {{nb}}", align="C")


@st.cache_resource(show_spinner=False)
def load_fonts() -> Dict[str, str]:
    """Return absolute paths to bundled Noto fonts."""

    available: Dict[str, str] = {}
    missing: List[str] = []

    downloaded_roles: List[str] = []

    def _attempt_download(role: str) -> None:
        existing = any(
            (ASSETS_DIR / candidate).exists() and (ASSETS_DIR / candidate).stat().st_size > 0
            for candidate in FONT_CANDIDATES[role]
        )
        if existing:
            return

        for url in FONT_DOWNLOADS.get(role, ()):  # pragma: no branch - finite loop
            filename = url.rsplit("/", 1)[-1]
            destination = ASSETS_DIR / filename
            if destination.exists() and destination.stat().st_size > 0:
                break
            try:
                with contextlib.closing(urlopen(url, timeout=15)) as response:
                    data = response.read()
            except Exception:
                continue
            if not data or len(data) < 1024:
                continue
            try:
                destination.write_bytes(data)
            except OSError:
                continue
            else:
                downloaded_roles.append(role)
                break

    for role in FONT_CANDIDATES:
        _attempt_download(role)

    if downloaded_roles:
        st.info(
            "Downloaded font support for: "
            + ", ".join(sorted(set(downloaded_roles)))
            + "."
        )

    for role, candidates in FONT_CANDIDATES.items():
        found_path: str | None = None
        for directory in FONT_SEARCH_PATHS:
            try:
                if not directory.exists():
                    continue
            except OSError:
                continue
            for candidate in candidates:
                candidate_path = directory / candidate
                try:
                    if candidate_path.exists() and candidate_path.stat().st_size > 0:
                        found_path = str(candidate_path)
                        break
                except OSError:
                    continue
            if found_path:
                break
        if found_path:
            available[role] = found_path
        else:
            missing.append(role)

    if missing:
        st.warning(
            "Unicode-capable fonts not found for: "
            + ", ".join(sorted(missing))
            + ". The PDF will fall back to limited built-in fonts; non-Latin text may not render correctly."
        )

    return available


def _prepare_pdf() -> StoryPDF:
    pdf = StoryPDF()
    fonts = load_fonts()

    def _register_font(
        family_name: str, font_key: str, fallback: str, *, unicode: bool = True
    ) -> str:
        font_path = fonts.get(font_key)
        if font_path:
            try:
                path_obj = Path(font_path)
                if path_obj.exists() and path_obj.stat().st_size > 0:
                    pdf.add_font(family_name, "", font_path, uni=unicode)
                    pdf.uses_unicode_fonts = True
                    return family_name
            except Exception:  # pragma: no cover - defensive fallback
                st.warning(
                    "Failed to load custom font '%s'; falling back to built-in fonts." % family_name
                )
        return fallback

    pdf.sans_family = _register_font("NotoSans", "sans", pdf.sans_family)
    pdf.serif_family = _register_font("NotoSerif", "serif", pdf.serif_family)

    return pdf


def _write_story_content(pdf: StoryPDF, story: Dict[str, Any], params: Dict[str, Any]) -> None:
    header_lines = [
        story.get("title", "Untitled Story"),
        f"Level: {params.get('level_schema')} {params.get('level')}",
        f"Languages: {params.get('learning_language')} for {params.get('native_language')} speakers",
    ]
    topic_text = ", ".join(params.get("topics", []))
    if topic_text:
        header_lines.append(f"Topics: {topic_text}")

    pdf.set_font(pdf.sans_family, size=20)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 10, header_lines[0])
    pdf.ln(2)

    pdf.set_font(pdf.sans_family, size=12)
    for meta in header_lines[1:]:
        pdf.multi_cell(0, 6, meta)
    pdf.ln(4)

    summary = story.get("summary")
    if summary:
        pdf.set_font(pdf.serif_family, size=12)
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 7, summary)
        pdf.ln(3)

    reading_sections = story.get("reading_sections") or []
    if reading_sections:
        pdf.set_font(pdf.sans_family, size=14)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 8, "Reading Support")
        pdf.ln(1)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font(pdf.serif_family, size=12)

    line_height = 7
    for section in reading_sections:
        original = section.get("original", "").strip()
        phonetics = section.get("phonetics", "").strip()
        translation = section.get("translation", "").strip()
        if original:
            pdf.multi_cell(0, line_height, original)
        if phonetics:
            pdf.set_font(pdf.sans_family, size=11)
            pdf.set_text_color(90, 90, 90)
            pdf.multi_cell(0, line_height, phonetics)
            pdf.set_font(pdf.serif_family, size=12)
            pdf.set_text_color(0, 0, 0)
        if translation:
            pdf.set_font(pdf.sans_family, size=11)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, line_height, translation)
            pdf.set_font(pdf.serif_family, size=12)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    if not reading_sections:
        body_text = story.get("body", "")
        if body_text:
            for paragraph in body_text.split("\n"):
                cleaned = paragraph.strip()
                if not cleaned:
                    pdf.ln(line_height / 2)
                    continue
                pdf.multi_cell(0, line_height, cleaned)
                pdf.ln(1)

    full_translation = story.get("translation")
    if full_translation:
        pdf.ln(1)
        pdf.set_font(pdf.sans_family, size=14)
        pdf.multi_cell(0, 8, "Full Story Translation")
        pdf.ln(1)
        pdf.set_font(pdf.sans_family, size=11)
        pdf.set_text_color(40, 40, 40)
        for paragraph in full_translation.split("\n"):
            cleaned = paragraph.strip()
            if not cleaned:
                pdf.ln(line_height / 2)
                continue
            pdf.multi_cell(0, line_height, cleaned)
            pdf.ln(1)
        pdf.set_font(pdf.serif_family, size=12)
        pdf.set_text_color(0, 0, 0)

    glossary = story.get("glossary") or []
    if params.get("include_glossary") and glossary:
        pdf.ln(3)
        pdf.set_font(pdf.sans_family, size=14)
        pdf.multi_cell(0, 8, "Glossary")
        pdf.ln(1)
        pdf.set_font(pdf.serif_family, size=12)
        for entry in glossary:
            term = entry.get("term")
            definition = entry.get("definition")
            if not term or not definition:
                continue
            pdf.multi_cell(0, line_height, f"• {term}: {definition}")
        pdf.ln(2)

    grammar_notes = story.get("grammar_notes") or []
    if grammar_notes:
        pdf.ln(2)
        pdf.set_font(pdf.sans_family, size=14)
        pdf.multi_cell(0, 8, "Grammar Notes")
        pdf.ln(1)
        pdf.set_font(pdf.serif_family, size=12)
        for note in grammar_notes:
            pdf.multi_cell(0, line_height, f"• {note}")
        pdf.ln(2)

    practice_ideas = story.get("practice_ideas") or []
    if practice_ideas:
        pdf.ln(1)
        pdf.set_font(pdf.sans_family, size=14)
        pdf.multi_cell(0, 8, "Practice Ideas")
        pdf.ln(1)
        pdf.set_font(pdf.serif_family, size=12)
        for idea in practice_ideas:
            pdf.multi_cell(0, line_height, f"• {idea}")
        pdf.ln(2)

    extra_notes = story.get("extra_notes") or []
    if extra_notes:
        pdf.ln(1)
        pdf.set_font(pdf.sans_family, size=14)
        pdf.multi_cell(0, 8, "Strategy & Culture Notes")
        pdf.ln(1)
        pdf.set_font(pdf.serif_family, size=12)
        for note in extra_notes:
            pdf.multi_cell(0, line_height, f"• {note}")
        pdf.ln(2)


def _pdf_to_bytes(pdf: StoryPDF) -> bytes:
    data = pdf.output(dest="S")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return data.encode("latin1")


@st.cache_data(show_spinner=False)
def story_to_pdf(story: Dict[str, Any], params: Dict[str, Any]) -> bytes:
    """Create a PDF document for a single story."""

    pdf = _prepare_pdf()
    pdf.set_title(story.get("title", "Story"))
    pdf.add_page()
    _write_story_content(pdf, story, params)
    return _pdf_to_bytes(pdf)


@st.cache_data(show_spinner=False)
def stories_to_single_pdf(stories: Sequence[Dict[str, Any]], params: Dict[str, Any]) -> bytes:
    """Create a combined PDF containing all stories with a table of contents."""

    preview = _prepare_pdf()
    preview.add_page()  # Cover placeholder
    preview.add_page()  # TOC placeholder

    entries: List[Dict[str, Any]] = []
    for idx, story in enumerate(stories, start=1):
        preview.add_page()
        start_page = preview.page_no()
        _write_story_content(preview, story, params)
        entries.append(
            {
                "index": idx,
                "title": story.get("title", f"Story {idx}"),
                "page": start_page,
            }
        )

    pdf = _prepare_pdf()
    pdf.set_title("Graded Reader Collection")

    pdf.add_page()
    pdf.set_font(pdf.sans_family, size=26)
    pdf.cell(0, 20, "Graded Reader Collection", ln=1, align="C")
    pdf.set_font(pdf.sans_family, size=14)
    pdf.multi_cell(0, 8, f"Level: {params.get('level_schema')} {params.get('level')}", align="C")
    pdf.multi_cell(0, 8, f"Learning: {params.get('learning_language')}", align="C")
    pdf.multi_cell(0, 8, f"Native language: {params.get('native_language')}", align="C")

    pdf.add_page()
    pdf.set_font(pdf.sans_family, size=20)
    pdf.multi_cell(0, 10, "Table of Contents")
    pdf.ln(5)
    pdf.set_font(pdf.sans_family, size=12)
    for entry in entries:
        title = entry["title"]
        page_num = entry["page"]
        line = f"{entry['index']}. {title}"[:90]
        pdf.cell(0, 8, f"{line} ...... {page_num}", ln=1)

    for story in stories:
        pdf.add_page()
        _write_story_content(pdf, story, params)

    return _pdf_to_bytes(pdf)


__all__ = ["load_fonts", "story_to_pdf", "stories_to_single_pdf"]
