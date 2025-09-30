"""PDF generation helpers for graded reader stories."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import streamlit as st
from fpdf import FPDF


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


class StoryPDF(FPDF):
    """FPDF subclass that renders footers with page numbers."""

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.alias_nb_pages()

    def footer(self) -> None:  # type: ignore[override]
        self.set_y(-15)
        self.set_font("NotoSans", size=10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"{self.page_no()} / {{nb}}", align="C")


@st.cache_resource(show_spinner=False)
def load_fonts() -> Dict[str, str]:
    """Return absolute paths to bundled Noto fonts."""

    fonts = {
        "sans": str(ASSETS_DIR / "NotoSans-Regular.ttf"),
        "serif": str(ASSETS_DIR / "NotoSerif-Regular.ttf"),
    }
    for font_path in fonts.values():
        if not Path(font_path).exists():
            raise FileNotFoundError(
                "Required font file is missing. Ensure Noto fonts are bundled."
            )
    return fonts


def _prepare_pdf() -> StoryPDF:
    pdf = StoryPDF()
    fonts = load_fonts()
    pdf.add_font("NotoSans", "", fonts["sans"], uni=True)
    pdf.add_font("NotoSerif", "", fonts["serif"], uni=True)
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

    pdf.set_font("NotoSans", size=20)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 10, header_lines[0])
    pdf.ln(2)

    pdf.set_font("NotoSans", size=12)
    for meta in header_lines[1:]:
        pdf.multi_cell(0, 6, meta)
    pdf.ln(4)

    pdf.set_font("NotoSerif", size=12)
    pdf.set_text_color(0, 0, 0)

    line_height = 7
    for paragraph in story.get("body", "").split("\n"):
        cleaned = paragraph.strip()
        if not cleaned:
            pdf.ln(line_height / 2)
            continue
        pdf.multi_cell(0, line_height, cleaned)
        pdf.ln(1)

    glossary = story.get("glossary") or []
    if params.get("include_glossary") and glossary:
        pdf.ln(3)
        pdf.set_font("NotoSans", size=14)
        pdf.multi_cell(0, 8, "Glossary")
        pdf.ln(1)
        pdf.set_font("NotoSerif", size=12)
        for entry in glossary:
            term = entry.get("term")
            definition = entry.get("definition")
            if not term or not definition:
                continue
            pdf.multi_cell(0, line_height, f"• {term}: {definition}")
        pdf.ln(2)


def _pdf_to_bytes(pdf: StoryPDF) -> bytes:
    buffer = pdf.output(dest="S").encode("latin1")
    return buffer


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

    pdf = _prepare_pdf()
    pdf.set_title("Graded Reader Collection")

    pdf.add_page()
    pdf.set_font("NotoSans", size=26)
    pdf.cell(0, 20, "Graded Reader Collection", ln=1, align="C")
    pdf.set_font("NotoSans", size=14)
    pdf.multi_cell(0, 8, f"Level: {params.get('level_schema')} {params.get('level')}", align="C")
    pdf.multi_cell(0, 8, f"Learning: {params.get('learning_language')}", align="C")
    pdf.multi_cell(0, 8, f"Native language: {params.get('native_language')}", align="C")

    pdf.add_page()
    toc_page = pdf.page_no()
    pdf.set_font("NotoSans", size=20)
    pdf.multi_cell(0, 10, "Table of Contents")
    pdf.ln(5)

    entries: List[Dict[str, Any]] = []
    for idx, story in enumerate(stories, start=1):
        pdf.add_page()
        start_page = pdf.page_no()
        entries.append({"index": idx, "title": story.get("title", f"Story {idx}"), "page": start_page})
        _write_story_content(pdf, story, params)

    # Populate table of contents after all stories are added
    current_page = pdf.page_no()
    pdf.set_page(toc_page)
    pdf.set_y(pdf.t_margin + 20)
    pdf.set_font("NotoSans", size=12)
    for entry in entries:
        title = entry["title"]
        page_num = entry["page"]
        line = f"{entry['index']}. {title}"[:90]
        pdf.cell(0, 8, f"{line} ...... {page_num}", ln=1)
    pdf.set_page(current_page)

    return _pdf_to_bytes(pdf)


__all__ = ["load_fonts", "story_to_pdf", "stories_to_single_pdf"]
