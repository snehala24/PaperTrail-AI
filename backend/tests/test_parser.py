import pymupdf as fitz
import pytest
from backend.app.ingestion.parser import extract_title_from_page_1 as parser_extract_title_from_page_1, parse_pdf_bytes
from backend.app.schemas import ChunkKind


def create_sample_pdf(pages_text: list[str], title: str = "Attention Is All You Need") -> bytes:
    """Helper to generate a clean multi-page PDF in-memory using PyMuPDF."""
    doc = fitz.open()
    doc.set_metadata({"title": title})
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_parse_valid_pdf():
    pages = [
        "1. Introduction\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        "2. Background\nThe goal of reducing sequential computation also forms the foundation of the Extended Neural GPU.\n\nTable 1: Maximum path lengths\nLayer Type: Self-Attention | Path: O(1)",
        "3. Conclusion\nWe presented the Transformer, the first sequence transduction model based entirely on attention.",
    ]
    pdf_bytes = create_sample_pdf(pages, title="Attention Paper")
    document, chunks = parse_pdf_bytes(pdf_bytes, filename="attention.pdf")

    assert document.title == "Attention Paper"
    assert document.filename == "attention.pdf"
    assert document.page_count == 3
    assert len(document.sha256) == 64
    assert document.extraction_warnings == []
    assert len(chunks) >= 3

    # Verify chunk structure
    assert all(c.doc_id == document.doc_id for c in chunks)
    assert any("Introduction" in (c.section_heading or "") or "Introduction" in c.text for c in chunks)
    assert any(c.page == 1 for c in chunks)
    assert any(c.page == 2 for c in chunks)
    assert any(c.page == 3 for c in chunks)


def test_parse_pdf_with_blank_page_records_warning():
    """Rule: never silently drop extraction failures or empty pages."""
    pages = [
        "1. Abstract\nThis paper proposes a new method.",
        "",  # Blank page 2
        "2. Results\nThe accuracy was 94.5%.",
    ]
    pdf_bytes = create_sample_pdf(pages, title="Paper With Blank Page")
    document, chunks = parse_pdf_bytes(pdf_bytes, filename="blank_page_paper.pdf")

    assert document.page_count == 3
    # Must record warning for page 2
    assert any("page 2" in w.lower() for w in document.extraction_warnings)
    # The valid pages must still be extracted
    assert any(c.page == 1 for c in chunks)
    assert any(c.page == 3 for c in chunks)


def test_chunk_kind_detection():
    pages = [
        "Table 1: Hyperparameters and BLEU comparison across architectures.\n| Epochs | Rate | BLEU |\n| 10 | 0.001 | 28.4 |",
        "Figure 2: The Transformer - model architecture with Multi-Head Attention blocks.",
        "Regular paragraph explaining the attention mechanism and dot-product calculations.",
    ]
    pdf_bytes = create_sample_pdf(pages, title="Table and Figure Paper")
    _, chunks = parse_pdf_bytes(pdf_bytes, filename="tables_figures.pdf")

    table_chunks = [c for c in chunks if c.kind == ChunkKind.TABLE]
    figure_chunks = [c for c in chunks if c.kind == ChunkKind.FIGURE_REF]
    text_chunks = [c for c in chunks if c.kind == ChunkKind.TEXT]

    assert len(table_chunks) >= 1
    assert len(figure_chunks) >= 1
    assert len(text_chunks) >= 1


def test_parse_corrupt_pdf():
    corrupt_bytes = b"Not a valid PDF byte sequence at all!"
    with pytest.raises(ValueError) as exc_info:
        parse_pdf_bytes(corrupt_bytes, filename="corrupt.pdf")
    assert "Unable to parse PDF" in str(exc_info.value)


def create_pdf_with_images(
    pages_text: list[str],
    image_pages: set[int],
    title: str = "Paper With Figures",
) -> bytes:
    """Multi-page PDF with an embedded raster image on each requested 1-indexed page."""
    doc = fitz.open()
    doc.set_metadata({"title": title})
    for idx, text in enumerate(pages_text):
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
        if (idx + 1) in image_pages:
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
            pix.set_rect(pix.irect, (200, 30, 30))
            page.insert_image(fitz.Rect(50, 100, 150, 200), pixmap=pix)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _graphics_warning(document) -> str:
    return " ".join(w for w in document.extraction_warnings if "figure graphics" in w)


def test_image_page_with_indexed_text_is_reported_as_graphics_only():
    """A figure page whose text was indexed must be reported precisely: graphics unread, text indexed."""
    pages = [
        "1. Introduction\nThe dominant sequence transduction models are based on recurrent networks.",
        "Figure 3 caption. The encoder on the left, the decoder on the right.",
    ]
    document, chunks = parse_pdf_bytes(
        create_pdf_with_images(pages, image_pages={2}), filename="figure_page.pdf"
    )
    assert document.page_count == 2
    warning = _graphics_warning(document)
    assert "1 page(s)" in warning  # only page 2 carries graphics
    assert "pages 2" in warning
    assert any(c.page == 1 for c in chunks)


def test_page_that_failed_chunking_is_not_reported_as_read_graphics(monkeypatch):
    """
    A page whose extraction raised records a failure warning; claiming its surrounding text was indexed
    would overstate coverage, so it must not appear in the graphics-with-text diagnostic.
    """
    from backend.app.ingestion import parser as parser_module

    original_chunk_page_blocks = parser_module.chunk_page_blocks

    def exploding_chunk_page_blocks(**kwargs):
        if kwargs.get("page") == 1:
            raise RuntimeError("simulated chunker crash")
        return original_chunk_page_blocks(**kwargs)

    monkeypatch.setattr(parser_module, "chunk_page_blocks", exploding_chunk_page_blocks)

    pages = [
        "1. Introduction\nRecurrent networks dominate sequence transduction today.",
        "Figure 3 caption. The encoder on the left, the decoder on the right.",
    ]
    document, chunks = parse_pdf_bytes(
        create_pdf_with_images(pages, image_pages={1, 2}), filename="crashing_page.pdf"
    )

    assert any("Page 1 failed extraction" in w for w in document.extraction_warnings)
    assert not any(c.page == 1 for c in chunks)
    warning = _graphics_warning(document)
    assert "pages 2" in warning
    assert "1" not in warning.split("pages")[1]


def _page_with_spans(specs: list[tuple[str, float, float, float]]):
    """Build a one-page PDF whose title line is written as separate font-size spans."""
    doc = fitz.open()
    page = doc.new_page()
    for text, fontsize, x, y in specs:
        page.insert_text((x, y), text, fontsize=fontsize)
    return doc, page


def test_small_caps_title_keeps_the_smaller_spans_of_the_same_line():
    """
    Small-caps titles render capitals at the title size and the remaining letters at a reduced size
    on the same line. Selecting span-by-span dropped the reduced letters and yielded "L".
    """
    doc, page = _page_with_spans(
        [
            ("L", 17.2, 50, 80),
            ("O", 13.8, 62, 80),
            ("RA: LOW-RANK ADAPTATION", 17.2, 70, 80),
            ("Edward Hu, Yelong Shen", 10.0, 50, 120),
        ]
    )
    try:
        title = parser_extract_title_from_page_1(page)
    finally:
        doc.close()
    assert title is not None
    assert "LOW-RANK ADAPTATION" in title
    # The reduced-size "O" belongs to the title line; the smaller author line does not.
    assert "".join(title.split()).startswith("LORA")
    assert "Edward" not in title


def test_title_wrapped_mid_word_is_rejoined():
    """A hard line break inside an all-caps title leaves '…LAN-' + 'GUAGE MODELS'; the title must read whole."""
    doc, page = _page_with_spans(
        [
            ("LORA: LOW-RANK ADAPTATION OF LARGE LAN-", 17.2, 50, 80),
            ("GUAGE MODELS", 17.2, 50, 100),
            ("Abstract", 11.0, 50, 140),
        ]
    )
    try:
        title = parser_extract_title_from_page_1(page)
    finally:
        doc.close()
    assert title == "LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS"
