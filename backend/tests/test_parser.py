import pymupdf as fitz
import pytest
from backend.app.ingestion.parser import parse_pdf_bytes
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
