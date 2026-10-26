import io
import os
import re
from unittest.mock import patch, MagicMock
import pytest
from pydantic import ValidationError
import pymupdf as fitz
from fastapi.testclient import TestClient

from backend.app.schemas import CitationCheck, Document
from backend.app.schemas.document import CHUNKER_VERSION
from backend.app.core.config import settings, get_gemini_model
from backend.app.ingestion.parser import parse_pdf_bytes, extract_title_from_page_1
from backend.app.main import app


client = TestClient(app)


def test_citation_check_nonexistent_supports_false_raises():
    """A3: When exists is False, supports must be None. supports=False with exists=False must raise."""
    with pytest.raises(ValidationError) as exc:
        CitationCheck(
            claim_id="claim-1",
            chunk_id="chunk-missing",
            exists=False,
            supports=False,
            notes="Evaluated as false support but chunk is missing",
        )
    assert "Citation support must be None" in str(exc.value)


def test_citation_check_nonexistent_supports_none_valid():
    """A3: When exists is False and supports is None, CitationCheck is valid."""
    check = CitationCheck(
        claim_id="claim-1",
        chunk_id="chunk-missing",
        exists=False,
        supports=None,
        notes="Non-existent chunk",
    )
    assert check.exists is False
    assert check.supports is None


def test_page_level_extraction_error_continues():
    """A2: Catch per-page extraction error, add warning with page number, and continue."""
    pdf = fitz.open()
    p1 = pdf.new_page()
    p1.insert_text((50, 100), "Page 1 Content for testing extraction error resilience.")
    p2 = pdf.new_page()
    p2.insert_text((50, 100), "Page 2 Content for testing extraction error resilience.")
    pdf_bytes = pdf.write()
    pdf.close()

    orig_open = fitz.open

    def failing_open(*args, **kwargs):
        doc = orig_open(*args, **kwargs)
        orig_load = doc.load_page

        def custom_load(idx):
            page = orig_load(idx)
            if idx == 1:
                page.get_text = MagicMock(side_effect=RuntimeError("Simulated font corrupt error"))
            return page

        doc.load_page = custom_load
        return doc

    with patch("pymupdf.open", side_effect=failing_open), patch("backend.app.ingestion.parser.fitz.open", side_effect=failing_open):
        parsed_doc, chunks = parse_pdf_bytes(pdf_bytes, "test_resilient.pdf")

    assert parsed_doc.page_count == 2
    assert any("Page 2 failed extraction" in w for w in parsed_doc.extraction_warnings)
    # Page 1 should still produce chunks
    assert any(c.page == 1 for c in chunks)


def test_image_blocks_warning():
    """A7: If a page has image blocks, add ONE summarised warning per document."""
    pdf = fitz.open()
    p1 = pdf.new_page()
    p1.insert_text((50, 100), "Page 1 Text")
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 10, 10), 0)
    pix.clear_with(255)
    p1.insert_image(fitz.Rect(50, 200, 100, 250), pixmap=pix)
    pdf_bytes = pdf.write()
    pdf.close()

    parsed_doc, _ = parse_pdf_bytes(pdf_bytes, "test_images.pdf")
    img_warnings = [w for w in parsed_doc.extraction_warnings if "contain image blocks" in w]
    assert len(img_warnings) == 1
    assert "1 pages contain image blocks that were not extracted: pages 1" in img_warnings[0]


def test_title_extraction_largest_font_top_35_percent():
    """A6: If metadata title is empty or filename-like, extract largest font text on page 1 within top 35%."""
    pdf = fitz.open()
    p = pdf.new_page()
    # Small header at top
    p.insert_text((50, 40), "Running Header", fontsize=9)
    # Large Title in top 35%
    p.insert_text((50, 100), "My Novel Research Paper Title", fontsize=18)
    # Abstract below title
    p.insert_text((50, 200), "Abstract: We present a method...", fontsize=10)
    pdf_bytes = pdf.write()
    pdf.close()

    parsed_doc, _ = parse_pdf_bytes(pdf_bytes, "2401.0001.pdf")
    assert parsed_doc.title == "My Novel Research Paper Title"


def test_gemini_model_missing_raises():
    """A5: GEMINI_MODEL must come from env/.env without stale default; fail clearly when missing."""
    with patch.object(settings, "gemini_model", None):
        with pytest.raises(ValueError) as exc:
            get_gemini_model()
        assert "GEMINI_MODEL is not set" in str(exc.value)


from fastapi import HTTPException
from backend.app.api.documents import upload_document, get_document, get_document_chunks
from backend.app.api.search import search_corpus
from backend.app.schemas.retrieval import SearchRequest


class DummyUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = "application/pdf"):
        self.filename = filename
        self.content_type = content_type
        self._content = content
        self.file = io.BytesIO(content)

    def read(self) -> bytes:
        return self._content


def test_upload_hardening_max_upload_mb():
    """A8: Content exceeding MAX_UPLOAD_MB returns 413."""
    with patch.object(settings, "max_upload_mb", 1):
        huge_bytes = b"%PDF-" + b"x" * (2 * 1024 * 1024)
        dummy = DummyUploadFile(filename="huge.pdf", content=huge_bytes)
        with pytest.raises(HTTPException) as exc:
            upload_document(file=dummy)
        assert exc.value.status_code == 413
        assert "exceeds maximum upload size" in exc.value.detail


def test_upload_hardening_magic_bytes():
    """A8: Content not starting with %PDF- returns 400."""
    dummy = DummyUploadFile(filename="fake.pdf", content=b"NOT_A_PDF_CONTENT")
    with pytest.raises(HTTPException) as exc:
        upload_document(file=dummy)
    assert exc.value.status_code == 400
    assert "Invalid PDF" in exc.value.detail


def test_upload_hardening_max_pages():
    """A8: Document exceeding MAX_PAGES returns 400."""
    pdf = fitz.open()
    for _ in range(5):
        pdf.new_page()
    pdf_bytes = pdf.write()
    pdf.close()

    with patch.object(settings, "max_pages", 3):
        dummy = DummyUploadFile(filename="toomany.pdf", content=pdf_bytes)
        with pytest.raises(HTTPException) as exc:
            upload_document(file=dummy)
        assert exc.value.status_code == 400
        assert "maximum page limit" in exc.value.detail


def test_validate_doc_id_regex():
    """A8: Validate doc_id with ^doc_[0-9a-f]{12}$ on every route."""
    with pytest.raises(HTTPException) as exc1:
        get_document("invalid_doc_id_here")
    assert exc1.value.status_code == 400
    assert "Invalid doc_id format" in exc1.value.detail

    with pytest.raises(HTTPException) as exc2:
        get_document_chunks("invalid_doc_id_here")
    assert exc2.value.status_code == 400
    assert "Invalid doc_id format" in exc2.value.detail

    with pytest.raises(HTTPException) as exc3:
        search_corpus(SearchRequest(query="test query", doc_ids=["doc_valid123456", "invalid!id"]))
    assert exc3.value.status_code == 400
    assert "Invalid doc_id in filter" in exc3.value.detail
