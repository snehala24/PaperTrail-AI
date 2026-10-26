import io
from unittest.mock import patch
import pymupdf as fitz
import pytest
from fastapi import HTTPException
from backend.app.api.health import get_health
from backend.app.api.documents import upload_document, list_documents, get_document_chunks
from backend.app.api.search import search_corpus
from backend.app.schemas.retrieval import SearchRequest, SearchResponse
from backend.app.schemas import DocumentUploadResponse, Document, Chunk


class DummyUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = "application/pdf"):
        self.filename = filename
        self.content_type = content_type
        self._content = content

    async def read(self) -> bytes:
        return self._content


def run_sync(coro):
    """Run a simple async coroutine to completion synchronously without an OS event loop."""
    try:
        val = None
        while True:
            val = coro.send(val)
            if hasattr(val, "__await__"):
                sub = val.__await__()
                try:
                    sub_val = None
                    while True:
                        sub_val = sub.send(sub_val)
                except StopIteration as s:
                    val = s.value
    except StopIteration as e:
        return e.value


def create_in_memory_pdf(text: str = "Test research content.") -> bytes:
    doc = fitz.open()
    doc.set_metadata({"title": "Test Title"})
    p1 = doc.new_page()
    p1.insert_text((50, 72), "1. Abstract\n" + text)
    p2 = doc.new_page()
    p2.insert_text((50, 72), "2. Methodology\nSection content details.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_get_health():
    res = run_sync(get_health())
    assert res.status == "ok"
    assert res.version == "0.1.0"
    assert res.timestamp is not None


def test_post_documents_success():
    pdf_bytes = create_in_memory_pdf("Experimental findings demonstrate 15% latency reduction.")
    upload_file = DummyUploadFile(filename="test_paper.pdf", content=pdf_bytes)

    with patch("backend.app.api.documents.retriever.index_chunks"):
        res = run_sync(upload_document(file=upload_file))

    assert isinstance(res, DocumentUploadResponse)
    assert res.chunk_count > 0
    assert res.document.filename == "test_paper.pdf"
    assert res.document.title == "Test Title"
    assert res.document.page_count == 2
    assert len(res.document.sha256) == 64
    assert isinstance(res.document.extraction_warnings, list)


def test_post_documents_deduplication():
    pdf_bytes = create_in_memory_pdf("Unique text for deduplication verification.")
    uf1 = DummyUploadFile(filename="dedup_paper.pdf", content=pdf_bytes)
    uf2 = DummyUploadFile(filename="dedup_paper.pdf", content=pdf_bytes)

    with patch("backend.app.api.documents.retriever.index_chunks"):
        res1 = run_sync(upload_document(file=uf1))
        res2 = run_sync(upload_document(file=uf2))

    assert res1.document.doc_id == res2.document.doc_id
    assert res1.document.sha256 == res2.document.sha256


def test_get_documents_and_chunks():
    pdf_bytes = create_in_memory_pdf("Inspection text for listing and chunk retrieval.")
    uf = DummyUploadFile(filename="inspect_paper.pdf", content=pdf_bytes)

    with patch("backend.app.api.documents.retriever.index_chunks"):
        upload_res = run_sync(upload_document(file=uf))

    doc_id = upload_res.document.doc_id

    docs = run_sync(list_documents())
    assert any(d.doc_id == doc_id for d in docs)

    chunks = run_sync(get_document_chunks(doc_id))
    assert len(chunks) > 0
    assert chunks[0].doc_id == doc_id


def test_post_documents_non_pdf():
    uf = DummyUploadFile(filename="notes.txt", content=b"Plain text notes", content_type="text/plain")
    with pytest.raises(HTTPException) as exc_info:
        run_sync(upload_document(file=uf))
    assert exc_info.value.status_code == 400
    assert "Only PDF files are supported" in exc_info.value.detail


def test_post_documents_empty_file():
    uf = DummyUploadFile(filename="empty.pdf", content=b"")
    with pytest.raises(HTTPException) as exc_info:
        run_sync(upload_document(file=uf))
    assert exc_info.value.status_code == 400
    assert "Empty file" in exc_info.value.detail


def test_post_documents_corrupted_pdf():
    uf = DummyUploadFile(filename="corrupt.pdf", content=b"%PDF-corrupted-bytes-xyz")
    with pytest.raises(HTTPException) as exc_info:
        run_sync(upload_document(file=uf))
    assert exc_info.value.status_code == 400
    assert "Unable to parse PDF" in exc_info.value.detail


def test_get_document_chunks_not_found():
    with pytest.raises(HTTPException) as exc_info:
        run_sync(get_document_chunks("non-existent-doc-id-12345"))
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail


def test_post_search_endpoint():
    req = SearchRequest(query="attention mechanism", top_k=5)
    with patch("backend.app.api.search.retriever.search") as mock_search:
        mock_search.return_value = []
        res = run_sync(search_corpus(req))
        assert isinstance(res, SearchResponse)
        assert res.query == "attention mechanism"
        assert res.results == []
        mock_search.assert_called_once_with(
            query="attention mechanism",
            top_k=5,
            doc_ids=None,
            include_references=False,
            use_reranker=True,
        )
