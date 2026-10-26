import pymupdf as fitz
import pytest
from httpx import ASGITransport, AsyncClient
from backend.app.main import app


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


@pytest.mark.asyncio
async def test_get_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_post_documents_success():
    pdf_bytes = create_in_memory_pdf("Experimental findings demonstrate 15% latency reduction.")
    files = {"file": ("test_paper.pdf", pdf_bytes, "application/pdf")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/documents", files=files)

    assert response.status_code == 201
    data = response.json()
    assert "document" in data
    assert data["chunk_count"] > 0
    doc = data["document"]
    assert doc["filename"] == "test_paper.pdf"
    assert doc["title"] == "Test Title"
    assert doc["page_count"] == 2
    assert len(doc["sha256"]) == 64
    assert isinstance(doc["extraction_warnings"], list)


@pytest.mark.asyncio
async def test_post_documents_non_pdf():
    files = {"file": ("notes.txt", b"Plain text notes", "text/plain")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/documents", files=files)

    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


@pytest.mark.asyncio
async def test_post_documents_empty_file():
    files = {"file": ("empty.pdf", b"", "application/pdf")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/documents", files=files)

    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]


@pytest.mark.asyncio
async def test_post_documents_corrupted_pdf():
    files = {"file": ("corrupt.pdf", b"%PDF-corrupted-bytes-xyz", "application/pdf")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/documents", files=files)

    assert response.status_code == 400
    assert "Unable to parse PDF" in response.json()["detail"]
