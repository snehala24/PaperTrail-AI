import hashlib
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from backend.app.schemas import Document, Chunk, DocumentUploadResponse
from backend.app.ingestion.parser import parse_pdf_bytes, calculate_sha256
from backend.app.core.storage import storage
from backend.app.retrieval.hybrid import retriever

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Ingest a research paper in PDF format.
    - Computes SHA-256 and deduplicates: re-uploading an existing PDF returns the stored document.
    - Parses pages, merges blocks into passages (~800-1500 chars), isolates tables, tags references.
    - Persists document and chunks to data/.
    - Updates the global hybrid retrieval index.
    """
    filename = file.filename or "unknown.pdf"

    if not filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Received file: {filename}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file provided. Please upload a valid non-empty PDF document.",
        )

    sha256 = calculate_sha256(content)
    existing_doc = storage.get_document_by_sha256(sha256)
    if existing_doc:
        existing_chunks = storage.get_chunks(existing_doc.doc_id)
        return DocumentUploadResponse(
            document=existing_doc,
            chunk_count=len(existing_chunks),
        )

    try:
        document, chunks = parse_pdf_bytes(content, filename=filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    saved_doc = storage.save_document(document, chunks)
    # Rebuild/update retrieval index with all corpus chunks
    retriever.index_chunks(storage.get_all_chunks())

    return DocumentUploadResponse(
        document=saved_doc,
        chunk_count=len(chunks),
    )


@router.get("", response_model=List[Document])
async def list_documents() -> List[Document]:
    """Retrieve list of all ingested documents."""
    return storage.list_documents()


@router.get("/{doc_id}/chunks", response_model=List[Chunk])
async def get_document_chunks(doc_id: str) -> List[Chunk]:
    """Retrieve all chunks for a specific document."""
    doc = storage.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{doc_id}' not found.",
        )
    return storage.get_chunks(doc_id)
