import re
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from backend.app.schemas import Document, Chunk, DocumentUploadResponse, CHUNKER_VERSION
from backend.app.ingestion.parser import parse_pdf_bytes, calculate_sha256
from backend.app.core.storage import storage
from backend.app.core.config import settings
from backend.app.retrieval.hybrid import retriever

router = APIRouter(prefix="/documents", tags=["Documents"])

DOC_ID_REGEX = re.compile(r"^doc_[0-9a-f]{12}$")


def validate_doc_id(doc_id: str) -> None:
    if not DOC_ID_REGEX.match(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid doc_id format: '{doc_id}'. Expected format '^doc_[0-9a-f]{{12}}$'.",
        )


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Ingest a research paper in PDF format.
    - Synchronous def route offloaded automatically to worker threadpool by FastAPI.
    - Validates upload size against MAX_UPLOAD_MB and magic bytes %PDF-.
    - Computes SHA-256 and deduplicates: re-uploading an existing PDF returns the stored document,
      UNLESS the stored document was created with an older chunker version (in which case it re-ingests).
    - Persists document and chunks to data/.
    - Updates the global hybrid retrieval index.
    """
    filename = file.filename or "unknown.pdf"

    if not filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Received file: {filename}",
        )

    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    else:
        read_res = file.read()
        import inspect
        if inspect.iscoroutine(read_res):
            import asyncio
            content = asyncio.run(read_res)
        else:
            content = read_res

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file provided. Please upload a valid non-empty PDF document.",
        )

    # Validate upload size (MAX_UPLOAD_MB)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum upload size of {settings.max_upload_mb} MB.",
        )

    # Validate PDF magic bytes (%PDF-)
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file format. Content must begin with %PDF- header.",
        )

    sha256 = calculate_sha256(content)
    existing_doc = storage.get_document_by_sha256(sha256)
    if existing_doc and getattr(existing_doc, "chunker_version", None) == CHUNKER_VERSION:
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

    saved_doc = storage.save_document(document, chunks, force_overwrite=True)
    all_chunks = storage.get_all_chunks()
    retriever.index_chunks(all_chunks)

    return DocumentUploadResponse(
        document=saved_doc,
        chunk_count=len(chunks),
    )


@router.get("", response_model=List[Document])
def list_documents() -> List[Document]:
    """Retrieve list of all ingested documents."""
    return storage.list_documents()


@router.get("/{doc_id}", response_model=Document)
def get_document(doc_id: str) -> Document:
    """Retrieve metadata for a specific document."""
    validate_doc_id(doc_id)
    doc = storage.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{doc_id}' not found.",
        )
    return doc


@router.get("/{doc_id}/chunks", response_model=List[Chunk])
def get_document_chunks(doc_id: str) -> List[Chunk]:
    """Retrieve all chunks for a specific document."""
    validate_doc_id(doc_id)
    doc = storage.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{doc_id}' not found.",
        )
    return storage.get_chunks(doc_id)
