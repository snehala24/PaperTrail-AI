from fastapi import APIRouter, File, HTTPException, UploadFile, status
from backend.app.schemas import DocumentUploadResponse
from backend.app.ingestion.parser import parse_pdf_bytes

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """
    Ingest a research paper in PDF format.
    Parses pages, extracts structured text blocks/chunks, records extraction warnings,
    and returns the Document metadata contract and chunk count.
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

    try:
        document, chunks = parse_pdf_bytes(content, filename=filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return DocumentUploadResponse(
        document=document,
        chunk_count=len(chunks),
    )
