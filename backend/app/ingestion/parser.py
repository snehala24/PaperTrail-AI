import hashlib
from typing import List, Tuple
import pymupdf as fitz
from backend.app.schemas import Document, Chunk
from backend.app.ingestion.chunker import chunk_page_blocks


def calculate_sha256(content: bytes) -> str:
    """Calculate the 64-character hexadecimal SHA-256 hash of bytes."""
    return hashlib.sha256(content).hexdigest()


def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> Tuple[Document, List[Chunk]]:
    """
    Parses PDF bytes using PyMuPDF (fitz).
    Maintains page and section awareness, records extraction failures/warnings,
    and returns the Document contract along with its granular Chunks.
    """
    if not pdf_bytes:
        raise ValueError("Unable to parse PDF: Empty byte sequence received.")

    sha256 = calculate_sha256(pdf_bytes)
    doc_id = f"doc_{sha256[:12]}"
    extraction_warnings: List[str] = []

    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to parse PDF: {exc}") from exc

    page_count = len(pdf_doc)
    if page_count == 0:
        pdf_doc.close()
        extraction_warnings.append("Document has 0 pages.")
        doc = Document(
            doc_id=doc_id,
            title=filename,
            filename=filename,
            sha256=sha256,
            page_count=0,
            extraction_warnings=extraction_warnings,
        )
        return doc, []

    # Title extraction from metadata or first page header fallback
    meta_title = (pdf_doc.metadata.get("title") or "").strip()
    title = meta_title if meta_title else filename

    all_chunks: List[Chunk] = []
    current_section = None

    for page_idx in range(page_count):
        page_num = page_idx + 1
        page = pdf_doc[page_idx]

        # Extract structural blocks: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type == 0 indicates text
        raw_blocks = page.get_text("blocks")
        text_blocks: List[str] = [
            b[4].strip() for b in raw_blocks if len(b) > 4 and b[6] == 0 and b[4].strip()
        ]

        if not text_blocks:
            # Fallback to plain text extraction
            plain_text = page.get_text("text").strip()
            if plain_text:
                text_blocks = [plain_text]
            else:
                # Rule: Record extraction failures, never silently drop
                extraction_warnings.append(
                    f"Page {page_num} has no extractable text; may be blank, scanned, or image-only."
                )
                continue

        chunks, current_section = chunk_page_blocks(
            doc_id=doc_id,
            page=page_num,
            blocks=text_blocks,
            current_section=current_section,
        )
        all_chunks.extend(chunks)

    pdf_doc.close()

    document = Document(
        doc_id=doc_id,
        title=title,
        filename=filename,
        sha256=sha256,
        page_count=page_count,
        extraction_warnings=extraction_warnings,
    )

    return document, all_chunks
