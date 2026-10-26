import hashlib
import re
from typing import Dict, List, Set, Tuple
import pymupdf as fitz
from backend.app.schemas import Document, Chunk, CHUNKER_VERSION
from backend.app.ingestion.chunker import (
    chunk_page_blocks,
    dehyphenate,
    is_equation_fragment,
)


def calculate_sha256(content: bytes) -> str:
    """Calculate the 64-character hexadecimal SHA-256 hash of bytes."""
    return hashlib.sha256(content).hexdigest()


def detect_running_headers(pdf_doc: fitz.Document, page_count: int) -> Set[str]:
    """
    Identify running header lines that repeat on >= 50% of pages (for documents with >= 2 pages).
    """
    if page_count < 2:
        return set()

    line_page_counts: Dict[str, Set[int]] = {}
    for page_idx in range(page_count):
        page = pdf_doc[page_idx]
        for b in page.get_text("blocks"):
            if len(b) > 4 and b[6] == 0:
                for line in b[4].splitlines():
                    clean_line = line.strip()
                    if clean_line and len(clean_line) > 10:
                        line_page_counts.setdefault(clean_line, set()).add(page_idx)

    threshold = max(2, int(0.5 * page_count))
    return {line for line, pages in line_page_counts.items() if len(pages) >= threshold}


ARXIV_SIDEBAR_REGEX = re.compile(
    r"^\s*arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\s*(?:\[[\w\.\-]+\])?(?:\s+\d{1,2}\s+[A-Za-z]+\s+\d{4})?\s*$",
    re.IGNORECASE,
)


def clean_page_blocks(
    raw_blocks: List[Tuple],
    running_headers: Set[str],
    page_idx: int = 1,
) -> List[str]:
    """
    Clean extracted text blocks for a page:
    - Drops running headers on pages 2+ (preserves paper title on page 1).
    - Drops standalone page numbers (lines matching ^\\d+$).
    - Drops rotated arXiv sidebar watermark stamp without deleting bibliography citation lines.
    - Drops isolated equation noise (| {z }, Shape, Scale).
    - Dehyphenates line-break hyphens ('pow-\\nerful' -> 'powerful').
    """
    cleaned_blocks: List[str] = []

    for b in raw_blocks:
        if len(b) <= 4 or b[6] != 0:
            continue
        text = b[4].strip()
        if not text:
            continue

        lines = text.splitlines()
        valid_lines: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # Drop running headers on page 2+ (never drop document title on page 1)
            if page_idx > 0 and s in running_headers:
                continue
            # Drop standalone page numbers
            if re.match(r"^\d+$", s):
                continue
            # Drop rotated arXiv sidebar watermark (preserving inline/bibliography arXiv citations)
            if ARXIV_SIDEBAR_REGEX.match(s):
                continue
            # Drop equation fragment lines
            if is_equation_fragment(s):
                continue
            valid_lines.append(line)

        if not valid_lines:
            continue

        joined_text = "\n".join(valid_lines).strip()
        if not joined_text or is_equation_fragment(joined_text):
            continue

        # Dehyphenate line-break hyphens
        dehyphenated_text = dehyphenate(joined_text)
        cleaned_blocks.append(dehyphenated_text)

    return cleaned_blocks


def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> Tuple[Document, List[Chunk]]:
    """
    Parses PDF bytes using PyMuPDF (fitz).
    Maintains page and section awareness, records extraction failures/warnings,
    removes running headers, page numbers, and sidebar noise, dehyphenates line breaks,
    and returns the Document contract (with chunker_version) along with granular Chunks.
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
            chunker_version=CHUNKER_VERSION,
            extraction_warnings=extraction_warnings,
        )
        return doc, []

    # Title extraction from metadata or first page header fallback
    meta_title = (pdf_doc.metadata.get("title") or "").strip()
    title = meta_title if meta_title else filename

    # Pre-detect running headers repeated on >= 50% of pages
    running_headers = detect_running_headers(pdf_doc, page_count)

    all_chunks: List[Chunk] = []
    current_section = None
    in_references = False

    for page_idx in range(page_count):
        page_num = page_idx + 1
        page = pdf_doc[page_idx]

        raw_blocks = page.get_text("blocks")
        text_blocks = clean_page_blocks(raw_blocks, running_headers, page_idx=page_idx)

        if not text_blocks:
            plain_text = page.get_text("text").strip()
            if plain_text:
                dehyphed = dehyphenate(plain_text)
                text_blocks = [dehyphed]
            else:
                extraction_warnings.append(
                    f"Page {page_num} has no extractable text; may be blank, scanned, or image-only."
                )
                continue

        chunks, current_section, in_references = chunk_page_blocks(
            doc_id=doc_id,
            page=page_num,
            blocks=text_blocks,
            current_section=current_section,
            in_references=in_references,
        )
        all_chunks.extend(chunks)

    pdf_doc.close()

    document = Document(
        doc_id=doc_id,
        title=title,
        filename=filename,
        sha256=sha256,
        page_count=page_count,
        chunker_version=CHUNKER_VERSION,
        extraction_warnings=extraction_warnings,
    )

    return document, all_chunks
