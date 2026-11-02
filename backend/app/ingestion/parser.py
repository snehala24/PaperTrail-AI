import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple
import pymupdf as fitz
from backend.app.schemas import Document, Chunk, CHUNKER_VERSION
from backend.app.core.config import settings
from backend.app.ingestion.chunker import (
    chunk_page_blocks,
    dehyphenate,
    is_equation_fragment,
)


def calculate_sha256(content: bytes) -> str:
    """Calculate the 64-character hexadecimal SHA-256 hash of bytes."""
    return hashlib.sha256(content).hexdigest()


def is_filename_like_title(title: str) -> bool:
    """Check if metadata title looks like an uninformative filename or placeholder."""
    t = title.strip().lower()
    if not t or t in ("untitled", "default", "blank", "none", "pdf"):
        return True
    if re.search(r"\.(pdf|tex|docx?|html?|txt|md)$", t, re.IGNORECASE):
        return True
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", t):
        return True
    return False


def extract_title_from_page_1(page: fitz.Page) -> Optional[str]:
    """
    Extract the title from the largest-font *lines* within the top 35% of page 1.
    Excludes margin text and rotated arXiv sidebars.

    Lines, not individual spans, are the unit of selection: titles set in small caps
    (e.g. the arXiv LoRA paper) store the capital glyphs at the title size and the
    smaller-caps glyphs at a reduced size, so filtering span-by-span used to return
    only the capitals ("L RA: L -R A L L - M").
    """
    try:
        rect = page.rect
        top_35_y = rect.height * 0.35
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return None

    line_records: List[Tuple[float, float, str]] = []
    for b in blocks:
        if b.get("type") == 0:  # text block
            bbox = b.get("bbox", (0, 0, 0, 0))
            if bbox[1] < top_35_y:
                # Exclude margin text (e.g. rotated margin x < 45)
                if bbox[0] < 45:
                    continue
                for l in b.get("lines", []):
                    span_texts: List[str] = []
                    line_size = 0.0
                    for s in l.get("spans", []):
                        raw_text = s.get("text", "")
                        if ARXIV_SIDEBAR_REGEX.match(raw_text.strip()):
                            continue
                        # Spans are style runs inside one line, not words, so they are concatenated
                        # as-is: small caps store "L" + "O" + "RA" for "LORA", and the inter-word
                        # gaps arrive as their own blank spans that must not be stripped away.
                        span_texts.append(raw_text)
                        line_size = max(line_size, float(s.get("size", 0)))
                    joined = "".join(span_texts).strip()
                    if joined:
                        line_records.append((line_size, float(bbox[1]), joined))

    if not line_records:
        return None

    max_size = max(r[0] for r in line_records)
    title_parts = [r[2] for r in line_records if r[0] >= max_size - 1.0]
    title = " ".join(title_parts).strip()
    # A title wrapped mid-word leaves "…LAN-" + "GUAGE MODELS"; rejoin only when both
    # neighbours are capitals, which is how a hard line break inside an all-caps title reads.
    title = re.sub(r"(?<=[A-Z])-\s+(?=[A-Z])", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title if title else None


def detect_running_headers(pdf_doc: fitz.Document, page_count: int) -> Set[str]:
    """
    Identify running header lines that repeat on >= 50% of pages (for documents with >= 2 pages).
    """
    if page_count < 2:
        return set()

    line_page_counts: Dict[str, Set[int]] = {}
    for page_idx in range(page_count):
        try:
            page = pdf_doc.load_page(page_idx)
            for b in page.get_text("blocks"):
                if len(b) > 4 and b[6] == 0:
                    for line in b[4].splitlines():
                        clean_line = line.strip()
                        if clean_line and len(clean_line) > 10:
                            line_page_counts.setdefault(clean_line, set()).add(page_idx)
        except Exception:
            continue

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
    page_height: float = 792.0,
    doc_tokens_lower: Optional[Set[str]] = None,
) -> List[str]:
    """
    Clean extracted text blocks for a page:
    - Drops running headers on pages 2+ (preserves paper title on page 1).
    - Drops standalone page numbers (lines matching ^\\d+$) only in top/bottom ~8% margin.
    - Drops rotated arXiv sidebar watermark stamp without deleting bibliography citation lines.
    - Drops isolated equation noise (| {z }, Shape, Scale).
    - Dehyphenates line-break hyphens ('pow-\\nerful' -> 'powerful') only when plausible.
    """
    cleaned_blocks: List[str] = []

    for b in raw_blocks:
        if len(b) <= 4 or b[6] != 0:
            continue
        text = b[4].strip()
        if not text:
            continue

        y0 = b[1] if len(b) > 3 else 0.0
        y1 = b[3] if len(b) > 3 else page_height
        is_header_footer_area = (y1 <= 0.12 * page_height) or (y0 >= 0.88 * page_height)

        lines = text.splitlines()
        valid_lines: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # Drop running headers on page 2+ (never drop document title on page 1)
            if page_idx > 0 and s in running_headers:
                continue
            # Drop standalone page numbers only if located in header/footer margin
            if re.match(r"^\d+$", s) and is_header_footer_area:
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

        # Dehyphenate line-break hyphens with document vocabulary awareness
        dehyphenated_text = dehyphenate(joined_text, doc_tokens_lower=doc_tokens_lower)
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

    # Page limit check (MAX_PAGES)
    if page_count > settings.max_pages:
        pdf_doc.close()
        raise ValueError(
            f"Document exceeds maximum page limit of {settings.max_pages} pages (received {page_count} pages)."
        )

    # Title extraction: metadata -> largest font in top 35% of page 1 -> filename
    meta_title = (pdf_doc.metadata.get("title") or "").strip()
    title = None
    if meta_title and not is_filename_like_title(meta_title):
        title = meta_title
    elif page_count > 0:
        try:
            title = extract_title_from_page_1(pdf_doc.load_page(0))
        except Exception:
            title = None
    if not title:
        title = filename

    # Pre-detect running headers repeated on >= 50% of pages
    running_headers = detect_running_headers(pdf_doc, page_count)

    # Pre-extract clean document tokens for vocabulary-aware dehyphenation
    raw_blocks_list: List[str] = []
    for page_idx in range(page_count):
        try:
            raw_blocks_list.extend(
                b[4] for b in pdf_doc[page_idx].get_text("blocks")
                if len(b) > 4 and b[6] == 0
            )
        except Exception:
            pass
    all_raw_blocks_text = " ".join(raw_blocks_list)
    raw_clean_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b(?!\s*-\s*\n)", all_raw_blocks_text))
    doc_tokens_lower: Set[str] = {t.lower() for t in raw_clean_tokens}

    all_chunks: List[Chunk] = []
    current_section = None
    in_references = False
    image_pages: Set[int] = set()
    textless_pages: Set[int] = set()
    failed_pages: Set[int] = set()

    for page_idx in range(page_count):
        page_num = page_idx + 1
        try:
            page = pdf_doc.load_page(page_idx)

            # Detect image blocks
            raw_blocks = page.get_text("blocks")
            has_image = any(len(b) > 6 and b[6] == 1 for b in raw_blocks)
            if not has_image:
                try:
                    has_image = len(page.get_images()) > 0
                except Exception:
                    pass
            if has_image:
                image_pages.add(page_num)

            page_height = getattr(page.rect, "height", 792.0)
            text_blocks = clean_page_blocks(
                raw_blocks,
                running_headers,
                page_idx=page_idx,
                page_height=page_height,
                doc_tokens_lower=doc_tokens_lower,
            )

            if not text_blocks:
                plain_text = page.get_text("text").strip()
                if plain_text:
                    dehyphed = dehyphenate(plain_text, doc_tokens_lower=doc_tokens_lower)
                    text_blocks = [dehyphed]
                else:
                    textless_pages.add(page_num)
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
        except Exception as exc:
            extraction_warnings.append(f"Page {page_num} failed extraction: {exc}")
            failed_pages.add(page_num)
            continue

    if image_pages:
        # Graphics are never interpreted, but the text on those same pages usually is.
        # Say precisely which is which so the diagnostic cannot overstate the gap:
        # pages with no text, and pages whose chunking raised, are excluded because on
        # those pages nothing was actually indexed.
        graphics_pages_with_text = sorted(image_pages - textless_pages - failed_pages)
        if graphics_pages_with_text:
            page_str = ", ".join(str(p) for p in graphics_pages_with_text)
            extraction_warnings.append(
                f"{len(graphics_pages_with_text)} page(s) contain figure graphics that are not "
                f"read as data; captions and surrounding text on those pages were indexed: pages {page_str}."
            )

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
