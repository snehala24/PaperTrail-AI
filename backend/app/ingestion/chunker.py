import re
from typing import List, Optional, Tuple
from backend.app.schemas import Chunk, ChunkKind

# Section heading patterns: e.g. "1. Introduction", "Section 2. Related Work", "Abstract", "References"
SECTION_REGEX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+[\w\s]{2,60}|abstract|introduction|background|methodology|methods|experiments?|results?|discussion|conclusion|related\s+work|acknowledgments?)$",
    re.IGNORECASE
)

REFERENCES_REGEX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography)\b",
    re.IGNORECASE
)

TABLE_REGEX = re.compile(r"^\s*(?:table\s*\d+|\|\s*[^|]+\s*\|)", re.IGNORECASE)
FIGURE_REGEX = re.compile(r"^\s*(?:figure\s*\d+|fig\.\s*\d+)", re.IGNORECASE)

# Target passage bounds in characters (~200-400 tokens)
MIN_CHUNK_CHARS = 800
MAX_CHUNK_CHARS = 1500
MAX_TABLE_CHARS = 2500


def identify_chunk_kind(text: str, in_references: bool = False) -> ChunkKind:
    """Classify chunk content into text, table, figure_ref, or references."""
    if in_references:
        return ChunkKind.REFERENCES
    stripped = text.strip()
    if TABLE_REGEX.match(stripped) or ("|" in stripped and "\n|" in stripped):
        return ChunkKind.TABLE
    if FIGURE_REGEX.match(stripped):
        return ChunkKind.FIGURE_REF
    return ChunkKind.TEXT


def split_oversized_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text longer than max_chars along paragraph or sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    # Split by double newline first
    paragraphs = text.split("\n\n")
    current = ""

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if len(current) + len(p_clean) + 2 <= max_chars:
            current = f"{current}\n\n{p_clean}" if current else p_clean
        else:
            if current:
                parts.append(current)
            if len(p_clean) <= max_chars:
                current = p_clean
            else:
                # Split paragraph by sentence
                sentences = re.split(r"(?<=[.!?])\s+", p_clean)
                sent_buff = ""
                for s in sentences:
                    if len(sent_buff) + len(s) + 1 <= max_chars:
                        sent_buff = f"{sent_buff} {s}" if sent_buff else s
                    else:
                        if sent_buff:
                            parts.append(sent_buff)
                        sent_buff = s
                current = sent_buff

    if current:
        parts.append(current)

    return parts if parts else [text[:max_chars]]


def chunk_page_blocks(
    doc_id: str,
    page: int,
    blocks: List[str],
    current_section: Optional[str] = None,
    in_references: bool = False,
    start_chunk_idx: int = 1,
) -> Tuple[List[Chunk], Optional[str], bool]:
    """
    Chunks a list of text blocks from a single page into passages of ~800-1500 chars.
    - Never crosses a section heading.
    - Keeps tables as isolated chunks (length capped).
    - Tags references with ChunkKind.REFERENCES once the References heading is encountered.
    """
    chunks: List[Chunk] = []
    chunk_idx = start_chunk_idx
    active_section = current_section
    active_in_refs = in_references

    text_buffer: List[str] = []
    buffer_len = 0

    def flush_buffer():
        nonlocal chunk_idx, text_buffer, buffer_len
        if not text_buffer:
            return
        merged_text = "\n\n".join(text_buffer).strip()
        text_buffer = []
        buffer_len = 0

        if not merged_text:
            return

        split_passages = split_oversized_text(merged_text, MAX_CHUNK_CHARS)
        for passage in split_passages:
            kind = ChunkKind.REFERENCES if active_in_refs else ChunkKind.TEXT
            chunk_id = f"{doc_id}_p{page}_c{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page,
                    section_heading=active_section,
                    text=passage,
                    kind=kind,
                )
            )
            chunk_idx += 1

    for block in blocks:
        text = block.strip()
        if not text:
            continue

        first_line = text.split("\n", 1)[0].strip()

        # Check for References heading
        if REFERENCES_REGEX.match(first_line):
            flush_buffer()
            active_section = "References"
            active_in_refs = True
            # Strip heading from text if heading is on its own line
            remaining = text[len(first_line):].strip()
            if remaining:
                text_buffer.append(remaining)
                buffer_len += len(remaining)
            continue

        # Check for other Section Headings (Never cross a section heading)
        if not active_in_refs and SECTION_REGEX.match(first_line) and len(first_line) < 80:
            flush_buffer()
            active_section = first_line
            remaining = text[len(first_line):].strip()
            if remaining:
                text_buffer.append(remaining)
                buffer_len += len(remaining)
            continue

        # Check for Tables (Keep tables as their own chunks)
        if identify_chunk_kind(text, active_in_refs) == ChunkKind.TABLE:
            flush_buffer()
            table_text = text[:MAX_TABLE_CHARS]
            chunk_id = f"{doc_id}_p{page}_c{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page,
                    section_heading=active_section,
                    text=table_text,
                    kind=ChunkKind.TABLE,
                )
            )
            chunk_idx += 1
            continue

        # Check for Figure references
        if identify_chunk_kind(text, active_in_refs) == ChunkKind.FIGURE_REF:
            flush_buffer()
            chunk_id = f"{doc_id}_p{page}_c{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page,
                    section_heading=active_section,
                    text=text,
                    kind=ChunkKind.FIGURE_REF,
                )
            )
            chunk_idx += 1
            continue

        # Regular Text / References block
        projected_len = buffer_len + len(text) + (2 if buffer_len > 0 else 0)
        if projected_len <= MAX_CHUNK_CHARS:
            text_buffer.append(text)
            buffer_len = projected_len
        else:
            # Buffer full; flush and start new buffer
            flush_buffer()
            text_buffer.append(text)
            buffer_len = len(text)

    flush_buffer()
    return chunks, active_section, active_in_refs
