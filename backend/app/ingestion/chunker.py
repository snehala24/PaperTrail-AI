import re
from typing import List, Optional, Tuple
from backend.app.schemas import Chunk, ChunkKind

# Section heading patterns: e.g. "1. Introduction", "Section 2. Related Work", "Abstract", "References"
SECTION_REGEX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+[\w\s]{2,60}|abstract|introduction|background|methodology|experiments?|results?|discussion|conclusion|references)$",
    re.IGNORECASE
)

TABLE_REGEX = re.compile(r"^\s*(?:table\s*\d+|\|\s*[^|]+\s*\|)", re.IGNORECASE)
FIGURE_REGEX = re.compile(r"^\s*(?:figure\s*\d+|fig\.\s*\d+)", re.IGNORECASE)


def identify_chunk_kind(text: str) -> ChunkKind:
    """Classify chunk content into text, table, or figure_ref."""
    stripped = text.strip()
    if TABLE_REGEX.match(stripped) or ("|" in stripped and "\n|" in stripped):
        return ChunkKind.TABLE
    if FIGURE_REGEX.match(stripped):
        return ChunkKind.FIGURE_REF
    return ChunkKind.TEXT


def chunk_page_blocks(
    doc_id: str,
    page: int,
    blocks: List[str],
    current_section: Optional[str] = None
) -> Tuple[List[Chunk], Optional[str]]:
    """
    Chunks a list of text blocks from a single page.
    Tracks active section headings across blocks and assigns chunk IDs.
    """
    chunks: List[Chunk] = []
    chunk_idx = 1
    active_section = current_section

    for block in blocks:
        text = block.strip()
        if not text:
            continue

        first_line = text.split("\n", 1)[0].strip()
        if SECTION_REGEX.match(first_line) and len(first_line) < 80:
            active_section = first_line

        kind = identify_chunk_kind(text)
        chunk_id = f"{doc_id}_p{page}_c{chunk_idx}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page=page,
                section_heading=active_section,
                text=text,
                kind=kind,
            )
        )
        chunk_idx += 1

    return chunks, active_section
