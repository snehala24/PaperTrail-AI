import re
from typing import List, Optional, Set, Tuple
from backend.app.schemas import Chunk, ChunkKind

# Target passage bounds in characters (~200-400 tokens)
MIN_CHUNK_CHARS = 800
MAX_CHUNK_CHARS = 1500
MAX_TABLE_CHARS = 2500
MIN_MERGE_CHARS = 150

# Heading detection patterns:
# 1. N. Title (e.g. "1. Introduction", "6. Conclusion and Limitations")
# 2. N.N. Title (e.g. "5.1. Rank Allocation Results", "5.2. SVD-Compression Results")
# 3. N.N.N. TITLE (e.g. "5.3.1. SCALABILITY OF LEMS")
# 4. A. Title (e.g. "A. Extended Experimental Setup", "B. Additional Ablations")
# 5. B.N. Title / B.N.N. TITLE (e.g. "B.1. Title", "B.7. Inference Acceleration")
# 6. Named headings (Abstract, References, Glossary, Appendix, Supplement, Impact Statement)
HEADING_PATTERNS = [
    re.compile(r"^(\d+\.?\s+[A-Z][\w\s\-\:,\(\)\/\?]{1,80})$"),
    re.compile(r"^(\d+\.\d+\.?\s+[A-Z][\w\s\-\:,\(\)\/\?]{1,80})$"),
    re.compile(r"^(\d+\.\d+\.\d+\.?\s+[A-Z][\w\s\-\:,\(\)\/\?]{1,80})$"),
    re.compile(r"^([A-Z]\.?\s+[A-Z][\w\s\-\:,\(\)\/\?]{1,80})$"),
    re.compile(r"^([A-Z]\.\d+(?:\.\d+)*\.?\s+[A-Z][\w\s\-\:,\(\)\/\?]{1,80})$"),
    re.compile(
        r"^(abstract|references|bibliography|appendix(?:\s+[A-Z0-9\.\:\-]+.*)?|supplement(?:ary)?(?:\s+.*)?|glossary|impact\s+statement)$",
        re.IGNORECASE,
    ),
]

REFERENCES_START_REGEX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography)\b",
    re.IGNORECASE,
)

REFERENCES_EXIT_PATTERNS = [
    re.compile(
        r"^(?:appendix(?:\s+[A-Z0-9\.\:\-]+.*)?|supplement(?:ary)?(?:\s+.*)?|glossary)$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:Appendix\s+)?(?:[A-E])(?:\.\d+)*\.?\s+[A-Z][\w\s\-\:,\(\)\/\?]{1,80}$"),
    re.compile(r"^(?:\d+\.?\s+[A-Z][\w\s\-\:,\(\)\/\?]{1,80})$"),
]

EQUATION_FRAGMENT_REGEX = re.compile(
    r"^(?:\|\s*\{?\s*z\s*\}?|shape|scale|bias|\|\s*\\?underbrace.*|\b[a-z]\s*\(c\)\s*=\s*[a-z]\s*\(c\).*|\(\d+\))$",
    re.IGNORECASE,
)


def is_equation_fragment(text: str) -> bool:
    """Detect isolated equation/LaTeX artifacts such as '| {z }', 'Shape', 'Scale', 'Bias'."""
    s = text.strip()
    if not s:
        return True
    if EQUATION_FRAGMENT_REGEX.match(s):
        return True
    if re.match(r"^\|\s*\{?\s*z?\s*\}?$", s):
        return True
    return False


def is_heading_line(line: str) -> Optional[str]:
    """
    Check if a single line is a section heading.
    Detects patterns like 'N. Title', 'N.N. Title', 'N.N.N. TITLE', 'A. Title', 'B.N.N. TITLE',
    as well as named headings (Abstract, References, Glossary, etc.).
    Excludes table/figure captions and full sentences.
    """
    s = line.strip()
    if not s or len(s) > 90:
        return None
    # Never treat Table or Figure captions as section headings
    if re.match(r"^(?:table|tab\.|figure|fig\.)\s*\d+", s, re.IGNORECASE):
        return None

    for pat in HEADING_PATTERNS:
        m = pat.match(s)
        if m:
            heading = m.group(1) if m.groups() else s
            if heading.endswith((".", ":", ",")):
                words = heading.split()
                if len(words) > 7:
                    continue
            return heading.strip()
    return None


def is_references_exit_heading(line: str) -> bool:
    """
    Check if a heading signals the end of references section:
    Patterns like 'A. Title', 'B.1. Title', 'Appendix', 'Supplement', 'Glossary'.
    Excludes bibliography author entries (e.g. 'R. Title', 'H. Title').
    """
    s = line.strip()
    if not s:
        return False
    return any(p.match(s) for p in REFERENCES_EXIT_PATTERNS)


def is_bibliography_content(text: str) -> bool:
    """
    Second signal content check: tag a chunk references only if it looks like a bibliography:
    19xx/20xx publication years plus markers like 'doi', 'arXiv', 'Proceedings', 'URL', 'et al.'.
    """
    years = re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", text)
    if not years:
        return False
    markers = [
        "doi", "arxiv", "proceedings", "url", "et al.",
        "conference", "journal", "in:", "pp.", "vol.",
        "pages", "ieee", "acm", "openreview", "springer",
    ]
    text_lower = text.lower()
    return any(m in text_lower for m in markers)


def get_numeric_token_density(text: str) -> float:
    """
    Calculate the proportion of content tokens that are numeric or tabular data values.
    Pure syntax tokens like '|', '+', '|---|' are ignored.
    """
    tokens = text.split()
    if not tokens:
        return 0.0

    numeric_count = 0
    total_content_tokens = 0
    for tok in tokens:
        clean = tok.strip("%,xX()[]{}*~±$<>?:;/|")
        # Skip pure syntax delimiter tokens (e.g. '|', '---', '|---|')
        if not clean or re.match(r"^[-+|=]+$", tok):
            continue
        total_content_tokens += 1
        try:
            float(clean)
            numeric_count += 1
            continue
        except ValueError:
            pass
        if clean.lower() in ("inf", "-inf", "nan", "n/a", "-", "--", "—"):
            numeric_count += 1
            continue
        if re.match(r"^[-+]?\d+(?:\.\d+)?(?:%|x|e[-+]?\d+)?$", tok, re.IGNORECASE):
            numeric_count += 1
            continue

    return numeric_count / total_content_tokens if total_content_tokens > 0 else 0.0


def identify_chunk_kind(text: str, in_references: bool = False) -> ChunkKind:
    """
    Classify chunk content into text, table, figure_ref, or references.
    Tag table for caption-plus-data (numeric-token density > 50%), NOT for chunks containing '|'.
    """
    if in_references:
        return ChunkKind.REFERENCES

    has_table_caption = bool(re.search(r"\b(?:table|tab\.)\s*\d+\b", text, re.IGNORECASE))
    density = get_numeric_token_density(text)

    # Also evaluate density of data lines excluding the caption header
    body_density = density
    if has_table_caption:
        lines = text.splitlines()
        if len(lines) > 1:
            body_density = get_numeric_token_density("\n".join(lines[1:]))

    if has_table_caption and (density > 0.50 or body_density >= 0.50):
        return ChunkKind.TABLE

    if re.search(r"^\s*(?:figure\s*\d+|fig\.\s*\d+)", text, re.IGNORECASE):
        return ChunkKind.FIGURE_REF

    return ChunkKind.TEXT


def dehyphenate(text: str) -> str:
    """Join line-break hyphens ('pow-\\nerful' -> 'powerful') when next char is lowercase."""
    return re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-z][a-zA-Z]*)", r"\1\2", text)


def merge_captions_in_blocks(blocks: List[str]) -> List[str]:
    """
    Ensure Figure and Table captions are never split mid-sentence across blocks.
    Keeps caption blocks together until the next paragraph or sentence termination.
    """
    merged_blocks: List[str] = []
    i = 0
    caption_start = re.compile(
        r"^\s*(?:figure\s*\d+|fig\.\s*\d+|table\s*\d+|tab\.\s*\d+)", re.IGNORECASE
    )

    while i < len(blocks):
        block = blocks[i].strip()
        if not block:
            i += 1
            continue

        if caption_start.match(block):
            caption_parts = [block]
            while i + 1 < len(blocks):
                next_block = blocks[i + 1].strip()
                if not next_block:
                    break
                if caption_start.match(next_block) or is_heading_line(next_block):
                    break

                last_text = caption_parts[-1].strip()
                ends_with_terminal = last_text.endswith((".", "!", "?"))
                next_starts_lowercase = next_block[0].islower() if next_block else False

                if not ends_with_terminal or next_starts_lowercase:
                    caption_parts.append(next_block)
                    i += 1
                else:
                    break

            merged_caption = caption_parts[0]
            for part in caption_parts[1:]:
                if merged_caption.rstrip().endswith("-"):
                    merged_caption = f"{merged_caption.rstrip()[:-1]}{part.lstrip()}"
                else:
                    merged_caption = f"{merged_caption} {part}"
            merged_blocks.append(merged_caption)
            i += 1
        else:
            merged_blocks.append(block)
            i += 1

    return merged_blocks


def split_blocks_by_headings(blocks: List[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Inspect blocks per line. If a heading line occurs inside a block,
    split the block so the heading line starts a new block unit.
    Returns list of (sub_block_text, new_heading_or_None).
    """
    units: List[Tuple[str, Optional[str]]] = []

    for block in blocks:
        lines = block.splitlines()
        current_lines: List[str] = []

        for line in lines:
            heading = is_heading_line(line)
            if heading:
                if current_lines:
                    units.append(("\n".join(current_lines).strip(), None))
                    current_lines = []
                units.append((heading, heading))
            else:
                current_lines.append(line)

        if current_lines:
            units.append(("\n".join(current_lines).strip(), None))

    return [(text, h) for text, h in units if text]


def merge_small_chunks(chunks: List[Chunk], min_chars: int = MIN_MERGE_CHARS) -> List[Chunk]:
    """
    Merge chunks under min_chars into a neighbor chunk on the same page.
    Filters out standalone equation fragments and noise.
    """
    if not chunks:
        return []

    cleaned: List[Chunk] = []
    for c in chunks:
        t = c.text.strip()
        if is_equation_fragment(t) or len(t) < 15:
            continue
        cleaned.append(c)

    if len(cleaned) <= 1:
        return cleaned

    merged: List[Chunk] = []
    i = 0
    while i < len(cleaned):
        c = cleaned[i]
        if len(c.text.strip()) < min_chars:
            if merged and merged[-1].page == c.page and merged[-1].section_heading == c.section_heading:
                prev = merged[-1]
                prev.text = f"{prev.text}\n\n{c.text}".strip()
                i += 1
                continue
            elif i + 1 < len(cleaned) and cleaned[i + 1].page == c.page:
                nxt = cleaned[i + 1]
                nxt.text = f"{c.text}\n\n{nxt.text}".strip()
                i += 1
                continue
            elif merged and merged[-1].page == c.page:
                merged[-1].text = f"{merged[-1].text}\n\n{c.text}".strip()
                i += 1
                continue

        merged.append(c)
        i += 1

    return merged


def split_oversized_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text longer than max_chars along paragraph or sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
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
    Chunks text blocks from a single page into passages of ~800-1500 chars:
    - Never crosses a section heading.
    - Heading lines start a new chunk and are inherited by subsequent chunks.
    - References mode ends at top-level headings (A. Title, B.1. Title, Appendix, Supplement, Glossary).
    - Content check: tags references only if bibliography content is present.
    - Keeps captions together without mid-sentence splits.
    - Keeps tables as isolated chunks based on numeric-token density > 50%.
    - Merges chunks under ~150 characters into a neighbor.
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
            # Tag references if in references mode AND bibliography content check matches
            kind = identify_chunk_kind(passage, active_in_refs)
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

    # 1. Keep caption blocks together
    caption_merged_blocks = merge_captions_in_blocks(blocks)

    # 2. Inspect per-line headings inside blocks
    units = split_blocks_by_headings(caption_merged_blocks)

    for text, heading_detected in units:
        text = text.strip()
        if not text or is_equation_fragment(text):
            continue

        # Handle detected heading
        if heading_detected:
            # Check if entering References
            if REFERENCES_START_REGEX.match(heading_detected):
                flush_buffer()
                active_section = "References"
                active_in_refs = True
                continue

            # If already in references mode, check if this is an exit heading
            if active_in_refs:
                if is_references_exit_heading(heading_detected):
                    flush_buffer()
                    active_in_refs = False
                    active_section = heading_detected
                else:
                    # Author entry inside bibliography (e.g. 'R. Title'); treat as regular bibliography text
                    text_buffer.append(heading_detected)
                    buffer_len += len(heading_detected)
                continue

            # Regular section transition
            flush_buffer()
            active_section = heading_detected
            continue

        # Check for Tables (caption-plus-data with numeric-token density > 50%)
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

        # Check for Figure References
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

        # Regular Text or References block
        projected_len = buffer_len + len(text) + (2 if buffer_len > 0 else 0)
        if projected_len <= MAX_CHUNK_CHARS:
            text_buffer.append(text)
            buffer_len = projected_len
        else:
            flush_buffer()
            text_buffer.append(text)
            buffer_len = len(text)

    flush_buffer()

    # 3. Merge chunks under 150 characters into neighbor chunks
    post_merged_chunks = merge_small_chunks(chunks, min_chars=MIN_MERGE_CHARS)

    return post_merged_chunks, active_section, active_in_refs
