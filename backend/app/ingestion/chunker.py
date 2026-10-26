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
# 5. B.N. Title / B.N.N. TITLE (e.g. "B.1. Title", "B.4. Llama 3.1-70B Results", "B.5.1. BITSTACK")
# 6. Named headings (Abstract, References, Glossary, Appendix, Supplement, Impact Statement)
HEADING_PATTERNS = [
    re.compile(r"^(\d{1,2}\.\s+[A-Z][A-Za-z0-9\s\-\(\)\/\,\?]{2,70})$"),
    re.compile(r"^(\d{1,2}\.\d{1,2}\.?\s+[A-Z][A-Za-z0-9\s\-\(\)\/\.\,\?]{2,70})$"),
    re.compile(r"^(\d{1,2}\.\d{1,2}\.\d{1,2}\.?\s+[A-Z][A-Za-z0-9\s\-\(\)\/\.\,\?]{2,70})$"),
    re.compile(r"^([A-Z]\.\s+[A-Z][A-Za-z0-9\s\-\(\)\/\,\?]{2,70})$"),
    re.compile(r"^([A-Z]\.\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z][A-Za-z0-9\s\-\(\)\/\.\,\?]{2,70})$"),
    re.compile(
        r"^(abstract|references|bibliography|glossary|impact\s+statement|appendix|supplement(?:ary)?(?:\s+(?:material|information))?)$",
        re.IGNORECASE,
    ),
]

NON_HEADING_UNITS = {
    "gb", "mb", "kb", "tb", "pb", "pm", "am", "gpu", "tpu", "ram", "vram", "cpu",
    "sec", "ms", "ns", "hz", "ghz", "mhz", "w", "kw", "mw", "v", "db", "flops", "params",
}

REFERENCES_START_REGEX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography)\b",
    re.IGNORECASE,
)

REFERENCES_EXIT_PATTERNS = [
    re.compile(
        r"^(?:appendix(?:\s+[A-Z0-9\.\:\-]+.*)?|supplement(?:ary)?(?:\s+(?:material|information))?|glossary)$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:Appendix\s+)?(?:[A-E])\.\s+[A-Z][A-Za-z0-9\s\-\(\)\/\,\?]{2,70}$"),
    re.compile(r"^(?:Appendix\s+)?(?:[A-E])\.\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z][A-Za-z0-9\s\-\(\)\/\.\,\?]{2,70}$"),
]

EQUATION_FRAGMENT_REGEX = re.compile(
    r"^(?:\|\s*\{?\s*z\s*\}?|shape|scale|bias|\|\s*\\?underbrace.*|\b[a-z]\s*\(c\)\s*=\s*[a-z]\s*\(c\).*)$",
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
    Excludes table/figure captions, list items with explanatory text, units, and dates.
    """
    s = line.strip()
    if not s or len(s) > 85:
        return None
    # Exclude lines ending with hyphen (line break continuation)
    if s.endswith("-"):
        return None
    # Never treat Table or Figure captions as section headings
    if re.match(r"^(?:table|tab\.|figure|fig\.)\s*\d+", s, re.IGNORECASE):
        return None
    # Exclude list items with colon followed by explanatory prose, e.g. '1. Granularity (Shape): Solving...'
    if ":" in s:
        parts = s.split(":", 1)
        after_colon = parts[1].strip()
        if after_colon:
            words = after_colon.split()
            # If colon is followed by prose (> 2 words or starts with lowercase word)
            if len(words) > 2 or (words and words[0][0].islower()):
                return None
    # Exclude full sentences starting with numbers and conjunctions like '3. While gains are marginal...'
    if re.match(r"^\d+\.\s+(?:while|because|although|since|if|however|therefore|moreover)\b", s, re.IGNORECASE):
        return None

    for pat in HEADING_PATTERNS:
        m = pat.match(s)
        if m:
            heading = m.group(1) if m.groups() else s
            # Check unit filter for lines like '5.07 GB' or '5.00 PM, 15 March 2021'
            title_part = re.sub(r"^(?:[A-Z0-9\.]+\s+)", "", heading).strip()
            first_word = title_part.split()[0].lower().rstrip(".,;") if title_part.split() else ""
            if first_word in NON_HEADING_UNITS:
                return None
            # Exclude lines containing dates like ', 15 March 2021'
            if re.search(r",\s*\d{1,2}\s+[A-Za-z]+\s+\d{4}", heading):
                return None
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
    Tag figure_ref only for standalone figure captions/references (short, <= 350 chars with caption punctuation).
    Body paragraphs discussing results are always tagged text.
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

    # Standalone figure caption requires punctuation ('Figure N:' or 'Figure N.') and short length (<= 350 chars)
    is_fig_caption = bool(re.match(r"^\s*(?:figure|fig\.)\s*\d+[:.]\s+", text, re.IGNORECASE))
    if is_fig_caption and len(text.strip()) <= 350:
        return ChunkKind.FIGURE_REF

    return ChunkKind.TEXT


def dehyphenate(text: str, doc_tokens_lower: Optional[Set[str]] = None) -> str:
    """
    Join line-break hyphens ('pow-\\nerful' -> 'powerful') only when plausible:
    1. Preserves suspended hyphens with coordinating conjunctions ('layer-\\nand' -> 'layer- and')
    2. Preserves multi-hyphen compounds ('mixture-\\nof-experts' -> 'mixture-of-experts')
    3. Preserves acronym/title-case compounds ('KFAC-\\nSVD' -> 'KFAC-SVD', 'ARC-\\nChallenge' -> 'ARC-Challenge')
    4. Joins when the combined word appears elsewhere in the document as a natural token
    5. Preserves hyphen when both fragments are independent words in doc, and joined is not ('sequence-\\ngradient' -> 'sequence-gradient')
    6. Joins syllable-break fragments ('pow-\\nerful' -> 'powerful', 'ap-\\nproximation' -> 'approximation')
    """
    if not text:
        return ""

    if doc_tokens_lower is None:
        clean = set(re.findall(r"\b[a-zA-Z]{3,}\b(?!\s*-\s*\n)", text))
        doc_tokens_lower = {t.lower() for t in clean}

    def _replace(match: re.Match) -> str:
        w1 = match.group(1)
        w2 = match.group(2)

        # 1. Suspended hyphen followed by coordinating conjunction ('layer-\nand' -> 'layer- and')
        conj_match = re.match(r"^(and|or|nor|vs|to)$", w2, re.IGNORECASE)
        if conj_match:
            return f"{w1}- {w2}"

        # 2. Multi-hyphen compound ('mixture-\nof-experts' -> 'mixture-of-experts')
        if "-" in w2:
            return f"{w1}-{w2}"

        # 3. Uppercase acronym or title-case compound ('KFAC-\nSVD', 'ARC-\nChallenge')
        if w1.isupper() or (len(w1) >= 2 and w1[0].isupper() and w2[0].isupper()):
            return f"{w1}-{w2}"

        joined = w1 + w2
        joined_lower = joined.lower()

        # 4. If joined word appears naturally in document: join!
        if joined_lower in doc_tokens_lower:
            return joined

        w1_lower = w1.lower()
        w2_lower = w2.lower()

        # 5. If BOTH w1 and w2 are independent words, and joined is NOT:
        # It is a hyphenated compound ('sequence-gradient', 'loss-aware', 'layer-output')
        if w1_lower in doc_tokens_lower and w2_lower in doc_tokens_lower and joined_lower not in doc_tokens_lower:
            return f"{w1}-{w2}"

        # 6. Default: split syllables of a single word ('pow-erful', 'approx-imation')
        return joined

    pattern = re.compile(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+(?:-[a-zA-Z]+)?)")
    return pattern.sub(_replace, text)


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
    Also stitches 2-line wrapped heading titles (e.g. '4.2. Global Rank Allocation...\nModeling (LEMS)').
    Returns list of (sub_block_text, new_heading_or_None).
    """
    units: List[Tuple[str, Optional[str]]] = []

    for block in blocks:
        lines = block.splitlines()
        current_lines: List[str] = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            heading = is_heading_line(line)
            if heading:
                # Check if next line is a continuation of this heading in the same block
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    if next_line and len(next_line) < 40 and not is_heading_line(next_line):
                        if next_line[0].isupper() or next_line.startswith("("):
                            heading = f"{heading} {next_line}"
                            line = f"{line} {next_line}"
                            idx += 1
                if current_lines:
                    units.append(("\n".join(current_lines).strip(), None))
                    current_lines = []
                units.append((line, heading))
            else:
                current_lines.append(line)
            idx += 1

        if current_lines:
            units.append(("\n".join(current_lines).strip(), None))

    return [(text, h) for text, h in units if text]


def merge_small_chunks(chunks: List[Chunk], min_chars: int = MIN_MERGE_CHARS) -> List[Chunk]:
    """
    Merge chunks under min_chars into a neighbor text chunk on the same page and section.
    Preserves figure_ref and table chunks isolated so they never absorb body prose.
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
        # Never merge table or figure_ref chunks into neighbors or absorb into them
        if c.kind in (ChunkKind.FIGURE_REF, ChunkKind.TABLE):
            merged.append(c)
            i += 1
            continue

        if len(c.text.strip()) < min_chars:
            # Only merge into previous if prev has same page, same section, and is TEXT
            if (
                merged
                and merged[-1].page == c.page
                and merged[-1].section_heading == c.section_heading
                and merged[-1].kind == ChunkKind.TEXT
            ):
                merged[-1].text = f"{merged[-1].text}\n\n{c.text}".strip()
                i += 1
                continue
            # Or merge into next if next has same page, same section, and is TEXT
            elif (
                i + 1 < len(cleaned)
                and cleaned[i + 1].page == c.page
                and cleaned[i + 1].section_heading == c.section_heading
                and cleaned[i + 1].kind == ChunkKind.TEXT
            ):
                nxt = cleaned[i + 1]
                nxt.text = f"{c.text}\n\n{nxt.text}".strip()
                i += 1
                continue
            # Fallback: merge into prev on same page if prev is TEXT
            elif (
                merged
                and merged[-1].page == c.page
                and merged[-1].kind == ChunkKind.TEXT
            ):
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
                text_buffer.append(text)
                buffer_len += len(text)
                continue

            # If already in references mode, check if this is an exit heading
            if active_in_refs:
                if is_references_exit_heading(heading_detected):
                    flush_buffer()
                    active_in_refs = False
                    active_section = heading_detected
                    text_buffer.append(text)
                    buffer_len += len(text)
                else:
                    # Author entry inside bibliography (e.g. 'R. Title'); treat as regular bibliography text
                    text_buffer.append(heading_detected)
                    buffer_len += len(heading_detected)
                continue

            # Regular section transition: flush prior section buffer and start new chunk with heading line retained
            flush_buffer()
            active_section = heading_detected
            text_buffer.append(text)
            buffer_len += len(text)
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

    # 4. Re-number chunks sequentially so chunk IDs per page are strictly gapless (c1, c2, c3...)
    renumbered_chunks: List[Chunk] = []
    for idx, c in enumerate(post_merged_chunks, start=start_chunk_idx):
        c.chunk_id = f"{doc_id}_p{page}_c{idx}"
        renumbered_chunks.append(c)

    return renumbered_chunks, active_section, active_in_refs


def chunk_index_text(chunk: Chunk) -> str:
    """
    Format section heading and chunk text for indexing and retrieval
    without duplicating the heading if it is already present at the start of text.
    """
    text = chunk.text.strip()
    heading = (chunk.section_heading or "").strip()
    if not heading:
        return text

    # Check if the text already begins with the heading
    first_line = text.split("\n", 1)[0].strip()
    if first_line.lower() == heading.lower() or text.lower().startswith(heading.lower()):
        return text

    return f"{heading}\n{text}"
