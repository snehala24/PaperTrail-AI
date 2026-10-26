import pytest
from backend.app.ingestion.chunker import chunk_page_blocks, ChunkKind
from backend.app.schemas import Chunk


def test_chunker_merges_small_consecutive_blocks():
    """Consecutive blocks within same page & section should be merged into 800-1500 chars."""
    doc_id = "doc-test"
    page = 1
    # 5 blocks of 250 characters each (total 1250 chars)
    blocks = [
        "Block 1: " + "a" * 240,
        "Block 2: " + "b" * 240,
        "Block 3: " + "c" * 240,
        "Block 4: " + "d" * 240,
        "Block 5: " + "e" * 240,
    ]
    chunks, _, _ = chunk_page_blocks(doc_id, page, blocks, current_section="Introduction")

    # Instead of 5 separate chunks, they should be merged into 1 or 2 passages
    assert len(chunks) < 5
    assert len(chunks) in (1, 2)
    for c in chunks:
        assert len(c.text) >= 400
        assert c.section_heading == "Introduction"
        assert c.kind == ChunkKind.TEXT


def test_chunker_never_crosses_section_heading():
    """A new section heading must force a flush and never be merged into the previous section."""
    doc_id = "doc-test"
    page = 2
    blocks = [
        "1. Introduction\nFirst section content.",
        "More details on the introduction.",
        "2. Related Work\nSecond section starts here.",
        "Discussion of prior literature.",
    ]
    chunks, active_sec, _ = chunk_page_blocks(doc_id, page, blocks)

    intro_chunks = [c for c in chunks if "Introduction" in (c.section_heading or "")]
    related_chunks = [c for c in chunks if "Related Work" in (c.section_heading or "")]

    assert len(intro_chunks) >= 1
    assert len(related_chunks) >= 1
    # No chunk should contain both sections
    for c in intro_chunks:
        assert "Related Work" not in c.text
    for c in related_chunks:
        assert "First section content" not in c.text


def test_chunker_keeps_tables_isolated():
    """Tables must be isolated into dedicated table chunks."""
    doc_id = "doc-test"
    page = 3
    blocks = [
        "We report the experimental comparisons across multiple architectures below:",
        "Table 1: Benchmark Results\n| Model | Accuracy | F1 |\n| Baseline | 82.1 | 80.5 |\n| Proposed | 88.4 | 87.2 |",
        "As seen in the results table, the proposed model significantly outperforms the baseline.",
    ]
    chunks, _, _ = chunk_page_blocks(doc_id, page, blocks, current_section="Results")

    table_chunks = [c for c in chunks if c.kind == ChunkKind.TABLE]
    assert len(table_chunks) == 1
    assert "Table 1" in table_chunks[0].text
    # Preceding and following text should not be merged into the table chunk
    assert "We report the experimental" not in table_chunks[0].text
    assert "significantly outperforms" not in table_chunks[0].text


def test_chunker_tags_references_kind():
    """Everything after References heading must be tagged ChunkKind.REFERENCES."""
    doc_id = "doc-test"
    page = 10
    blocks = [
        "6. Conclusion\nWe have introduced PaperTrail AI.",
        "References\n[1] Vaswani et al. Attention is all you need. NeurIPS 2017.",
        "[2] Devlin et al. BERT: Pre-training of deep bidirectional transformers. NAACL 2019.",
    ]
    chunks, active_sec, in_refs = chunk_page_blocks(doc_id, page, blocks, current_section="Conclusion")

    ref_chunks = [c for c in chunks if c.kind == ChunkKind.REFERENCES]
    assert len(ref_chunks) >= 1
    for rc in ref_chunks:
        assert rc.kind == ChunkKind.REFERENCES
        assert rc.section_heading == "References"


def test_chunker_34_page_target_chunk_count():
    """A realistic 34-page document must produce between 80 and 200 chunks (not 963!)."""
    doc_id = "doc-34pages"
    all_chunks = []
    current_section = "1. Introduction"
    in_references = False

    # Simulate 34 pages with ~5-8 text blocks per page (averaging ~3000 chars per page)
    for page in range(1, 35):
        if page == 30:
            current_section = "References"
            in_references = True

        blocks = []
        if page == 1:
            blocks.append("1. Introduction")
        elif page == 10:
            blocks.append("2. Methodology")
        elif page == 20:
            blocks.append("3. Experiments")
        elif page == 30:
            blocks.append("References")

        # Add 6 blocks of 400 chars each = ~2400 chars per page
        for b_idx in range(6):
            if in_references:
                blocks.append(f"[{page * 6 + b_idx}] Author et al. Paper title and conference proceedings, 202{b_idx}.")
            else:
                blocks.append(f"Page {page} paragraph {b_idx}: " + ("word " * 50))

        page_chunks, current_section, in_references = chunk_page_blocks(
            doc_id=doc_id,
            page=page,
            blocks=blocks,
            current_section=current_section,
            in_references=in_references,
        )
        all_chunks.extend(page_chunks)

    # 34 pages should produce between 60 and 200 chunks
    assert 60 <= len(all_chunks) <= 200, f"Expected 60-200 chunks for 34 pages, got {len(all_chunks)}"
