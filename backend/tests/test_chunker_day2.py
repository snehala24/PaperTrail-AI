import io
import pytest
from backend.app.schemas import Chunk, ChunkKind, Document, CHUNKER_VERSION
from backend.app.schemas.eval import normalize_text_span, is_chunk_match
from backend.app.ingestion.chunker import (
    chunk_page_blocks,
    is_heading_line,
    is_references_exit_heading,
    is_bibliography_content,
    get_numeric_token_density,
    identify_chunk_kind,
    dehyphenate,
    merge_captions_in_blocks,
    merge_small_chunks,
    is_equation_fragment,
)
from backend.app.ingestion.parser import clean_page_blocks, parse_pdf_bytes
from backend.app.core.storage import DocumentStorage
import pymupdf as fitz


def test_references_mode_ends_at_next_top_level_heading():
    """References mode ends at top-level headings (Glossary, Appendix, A. Title, B.1. Title)."""
    doc_id = "doc-ref-test"
    blocks_p9 = [
        "6. Conclusion\nSummary of paper discussing results and future directions in depth." * 3,
        "References\nAkiba et al. Optuna hyperparameter optimization. In Proceedings of 2019." * 3,
        "Devlin et al. BERT. In Proceedings of NAACL 2019." * 3,
    ]
    chunks_p9, sec_p9, in_refs_p9 = chunk_page_blocks(doc_id, 9, blocks_p9, current_section="6. Conclusion")
    assert in_refs_p9 is True
    ref_chunks = [c for c in chunks_p9 if c.kind == ChunkKind.REFERENCES]
    assert len(ref_chunks) >= 1

    # Page 12: Glossary heading ends references mode
    blocks_p12 = [
        "Glossary\nARS Adaptive rank selection abbreviations and definitions explained." * 3,
        "ASVD Activation-aware singular value decomposition definition details." * 3,
    ]
    chunks_p12, sec_p12, in_refs_p12 = chunk_page_blocks(
        doc_id, 12, blocks_p12, current_section=sec_p9, in_references=in_refs_p9
    )
    assert in_refs_p12 is False
    assert sec_p12 == "Glossary"
    assert all(c.kind == ChunkKind.TEXT for c in chunks_p12)

    # Page 14: Appendix A ends references if not already ended
    blocks_p14 = [
        "A. Extended Experimental Setup\nDetails on models and search baselines across architectures." * 3,
        "B.1. Extended Explanations\nAblations on algorithm components and hyperparameter sensitivity." * 3,
    ]
    chunks_p14, sec_p14, in_refs_p14 = chunk_page_blocks(
        doc_id, 14, blocks_p14, current_section="References", in_references=True
    )
    assert in_refs_p14 is False
    assert any("A. Extended Experimental Setup" in (c.section_heading or "") for c in chunks_p14)
    assert all(c.kind == ChunkKind.TEXT for c in chunks_p14)


def test_references_content_check():
    """Verify bibliography content check requires years and publication markers."""
    bib_text = "Vaswani et al. Attention is all you need. In Proceedings of NeurIPS 2017. URL https://arxiv.org/abs/1706.03762."
    assert is_bibliography_content(bib_text) is True

    # Regular text without years or markers
    normal_text = "In this work, we address both challenges by introducing LEMS for global rank allocation."
    assert is_bibliography_content(normal_text) is False


def test_heading_detection_per_line_and_inheritance():
    """Detects headings per line inside blocks; subsequent chunks inherit the heading."""
    doc_id = "doc-heading-test"
    blocks = [
        "Preceding paragraph in evaluation section providing comprehensive context." * 3,
        "5.1. Rank Allocation Results\nContent for 5.1 discussing results in extensive detail across benchmarks." * 3,
        "5.2. SVD-Compression Results\nWe evaluate KFAC-SVD across three LLM families in rigorous tests." * 3,
        "Additional discussion of compression performance across models." * 3,
        "B.7. Inference Acceleration\nInference throughput reaches a substantial 2.01x speedup over baseline." * 3,
    ]
    chunks, _, _ = chunk_page_blocks(doc_id, 1, blocks, current_section="5. Evaluation")

    c_51 = [c for c in chunks if c.section_heading == "5.1. Rank Allocation Results"]
    c_52 = [c for c in chunks if c.section_heading == "5.2. SVD-Compression Results"]
    c_b7 = [c for c in chunks if c.section_heading == "B.7. Inference Acceleration"]

    assert len(c_51) >= 1
    assert len(c_52) >= 1
    assert len(c_b7) >= 1

    # 5.2 content must not be labeled 5.1
    for c in c_52:
        assert "5.1" not in c.section_heading
        assert "KFAC-SVD across three LLM families" in c.text

    # B.7 content must be labeled B.7
    assert "2.01x speedup" in c_b7[0].text
    assert c_b7[0].section_heading == "B.7. Inference Acceleration"


def test_noise_filtering_running_headers_and_page_numbers():
    """Running headers, standalone page numbers, and rotated arXiv sidebar text are dropped."""
    running_headers = {"Advancing SVD-based LLM Compression via Layer-Wise Error Model Search"}
    raw_blocks = [
        (0, 0, 100, 20, "Advancing SVD-based LLM Compression via Layer-Wise Error Model Search\n", 0, 0),
        (0, 50, 100, 100, "1. Introduction\nValid research text describing models.\n", 1, 0),
        (0, 700, 100, 720, "14\n", 2, 0),
        (0, 750, 100, 780, "arXiv:2610.32878v1 [cs.AI] 23 Oct 2026\n", 3, 0),
    ]

    cleaned = clean_page_blocks(raw_blocks, running_headers)
    assert len(cleaned) == 1
    assert "1. Introduction" in cleaned[0]
    assert "14" not in cleaned[0]
    assert "arXiv:2610" not in cleaned[0]
    assert "Advancing SVD-based" not in cleaned[0]


def test_page_1_title_not_dropped_by_running_headers():
    """Title on page 1 (page_idx=0) is preserved even if it matches repeating running header text."""
    running_headers = {"Advancing SVD-based LLM Compression via Layer-Wise Error Model Search"}
    raw_blocks = [
        (0, 0, 100, 20, "Advancing SVD-based LLM Compression via Layer-Wise Error Model Search\n", 0, 0),
        (0, 50, 100, 100, "1. Introduction\nValid research text.\n", 1, 0),
    ]
    cleaned = clean_page_blocks(raw_blocks, running_headers, page_idx=0)
    assert len(cleaned) == 2
    assert "Advancing SVD-based" in cleaned[0]
    assert "1. Introduction" in cleaned[1]


def test_caption_never_split_mid_sentence():
    """Never split Figure/Table caption mid-sentence across consecutive blocks."""
    blocks = [
        "Figure 11. Perplexity of Qwen3-8B when compressed using dif-\n",
        "ferent sample counts and different sampling seeds when removing 20% of parameters with KFAC-SVD + LEMS.\n",
        "Subsequent paragraph text analyzing the figure observations in detail.",
    ]
    merged = merge_captions_in_blocks(blocks)
    assert len(merged) == 2
    assert "Figure 11. Perplexity of Qwen3-8B when compressed using different sample counts" in merged[0]
    assert "Subsequent paragraph text" in merged[1]


def test_merge_small_chunks_and_drop_equation_fragments():
    """Equation fragments (| {z }, Shape, Scale) dropped; chunks < 150 chars merged into neighbor."""
    assert is_equation_fragment("| {z }") is True
    assert is_equation_fragment("Shape") is True
    assert is_equation_fragment("Scale") is True
    assert is_equation_fragment("Bias") is True

    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            page=1,
            section_heading="4. Methodology",
            text="First substantial paragraph discussing the formulation of global rank allocation in detail." * 2,
            kind=ChunkKind.TEXT,
        ),
        Chunk(
            chunk_id="c2",
            doc_id="d1",
            page=1,
            section_heading="4. Methodology",
            text="| {z }",
            kind=ChunkKind.TEXT,
        ),
        Chunk(
            chunk_id="c3",
            doc_id="d1",
            page=1,
            section_heading="4. Methodology",
            text="Shape",
            kind=ChunkKind.TEXT,
        ),
        Chunk(
            chunk_id="c4",
            doc_id="d1",
            page=1,
            section_heading="4. Methodology",
            text="Short note continuing.",
            kind=ChunkKind.TEXT,
        ),
    ]

    result = merge_small_chunks(chunks, min_chars=150)
    assert len(result) == 1
    # Equation fragments are dropped
    assert "| {z }" not in result[0].text
    assert result[0].text.strip() != "Shape"
    # Small chunk merged into c1
    assert "Short note continuing." in result[0].text


def test_table_kind_numeric_density():
    """Tag table for caption-plus-data (numeric-token density > 50%), not for pipes."""
    # Table caption + numeric data (>50% numeric density)
    table_text = (
        "Table 1. Benchmark Results\n"
        "Baseline 5.25 8.10 27.72 63.95 6.14 9.47 9.90 63.34 9.71 15.52 15.43 62.03\n"
        "Proposed 4.12 6.80 22.10 68.20 5.01 7.20 8.10 67.50 8.10 12.30 11.20 66.80"
    )
    density = get_numeric_token_density(table_text)
    assert density > 0.50
    assert identify_chunk_kind(table_text) == ChunkKind.TABLE

    # Math or text with pipe '|' but < 50% numeric density
    pipe_text = "Let ||x|| be the Euclidean norm such that |x - y| < epsilon in equation (5)."
    assert get_numeric_token_density(pipe_text) < 0.50
    assert identify_chunk_kind(pipe_text) == ChunkKind.TEXT


def test_dehyphenation_and_gold_matcher():
    """Join line-break hyphens when next char is lowercase; normalize hyphens in matcher."""
    raw = "Low-rank compression offers a pow-\nerful strategy for mod-\nels."
    dehyphed = dehyphenate(raw)
    assert "powerful" in dehyphed
    assert "models" in dehyphed

    # Regression tests for glued-words defects:
    # 1. Multi-hyphen compound must preserve hyphens
    moe_raw = "vision-language models (VLMs) and mixture-\nof-experts (MoE) models."
    moe_dehyphed = dehyphenate(moe_raw)
    assert "mixture-of-experts" in moe_dehyphed
    assert "mixtureof" not in moe_dehyphed

    # 2. Suspended hyphen with coordinating conjunction
    layer_raw = "it cannot capture layer-\nand type-specific importance deviations."
    layer_dehyphed = dehyphenate(layer_raw)
    assert "layer- and type-specific" in layer_dehyphed
    assert "layerand" not in layer_dehyphed

    # 3. Hyphenated compound modifier in context
    seq_raw = "sequence data and gradient computation for sequence-\ngradient fitting."
    seq_dehyphed = dehyphenate(seq_raw)
    assert "sequence-gradient" in seq_dehyphed
    assert "sequencegradient" not in seq_dehyphed

    # 4. Acronym compound
    kfac_raw = "addressing the KFAC-\nSVD rank bottleneck."
    kfac_dehyphed = dehyphenate(kfac_raw)
    assert "KFAC-SVD" in kfac_dehyphed
    assert "KFACSVD" not in kfac_dehyphed

    # Span matcher with hyphens and whitespace normalization
    chunk_text = "The system delivers state-of-the-art compression with powerful latency reduction."
    gold_span_hyphen = "state‑of‑the‑art"  # Non-breaking unicode hyphen
    gold_span_linebreak = "pow-\nerful"

    assert is_chunk_match(1, chunk_text, 1, gold_span_hyphen) is True
    assert is_chunk_match(1, chunk_text, 1, gold_span_linebreak) is True


def test_reingest_chunker_version(tmp_path):
    """If stored document has older chunker version, save_document overwrites record."""
    store = DocumentStorage(base_dir=tmp_path)
    old_doc = Document(
        doc_id="doc_old",
        title="Old Paper",
        filename="old.pdf",
        sha256="a" * 64,
        page_count=5,
        chunker_version="0.1.0",
    )
    old_chunks = [
        Chunk(chunk_id="c1", doc_id="doc_old", page=1, text="Old chunk text", kind=ChunkKind.TEXT)
    ]
    store.save_document(old_doc, old_chunks)

    # Ingest document with updated chunker version (0.2.0)
    new_doc = Document(
        doc_id="doc_old",
        title="Updated Paper",
        filename="old.pdf",
        sha256="a" * 64,
        page_count=5,
        chunker_version="0.2.0",
    )
    new_chunks = [
        Chunk(chunk_id="c1", doc_id="doc_old", page=1, text="New merged chunk text", kind=ChunkKind.TEXT),
        Chunk(chunk_id="c2", doc_id="doc_old", page=2, text="Second chunk text", kind=ChunkKind.TEXT),
    ]
    saved = store.save_document(new_doc, new_chunks)
    assert saved.chunker_version == "0.2.0"

    # Reload from storage to verify persistence was updated
    loaded = store.get_document("doc_old")
    assert loaded.chunker_version == "0.2.0"
    loaded_chunks = store.get_chunks("doc_old")
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0].text == "New merged chunk text"


def test_chunker_v022_equation_numbers_survive():
    """C1: Standalone equation numbers like (15) are NOT treated as equation fragments."""
    assert is_equation_fragment("(15)") is False
    assert is_equation_fragment("(1)") is False
    assert is_equation_fragment("| {z }") is True
    assert is_equation_fragment("Shape") is True


def test_page_number_noise_margin():
    """C2: Drop bare digit-only lines only in top/bottom ~8% margin of page."""
    # Top margin (y1 <= 0.08 * 800 = 64) -> dropped
    top_block = [(50, 20, 100, 35, "14", 0, 0)]
    cleaned_top = clean_page_blocks(top_block, running_headers=set(), page_idx=2, page_height=800.0)
    assert cleaned_top == []

    # Bottom margin (y0 >= 0.92 * 800 = 736) -> dropped
    bottom_block = [(50, 750, 100, 770, "14", 0, 0)]
    cleaned_bottom = clean_page_blocks(bottom_block, running_headers=set(), page_idx=2, page_height=800.0)
    assert cleaned_bottom == []

    # Middle of page (y0=300, y1=320) -> kept
    mid_block = [(50, 300, 100, 320, "14", 0, 0)]
    cleaned_mid = clean_page_blocks(mid_block, running_headers=set(), page_idx=2, page_height=800.0)
    assert cleaned_mid == ["14"]


def test_caption_isolation_never_merges_body_prose():
    """C3: When a caption ends, do NOT merge following body prose into the caption chunk."""
    fig_chunk = Chunk(
        chunk_id="doc_p1_c1",
        doc_id="doc",
        page=1,
        text="Figure 1: Architecture diagram showing memory hierarchy.",
        kind=ChunkKind.FIGURE_REF,
    )
    small_prose_chunk = Chunk(
        chunk_id="doc_p1_c2",
        doc_id="doc",
        page=1,
        text="This is a short body paragraph.",
        kind=ChunkKind.TEXT,
    )
    merged = merge_small_chunks([fig_chunk, small_prose_chunk], min_chars=150)
    # Figure chunk must remain isolated and not merge with body prose
    assert len(merged) == 2
    assert merged[0].kind == ChunkKind.FIGURE_REF
    assert merged[1].kind == ChunkKind.TEXT
    assert "body paragraph" not in merged[0].text


def test_gapless_chunk_ids_after_merging():
    """C5: Sequential 1-based chunk IDs per page are strictly gapless (c1, c2, c3...)."""
    doc_id = "doc_gapless"
    blocks = [
        "1. Introduction\nFirst section lead paragraph with comprehensive details about method." * 4,
        "Figure 1: Isolated figure diagram caption with overview.",
        "Short paragraph that merges into previous text chunk.",
        "Table 1: Benchmark results\n10 20 30 40 50 60 70 80 90 100",
    ]
    chunks, _, _ = chunk_page_blocks(doc_id, page=7, blocks=blocks, current_section="1. Introduction")
    chunk_numbers = [int(c.chunk_id.split("_c")[-1]) for c in chunks]
    expected_numbers = list(range(1, len(chunks) + 1))
    assert chunk_numbers == expected_numbers


def test_chunk_index_text_helper():
    """C6: chunk_index_text formats section + text without duplicating leading headings."""
    from backend.app.ingestion.chunker import chunk_index_text

    # Heading not in text -> prepend
    c1 = Chunk(
        chunk_id="c1",
        doc_id="d1",
        page=1,
        section_heading="2. Methodology",
        text="We propose an optimization algorithm.",
        kind=ChunkKind.TEXT,
    )
    assert chunk_index_text(c1) == "2. Methodology\nWe propose an optimization algorithm."

    # Heading already at start of text -> do not duplicate
    c2 = Chunk(
        chunk_id="c2",
        doc_id="d1",
        page=1,
        section_heading="2. Methodology",
        text="2. Methodology\nWe propose an optimization algorithm.",
        kind=ChunkKind.TEXT,
    )
    assert chunk_index_text(c2) == "2. Methodology\nWe propose an optimization algorithm."


def test_real_paper_verification_c8():
    """C8: Real paper verification on data/2610.32878v1.pdf."""
    from pathlib import Path
    pdf_path = Path("data/2610.32878v1.pdf")
    if not pdf_path.exists():
        pytest.skip("Real paper data/2610.32878v1.pdf not present")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    doc, chunks = parse_pdf_bytes(pdf_bytes, "2610.32878v1.pdf")
    # Total chunks in [80, 200]
    assert 80 <= len(chunks) <= 200
    # References chunks only on pages 9-11
    ref_pages = sorted(list(set(c.page for c in chunks if c.kind == ChunkKind.REFERENCES)))
    assert ref_pages == [9, 10, 11]
    # No figure_ref chunk > 700 chars
    assert all(len(c.text) <= 700 for c in chunks if c.kind == ChunkKind.FIGURE_REF)
    # (15) appears in page-5 chunk text
    p5_text = "\n".join([c.text for c in chunks if c.page == 5])
    assert "(15)" in p5_text
    # Chunk IDs are contiguous per page
    page_chunks = {}
    for c in chunks:
        page_chunks.setdefault(c.page, []).append(int(c.chunk_id.split("_c")[-1]))
    for p, ids in page_chunks.items():
        assert ids == list(range(1, len(ids) + 1))

