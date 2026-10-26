import pytest
from pydantic import ValidationError
from backend.app.schemas import (
    Document,
    Chunk,
    ChunkKind,
    Claim,
    EvidenceLink,
    EvidenceRelation,
    CitationCheck,
    Answer,
    HealthResponse,
    DocumentUploadResponse,
)

SAMPLE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ---------------------------------------------------------
# Document Schema Tests
# ---------------------------------------------------------

def test_document_valid():
    doc = Document(
        doc_id="doc-001",
        title="Attention Is All You Need",
        filename="vaswani2017.pdf",
        sha256=SAMPLE_SHA256,
        page_count=15,
        extraction_warnings=["Header font irregular on page 3"],
    )
    assert doc.doc_id == "doc-001"
    assert doc.title == "Attention Is All You Need"
    assert doc.sha256 == SAMPLE_SHA256
    assert doc.page_count == 15
    assert len(doc.extraction_warnings) == 1


def test_document_default_warnings():
    doc = Document(
        doc_id="doc-002",
        title="BERT",
        filename="devlin2018.pdf",
        sha256=SAMPLE_SHA256,
        page_count=16,
    )
    assert doc.extraction_warnings == []


def test_document_invalid_sha256():
    with pytest.raises(ValidationError):
        Document(
            doc_id="doc-003",
            title="Invalid Hash Doc",
            filename="test.pdf",
            sha256="not-a-valid-sha256-hash",
            page_count=1,
        )


def test_document_negative_page_count():
    with pytest.raises(ValidationError):
        Document(
            doc_id="doc-004",
            title="Negative Page Doc",
            filename="test.pdf",
            sha256=SAMPLE_SHA256,
            page_count=-1,
        )


# ---------------------------------------------------------
# Chunk Schema Tests
# ---------------------------------------------------------

def test_chunk_valid():
    chunk = Chunk(
        chunk_id="chk-001",
        doc_id="doc-001",
        page=1,
        section_heading="1. Introduction",
        text="Recurrent neural networks have been firmly established...",
        kind=ChunkKind.TEXT,
    )
    assert chunk.chunk_id == "chk-001"
    assert chunk.page == 1
    assert chunk.kind == ChunkKind.TEXT


def test_chunk_table_kind():
    chunk = Chunk(
        chunk_id="chk-002",
        doc_id="doc-001",
        page=8,
        section_heading="Table 1: BLEU scores",
        text="| Model | BLEU |\n| --- | --- |\n| Transformer | 28.4 |",
        kind=ChunkKind.TABLE,
    )
    assert chunk.kind == ChunkKind.TABLE


def test_chunk_invalid_page():
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chk-003",
            doc_id="doc-001",
            page=0,  # pages must be >= 1
            text="Invalid page chunk",
        )


def test_chunk_empty_text():
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chk-004",
            doc_id="doc-001",
            page=1,
            text="",
        )


# ---------------------------------------------------------
# Claim Schema Tests
# ---------------------------------------------------------

def test_claim_valid():
    claim = Claim(
        claim_id="clm-001",
        answer_id="ans-001",
        text="Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
    )
    assert claim.claim_id == "clm-001"
    assert "BLEU" in claim.text


def test_claim_empty_text():
    with pytest.raises(ValidationError):
        Claim(
            claim_id="clm-002",
            answer_id="ans-001",
            text="",
        )


# ---------------------------------------------------------
# EvidenceLink Schema Tests
# ---------------------------------------------------------

def test_evidence_link_supports():
    link = EvidenceLink(
        claim_id="clm-001",
        chunk_id="chk-002",
        relation=EvidenceRelation.SUPPORTS,
        rationale="Table 1 explicitly lists Transformer (base model) at 28.4 BLEU.",
        confidence=0.98,
    )
    assert link.relation == EvidenceRelation.SUPPORTS
    assert link.conflict_reason is None
    assert link.confidence == 0.98


def test_evidence_link_apparent_conflict_valid():
    link = EvidenceLink(
        claim_id="clm-001",
        chunk_id="chk-005",
        relation=EvidenceRelation.CONTRADICTS,
        rationale="Subsequent study reports lower score of 26.1.",
        confidence=0.85,
        conflict_reason="Evaluated on a different sub-corpus test split without checkpoint averaging.",
    )
    assert link.relation == EvidenceRelation.CONTRADICTS
    assert link.conflict_reason is not None


def test_evidence_link_apparent_conflict_requires_reason():
    """Rule 3: Conflicts are labelled 'apparent' with a reason; never assert definitive contradiction."""
    with pytest.raises(ValidationError) as exc_info:
        EvidenceLink(
            claim_id="clm-001",
            chunk_id="chk-005",
            relation=EvidenceRelation.CONTRADICTS,
            rationale="Subsequent study contradicts the score.",
            confidence=0.85,
            conflict_reason=None,
        )
    assert "conflict_reason is required" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        EvidenceLink(
            claim_id="clm-001",
            chunk_id="chk-005",
            relation=EvidenceRelation.WEAKENS,
            rationale="Subsequent study casts doubt.",
            confidence=0.85,
            conflict_reason="",
        )
    assert "conflict_reason is required" in str(exc_info.value)


def test_evidence_link_invalid_confidence():
    with pytest.raises(ValidationError):
        EvidenceLink(
            claim_id="clm-001",
            chunk_id="chk-001",
            relation=EvidenceRelation.SUPPORTS,
            rationale="Valid rationale",
            confidence=1.5,  # must be <= 1.0
        )


# ---------------------------------------------------------
# CitationCheck Schema Tests
# ---------------------------------------------------------

def test_citation_check_valid_supported():
    check = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-002",
        exists=True,
        supports=True,
        notes="Exact match in table row.",
    )
    assert check.exists is True
    assert check.supports is True


def test_citation_check_valid_nonexistent():
    check = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-missing",
        exists=False,
        supports=None,
        notes="Chunk ID not found in retrieved index.",
    )
    assert check.exists is False
    assert check.supports is None


def test_citation_check_cannot_support_if_nonexistent():
    """A citation cannot be marked as supporting if the referenced chunk does not exist."""
    with pytest.raises(ValidationError) as exc_info:
        CitationCheck(
            claim_id="clm-001",
            chunk_id="chk-missing",
            exists=False,
            supports=True,
        )
    assert "cannot support a claim if the chunk does not exist" in str(exc_info.value)


# ---------------------------------------------------------
# Answer Schema Tests
# ---------------------------------------------------------

def test_answer_valid():
    claim = Claim(
        claim_id="clm-001",
        answer_id="ans-001",
        text="Transformers rely entirely on self-attention mechanisms.",
    )
    answer = Answer(
        answer_id="ans-001",
        question="What is the architectural core of the Transformer?",
        claims=[claim],
        abstained=False,
    )
    assert answer.abstained is False
    assert len(answer.claims) == 1


def test_answer_honest_abstention_valid():
    """Rule 4: Abstain when retrieved evidence is insufficient."""
    answer = Answer(
        answer_id="ans-002",
        question="What was the learning rate on TPU v4 clusters in Vaswani et al.?",
        claims=[],
        abstained=True,
        abstain_reason="Insufficient evidence: Vaswani et al. (2017) tested on 8 P100 GPUs, not TPU v4.",
    )
    assert answer.abstained is True
    assert "TPU v4" in (answer.abstain_reason or "")


def test_answer_abstention_requires_reason():
    with pytest.raises(ValidationError) as exc_info:
        Answer(
            answer_id="ans-003",
            question="Unknown question",
            claims=[],
            abstained=True,
            abstain_reason=None,
        )
    assert "abstain_reason is required when abstained=True" in str(exc_info.value)


# ---------------------------------------------------------
# API Schema Tests
# ---------------------------------------------------------

def test_health_response():
    resp = HealthResponse(status="ok", version="0.1.0")
    assert resp.status == "ok"
    assert resp.version == "0.1.0"
    assert resp.timestamp is not None


def test_document_upload_response():
    doc = Document(
        doc_id="doc-001",
        title="Test Doc",
        filename="test.pdf",
        sha256=SAMPLE_SHA256,
        page_count=3,
    )
    resp = DocumentUploadResponse(document=doc, chunk_count=12)
    assert resp.chunk_count == 12
    assert resp.document.doc_id == "doc-001"
