import pytest
from pydantic import ValidationError
from backend.app.schemas import (
    Document,
    Chunk,
    ChunkKind,
    Claim,
    EvidenceLink,
    EvidenceRelation,
    ClaimVerificationStatus,
    CitationCheck,
    Answer,
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
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
    assert claim.chunk_ids == []
    assert claim.quote is None
    assert claim.status == ClaimVerificationStatus.UNVERIFIED


def test_claim_empty_text():
    with pytest.raises(ValidationError):
        Claim(
            claim_id="clm-002",
            answer_id="ans-001",
            text="",
        )


def test_claim_with_chunk_ids_quote_and_status():
    claim = Claim(
        claim_id="clm-003",
        answer_id="ans-001",
        text="LEMS uses the Gurobi solver for ILP.",
        chunk_ids=["doc_c67573ef50cc_p6_c2"],
        quote="we use the Gurobi solver",
        status=ClaimVerificationStatus.VERIFIED,
    )
    assert claim.chunk_ids == ["doc_c67573ef50cc_p6_c2"]
    assert claim.quote == "we use the Gurobi solver"
    assert claim.status == ClaimVerificationStatus.VERIFIED


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
    assert check.status == ClaimVerificationStatus.INVALID_REFERENCE


def test_citation_check_cannot_support_if_nonexistent():
    """A citation cannot be marked as supporting if the referenced chunk does not exist."""
    with pytest.raises(ValidationError) as exc_info:
        CitationCheck(
            claim_id="clm-001",
            chunk_id="chk-missing",
            exists=False,
            supports=True,
        )
    assert "Citation support must be None" in str(exc_info.value)


def test_citation_check_intermediate_representation():
    # Intermediate check after Layer 1 passes: supports is None, status is UNVERIFIED
    check_l1 = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-002",
        exists=True,
        supports=None,
    )
    assert check_l1.exists is True
    assert check_l1.supports is None
    assert check_l1.status == ClaimVerificationStatus.UNVERIFIED

    # Intermediate check with supports=True alone: does NOT auto-promote to VERIFIED
    check_l3 = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-002",
        exists=True,
        supports=True,
    )
    assert check_l3.status == ClaimVerificationStatus.UNVERIFIED


def test_citation_check_verified_requires_supports_true():
    # Status can only be VERIFIED when supports is True
    check_verified = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-002",
        exists=True,
        supports=True,
        status=ClaimVerificationStatus.VERIFIED,
    )
    assert check_verified.status == ClaimVerificationStatus.VERIFIED

    # Fails if supports is None
    with pytest.raises(ValidationError) as exc_info:
        CitationCheck(
            claim_id="clm-001",
            chunk_id="chk-002",
            exists=True,
            supports=None,
            status=ClaimVerificationStatus.VERIFIED,
        )
    assert "cannot have status 'verified' unless exists=True and supports=True" in str(exc_info.value)

    # Fails if supports is False
    with pytest.raises(ValidationError) as exc_info:
        CitationCheck(
            claim_id="clm-001",
            chunk_id="chk-002",
            exists=True,
            supports=False,
            status=ClaimVerificationStatus.VERIFIED,
        )
    assert "cannot have status 'verified' unless exists=True and supports=True" in str(exc_info.value)


def test_citation_check_unsupported_status():
    check_unsupported = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-002",
        exists=True,
        supports=False,
        status=ClaimVerificationStatus.UNSUPPORTED,
    )
    assert check_unsupported.status == ClaimVerificationStatus.UNSUPPORTED

    with pytest.raises(ValidationError) as exc_info:
        CitationCheck(
            claim_id="clm-001",
            chunk_id="chk-002",
            exists=True,
            supports=True,
            status=ClaimVerificationStatus.UNSUPPORTED,
        )
    assert "cannot have status 'unsupported' when supports=True" in str(exc_info.value)


def test_citation_check_exists_true_cannot_be_invalid_reference():
    with pytest.raises(ValidationError) as exc_info:
        CitationCheck(
            claim_id="clm-001",
            chunk_id="chk-002",
            exists=True,
            status=ClaimVerificationStatus.INVALID_REFERENCE,
        )
    assert "Citation cannot have status 'invalid_reference' when exists=True" in str(exc_info.value)


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
# AnswerRequest Schema Tests
# ---------------------------------------------------------

def test_answer_request_valid():
    req = AnswerRequest(
        question="Which solver does LEMS use?",
        doc_ids=["doc_c67573ef50cc"],
        top_k=8,
    )
    assert req.question == "Which solver does LEMS use?"
    assert req.doc_ids == ["doc_c67573ef50cc"]
    assert req.top_k == 8


def test_answer_request_default_top_k():
    req = AnswerRequest(question="What is LEMS?")
    assert req.top_k == 8
    assert req.doc_ids is None


def test_answer_request_empty_question_fails():
    with pytest.raises(ValidationError):
        AnswerRequest(question="")

    with pytest.raises(ValidationError):
        AnswerRequest(question="   \n\t  ")


def test_answer_request_invalid_doc_id_regex():
    with pytest.raises(ValidationError) as exc_info:
        AnswerRequest(question="Valid question", doc_ids=["invalid_doc_123"])
    assert "Invalid doc_id in filter" in str(exc_info.value)


def test_answer_request_top_k_bounds():
    req10 = AnswerRequest(question="Test question", top_k=10)
    assert req10.top_k == 10

    with pytest.raises(ValidationError):
        AnswerRequest(question="Test question", top_k=11)

    with pytest.raises(ValidationError):
        AnswerRequest(question="Test question", top_k=0)


# ---------------------------------------------------------
# AnswerResponse Schema Tests
# ---------------------------------------------------------

def test_answer_response_answered_valid():
    link = EvidenceLink(
        claim_id="clm-001",
        chunk_id="chk-002",
        relation=EvidenceRelation.SUPPORTS,
        rationale="Explicit mention",
        confidence=0.95,
    )
    check = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-002",
        exists=True,
        supports=True,
        status=ClaimVerificationStatus.VERIFIED,
    )
    resp = AnswerResponse(
        answer="LEMS uses the Gurobi solver.",
        evidence_links=[link],
        citation_checks=[check],
        status=AnswerStatus.ANSWERED,
        abstain_reason=None,
        timings={"retrieval": 0.05, "generation": 0.42, "total": 0.47},
    )
    assert resp.status == AnswerStatus.ANSWERED
    assert resp.abstain_reason is None
    assert len(resp.evidence_links) == 1
    assert len(resp.citation_checks) == 1
    assert resp.citation_checks[0].status == ClaimVerificationStatus.VERIFIED
    assert resp.timings["total"] == 0.47


def test_answer_response_abstained_valid():
    resp = AnswerResponse(
        answer="The provided passages do not contain information regarding compute costs.",
        evidence_links=[],
        citation_checks=[],
        status=AnswerStatus.ABSTAINED,
        abstain_reason="Insufficient evidence: financial expenditures not reported.",
        timings={"retrieval": 0.04, "total": 0.05},
    )
    assert resp.status == AnswerStatus.ABSTAINED
    assert "Insufficient evidence" in (resp.abstain_reason or "")


def test_answer_response_abstained_requires_reason():
    with pytest.raises(ValidationError) as exc_info:
        AnswerResponse(
            answer="Abstaining.",
            evidence_links=[],
            citation_checks=[],
            status=AnswerStatus.ABSTAINED,
            abstain_reason=None,
        )
    assert "abstain_reason is required when status is 'abstained'" in str(exc_info.value)


def test_answer_response_json_serialization():
    check = CitationCheck(
        claim_id="clm-001",
        chunk_id="chk-001",
        exists=False,
    )
    resp = AnswerResponse(
        answer="Abstaining due to invalid reference.",
        evidence_links=[],
        citation_checks=[check],
        status=AnswerStatus.ABSTAINED,
        abstain_reason="Citation references missing chunk.",
        timings={"total": 0.12},
    )
    data = resp.model_dump(mode="json")
    assert data["status"] == "abstained"
    assert data["citation_checks"][0]["status"] == "invalid_reference"
    assert data["citation_checks"][0]["supports"] is None
    assert data["timings"]["total"] == 0.12


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
