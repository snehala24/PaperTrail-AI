"""
Focused Regression Test Suite for Day 4 Cross-Paper Intelligence.

Verifies:
1. Multi-paper claim-to-evidence mapping:
   - Each claim is mapped to its source document (doc_id), 1-indexed page, and exact verbatim quote.
   - Verified EvidenceLink objects capture doc_id, page, and quote alongside claim_id and chunk_id.
2. Source-document isolation:
   - Document filters strictly isolate citations; citations outside allowed doc_ids fail Layer 1.
3. Apparent-conflict handling:
   - Cross-paper apparent conflict (e.g. Q-030) between LEMS degradation claim and GFWSVD outperforming baselines.
   - Correctly identifies differing experimental conditions, flags source disagreement, and populates ClaimComparison.
4. Unsupported comparisons & honest abstention:
   - Incomplete cross-paper retrieval (only 1 of 2 requested papers returned) honestly abstains.
   - Self-abstention when evidence cannot resolve the comparison honestly abstains.
   - Fabricated or altered quotes in cross-paper claims trigger fail-closed zero-tolerance abstention.
   - Unentailed cross-paper comparative claims fail Layer 3 and trigger fail-closed zero-tolerance abstention.
5. Multi-paper citation verification:
   - 3-layer verification across multi-paper retrieved context.

ZERO LIVE GEMINI CALLS: Deterministic and fully offline using MockLLMProvider and mock retrievers.
Driven synchronously via run_sync to prevent Windows socketpair deadlocks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest

from backend.app.citations.verifier import CitationVerifier
from backend.app.claims.service import AnswerService
from backend.app.llm.base import LLMMessage, LLMProvider, LLMResponse
from backend.app.schemas import (
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    Chunk,
    ChunkKind,
    CitationCheck,
    Claim,
    ClaimComparison,
    ClaimVerificationStatus,
    ComparisonType,
    EvidenceLink,
    EvidenceRelation,
    ScoredChunk,
)


def run_sync(coro):
    """Drive a coroutine to completion synchronously without requiring an OS socketpair."""
    if not hasattr(coro, "send"):
        return coro
    try:
        val = None
        while True:
            val = coro.send(val)
            if hasattr(val, "__await__"):
                sub = val.__await__()
                try:
                    sub_val = None
                    while True:
                        sub_val = sub.send(sub_val)
                except StopIteration as s:
                    val = s.value
            elif hasattr(val, "send"):
                val = run_sync(val)
    except StopIteration as e:
        return e.value


def make_scored_chunk(
    chunk_id: str = "doc_c67573ef50cc_p1_c1",
    doc_id: str = "doc_c67573ef50cc",
    page: int = 1,
    text: str = "Evidence text here.",
    rerank_score: float = 5.0,
    kind: ChunkKind = ChunkKind.TEXT,
) -> ScoredChunk:
    """Helper fixture to create ScoredChunk instances."""
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=page,
        kind=kind,
        text=text,
        char_start=0,
        char_end=len(text),
        rerank_score=rerank_score,
        bm25_rank=1,
        dense_rank=1,
        fused_rank=1,
    )


class MockRetriever:
    """Mock retriever returning predetermined ScoredChunk results."""

    def __init__(self, results: Optional[List[ScoredChunk]] = None):
        self.results = results if results is not None else []
        self.chunks_by_id = {c.chunk_id: c for c in self.results}

    def search(
        self,
        query: str,
        top_k: int = 8,
        doc_ids: Optional[List[str]] = None,
        include_references: bool = False,
        use_reranker: bool = True,
    ) -> List[ScoredChunk]:
        if doc_ids is not None:
            allowed = set(doc_ids)
            return [c for c in self.results if c.doc_id in allowed][:top_k]
        return self.results[:top_k]


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM provider for generator and verifier."""

    def __init__(self, responses: Optional[List[Union[str, LLMResponse]]] = None, default_response: str = "{}"):
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.call_history: List[List[LLMMessage]] = []

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_history.append(messages)
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, LLMResponse):
                return item
            content = item
        else:
            content = self.default_response
        return LLMResponse(
            content=content,
            model="mock-gemini-2.5-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


# =========================================================================
# 1. Multi-Paper Claim-to-Evidence Mapping (Q-029)
# =========================================================================

def test_cross_paper_evidence_mapping_q029():
    """
    Test 1: Multi-paper claim-to-evidence mapping for Q-029.
    Both papers describe the Kronecker-factor estimation via Lanczos rank-1 SVD.
    Verifies:
    - Claim 1 is mapped to LEMS (doc_c67573ef50cc, p.3) with exact verbatim quote.
    - Claim 2 is mapped to GFWSVD (doc_7fdb1200e823, p.4) with exact verbatim quote.
    - EvidenceLink models contain doc_id, page, and quote.
    - comparisons list contains ClaimComparison with ComparisonType.SUPPORTS.
    """
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id="doc_c67573ef50cc",
        page=3,
        text=(
            "By fitting a rank-1 approximation to bF in a rearranged space using Lanczos "
            "(Lanczos, 1950), GFWSVD creates a Kronecker product A ⊗G."
        ),
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p4_c3",
        doc_id="doc_7fdb1200e823",
        page=4,
        text=(
            "Standard SVD is computationally intractable for large matrices, so we employ truncated "
            "SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction"
        ),
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    gen_json = json.dumps({
        "answer": (
            "Both papers describe the Kronecker-factor estimation similarly: LEMS notes that GFWSVD fits a "
            "rank-1 approximation using Lanczos, and the GFWSVD paper confirms employing truncated SVD via Lanczos."
        ),
        "claims": [
            {
                "claim_id": "c1",
                "text": "GFWSVD fits a rank-1 approximation in a rearranged space using Lanczos.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "By fitting a rank-1 approximation to bF in a rearranged space using Lanczos (Lanczos, 1950), GFWSVD creates a Kronecker product A ⊗G",
            },
            {
                "claim_id": "c2",
                "text": "GFWSVD employs truncated SVD using the Lanczos method to avoid explicit matrix construction.",
                "chunk_ids": ["doc_7fdb1200e823_p4_c3"],
                "quote": "so we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "c1",
                "claim_id_b": "c2",
                "comparison_type": "supports",
                "summary": "Both papers agree that GFWSVD computes Kronecker factors using Lanczos rank-1 truncated SVD.",
                "conflict_reason": None,
            }
        ],
    })

    verif_ok = json.dumps({"supported": True, "rationale": "Direct verbatim quote and factual entailment."})
    llm = MockLLMProvider([gen_json, verif_ok, verif_ok])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Do the two papers describe the Kronecker-factor estimation the same way?",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 2
    assert len(resp.citation_checks) == 2
    assert all(chk.status == ClaimVerificationStatus.VERIFIED for chk in resp.citation_checks)

    # Check multi-paper evidence mapping
    link1 = next(l for l in resp.evidence_links if l.claim_id == "c1")
    assert link1.doc_id == "doc_c67573ef50cc"
    assert link1.page == 3
    assert link1.quote == "By fitting a rank-1 approximation to bF in a rearranged space using Lanczos (Lanczos, 1950), GFWSVD creates a Kronecker product A ⊗G"
    assert link1.relation == EvidenceRelation.SUPPORTS

    link2 = next(l for l in resp.evidence_links if l.claim_id == "c2")
    assert link2.doc_id == "doc_7fdb1200e823"
    assert link2.page == 4
    assert link2.quote == "so we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction"
    assert link2.relation == EvidenceRelation.SUPPORTS

    # Check cross-paper comparison
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    assert comp.comparison_type == ComparisonType.SUPPORTS
    assert comp.doc_id_a == "doc_c67573ef50cc"
    assert comp.doc_id_b == "doc_7fdb1200e823"
    assert "Lanczos" in comp.summary


# =========================================================================
# 2. Source-Document Isolation
# =========================================================================

def test_source_document_isolation():
    """
    Test 2: Document filtering isolates sources strictly.
    When a request filters for doc_c67573ef50cc, citing a chunk from doc_7fdb1200e823
    fails Layer 1 (invalid_reference) and triggers zero-tolerance abstention.
    """
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id="doc_c67573ef50cc",
        page=3,
        text="LEMS uses token-wise statistics.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p4_c3",
        doc_id="doc_7fdb1200e823",
        page=4,
        text="GFWSVD uses Lanczos SVD.",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    # Model attempts to cite chunk_gfw while doc_ids filter is restricted to doc_c67573ef50cc
    gen_json = json.dumps({
        "answer": "GFWSVD uses Lanczos.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "GFWSVD uses Lanczos.",
                "chunk_ids": ["doc_7fdb1200e823_p4_c3"],
                "quote": "GFWSVD uses Lanczos SVD.",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
    })

    llm = MockLLMProvider([gen_json])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How does LEMS work?",
        doc_ids=["doc_c67573ef50cc"],  # doc_7fdb1200e823 is excluded
    )
    resp: AnswerResponse = run_sync(service.answer(req))

    # Must abstain fail-closed because chunk_gfw is from excluded document
    assert resp.status == AnswerStatus.ABSTAINED
    assert "zero-tolerance" in resp.abstain_reason.lower()
    chk = resp.citation_checks[0]
    assert chk.exists is False
    assert chk.status == ClaimVerificationStatus.INVALID_REFERENCE
    assert "Layer 1 failure" in chk.notes


# =========================================================================
# 3. Cross-Paper Apparent Conflict Handling (Q-030)
# =========================================================================

def test_cross_paper_apparent_conflict_q030():
    """
    Test 3: Cross-paper apparent conflict handling for Q-030.
    LEMS reports GFWSVD degrades significantly under identical zero-shot conditions,
    whereas GFWSVD reports outperforming baselines under fine-tuning/regularization.
    Verifies:
    - Both claims are verified against their respective documents.
    - Apparent conflict cause (differing conditions) is captured in conflict_reason.
    - 'sources disagree' flag and conflicting chunks are present in answer text.
    - ClaimComparison model records APPARENT_CONFLICT with resolution reason.
    """
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p6_c5",
        doc_id="doc_c67573ef50cc",
        page=6,
        text="GFWSVD degrades significantly under identical conditions, reaffirming its conditioning challenges.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p9_c4",
        doc_id="doc_7fdb1200e823",
        page=9,
        text=(
            "Our results demonstrate that GFWSVD consistently outperforms diagonal and "
            "activation-based baselines across a wide range of compression regimes."
        ),
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    gen_json = json.dumps({
        "answer": (
            "LEMS reports that GFWSVD degrades significantly under identical evaluation conditions, "
            "whereas GFWSVD's own paper reports consistently outperforming diagonal and activation baselines. "
            "This apparent conflict is caused by differing experimental conditions: LEMS evaluates zero-shot uniform "
            "compression without fine-tuning, whereas GFWSVD evaluates with regularization and fine-tuning."
        ),
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS reports GFWSVD degrades significantly under identical conditions.",
                "chunk_ids": ["doc_c67573ef50cc_p6_c5"],
                "quote": "GFWSVD degrades significantly under identical conditions",
            },
            {
                "claim_id": "c2",
                "text": "GFWSVD reports consistently outperforming diagonal and activation-based baselines.",
                "chunk_ids": ["doc_7fdb1200e823_p9_c4"],
                "quote": "Our results demonstrate that GFWSVD consistently outperforms diagonal and activation-based baselines across a wide range of compression regimes.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [
            {
                "chunk_id_a": "doc_c67573ef50cc_p6_c5",
                "chunk_id_b": "doc_7fdb1200e823_p9_c4",
                "conflict_type": "apparent_conflict",
                "conflict_reason": (
                    "Differing experimental conditions: LEMS evaluates uniform compression without fine-tuning, "
                    "whereas GFWSVD evaluates under fine-tuning and regularization across models."
                ),
            }
        ],
        "comparisons": [
            {
                "claim_id_a": "c1",
                "claim_id_b": "c2",
                "comparison_type": "apparent_conflict",
                "summary": "Apparent conflict between reported degradation in LEMS and reported baseline outperformance in GFWSVD.",
                "conflict_reason": (
                    "Differing experimental conditions: LEMS evaluates zero-shot compression without fine-tuning, "
                    "whereas GFWSVD evaluates with regularization and fine-tuning across models."
                ),
            }
        ],
    })

    verif_ok = json.dumps({"supported": True, "rationale": "Exact match and direct entailment."})
    llm = MockLLMProvider([gen_json, verif_ok, verif_ok])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Does GFWSVD's own reported performance match the LEMS paper's claim that it degrades?",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert "sources disagree" in resp.answer.lower()
    assert "doc_c67573ef50cc_p6_c5" in resp.answer
    assert "doc_7fdb1200e823_p9_c4" in resp.answer

    # Evidence links are verified and mapped to documents
    assert len(resp.evidence_links) == 2
    lems_link = next(l for l in resp.evidence_links if l.chunk_id == "doc_c67573ef50cc_p6_c5")
    assert lems_link.doc_id == "doc_c67573ef50cc"
    assert lems_link.page == 6

    gfw_link = next(l for l in resp.evidence_links if l.chunk_id == "doc_7fdb1200e823_p9_c4")
    assert gfw_link.doc_id == "doc_7fdb1200e823"
    assert gfw_link.page == 9

    # Comparisons capture apparent conflict with differing conditions reason
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    assert comp.comparison_type == ComparisonType.APPARENT_CONFLICT
    assert comp.conflict_reason is not None
    assert "Differing experimental conditions" in comp.conflict_reason


# =========================================================================
# 4. Unsupported Comparisons & Honest Abstention
# =========================================================================

def test_unsupported_comparison_incomplete_retrieval_abstains():
    """
    Test 4a: Honest abstention when multi-document comparison is requested but
    retrieved chunks only contain evidence from one paper.
    """
    # Only doc_c67573ef50cc chunks retrieved; doc_7fdb1200e823 is missing
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p1_c1",
        doc_id="doc_c67573ef50cc",
        page=1,
        text="LEMS achieves high compression.",
    )
    retriever = MockRetriever([chunk_lems])
    llm = MockLLMProvider()
    service = AnswerService(retriever=retriever, llm_provider=llm)

    req = AnswerRequest(
        question="Compare the training compute cost between the two papers.",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert "Cross-paper evidence incomplete" in resp.abstain_reason
    assert "doc_7fdb1200e823" in resp.abstain_reason
    # LLM was never called because fail-closed check caught the missing document
    assert len(llm.call_history) == 0


def test_unsupported_comparison_generator_self_abstains():
    """
    Test 4b: Honest abstention when retrieved evidence does not contain sufficient
    facts to answer a comparative question (generator self-abstains).
    """
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p1_c1",
        doc_id="doc_c67573ef50cc",
        text="LEMS description.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p1_c1",
        doc_id="doc_7fdb1200e823",
        text="GFWSVD description.",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    gen_json = json.dumps({
        "answer": "Neither paper reports the monetary compute cost or carbon emissions in dollars.",
        "claims": [],
        "abstained": True,
        "abstain_reason": "Neither paper reports monetary training cost or carbon footprint.",
        "conflicts": [],
    })
    llm = MockLLMProvider([gen_json])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How do the dollar training costs compare between LEMS and GFWSVD?",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert "monetary training cost" in resp.abstain_reason.lower()
    assert resp.evidence_links == []
    assert resp.citation_checks == []


def test_unsupported_comparison_hallucinated_quote_fails_closed():
    """
    Test 4c: Fail-closed behavior when a cross-paper claim includes an altered
    or fabricated quote (Layer 2 citation failure).
    """
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p1_c1",
        doc_id="doc_c67573ef50cc",
        text="LEMS uses integer linear programming for rank allocation.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p1_c1",
        doc_id="doc_7fdb1200e823",
        text="GFWSVD uses Lanczos method for Fisher decomposition.",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    # Model hallucinates quote on the GFWSVD side
    gen_json = json.dumps({
        "answer": "LEMS uses ILP while GFWSVD uses quadratic programming.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses integer linear programming.",
                "chunk_ids": ["doc_c67573ef50cc_p1_c1"],
                "quote": "LEMS uses integer linear programming",
            },
            {
                "claim_id": "c2",
                "text": "GFWSVD uses quadratic programming.",
                "chunk_ids": ["doc_7fdb1200e823_p1_c1"],
                "quote": "GFWSVD uses quadratic programming",  # FABRICATED QUOTE
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
    })

    verif_ok = json.dumps({"supported": True, "rationale": "Entailed."})
    llm = MockLLMProvider([gen_json, verif_ok])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How do optimization techniques compare between the papers?",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert "zero-tolerance" in resp.abstain_reason.lower()
    # Check that c2 failed Layer 2
    c2_check = next(c for c in resp.citation_checks if c.claim_id == "c2")
    assert c2_check.status == ClaimVerificationStatus.UNSUPPORTED
    assert "Layer 2 failure" in c2_check.notes


def test_unsupported_comparison_unentailed_claim_fails_closed():
    """
    Test 4d: Fail-closed behavior when a cross-paper claim passes quote matching
    but fails semantic entailment check (Layer 3 failure).
    """
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p1_c1",
        doc_id="doc_c67573ef50cc",
        text="LEMS evaluation covers Mistral, Llama 3, and Qwen3.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p1_c1",
        doc_id="doc_7fdb1200e823",
        text="GFWSVD was tested on BERT, Llama 2, and Llama 3.1.",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    gen_json = json.dumps({
        "answer": "LEMS proves GFWSVD is obsolete across all language tasks.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS proves GFWSVD is obsolete across all language tasks.",
                "chunk_ids": ["doc_c67573ef50cc_p1_c1"],
                "quote": "Mistral, Llama 3, and Qwen3.",  # Quote matches
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
    })

    verif_fail = json.dumps({"supported": False, "rationale": "Extreme extrapolation not entailed by evidence."})
    llm = MockLLMProvider([gen_json, verif_fail])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Does LEMS make GFWSVD obsolete?",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert "zero-tolerance" in resp.abstain_reason.lower()
    chk = resp.citation_checks[0]
    assert chk.status == ClaimVerificationStatus.UNSUPPORTED
    assert chk.supports is False
    assert "not entailed by evidence" in chk.notes.lower()


# =========================================================================
# 5. Cross-Paper Chunk Continuation & Attribution Regression (Test E)
# =========================================================================

def test_cross_paper_kronecker_estimation_continuation_and_attribution():
    """
    Test 5: Regression test for Day 4 Test E (cross-paper Kronecker-factor estimation).
    Verifies:
    1. Fail-closed: When claim_2 quotes text from a continuation chunk (p4_c2)
       but attributes it to the preceding chunk (p3_c5 where the quote is cut off),
       Layer 2 fails and the pipeline strictly abstains (AnswerStatus.ABSTAINED).
    2. Correct attribution: When each claim correctly cites its respective chunk
       and document (doc_7fdb1200e823_p4_c3 for GFWSVD and doc_c67573ef50cc_p4_c2 for LEMS),
       both pass Layer 1-3 verification and the pipeline returns AnswerStatus.ANSWERED
       with populated comparisons and evidence links.
    """
    chunk_p3c5 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id="doc_c67573ef50cc",
        page=3,
        text=(
            "By fitting a rank-1 approximation to bF in a rearranged space using Lanczos (Lanczos, 1950), "
            "GFWSVD creates a Kronecker product A ⊗G. Method. To avoid the rank bottleneck of GFWSVD’s "
            "per-sequence gradient-based estimator, we estimate Kronecker factors from token-wise second moments. "
            "For a linear layer with input x and output gradient δ = ∂L/∂y, the per-token"
        ),
    )
    chunk_p4c2 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p4_c2",
        doc_id="doc_c67573ef50cc",
        page=4,
        text=(
            "weight gradient is ∇WL = δx⊤. Following (Martens & Grosse, 2015), we approximate the expectation of "
            "the Kronecker product by the Kronecker product of expectations: F = E[(x⊗δ)(x⊗δ)⊤] ≈E[xx⊤]⊗E[δδ⊤] = A⊗G."
        ),
    )
    chunk_gfw_p4c3 = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p4_c3",
        doc_id="doc_7fdb1200e823",
        page=4,
        text="so we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction",
    )
    retriever = MockRetriever([chunk_p3c5, chunk_p4c2, chunk_gfw_p4c3])

    # Case A: Quote misattributed across page boundary (quote in p4_c2 cited against p3_c5) -> Fails closed
    mismatched_gen = json.dumps({
        "answer": "LEMS and GFWSVD use different Kronecker-factor estimation processes.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "GFWSVD fits a rank-1 approximation using Lanczos.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "By fitting a rank-1 approximation to bF in a rearranged space using Lanczos (Lanczos, 1950), GFWSVD creates a Kronecker product A ⊗G",
            },
            {
                "claim_id": "c2",
                "text": "KFAC-SVD approximates the expectation of the Kronecker product by the product of expectations.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],  # Misattribution: quote is physically in p4_c2!
                "quote": "we approximate the expectation of the Kronecker product by the Kronecker product of expectations",
            },
        ],
        "abstained": False,
        "conflicts": [],
    })
    llm_fail = MockLLMProvider([mismatched_gen, json.dumps({"supported": True, "rationale": "Directly supported."})])
    verifier_fail = CitationVerifier(llm_provider=llm_fail)
    service_fail = AnswerService(retriever=retriever, llm_provider=llm_fail, verifier=verifier_fail)

    req = AnswerRequest(
        question="Do LEMS and GFWSVD describe the Kronecker-factor estimation process in the same way?",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp_fail = run_sync(service_fail.answer(req))

    assert resp_fail.status == AnswerStatus.ABSTAINED
    assert "citations unsupported" in resp_fail.abstain_reason
    assert len(resp_fail.citation_checks) == 2
    c2_chk = next(c for c in resp_fail.citation_checks if c.claim_id == "c2")
    assert c2_chk.status == ClaimVerificationStatus.UNSUPPORTED
    assert "Layer 2 failure: Verbatim quote" in c2_chk.notes

    # Case B: Correct chunk attribution and cross-paper grounding -> Answered with valid comparisons
    valid_gen = json.dumps({
        "answer": (
            "LEMS and GFWSVD use different Kronecker-factor estimation processes: GFWSVD employs truncated SVD "
            "via the Lanczos method on sequence-level gradients, whereas LEMS (via KFAC-SVD) decouples the expectation "
            "using token-wise second moments."
        ),
        "claims": [
            {
                "claim_id": "c1",
                "text": "GFWSVD employs truncated SVD using the Lanczos method.",
                "chunk_ids": ["doc_7fdb1200e823_p4_c3"],
                "quote": "so we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction",
            },
            {
                "claim_id": "c2",
                "text": "LEMS approximates the expectation of the Kronecker product using token-wise statistics.",
                "chunk_ids": ["doc_c67573ef50cc_p4_c2"],
                "quote": "we approximate the expectation of the Kronecker product by the Kronecker product of expectations",
            },
        ],
        "abstained": False,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "c1",
                "claim_id_b": "c2",
                "comparison_type": "apparent_conflict",
                "summary": "Methodological difference in factor estimation: sequence-level Lanczos truncated SVD vs token-wise second-moment expectation decoupling.",
                "conflict_reason": "Differing methodological formulations: sequence-level Lanczos truncated SVD in GFWSVD vs token-wise second moments in LEMS.",
            }
        ],
    })
    llm_success = MockLLMProvider([
        valid_gen,
        json.dumps({"supported": True, "rationale": "Directly supported by GFWSVD Section 4.1."}),
        json.dumps({"supported": True, "rationale": "Directly supported by LEMS Section 4.1."}),
    ])
    verifier_success = CitationVerifier(llm_provider=llm_success)
    service_success = AnswerService(retriever=retriever, llm_provider=llm_success, verifier=verifier_success)

    resp_success = run_sync(service_success.answer(req))

    assert resp_success.status == AnswerStatus.ANSWERED
    assert len(resp_success.evidence_links) == 2
    assert all(c.status == ClaimVerificationStatus.VERIFIED for c in resp_success.citation_checks)

    # Check evidence link document attribution
    gfw_link = next(l for l in resp_success.evidence_links if l.chunk_id == "doc_7fdb1200e823_p4_c3")
    assert gfw_link.doc_id == "doc_7fdb1200e823"
    assert gfw_link.page == 4

    lems_link = next(l for l in resp_success.evidence_links if l.chunk_id == "doc_c67573ef50cc_p4_c2")
    assert lems_link.doc_id == "doc_c67573ef50cc"
    assert lems_link.page == 4

    # Check comparisons populated with both document IDs
    assert len(resp_success.comparisons) == 1
    comp = resp_success.comparisons[0]
    assert comp.comparison_type == ComparisonType.APPARENT_CONFLICT
    assert comp.doc_id_a == "doc_7fdb1200e823"
    assert comp.doc_id_b == "doc_c67573ef50cc"
    assert comp.conflict_reason is not None


# =========================================================================
# 5. Day 4 Corrections: Regression Tests for Retrieval & Source Grounding
# =========================================================================

def test_cross_paper_retrieval_balanced_coverage():
    """
    Regression Test 5A: Balanced cross-paper candidate pooling and retrieval.
    Verifies that when multiple doc_ids are requested, select_allowed_candidates
    allocates balanced evidence slots across all requested documents rather than allowing
    a single dominant paper to crowd out the other.
    """
    from backend.app.retrieval.hybrid import select_allowed_candidates
    from backend.app.schemas import Chunk

    chunk_lems_1 = Chunk(chunk_id="doc_lems_1", doc_id="doc_lems", page=1, text="Text 1", kind=ChunkKind.TEXT, char_start=0, char_end=6)
    chunk_lems_2 = Chunk(chunk_id="doc_lems_2", doc_id="doc_lems", page=2, text="Text 2", kind=ChunkKind.TEXT, char_start=0, char_end=6)
    chunk_lems_3 = Chunk(chunk_id="doc_lems_3", doc_id="doc_lems", page=3, text="Text 3", kind=ChunkKind.TEXT, char_start=0, char_end=6)
    chunk_gfw_1 = Chunk(chunk_id="doc_gfw_1", doc_id="doc_gfw", page=1, text="Text 4", kind=ChunkKind.TEXT, char_start=0, char_end=6)
    chunk_gfw_2 = Chunk(chunk_id="doc_gfw_2", doc_id="doc_gfw", page=2, text="Text 5", kind=ChunkKind.TEXT, char_start=0, char_end=6)

    raw_results = [
        (chunk_lems_1, 5.0, 1),
        (chunk_lems_2, 4.5, 2),
        (chunk_lems_3, 4.0, 3),
        (chunk_gfw_1, 3.0, 4),
        (chunk_gfw_2, 2.5, 5),
    ]

    selected = select_allowed_candidates(raw_results, max_candidates=4, doc_ids={"doc_lems", "doc_gfw"})
    doc_ids_returned = {c.doc_id for c, _, _ in selected}
    assert "doc_lems" in doc_ids_returned
    assert "doc_gfw" in doc_ids_returned
    assert len(selected) == 4
    lems_count = sum(1 for c, _, _ in selected if c.doc_id == "doc_lems")
    gfw_count = sum(1 for c, _, _ in selected if c.doc_id == "doc_gfw")
    assert lems_count == 2
    assert gfw_count == 2


def test_cross_paper_source_aware_primary_attribution():
    """
    Regression Test 5B: Source-aware primary evidence attribution.
    Verifies that claims regarding each paper are independently grounded in each
    paper's primary chunks with correct doc_id, page, and verbatim quotes.
    """
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p4_c3",
        doc_id="doc_7fdb1200e823",
        page=4,
        text="so we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction",
    )
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id="doc_c67573ef50cc",
        page=3,
        text="we estimate Kronecker factors from token-wise second moments",
    )
    retriever = MockRetriever([chunk_gfw, chunk_lems])

    gen_json = json.dumps({
        "answer": "GFWSVD uses Lanczos truncated SVD, whereas LEMS estimates factors from token-wise second moments.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "GFWSVD employs truncated SVD using the Lanczos method.",
                "chunk_ids": ["doc_7fdb1200e823_p4_c3"],
                "quote": "so we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction",
            },
            {
                "claim_id": "c2",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "c1",
                "claim_id_b": "c2",
                "comparison_type": "supports",
                "summary": "Both papers specify distinct low-rank Kronecker factor estimators.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "GFWSVD Section 4.1 supports c1."}),
        json.dumps({"supported": True, "rationale": "LEMS Section 4.1 supports c2."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Do LEMS and GFWSVD describe the Kronecker-factor estimation process in the same way?",
        doc_ids=["doc_7fdb1200e823", "doc_c67573ef50cc"],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 2
    assert all(c.status == ClaimVerificationStatus.VERIFIED for c in resp.citation_checks)

    link_gfw = next(l for l in resp.evidence_links if l.claim_id == "c1")
    assert link_gfw.doc_id == "doc_7fdb1200e823"
    assert link_gfw.page == 4

    link_lems = next(l for l in resp.evidence_links if l.claim_id == "c2")
    assert link_lems.doc_id == "doc_c67573ef50cc"
    assert link_lems.page == 3


def test_cross_paper_structured_comparison_output_generation():
    """
    Regression Test 5C: Structured comparison output generation and validation.
    Verifies that when evidence supports a comparison, ClaimComparison is returned
    with valid claim IDs, doc_ids, and non-empty summary. Also tests fallback derivation.
    """
    doc_a = "doc_111111111111"
    doc_b = "doc_222222222222"
    chunk_a = make_scored_chunk(chunk_id="chk_a", doc_id=doc_a, page=1, text="Method A uses Lanczos SVD.")
    chunk_b = make_scored_chunk(chunk_id="chk_b", doc_id=doc_b, page=2, text="Method B uses token-wise statistics.")
    retriever = MockRetriever([chunk_a, chunk_b])

    # LLM returns claims but omits comparisons array; service fallback derives it
    gen_json = json.dumps({
        "answer": "Method A uses Lanczos SVD, while Method B uses token-wise statistics.",
        "claims": [
            {"claim_id": "c1", "text": "Method A uses Lanczos SVD.", "chunk_ids": ["chk_a"], "quote": "Method A uses Lanczos SVD."},
            {"claim_id": "c2", "text": "Method B uses token-wise statistics.", "chunk_ids": ["chk_b"], "quote": "Method B uses token-wise statistics."},
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact match for c1."}),
        json.dumps({"supported": True, "rationale": "Exact match for c2."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="Compare Method A and Method B", doc_ids=[doc_a, doc_b])
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.comparisons) >= 1
    comp = resp.comparisons[0]
    assert comp.claim_id_a == "c1"
    assert comp.claim_id_b == "c2"
    assert comp.doc_id_a == doc_a
    assert comp.doc_id_b == doc_b
    assert comp.comparison_type == ComparisonType.UNRESOLVED
    assert "unresolved" in comp.summary.lower()


def test_cross_paper_missing_evidence_unresolved_abstention():
    """
    Regression Test 5D: Missing evidence produces honest abstention outcome.
    When a comparative request is made across multiple papers, but retrieval fails
    to return chunks from one of the papers, the system honestly abstains rather
    than fabricating a comparison.
    """
    chunk_lems = make_scored_chunk(chunk_id="doc_c67573ef50cc_p3_c5", doc_id="doc_c67573ef50cc", page=3, text="LEMS evidence.")
    # GFWSVD chunk is missing from retrieval
    retriever = MockRetriever([chunk_lems])

    llm = MockLLMProvider([])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Compare LEMS versus GFWSVD factor estimation",
        doc_ids=["doc_c67573ef50cc", "doc_7fdb1200e823"],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert resp.abstain_reason is not None
    assert "Cross-paper evidence incomplete" in resp.abstain_reason
    assert "doc_7fdb1200e823" in resp.abstain_reason
    assert len(resp.comparisons) == 0


def test_cross_paper_single_paper_regression_unaffected():
    """
    Regression Test 5E: Existing single-paper answers and citation verification remain unaffected.
    Verifies that single-paper requests without comparison filters operate normally
    and citation verification succeeds without spurious comparisons.
    """
    chunk = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p6_c2",
        doc_id="doc_c67573ef50cc",
        page=6,
        text="All experiments are run on a single NVIDIA H200 GPU.",
    )
    retriever = MockRetriever([chunk])

    gen_json = json.dumps({
        "answer": "All experiments were executed on an NVIDIA H200 GPU.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "All experiments are run on a single NVIDIA H200 GPU.",
                "chunk_ids": ["doc_c67573ef50cc_p6_c2"],
                "quote": "All experiments are run on a single NVIDIA H200 GPU.",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact match."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="Which GPU was used?", doc_ids=["doc_c67573ef50cc"])
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 1
    assert resp.evidence_links[0].doc_id == "doc_c67573ef50cc"
    assert len(resp.comparisons) == 0

# =========================================================================
# 6. Final Day 4 Corrective Pass: Regression Tests
# =========================================================================

def test_cross_paper_conflicting_claims_omitted_comparison_does_not_default_to_supports():
    """
    Regression Test 6A: Conflicting claims when both comparisons and conflicts are omitted
    must NOT default to SUPPORTS.
    Verifies that Fallback 2 labels the comparison as UNRESOLVED and does not falsely
    infer consensus or support solely from distinct document IDs.
    """
    doc_a = "doc_111111111111"
    doc_b = "doc_222222222222"
    chunk_a = make_scored_chunk(
        chunk_id="chk_a",
        doc_id=doc_a,
        page=1,
        text="Method degrades severely under zero-shot evaluation.",
    )
    chunk_b = make_scored_chunk(
        chunk_id="chk_b",
        doc_id=doc_b,
        page=2,
        text="Method consistently outperforms baselines across all tasks.",
    )
    retriever = MockRetriever([chunk_a, chunk_b])

    # LLM returns conflicting statements but omits both conflicts and comparisons
    gen_json = json.dumps({
        "answer": "Paper A reports severe degradation, whereas Paper B reports outperforming baselines.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "Method degrades severely under zero-shot evaluation.",
                "chunk_ids": ["chk_a"],
                "quote": "Method degrades severely under zero-shot evaluation.",
            },
            {
                "claim_id": "c2",
                "text": "Method consistently outperforms baselines across all tasks.",
                "chunk_ids": ["chk_b"],
                "quote": "Method consistently outperforms baselines across all tasks.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact match for c1."}),
        json.dumps({"supported": True, "rationale": "Exact match for c2."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="Compare Paper A and Paper B findings", doc_ids=[doc_a, doc_b])
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    assert comp.comparison_type != ComparisonType.SUPPORTS
    assert comp.comparison_type == ComparisonType.UNRESOLVED
    assert comp.claim_id_a == "c1"
    assert comp.claim_id_b == "c2"
    assert comp.doc_id_a == doc_a
    assert comp.doc_id_b == doc_b
    assert "unresolved" in comp.summary.lower()


def test_cross_paper_three_documents_asymmetric_candidate_counts():
    """
    Regression Test 6B: Three requested documents with asymmetric candidate counts,
    including one document with zero relevant candidates.
    Verifies that candidate allocation does not starve documents with fewer candidates,
    does not crash on 0-candidate documents, and backfills slots stably.
    """
    from backend.app.retrieval.hybrid import select_allowed_candidates
    from backend.app.schemas import Chunk

    doc_1 = "doc_111111111111"
    doc_2 = "doc_222222222222"
    doc_3 = "doc_333333333333"

    # doc_1 has 5 candidates (high scores)
    c1_chunks = [
        Chunk(chunk_id=f"c1_{i}", doc_id=doc_1, page=i, text=f"Doc 1 chunk {i}", kind=ChunkKind.TEXT, char_start=0, char_end=15)
        for i in range(1, 6)
    ]
    # doc_2 has 1 candidate (lower score)
    c2_chunk = Chunk(chunk_id="c2_1", doc_id=doc_2, page=1, text="Doc 2 chunk 1", kind=ChunkKind.TEXT, char_start=0, char_end=15)
    # doc_3 has 0 candidates

    raw_results = [
        (c1_chunks[0], 10.0, 1),
        (c1_chunks[1], 9.0, 2),
        (c1_chunks[2], 8.0, 3),
        (c1_chunks[3], 7.0, 4),
        (c1_chunks[4], 6.0, 5),
        (c2_chunk, 4.0, 6),
    ]

    selected = select_allowed_candidates(
        raw_results=raw_results,
        max_candidates=5,
        doc_ids={doc_1, doc_2, doc_3},
    )

    selected_doc_ids = {c.doc_id for c, _, _ in selected}
    assert doc_1 in selected_doc_ids
    assert doc_2 in selected_doc_ids
    assert doc_3 not in selected_doc_ids  # 0 candidates available

    # doc_2 must not be starved despite lower score
    doc_2_count = sum(1 for c, _, _ in selected if c.doc_id == doc_2)
    assert doc_2_count == 1

    # Remaining 4 slots filled by doc_1
    doc_1_count = sum(1 for c, _, _ in selected if c.doc_id == doc_1)
    assert doc_1_count == 4
    assert len(selected) == 5

    # Monotonic original rank ordering preserved
    ranks = [r for _, _, r in selected]
    assert ranks == sorted(ranks)


def test_cross_paper_both_documents_retrieved_only_one_cited_secondary_qualification():
    """
    Regression Test 6C: Both documents retrieved but verified claims cite only one document.
    Asserts that the secondary-source qualification note is appended to the answer text.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="GFWSVD creates a Kronecker product A ⊗ G via Lanczos approximation.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p4_c3",
        doc_id=doc_gfw,
        page=4,
        text="We employ truncated SVD using the Lanczos method to avoid matrix construction.",
    )
    # Both documents are present in retrieval results
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    # LLM generator only formulates a claim citing LEMS describing GFWSVD
    gen_json = json.dumps({
        "answer": "GFWSVD creates a Kronecker product via Lanczos approximation according to LEMS.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "GFWSVD creates a Kronecker product via Lanczos approximation.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "GFWSVD creates a Kronecker product A ⊗ G via Lanczos approximation.",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact match in LEMS chunk."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How does GFWSVD estimate factors compared to LEMS?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 1
    assert resp.evidence_links[0].doc_id == doc_lems

    # Secondary-source qualification must be present alerting the user
    expected_note = (
        f"Note: Claims regarding {doc_gfw} rely on descriptions in {doc_lems} "
        f"rather than independent primary-source evidence from {doc_gfw}."
    )
    assert expected_note in resp.answer
    assert not resp.answer.endswith("..")
    assert resp.answer.endswith(".")


def test_cross_paper_multi_doc_non_comparative_preserves_comparative_abstention():
    """
    Regression Test 6D: Multi-document non-comparative question handling.
    Verifies that a non-comparative query across multiple documents is not erroneously
    blocked by the comparative-query retrieval sufficiency check, while ensuring
    comparative queries still trigger honest abstention when evidence is missing.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p6_c2",
        doc_id=doc_lems,
        page=6,
        text="All experiments are run on a single NVIDIA H200 GPU.",
    )
    # Only LEMS chunk in retrieval; GFWSVD has 0 chunks
    retriever = MockRetriever([chunk_lems])

    # Case 1: Non-comparative query asking a factual question across doc_ids
    gen_json = json.dumps({
        "answer": "The experiments utilized an NVIDIA H200 GPU.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "All experiments are run on a single NVIDIA H200 GPU.",
                "chunk_ids": ["doc_c67573ef50cc_p6_c2"],
                "quote": "All experiments are run on a single NVIDIA H200 GPU.",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })
    llm1 = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact match."}),
    ])
    verifier1 = CitationVerifier(llm_provider=llm1)
    service1 = AnswerService(retriever=retriever, llm_provider=llm1, verifier=verifier1)

    non_comp_req = AnswerRequest(
        question="Which GPU hardware was used for running the experiments?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp1 = run_sync(service1.answer(non_comp_req))
    # Must answer rather than abstaining because query is not comparative
    assert resp1.status == AnswerStatus.ANSWERED
    assert len(resp1.evidence_links) == 1
    # Secondary-source qualification note informs user that doc_gfw was not cited
    assert "rather than independent primary-source evidence" in resp1.answer

    # Case 2: Comparative query with the same retriever missing doc_gfw chunks
    llm2 = MockLLMProvider([])
    verifier2 = CitationVerifier(llm_provider=llm2)
    service2 = AnswerService(retriever=retriever, llm_provider=llm2, verifier=verifier2)

    comp_req = AnswerRequest(
        question="Compare the GPU hardware used between LEMS and GFWSVD",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp2 = run_sync(service2.answer(comp_req))
    # Must abstain because evidence is incomplete for a comparative query
    assert resp2.status == AnswerStatus.ABSTAINED
    assert "Cross-paper evidence incomplete" in resp2.abstain_reason

# =========================================================================
# 7. Day 4 Corrective Pass 2: Methodological Contrast & Candidate Scaling
# =========================================================================

def test_cross_paper_neutral_methodological_contrast_demoted_to_unresolved():
    """
    Regression Test 7A: Neutral methodological contrast without documented conflict reason
    must be demoted to UNRESOLVED rather than falsely labelled apparent_conflict.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="To avoid the rank bottleneck, we estimate Kronecker factors from token-wise second moments.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p4_c3",
        doc_id=doc_gfw,
        page=4,
        text="we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    # LLM returns apparent_conflict for different estimation algorithms, but conflict_reason is null
    gen_json = json.dumps({
        "answer": "GFWSVD uses Lanczos truncated SVD, whereas LEMS estimates factors from token-wise moments.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "GFWSVD employs truncated SVD using the Lanczos method.",
                "chunk_ids": ["doc_7fdb1200e823_p4_c3"],
                "quote": "we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction",
            },
            {
                "claim_id": "c2",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "c1",
                "claim_id_b": "c2",
                "comparison_type": "apparent_conflict",
                "summary": "GFWSVD uses Lanczos SVD, whereas LEMS uses token-wise moments.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact match for c1."}),
        json.dumps({"supported": True, "rationale": "Exact match for c2."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Do LEMS and GFWSVD describe factor estimation the same way?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    # Guard: apparent_conflict without conflict_reason must be demoted to UNRESOLVED
    assert comp.comparison_type == ComparisonType.UNRESOLVED
    assert comp.conflict_reason is None


def test_cross_paper_genuine_apparent_conflict_retains_apparent_conflict():
    """
    Regression Test 7B: Genuine apparent conflict with a documented conflict cause
    (e.g. differing evaluation conditions) is preserved as APPARENT_CONFLICT.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p6_c5",
        doc_id=doc_lems,
        page=6,
        text="GFWSVD degrades significantly under identical conditions.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p9_c4",
        doc_id=doc_gfw,
        page=9,
        text="GFWSVD consistently outperforms diagonal and activation-based baselines across all tasks.",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    reason_str = "Differing evaluation conditions: zero-shot uniform compression vs fine-tuning with regularization."
    gen_json = json.dumps({
        "answer": "LEMS reports GFWSVD degrades in zero-shot, whereas GFWSVD reports outperforming baselines under fine-tuning.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS reports GFWSVD degrades significantly under identical conditions.",
                "chunk_ids": ["doc_c67573ef50cc_p6_c5"],
                "quote": "GFWSVD degrades significantly under identical conditions.",
            },
            {
                "claim_id": "c2",
                "text": "GFWSVD consistently outperforms diagonal and activation-based baselines across all tasks.",
                "chunk_ids": ["doc_7fdb1200e823_p9_c4"],
                "quote": "GFWSVD consistently outperforms diagonal and activation-based baselines across all tasks.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "c1",
                "claim_id_b": "c2",
                "comparison_type": "apparent_conflict",
                "summary": "Reported performance differs due to evaluation regime.",
                "conflict_reason": reason_str,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact match for c1."}),
        json.dumps({"supported": True, "rationale": "Exact match for c2."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Does GFWSVD perform well across both papers?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    # Retains APPARENT_CONFLICT when a meaningful conflict_reason is present
    assert comp.comparison_type == ComparisonType.APPARENT_CONFLICT
    assert comp.conflict_reason == reason_str


def test_cross_paper_multi_document_candidate_pool_scaling():
    """
    Regression Test 7C: Multi-document retrieval candidate pool scaling.
    Verifies that when multiple documents are requested, candidate pool depth scales
    with len(doc_filter_set), ensuring balanced candidate allocation up to candidate_pool_size
    per document into RRF fusion without prematurely pruning specific technical passages.
    """
    from backend.app.retrieval.hybrid import HybridRetriever, select_allowed_candidates
    from backend.app.schemas import Chunk

    doc_a = "doc_111111111111"
    doc_b = "doc_222222222222"

    chunks_a = [
        Chunk(chunk_id=f"a_{i}", doc_id=doc_a, page=i, text=f"Chunk A {i}", kind=ChunkKind.TEXT, char_start=0, char_end=10)
        for i in range(1, 41)
    ]
    chunks_b = [
        Chunk(chunk_id=f"b_{i}", doc_id=doc_b, page=i, text=f"Chunk B {i}", kind=ChunkKind.TEXT, char_start=0, char_end=10)
        for i in range(1, 41)
    ]

    all_chunks = chunks_a + chunks_b
    retriever = HybridRetriever()
    retriever.index_chunks(all_chunks)

    # For N=2 documents, candidate_pool_size=10 should yield effective_pool_size=20 (10 per document)
    res = retriever.search(
        query="Chunk",
        top_k=8,
        doc_ids=[doc_a, doc_b],
        candidate_pool_size=10,
        use_reranker=False,
    )

    assert len(res) == 8
    # candidate_chunks pool should have been 20 (10 per doc)
    assert retriever.last_total_candidates == 20


class DeterministicMockReranker:
    """Mock reranker returning predetermined scores for testing context selection."""

    def __init__(self, score_map: Dict[str, float]):
        self.score_map = score_map

    def rerank(self, query: str, chunks: List[Chunk], top_k: Optional[int] = None) -> List[Tuple[Chunk, float, int]]:
        scored = [(c, self.score_map.get(c.chunk_id, -10.0)) for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = top_k if top_k is not None else len(scored)
        return [(c, s, rank) for rank, (c, s) in enumerate(scored[:top_k], 1)]


def test_cross_paper_target_scenario_evidence_coverage():
    """
    Regression Test 8A: Target Scenario Multi-Document Coverage.
    Verifies that when both requested documents contain genuinely relevant evidence,
    both receive guaranteed representation in the final context through the evidence allowance,
    and remaining slots are awarded by overall rerank score without duplicates or ordering inversion.
    """
    from backend.app.retrieval.hybrid import HybridRetriever
    from backend.app.schemas import Chunk

    doc_a = "doc_lems"
    doc_b = "doc_gfw"

    chunks_a = [
        Chunk(chunk_id=f"a_{i}", doc_id=doc_a, page=i, text=f"LEMS estimation passage {i}", kind=ChunkKind.TEXT, char_start=0, char_end=20)
        for i in range(1, 10)
    ]
    chunks_b = [
        Chunk(chunk_id=f"b_{i}", doc_id=doc_b, page=i, text=f"GFWSVD factor computation {i}", kind=ChunkKind.TEXT, char_start=0, char_end=20)
        for i in range(1, 10)
    ]

    # Scores: doc_a has high scores (4.0 to 2.0); doc_b has strong scores (3.0 to 1.5)
    score_map = {
        "a_1": 4.5, "a_2": 3.8, "a_3": 3.2, "a_4": 2.8, "a_5": 2.2,
        "b_1": 3.0, "b_2": 2.5, "b_3": 1.8, "b_4": 1.5, "b_5": 0.5,
    }

    retriever = HybridRetriever()
    retriever.index_chunks(chunks_a + chunks_b)
    retriever.reranker = DeterministicMockReranker(score_map)

    results = retriever.search(
        query="estimation",
        top_k=8,
        doc_ids=[doc_a, doc_b],
        use_reranker=True,
    )

    assert len(results) == 8
    doc_a_cids = [r.chunk_id for r in results if r.doc_id == doc_a]
    doc_b_cids = [r.chunk_id for r in results if r.doc_id == doc_b]

    # Both documents must have representation
    assert len(doc_a_cids) >= 2
    assert len(doc_b_cids) >= 2
    # doc_b's top relevant chunks must be present
    assert "b_1" in doc_b_cids
    assert "b_2" in doc_b_cids
    # Monotonic ordering strictly descending by score
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Zero duplicates
    assert len(set(r.chunk_id for r in results)) == 8


def test_cross_paper_irrelevant_second_paper_not_forced_into_context():
    """
    Regression Test 8B: Second paper has NO relevant evidence.
    Verifies that the multi-document retriever does NOT force irrelevant chunks into the
    final top-k context when a requested document contains no relevant evidence (all scores
    below the relevance threshold), preserving invariant that irrelevant evidence is not forced.
    """
    from backend.app.retrieval.hybrid import HybridRetriever
    from backend.app.schemas import Chunk

    doc_a = "doc_relevant"
    doc_b = "doc_irrelevant"

    chunks_a = [
        Chunk(chunk_id=f"rel_{i}", doc_id=doc_a, page=i, text=f"Relevant passage {i}", kind=ChunkKind.TEXT, char_start=0, char_end=20)
        for i in range(1, 15)
    ]
    chunks_b = [
        Chunk(chunk_id=f"irrel_{i}", doc_id=doc_b, page=i, text=f"Irrelevant baking recipe {i}", kind=ChunkKind.TEXT, char_start=0, char_end=20)
        for i in range(1, 15)
    ]

    # doc_a chunks score +4.0 down to +1.0; doc_b chunks score -10.5 (completely irrelevant)
    score_map = {f"rel_{i}": 4.0 - (i * 0.2) for i in range(1, 15)}
    score_map.update({f"irrel_{i}": -10.5 - (i * 0.1) for i in range(1, 15)})

    retriever = HybridRetriever()
    retriever.index_chunks(chunks_a + chunks_b)
    retriever.reranker = DeterministicMockReranker(score_map)

    results = retriever.search(
        query="Relevant passage",
        top_k=8,
        doc_ids=[doc_a, doc_b],
        use_reranker=True,
    )

    assert len(results) == 8
    doc_b_results = [r for r in results if r.doc_id == doc_b]
    # Irrelevant paper must have ZERO chunks in the final context
    assert len(doc_b_results) == 0
    # All 8 context slots must be awarded to relevant doc_a chunks
    assert all(r.doc_id == doc_a for r in results)
    assert all(r.rerank_score > 0.0 for r in results)


# =========================================================================
# 9. Day 4 Corrective Pass 4: Claim Attribution Integrity
# =========================================================================

def test_cross_paper_attribution_bleed_fails_verification_and_triggers_abstention():
    """
    Regression Test 9A: Reproduces exact Test E attribution-bleed failure.
    Chunk A (LEMS p3_c5) mentions per-sequence gradients, but NOT 'calibration dataset'.
    Chunk B (GFWSVD p5_c5) mentions 'calibration dataset'.
    Generator creates claim_1: 'GFWSVD determines factors using per-sequence gradients from a calibration dataset'
    citing Chunk A (LEMS).
    Layer 3 correctly marks claim_1 unsupported ('calibration dataset' not in cited chunk).
    Strict zero-tolerance policy triggers: Answer status is ABSTAINED, all claims and comparisons wiped.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="GFWSVD determines the factors A and G using k per-sequence gradients. We estimate Kronecker factors from token-wise second moments.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p5_c5",
        doc_id=doc_gfw,
        page=5,
        text="Given a calibration dataset D, the per-sample gradients are computed, and the corresponding Fisher factors are estimated according to Algorithm 1.",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    # Generator blends "calibration dataset" from GFWSVD chunk into claim_1 citing LEMS chunk
    gen_json = json.dumps({
        "answer": "GFWSVD uses per-sequence gradients from a calibration dataset, while LEMS uses token-wise moments.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "GFWSVD determines factors A and G using per-sequence gradients from a calibration dataset.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "GFWSVD determines the factors A and G using k per-sequence gradients.",
            },
            {
                "claim_id": "claim_2",
                "text": "GFWSVD estimates Fisher factors according to Algorithm 1.",
                "chunk_ids": ["doc_7fdb1200e823_p5_c5"],
                "quote": "Fisher factors are estimated according to Algorithm 1.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "comparison_type": "unresolved",
                "summary": "Both describe factor estimation.",
                "conflict_reason": None,
            }
        ],
    })

    # Verifier evaluation: claim_1 fails Layer 3 because 'calibration dataset' is not in chunk_lems
    llm = MockLLMProvider([
        gen_json,
        json.dumps({
            "supported": False,
            "rationale": "The evidence states that GFWSVD determines factors A and G using k per-sequence gradients, but it does not mention that these gradients are derived from a 'calibration dataset'.",
        }),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Do LEMS and GFWSVD describe the Kronecker-factor estimation process in the same way?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    # Assert fail-closed zero-tolerance abstention
    assert resp.status == AnswerStatus.ABSTAINED
    assert "Strict zero-tolerance policy" in resp.abstain_reason
    assert "claim_1" in resp.abstain_reason
    # All evidence links and comparisons must be wiped
    assert len(resp.evidence_links) == 0
    assert len(resp.comparisons) == 0
    assert "unsupported" in resp.citation_checks[0].status


def test_cross_paper_separate_valid_claims_support_methodological_comparison():
    """
    Regression Test 9B: Correct claim attribution integrity with dedicated claims.
    Claim 1 cites GFWSVD chunk (p5_c5) with exact quote from GFWSVD.
    Claim 2 cites LEMS chunk (p3_c5) with exact quote from LEMS.
    Both claims pass Layer 1, 2, and 3 verification.
    Structured comparison connects them as UNRESOLVED (neutral methodological contrast).
    Asserts: status=ANSWERED, both papers cited in evidence_links, comparisons populated.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="we estimate Kronecker factors from token-wise second moments.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p5_c5",
        doc_id=doc_gfw,
        page=5,
        text="Given a calibration dataset D, the per-sample gradients are computed, and the corresponding Fisher factors are estimated according to Algorithm 1.",
    )
    retriever = MockRetriever([chunk_lems, chunk_gfw])

    gen_json = json.dumps({
        "answer": "GFWSVD computes per-sample gradients given a calibration dataset via Algorithm 1, whereas LEMS estimates factors from token-wise second moments.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "GFWSVD computes per-sample gradients given a calibration dataset to estimate Fisher factors according to Algorithm 1.",
                "chunk_ids": ["doc_7fdb1200e823_p5_c5"],
                "quote": "Given a calibration dataset D, the per-sample gradients are computed, and the corresponding Fisher factors are estimated according to Algorithm 1.",
            },
            {
                "claim_id": "claim_2",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "comparison_type": "unresolved",
                "summary": "GFWSVD estimates factors from per-sample gradients via Algorithm 1, whereas LEMS estimates factors from token-wise second moments.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact entailment for GFWSVD claim."}),
        json.dumps({"supported": True, "rationale": "Exact entailment for LEMS claim."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Do LEMS and GFWSVD describe the Kronecker-factor estimation process in the same way?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 2
    cited_docs = {l.doc_id for l in resp.evidence_links}
    assert cited_docs == {doc_lems, doc_gfw}
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    assert comp.comparison_type == ComparisonType.UNRESOLVED
    assert comp.doc_id_a == doc_gfw
    assert comp.doc_id_b == doc_lems


def test_cross_paper_mismatched_source_attribution_fails_quote_or_doc_verification():
    """
    Regression Test 9C: Mismatched source attribution.
    A claim stating GFWSVD uses Lanczos quotes GFWSVD text, but attaches a chunk ID from LEMS
    where that verbatim quote does not exist (Layer 2 mismatch).
    Fails verification closed and triggers honest abstention.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="LEMS estimates Kronecker factors from token-wise moments.",
    )
    retriever = MockRetriever([chunk_lems])

    gen_json = json.dumps({
        "answer": "GFWSVD uses Lanczos SVD.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "GFWSVD uses truncated SVD using the Lanczos method.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "truncated SVD using the Lanczos method (Lanczos, 1950)",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })

    llm = MockLLMProvider([gen_json])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How does GFWSVD estimate Kronecker factors?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert len(resp.evidence_links) == 0
    # Check status is unsupported due to Layer 2 quote mismatch
    assert resp.citation_checks[0].status == ClaimVerificationStatus.UNSUPPORTED
    assert "Layer 2 failure" in resp.citation_checks[0].notes


# =============================================================================
# SECTION 10: STRUCTURED COMPARISON INTEGRITY (PASS 5)
# =============================================================================


def test_cross_paper_valid_distinct_document_comparison_retained():
    """
    Pass 5 Requirement: Valid cross-paper comparison linking verified claims from each requested paper.
    - User requests 2 papers (doc_a, doc_b).
    - Claim 1 cites doc_a; Claim 2 cites doc_b.
    - Both claims pass verification.
    - Comparison links claim_1 and claim_2 with doc_id_a=doc_a, doc_id_b=doc_b.
    - Assert: status=ANSWERED, comparison is retained and populated.
    """
    doc_a = "doc_aaaaaaaaaaaa"
    doc_b = "doc_bbbbbbbbbbbb"

    chunk_a = make_scored_chunk(
        chunk_id="doc_aaaaaaaaaaaa_p1_c1",
        doc_id=doc_a,
        page=1,
        text="Method A uses stochastic gradient descent with learning rate 0.01.",
    )
    chunk_b = make_scored_chunk(
        chunk_id="doc_bbbbbbbbbbbb_p1_c1",
        doc_id=doc_b,
        page=1,
        text="Method B uses Adam optimizer with beta 0.9.",
    )
    retriever = MockRetriever([chunk_a, chunk_b])

    gen_json = json.dumps({
        "answer": "Method A uses stochastic gradient descent whereas Method B uses Adam optimizer.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "Method A uses stochastic gradient descent.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c1"],
                "quote": "Method A uses stochastic gradient descent",
            },
            {
                "claim_id": "claim_2",
                "text": "Method B uses Adam optimizer.",
                "chunk_ids": ["doc_bbbbbbbbbbbb_p1_c1"],
                "quote": "Method B uses Adam optimizer",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_a,
                "doc_id_b": doc_b,
                "comparison_type": "unresolved",
                "summary": "Method A uses SGD whereas Method B uses Adam optimizer.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed by doc_A."}),
        json.dumps({"supported": True, "rationale": "Entailed by doc_B."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Compare optimization methods between Paper A and Paper B.",
        doc_ids=[doc_a, doc_b],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 2
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    assert comp.claim_id_a == "claim_1"
    assert comp.claim_id_b == "claim_2"
    assert comp.doc_id_a == doc_a
    assert comp.doc_id_b == doc_b
    assert comp.comparison_type == ComparisonType.UNRESOLVED
    assert "Method A uses SGD whereas Method B uses Adam" in comp.summary


def test_cross_paper_same_document_pair_misattributed_summary_suppressed():
    """
    Pass 5 Requirement: Same-document pair incorrectly presented as cross-paper comparison.
    - User requests 2 papers (doc_a, doc_b).
    - Claim 1 cites doc_a; Claim 2 also cites doc_a.
    - Both claims pass verification within doc_a.
    - Generator emits a comparison linking claim_1 and claim_2, but summary purports to compare
      Doc A and Doc B ("Doc A uses SGD whereas Doc B uses Adam optimizer").
    - Assert: The spurious same-document comparison is suppressed.
    """
    doc_a = "doc_aaaaaaaaaaaa"
    doc_b = "doc_bbbbbbbbbbbb"

    chunk_a1 = make_scored_chunk(
        chunk_id="doc_aaaaaaaaaaaa_p1_c1",
        doc_id=doc_a,
        page=1,
        text="Method A uses stochastic gradient descent.",
    )
    chunk_a2 = make_scored_chunk(
        chunk_id="doc_aaaaaaaaaaaa_p1_c2",
        doc_id=doc_a,
        page=1,
        text="In Method A, momentum parameter is set to 0.9.",
    )
    chunk_b = make_scored_chunk(
        chunk_id="doc_bbbbbbbbbbbb_p1_c1",
        doc_id=doc_b,
        page=1,
        text="Method B uses Adam optimizer.",
    )
    # Include doc_b chunk in retrieval pool so retrieval doesn't trigger incomplete-evidence abstention
    retriever = MockRetriever([chunk_a1, chunk_a2, chunk_b])

    gen_json = json.dumps({
        "answer": "Method A uses SGD with momentum 0.9.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "Method A uses stochastic gradient descent.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c1"],
                "quote": "Method A uses stochastic gradient descent",
            },
            {
                "claim_id": "claim_2",
                "text": "In Method A, momentum parameter is set to 0.9.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c2"],
                "quote": "momentum parameter is set to 0.9",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_a,
                "doc_id_b": doc_a,
                "comparison_type": "unresolved",
                "summary": "Doc A uses SGD whereas Doc B uses Adam optimizer.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed by doc_a."}),
        json.dumps({"supported": True, "rationale": "Entailed by doc_a."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Describe Method A parameters.",
        doc_ids=[doc_a, doc_b],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    # The misattributed same-doc comparison MUST be suppressed
    assert len(resp.comparisons) == 0


def test_cross_paper_summary_referencing_unrepresented_paper_filtered():
    """
    Pass 5 Requirement: Comparison whose summary refers to a paper not represented by linked claims.
    - User requests 3 papers (doc_a, doc_b, doc_c).
    - Claim 1 cites doc_a; Claim 2 cites doc_b.
    - Comparison links claim_1 and claim_2, but its summary asserts a comparison involving Paper C
      ("Doc A uses SGD, Doc B uses Adam, whereas Doc C uses RMSprop").
    - Because doc_c is not represented by either claim in this comparison, it is filtered out.
    """
    doc_a = "doc_aaaaaaaaaaaa"
    doc_b = "doc_bbbbbbbbbbbb"
    doc_c = "doc_cccccccccccc"

    chunk_a = make_scored_chunk(chunk_id="doc_aaaaaaaaaaaa_p1_c1", doc_id=doc_a, text="Doc A uses SGD.")
    chunk_b = make_scored_chunk(chunk_id="doc_bbbbbbbbbbbb_p1_c1", doc_id=doc_b, text="Doc B uses Adam.")
    chunk_c = make_scored_chunk(chunk_id="doc_cccccccccccc_p1_c1", doc_id=doc_c, text="Doc C uses RMSprop.")
    retriever = MockRetriever([chunk_a, chunk_b, chunk_c])

    gen_json = json.dumps({
        "answer": "Doc A uses SGD, Doc B uses Adam, and Doc C uses RMSprop.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "Doc A uses SGD.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c1"],
                "quote": "Doc A uses SGD.",
            },
            {
                "claim_id": "claim_2",
                "text": "Doc B uses Adam.",
                "chunk_ids": ["doc_bbbbbbbbbbbb_p1_c1"],
                "quote": "Doc B uses Adam.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_a,
                "doc_id_b": doc_b,
                "comparison_type": "unresolved",
                "summary": "Doc A uses SGD, Doc B uses Adam, whereas Doc C uses RMSprop.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed."}),
        json.dumps({"supported": True, "rationale": "Entailed."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Examine optimizers across papers.",
        doc_ids=[doc_a, doc_b, doc_c],
    )
    resp = run_sync(service.answer(req))

    # The comparison referencing unrepresented doc_c is suppressed
    assert len(resp.comparisons) == 0


def test_cross_paper_unverified_claims_suppress_comparison():
    """
    Pass 5 Requirement: Suppression of comparisons involving unverified claims.
    - Comparison links claim_1 and claim_2.
    - Claim 2 fails verification (e.g. quote mismatch Layer 2 failure).
    - Assert: The comparison is suppressed and does not appear in resp.comparisons.
      Under zero-tolerance policy, the overall answer status is ABSTAINED.
    """
    doc_a = "doc_aaaaaaaaaaaa"
    doc_b = "doc_bbbbbbbbbbbb"

    chunk_a = make_scored_chunk(chunk_id="doc_aaaaaaaaaaaa_p1_c1", doc_id=doc_a, text="Doc A uses SGD.")
    chunk_b = make_scored_chunk(chunk_id="doc_bbbbbbbbbbbb_p1_c1", doc_id=doc_b, text="Doc B uses Adam.")
    retriever = MockRetriever([chunk_a, chunk_b])

    gen_json = json.dumps({
        "answer": "Doc A uses SGD whereas Doc B uses Adam.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "Doc A uses SGD.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c1"],
                "quote": "Doc A uses SGD.",
            },
            {
                "claim_id": "claim_2",
                "text": "Doc B uses RMSprop.", # Mismatch: text has Adam, not RMSprop
                "chunk_ids": ["doc_bbbbbbbbbbbb_p1_c1"],
                "quote": "Doc B uses RMSprop.", # Fails quote match
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_a,
                "doc_id_b": doc_b,
                "comparison_type": "unresolved",
                "summary": "Doc A uses SGD whereas Doc B uses RMSprop.",
                "conflict_reason": None,
            }
        ],
    })

    # Verifier: claim_1 passes L1 & L2; LLM marks claim_1 supported. Claim 2 fails L2 quote match.
    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Compare optimizers.",
        doc_ids=[doc_a, doc_b],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert len(resp.comparisons) == 0


def test_single_document_within_paper_comparison_preserved():
    """
    Pass 5 Requirement: Preservation of legitimate within-paper comparisons.
    - Single-document inquiry (req.doc_ids = ["doc_aaaaaaaaaaaa"]).
    - Both claims cite doc_a (e.g. baseline model vs proposed model).
    - Both claims pass verification.
    - Comparison links claim_1 and claim_2 within doc_a without claiming to compare multiple papers.
    - Assert: status=ANSWERED, comparison is PRESERVED.
    """
    doc_a = "doc_aaaaaaaaaaaa"

    chunk_1 = make_scored_chunk(
        chunk_id="doc_aaaaaaaaaaaa_p1_c1",
        doc_id=doc_a,
        page=1,
        text="The baseline model achieves 70% accuracy on the validation benchmark.",
    )
    chunk_2 = make_scored_chunk(
        chunk_id="doc_aaaaaaaaaaaa_p1_c2",
        doc_id=doc_a,
        page=2,
        text="The proposed model achieves 85% accuracy on the validation benchmark.",
    )
    retriever = MockRetriever([chunk_1, chunk_2])

    gen_json = json.dumps({
        "answer": "The baseline model achieves 70% accuracy, whereas the proposed model achieves 85% accuracy.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "The baseline model achieves 70% accuracy on the validation benchmark.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c1"],
                "quote": "baseline model achieves 70% accuracy",
            },
            {
                "claim_id": "claim_2",
                "text": "The proposed model achieves 85% accuracy on the validation benchmark.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c2"],
                "quote": "proposed model achieves 85% accuracy",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_a,
                "doc_id_b": doc_a,
                "comparison_type": "unresolved",
                "summary": "The baseline model achieves 70% accuracy whereas the proposed model achieves 85% accuracy.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed."}),
        json.dumps({"supported": True, "rationale": "Entailed."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How does the proposed model compare to the baseline within this paper?",
        doc_ids=[doc_a],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 2
    # Legitimate within-paper comparison MUST be preserved
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    assert comp.claim_id_a == "claim_1"
    assert comp.claim_id_b == "claim_2"
    assert comp.doc_id_a == doc_a
    assert comp.doc_id_b == doc_a
    assert "baseline model achieves 70% accuracy whereas the proposed model achieves 85%" in comp.summary


def test_test_e_scenario_duplicate_same_doc_comparison_filtered_distinct_retained():
    """
    Pass 5 Specific Regression: Reproduces exact Test E condition where generator emitted:
    - Comparison 1: claim_1 (GFWSVD) vs claim_3 (LEMS) -> genuine cross-paper comparison.
    - Comparison 2: claim_2 (LEMS) vs claim_3 (LEMS) -> same-document pair within LEMS,
      with summary contrasting GFWSVD and LEMS.
    Asserts:
    - Comparison 1 is retained.
    - Comparison 2 is cleanly filtered out.
    - Exactly 1 valid cross-paper comparison is emitted in resp.comparisons.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p5_c5",
        doc_id=doc_gfw,
        page=5,
        text="Given a calibration dataset D, the per-sample gradients are computed, and the corresponding Fisher factors are estimated according to Algorithm 1.",
    )
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="GFWSVD (Chekalina et al., 2025) determines the factors A and G using k per-sequence gradients. To avoid the rank bottleneck of GFWSVD’s per-sequence gradient-based estimator, we estimate Kronecker factors from token-wise second moments.",
    )
    retriever = MockRetriever([chunk_gfw, chunk_lems])

    gen_json = json.dumps({
        "answer": "No, LEMS and GFWSVD do not describe factor estimation in the same way.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "GFWSVD computes per-sample gradients given a calibration dataset to estimate Fisher factors.",
                "chunk_ids": ["doc_7fdb1200e823_p5_c5"],
                "quote": "Given a calibration dataset D, the per-sample gradients are computed",
            },
            {
                "claim_id": "claim_2",
                "text": "GFWSVD determines factors using k per-sequence gradients.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "GFWSVD (Chekalina et al., 2025) determines the factors A and G using k per-sequence gradients",
            },
            {
                "claim_id": "claim_3",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_3",
                "doc_id_a": doc_gfw,
                "doc_id_b": doc_lems,
                "comparison_type": "unresolved",
                "summary": "GFWSVD estimates factors from per-sample gradients, whereas LEMS estimates factors from token-wise second moments.",
                "conflict_reason": None,
            },
            {
                "claim_id_a": "claim_2",
                "claim_id_b": "claim_3",
                "doc_id_a": doc_lems,
                "doc_id_b": doc_lems,
                "comparison_type": "unresolved",
                "summary": "GFWSVD uses per-sequence gradients to determine Kronecker factors, whereas LEMS uses token-wise second moments.",
                "conflict_reason": None,
            },
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed GFWSVD primary."}),
        json.dumps({"supported": True, "rationale": "Entailed LEMS secondary."}),
        json.dumps({"supported": True, "rationale": "Entailed LEMS primary."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Do LEMS and GFWSVD describe the Kronecker-factor estimation process in the same way?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 3
    # Exactly 1 comparison should remain: the cross-paper comparison linking doc_gfw and doc_lems
    assert len(resp.comparisons) == 1
    retained_comp = resp.comparisons[0]
    assert retained_comp.claim_id_a == "claim_1"
    assert retained_comp.claim_id_b == "claim_3"
    assert retained_comp.doc_id_a == doc_gfw
    assert retained_comp.doc_id_b == doc_lems
    assert retained_comp.comparison_type == ComparisonType.UNRESOLVED


# =============================================================================
# SECTION 11: PASS 6 MINIMAL CLAIMS & METHOD ATTRIBUTION INTEGRITY
# =============================================================================


def test_pass6_minimal_claims_two_paper_answer():
    """
    Pass 6 Requirement A: Minimal claim generation.
    A valid two-paper answer with exactly one verified claim per paper (GFWSVD and LEMS)
    must fully answer the comparison without requiring an unnecessary third claim.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p5_c5",
        doc_id=doc_gfw,
        page=5,
        text="Given a calibration dataset D, the per-sample gradients are computed, and the corresponding Fisher factors are estimated according to Algorithm 1.",
    )
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="we estimate Kronecker factors from token-wise second moments.",
    )
    retriever = MockRetriever([chunk_gfw, chunk_lems])

    gen_json = json.dumps({
        "answer": "GFWSVD computes per-sample gradients given a calibration dataset, whereas LEMS estimates factors from token-wise second moments.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "GFWSVD computes per-sample gradients given a calibration dataset to estimate Fisher factors according to Algorithm 1.",
                "chunk_ids": ["doc_7fdb1200e823_p5_c5"],
                "quote": "Given a calibration dataset D, the per-sample gradients are computed, and the corresponding Fisher factors are estimated according to Algorithm 1.",
            },
            {
                "claim_id": "claim_2",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_gfw,
                "doc_id_b": doc_lems,
                "comparison_type": "unresolved",
                "summary": "GFWSVD estimates factors from per-sample gradients via Algorithm 1, whereas LEMS estimates factors from token-wise second moments.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Exact primary grounding for GFWSVD."}),
        json.dumps({"supported": True, "rationale": "Exact primary grounding for LEMS."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Do LEMS and GFWSVD describe the Kronecker-factor estimation process in the same way?",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    # Exactly two minimal sufficient claims, one per requested paper
    assert len(resp.evidence_links) == 2
    cited_docs = {l.doc_id for l in resp.evidence_links}
    assert cited_docs == {doc_gfw, doc_lems}
    # Exactly one valid cross-paper comparison
    assert len(resp.comparisons) == 1
    comp = resp.comparisons[0]
    assert comp.claim_id_a == "claim_1"
    assert comp.claim_id_b == "claim_2"
    assert comp.doc_id_a == doc_gfw
    assert comp.doc_id_b == doc_lems
    assert comp.comparison_type == ComparisonType.UNRESOLVED


def test_pass6_appendix_attribution_unsupported_when_method_connection_missing():
    """
    Pass 6 Requirement B: Appendix attribution.
    An appendix chunk describing KFAC-SVD without establishing that it is LEMS's proposed
    method must not support a claim attributing that method to LEMS.
    Evaluates Layer 3 failure and verifies fail-closed zero-tolerance abstention.
    """
    doc_lems = "doc_c67573ef50cc"

    # Chunk text from appendix B.2 describing KFAC-SVD protocol without naming LEMS
    chunk_appendix = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p21_c2",
        doc_id=doc_lems,
        page=21,
        text="To collect token-wise activations X and output gradients, we run forward and backward passes on our calibration data. From those, we estimate the Kronecker factors, bA and bG.",
    )
    retriever = MockRetriever([chunk_appendix])

    gen_json = json.dumps({
        "answer": "LEMS runs forward and backward passes on calibration data to estimate Kronecker factors.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "LEMS runs forward and backward passes on calibration data to estimate Kronecker factors.",
                "chunk_ids": ["doc_c67573ef50cc_p21_c2"],
                "quote": "we run forward and backward passes on our calibration data. From those, we estimate the Kronecker factors",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })

    # Verifier evaluation: fails Layer 3 because evidence describes KFAC-SVD protocol, not LEMS
    llm = MockLLMProvider([
        gen_json,
        json.dumps({
            "supported": False,
            "rationale": "The evidence describes the procedure for KFAC-SVD, but it does not mention LEMS or attribute this process to LEMS.",
        }),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How does LEMS estimate Kronecker factors?",
        doc_ids=[doc_lems],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert "Strict zero-tolerance policy" in resp.abstain_reason
    assert "claim_1" in resp.abstain_reason
    assert len(resp.evidence_links) == 0
    assert len(resp.comparisons) == 0
    assert resp.citation_checks[0].status == ClaimVerificationStatus.UNSUPPORTED
    assert "KFAC-SVD" in resp.citation_checks[0].notes


def test_pass6_valid_appendix_attribution_supported_when_explicitly_connected():
    """
    Pass 6 Requirement C: Valid appendix attribution.
    Appendix evidence remains usable when the source explicitly connects the described method
    to the paper's own method. (Ensures appendix evidence is NOT globally banned).
    """
    doc_lems = "doc_c67573ef50cc"

    # Appendix chunk where the text explicitly connects the method to LEMS
    chunk_appendix = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p21_c2",
        doc_id=doc_lems,
        page=21,
        text="In our proposed LEMS framework, we run forward and backward passes on calibration data to estimate Kronecker factors.",
    )
    retriever = MockRetriever([chunk_appendix])

    gen_json = json.dumps({
        "answer": "In LEMS, forward and backward passes on calibration data are run to estimate Kronecker factors.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "LEMS runs forward and backward passes on calibration data to estimate Kronecker factors.",
                "chunk_ids": ["doc_c67573ef50cc_p21_c2"],
                "quote": "In our proposed LEMS framework, we run forward and backward passes on calibration data to estimate Kronecker factors.",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({
            "supported": True,
            "rationale": "The evidence explicitly connects this factor estimation procedure to the proposed LEMS framework.",
        }),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How does LEMS estimate Kronecker factors in the appendix?",
        doc_ids=[doc_lems],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 1
    assert resp.citation_checks[0].status == ClaimVerificationStatus.VERIFIED


def test_pass6_two_paper_independent_grounding_verified():
    """
    Pass 6 Requirement D: Two-paper grounding.
    Both requested papers must have independently verified primary-source claims.
    Ensures that when both are cited directly, no secondary-source qualification is appended.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p5_c5",
        doc_id=doc_gfw,
        page=5,
        text="Given a calibration dataset D, the per-sample gradients are computed.",
    )
    chunk_lems = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="we estimate Kronecker factors from token-wise second moments.",
    )
    retriever = MockRetriever([chunk_gfw, chunk_lems])

    gen_json = json.dumps({
        "answer": "GFWSVD uses per-sample gradients, whereas LEMS uses token-wise moments.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "GFWSVD computes per-sample gradients.",
                "chunk_ids": ["doc_7fdb1200e823_p5_c5"],
                "quote": "per-sample gradients are computed.",
            },
            {
                "claim_id": "claim_2",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_gfw,
                "doc_id_b": doc_lems,
                "comparison_type": "unresolved",
                "summary": "GFWSVD uses per-sample gradients whereas LEMS uses token-wise moments.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed."}),
        json.dumps({"supported": True, "rationale": "Entailed."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Compare factor estimation between LEMS and GFWSVD.",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    cited_docs = {l.doc_id for l in resp.evidence_links}
    assert cited_docs == {doc_lems, doc_gfw}
    assert "rather than independent primary-source" not in resp.answer


def test_pass6_comparison_integrity_distinct_documents_only():
    """
    Pass 6 Requirement E: Comparison integrity.
    Only verified claims from distinct documents can form a cross-paper comparison.
    A comparison linking claims from the same document in a multi-doc request is suppressed.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_lems_1 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="we estimate Kronecker factors from token-wise second moments.",
    )
    chunk_lems_2 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p4_c2",
        doc_id=doc_lems,
        page=4,
        text="LEMS avoids the rank bottleneck of prior methods.",
    )
    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p5_c5",
        doc_id=doc_gfw,
        page=5,
        text="per-sample gradients are computed.",
    )
    retriever = MockRetriever([chunk_lems_1, chunk_lems_2, chunk_gfw])

    gen_json = json.dumps({
        "answer": "LEMS estimates factors from token-wise moments and avoids bottlenecks.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments.",
            },
            {
                "claim_id": "claim_2",
                "text": "LEMS avoids the rank bottleneck of prior methods.",
                "chunk_ids": ["doc_c67573ef50cc_p4_c2"],
                "quote": "LEMS avoids the rank bottleneck of prior methods.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_lems,
                "doc_id_b": doc_lems,
                "comparison_type": "unresolved",
                "summary": "Comparing LEMS moments and bottleneck properties.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed."}),
        json.dumps({"supported": True, "rationale": "Entailed."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Describe LEMS estimation.",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    # In a multi-doc request, the same-doc comparison presented as cross-paper is suppressed
    assert len(resp.comparisons) == 0


def test_pass6_zero_tolerance_unnecessary_unsupported_claim_suppresses_answer():
    """
    Pass 6 Requirement F: Zero-tolerance regression.
    Even if two claims are valid, an unnecessary third claim that fails verification
    must trigger strict fail-closed abstention, suppressing answer, comparisons, and links.
    """
    doc_lems = "doc_c67573ef50cc"
    doc_gfw = "doc_7fdb1200e823"

    chunk_gfw = make_scored_chunk(
        chunk_id="doc_7fdb1200e823_p5_c5",
        doc_id=doc_gfw,
        page=5,
        text="Given a calibration dataset D, the per-sample gradients are computed.",
    )
    chunk_lems_1 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p3_c5",
        doc_id=doc_lems,
        page=3,
        text="we estimate Kronecker factors from token-wise second moments.",
    )
    chunk_lems_2 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p21_c2",
        doc_id=doc_lems,
        page=21,
        text="For KFAC-SVD we collect token-wise activations.",
    )
    retriever = MockRetriever([chunk_gfw, chunk_lems_1, chunk_lems_2])

    gen_json = json.dumps({
        "answer": "GFWSVD uses per-sample gradients, LEMS uses token-wise moments.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "GFWSVD computes per-sample gradients.",
                "chunk_ids": ["doc_7fdb1200e823_p5_c5"],
                "quote": "per-sample gradients are computed.",
            },
            {
                "claim_id": "claim_2",
                "text": "LEMS estimates Kronecker factors from token-wise second moments.",
                "chunk_ids": ["doc_c67573ef50cc_p3_c5"],
                "quote": "we estimate Kronecker factors from token-wise second moments.",
            },
            {
                "claim_id": "claim_3",
                "text": "LEMS collects activations for KFAC-SVD.",
                "chunk_ids": ["doc_c67573ef50cc_p21_c2"],
                "quote": "For KFAC-SVD we collect token-wise activations.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_gfw,
                "doc_id_b": doc_lems,
                "comparison_type": "unresolved",
                "summary": "GFWSVD uses per-sample gradients whereas LEMS uses token-wise moments.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed."}),
        json.dumps({"supported": True, "rationale": "Entailed."}),
        json.dumps({"supported": False, "rationale": "Chunk does not attribute KFAC-SVD to LEMS."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="Compare factor estimation between LEMS and GFWSVD.",
        doc_ids=[doc_lems, doc_gfw],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert "Strict zero-tolerance policy" in resp.abstain_reason
    assert "claim_3" in resp.abstain_reason
    assert len(resp.evidence_links) == 0
    assert len(resp.comparisons) == 0
    assert "I must abstain from answering" in resp.answer


def test_pass6_single_paper_claim_generation_and_within_paper_comparison_preserved():
    """
    Pass 6 Requirement G: Single-paper regression.
    Existing single-paper claim generation and legitimate within-paper comparisons
    continue working without regression.
    """
    doc_a = "doc_aaaaaaaaaaaa"

    chunk_1 = make_scored_chunk(
        chunk_id="doc_aaaaaaaaaaaa_p1_c1",
        doc_id=doc_a,
        page=1,
        text="The baseline method achieves 75% accuracy.",
    )
    chunk_2 = make_scored_chunk(
        chunk_id="doc_aaaaaaaaaaaa_p1_c2",
        doc_id=doc_a,
        page=2,
        text="Our proposed method achieves 88% accuracy.",
    )
    retriever = MockRetriever([chunk_1, chunk_2])

    gen_json = json.dumps({
        "answer": "The baseline achieves 75% accuracy, whereas our proposed method achieves 88% accuracy.",
        "claims": [
            {
                "claim_id": "claim_1",
                "text": "The baseline method achieves 75% accuracy.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c1"],
                "quote": "baseline method achieves 75% accuracy.",
            },
            {
                "claim_id": "claim_2",
                "text": "The proposed method achieves 88% accuracy.",
                "chunk_ids": ["doc_aaaaaaaaaaaa_p1_c2"],
                "quote": "Our proposed method achieves 88% accuracy.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": "claim_1",
                "claim_id_b": "claim_2",
                "doc_id_a": doc_a,
                "doc_id_b": doc_a,
                "comparison_type": "unresolved",
                "summary": "Baseline achieves 75% whereas proposed method achieves 88%.",
                "conflict_reason": None,
            }
        ],
    })

    llm = MockLLMProvider([
        gen_json,
        json.dumps({"supported": True, "rationale": "Entailed."}),
        json.dumps({"supported": True, "rationale": "Entailed."}),
    ])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(
        question="How does the proposed method compare to baseline?",
        doc_ids=[doc_a],
    )
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.evidence_links) == 2
    assert len(resp.comparisons) == 1
    assert resp.comparisons[0].doc_id_a == doc_a
    assert resp.comparisons[0].doc_id_b == doc_a
