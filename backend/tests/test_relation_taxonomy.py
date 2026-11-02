"""
Relation-taxonomy conformance suite: does the pipeline actually keep the five evidence
categories apart, and can it never harden a contextual disagreement into a contradiction?

Scope and honesty boundary
--------------------------
Every fixture below is SYNTHETIC. The chunk texts are invented for the test, the document ids are
reserved placeholders that cannot belong to an indexed paper (`doc_00000000000a/b`), and no claim in
them is a finding of any real paper. They exist to exercise the decision logic in `AnswerService`
under a scripted provider. Passing these tests proves that the relation machinery distinguishes the
categories *when a model names them correctly*; it does not prove that PaperTrail detects
contradictions in real research, which remains unestablished and is gated on human-reviewed
cross-paper gold labels (see docs/benchmark-contradictions.md).

The five categories the product must separate, and how each is represented:

  genuine contradiction  -> ComparisonType.CONTRADICTS          (both sides grounded + reason)
  apparent conflict      -> ComparisonType.APPARENT_CONFLICT    (both sides grounded + reason)
  compatible findings    -> ComparisonType.SUPPORTS             (both sides grounded)
  one-sided evidence     -> answer-level "sources disagree" note, NO ledger edge
  insufficient evidence  -> abstention, zero claims of any relationship

Plus the escalation rule that matters most for a research tool: a contextual disagreement is never
promoted to `contradicts`, and a relationship asserted without a reason is demoted to `unresolved`.

ZERO LIVE GEMINI CALLS: fully offline, deterministic, via MockLLMProvider / MockRetriever.
"""

import json
from typing import Any, Dict, List, Optional

from backend.app.citations.verifier import CitationVerifier
from backend.app.claims.service import AnswerService
from backend.app.schemas import AnswerRequest, AnswerResponse, AnswerStatus, ComparisonType

from backend.tests.test_day4_cross_paper import (
    MockLLMProvider,
    MockRetriever,
    make_scored_chunk,
    run_sync,
)

# Reserved placeholder ids: twelve hex zeros with a trailing letter. No ingested document can have
# one, so a passing fixture can never be mistaken for evidence about a real paper.
DOC_A = "doc_00000000000a"
DOC_B = "doc_00000000000b"
CHUNK_A = "doc_00000000000a_p1_c1"
CHUNK_B = "doc_00000000000b_p2_c1"
PHANTOM_CHUNK = "doc_00000000000b_p9_c9"

# Invented numbers and wording: unmistakably not a finding of any indexed paper.
TEXT_A = "Method Alpha records a 12.0 percent error rate on the shared held-out split."
TEXT_B = "Method Beta records a 4.0 percent error rate on the shared held-out split."

SUPPORTED = json.dumps({"supported": True, "rationale": "Quote is verbatim and entailed by the chunk."})


def _chunk(cid: str, doc: str, page: int, text: str):
    return make_scored_chunk(chunk_id=cid, doc_id=doc, page=page, text=text)


def _claim(cid: str, text: str, quote: str) -> Dict[str, Any]:
    return {"claim_id": cid, "text": text, "chunk_ids": [cid], "quote": quote}


def _run(
    gen_payload: Dict[str, Any],
    chunks: List[Any],
    doc_ids: Optional[List[str]] = None,
    verifier_verdicts: Optional[List[str]] = None,
    question: str = "How do the two methods compare on the held-out split?",
) -> AnswerResponse:
    """Drive one scripted answer through the real service and verifier."""
    retriever = MockRetriever(chunks)
    llm = MockLLMProvider(
        [json.dumps(gen_payload)]
        + (verifier_verdicts if verifier_verdicts is not None else [SUPPORTED, SUPPORTED, SUPPORTED])
    )
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=CitationVerifier(llm_provider=llm))
    return run_sync(service.answer(AnswerRequest(question=question, doc_ids=doc_ids)))


def _conflict_payload(conflict_type: str, reason: Optional[str]) -> Dict[str, Any]:
    """A two-claim answer whose sources both back verified claims, plus one flagged conflict."""
    return {
        "answer": "The two synthetic papers report different error rates on the same split.",
        "claims": [
            _claim(CHUNK_A, "Method Alpha records a 12.0 percent error rate.", TEXT_A),
            _claim(CHUNK_B, "Method Beta records a 4.0 percent error rate.", TEXT_B),
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [
            {
                "chunk_id_a": CHUNK_A,
                "chunk_id_b": CHUNK_B,
                "conflict_type": conflict_type,
                "conflict_reason": reason,
            }
        ],
        "comparisons": [],
    }


# =========================================================================
# Categories the ledger can express
# =========================================================================

def test_genuine_contradiction_category_is_reachable_when_both_sides_are_grounded():
    """
    The `contradicts` label is not a dead enum value: under a synthetic fixture with both sides
    grounded in verified claims and an explicit reason, the ledger carries CONTRADICTS.
    """
    resp = _run(_conflict_payload("contradicts", "Mutually exclusive error rates reported for one split."), [_chunk(CHUNK_A, DOC_A, 1, TEXT_A), _chunk(CHUNK_B, DOC_B, 2, TEXT_B)])

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.comparisons) == 1
    assert resp.comparisons[0].comparison_type == ComparisonType.CONTRADICTS
    assert resp.comparisons[0].conflict_reason


def test_apparent_conflict_category_is_kept_distinct_from_contradiction():
    """The same machinery, a different named relation: the label is preserved, not collapsed."""
    resp = _run(_conflict_payload("apparent_conflict", "Different calibration sets under the same metric."), [_chunk(CHUNK_A, DOC_A, 1, TEXT_A), _chunk(CHUNK_B, DOC_B, 2, TEXT_B)])

    assert resp.status == AnswerStatus.ANSWERED
    assert len(resp.comparisons) == 1
    assert resp.comparisons[0].comparison_type == ComparisonType.APPARENT_CONFLICT
    assert resp.comparisons[0].comparison_type != ComparisonType.CONTRADICTS


def test_compatible_findings_category_is_reported_as_supports_not_as_conflict():
    """Two papers agreeing is a `supports` edge, and no disagreement note is fabricated."""
    payload = {
        "answer": "Both synthetic papers report that the factorisation is estimated from calibration gradients.",
        "claims": [
            _claim(CHUNK_A, "Alpha estimates the factor from calibration gradients.", "Method Alpha records a 12.0 percent error rate on the shared held-out split."),
            _claim(CHUNK_B, "Beta estimates the factor from calibration gradients.", "Method Beta records a 4.0 percent error rate on the shared held-out split."),
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
        "comparisons": [
            {
                "claim_id_a": CHUNK_A,
                "claim_id_b": CHUNK_B,
                "doc_id_a": DOC_A,
                "doc_id_b": DOC_B,
                "comparison_type": "supports",
                "summary": "Both papers ground the factor in calibration gradients.",
                "conflict_reason": None,
            }
        ],
    }
    resp = _run(payload, [_chunk(CHUNK_A, DOC_A, 1, TEXT_A), _chunk(CHUNK_B, DOC_B, 2, TEXT_B)], doc_ids=[DOC_A, DOC_B])

    assert resp.status == AnswerStatus.ANSWERED
    assert [c.comparison_type for c in resp.comparisons] == [ComparisonType.SUPPORTS]
    assert "sources disagree" not in resp.answer.lower()


# =========================================================================
# Categories that must NOT produce a ledger edge
# =========================================================================

def test_one_sided_evidence_is_noted_but_never_becomes_a_relationship():
    """
    Only one side backs a verified claim. The reader is told the sources disagree; the ledger gets no
    edge, because a comparison naming a claim that does not exist would corrupt it.
    """
    payload = {
        "answer": "Method Alpha records a 12.0 percent error rate on the shared held-out split.",
        "claims": [_claim(CHUNK_A, "Method Alpha records a 12.0 percent error rate.", TEXT_A)],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [
            {
                "chunk_id_a": CHUNK_A,
                "chunk_id_b": CHUNK_B,
                "conflict_type": "apparent_conflict",
                "conflict_reason": "Only one side is grounded here.",
            }
        ],
        "comparisons": [],
    }
    resp = _run(payload, [_chunk(CHUNK_A, DOC_A, 1, TEXT_A), _chunk(CHUNK_B, DOC_B, 2, TEXT_B)])

    assert resp.status == AnswerStatus.ANSWERED
    assert resp.comparisons == []
    assert "sources disagree" in resp.answer.lower()
    assert CHUNK_B in resp.answer


def test_insufficient_evidence_abstains_and_claims_no_relationship_at_all():
    """Two papers requested, one retrieved: abstain, zero evidence links, zero comparisons."""
    resp = _run(
        {"answer": "", "claims": [], "abstained": True, "abstain_reason": "Evidence from the second paper is missing.", "conflicts": [], "comparisons": []},
        [_chunk(CHUNK_A, DOC_A, 1, TEXT_A)],
        doc_ids=[DOC_A, DOC_B],
        question="Do the two papers disagree about the error rate?",
    )

    assert resp.status == AnswerStatus.ABSTAINED
    assert resp.comparisons == []
    assert resp.evidence_links == []
    assert resp.citation_checks == []
    assert DOC_B in (resp.abstain_reason or ""), "the abstention must name the paper that produced no evidence"


# =========================================================================
# Escalation and demotion rules
# =========================================================================

def test_contextual_disagreement_is_never_escalated_to_contradicts():
    """
    The property the release depends on: when the model names a contextual disagreement, the stored
    relation stays `apparent_conflict`. Nothing downstream hardens it into `contradicts`.
    """
    resp = _run(_conflict_payload("apparent_conflict", "Different evaluation windows for the same metric."), [_chunk(CHUNK_A, DOC_A, 1, TEXT_A), _chunk(CHUNK_B, DOC_B, 2, TEXT_B)])

    assert resp.comparisons, "fixture should produce a comparison"
    types = {c.comparison_type for c in resp.comparisons}
    assert types == {ComparisonType.APPARENT_CONFLICT}
    assert ComparisonType.CONTRADICTS not in types
    assert resp.comparisons[0].summary and "contradiction" not in resp.comparisons[0].summary.lower()


def test_relationship_asserted_without_a_reason_is_demoted_to_unresolved():
    """A bare assertion of contradiction, with no documented reason, is not good enough for the ledger."""
    resp = _run(_conflict_payload("contradicts", None), [_chunk(CHUNK_A, DOC_A, 1, TEXT_A), _chunk(CHUNK_B, DOC_B, 2, TEXT_B)])

    assert [c.comparison_type for c in resp.comparisons] == [ComparisonType.UNRESOLVED]


def test_conflict_naming_an_unretrieved_chunk_is_dropped_rather_than_invented_into_a_note():
    """A conflict that cites a chunk this question never retrieved cannot be surfaced at all."""
    payload = _conflict_payload("apparent_conflict", "Different calibration sets under the same metric.")
    payload["conflicts"][0]["chunk_id_b"] = PHANTOM_CHUNK
    resp = _run(payload, [_chunk(CHUNK_A, DOC_A, 1, TEXT_A), _chunk(CHUNK_B, DOC_B, 2, TEXT_B)])

    assert resp.comparisons == []
    assert PHANTOM_CHUNK not in resp.answer
