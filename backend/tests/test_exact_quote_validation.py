import json
import asyncio
from typing import List

import pytest
from unittest.mock import AsyncMock

from backend.app.claims.service import AnswerService, EMERGENCY_RERANK_SCORE_FLOOR
from backend.app.schemas import (
    AnswerRequest,
    AnswerStatus,
    ClaimVerificationStatus,
    ScoredChunk,
)
from backend.app.citations.verifier import CitationVerifier

# Reuse mock helpers from existing tests (they are defined in test_api_answer.py)
# We'll duplicate minimal versions here to keep this file self‑contained.

class MockLLMProvider:
    """Deterministic mock LLM provider for generator and verifier calls."""

    def __init__(self, responses: List[object] = None, default_response: str = "{}"):
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.call_history = []
        self.call_kwargs = []

    async def generate(self, messages, temperature=0.0, max_tokens=None, **kwargs):
        # Record the call for possible inspection.
        self.call_history.append(messages)
        self.call_kwargs.append(dict(temperature=temperature, max_tokens=max_tokens, **kwargs))
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, str):
                content = item
            else:
                # Assume a ready LLMResponse‑like object with .content attribute
                return item
        else:
            raise AssertionError("MockLLMProvider: no more prepared responses for generate()")
        # Simple response mimicking LLMResponse
        class SimpleResp:
            def __init__(self, c):
                self.content = c
                self.model = "mock"
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.total_tokens = 0
        return SimpleResp(content)

class MockRetriever:
    """Mock retriever returning predetermined ScoredChunk results."""

    def __init__(self, results: List[ScoredChunk] = None):
        self.results = results or []
        self.chunks_by_id = {c.chunk_id: c for c in self.results}
        self.last_query = None
        self.last_top_k = None

    def search(self, query: str, top_k: int = 10, doc_ids=None, include_references=False, use_reranker=True, candidate_pool_size=None):
        self.last_query = query
        self.last_top_k = top_k
        filtered = self.results
        if doc_ids is not None:
            allowed = set(doc_ids)
            filtered = [c for c in filtered if c.doc_id in allowed]
        return filtered[:top_k]

def make_chunk(chunk_id="doc_c67573ef50cc_p6_c2", doc_id="doc_c67573ef50cc", page=6,
               text="LEMS delegates solving the allocation problem to the off‑the‑shelf Gurobi optimizer.") -> ScoredChunk:
    from backend.app.schemas import ChunkKind
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=page,
        section_heading="Methodology",
        text=text,
        kind=ChunkKind.TEXT,
        rerank_score=3.5,
    )

def run_sync(coro):
    """
    Execute an async coroutine to completion.

    Both branches of the previous implementation called asyncio.run, while asyncio.get_event_loop()
    only emitted a DeprecationWarning under pytest-asyncio, so the call is gone. asyncio.run owns its
    own loop, which is what keeps this suite safe on Windows.
    """
    return asyncio.run(coro)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_exact_quote_accepted():
    """A claim with an exact verbatim quote should pass and return ANSWERED."""
    chunk = make_chunk()
    retriever = MockRetriever([chunk])

    gen_json = json.dumps({
        "answer": "LEMS uses the Gurobi optimizer.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses Gurobi optimizer.",
                "chunk_ids": [chunk.chunk_id],
                "quote": "Gurobi optimizer",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
    })
    # Verifier response – a simple supported true payload.
    verif_json = json.dumps({"supported": True, "rationale": "OK"})

    llm = MockLLMProvider([gen_json, verif_json])
    verifier = CitationVerifier(llm_provider=llm)  # uses the same mock for verification
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="What optimizer does LEMS use?", top_k=5)
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert resp.abstain_reason is None
    assert any("Gurobi optimizer" in link.quote for link in resp.evidence_links)
    # Verify the claim was verified.
    assert any(chk.status == ClaimVerificationStatus.VERIFIED for chk in resp.citation_checks)


def test_paraphrased_quote_rejected():
    """A claim whose quote is paraphrased (not a verbatim substring) must cause abstention."""
    chunk = make_chunk()
    retriever = MockRetriever([chunk])

    gen_json = json.dumps({
        "answer": "LEMS uses a powerful optimizer.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses a powerful optimizer.",
                "chunk_ids": [chunk.chunk_id],
                "quote": "powerful optimizer",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
    })
    # No verifier response needed because we expect early abort.
    llm = MockLLMProvider([gen_json])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="What optimizer does LEMS use?", top_k=5)
    resp = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert resp.abstain_reason


def test_noncontiguous_quote_rejected():
    """A quote that skips words (non‑contiguous) must be rejected."""
    chunk = make_chunk()
    retriever = MockRetriever([chunk])

    # The chunk contains "Gurobi optimizer"; we provide a quote with a line‑break hyphen scenario.
    gen_json = json.dumps({
        "answer": "LEMS uses the Gurobi optimizer.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses Gurobi optimizer.",
                "chunk_ids": [chunk.chunk_id],
                "quote": "LEMS Gurobi optimizer",  # skips words; not a contiguous source span
            }
        ],
        "abstained": False,
        "abstain_reason": None,
    })
    llm = MockLLMProvider([gen_json])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="What optimizer does LEMS use?", top_k=5)
    resp = run_sync(service.answer(req))
    assert resp.status == AnswerStatus.ABSTAINED
    assert resp.abstain_reason


def test_missing_cited_chunk_rejected():
    """If the claim references a chunk_id not present in retrieved results, abort."""
    # No chunks returned by retriever.
    retriever = MockRetriever([])
    gen_json = json.dumps({
        "answer": "LEMS uses the Gurobi optimizer.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses Gurobi optimizer.",
                "chunk_ids": ["nonexistent_chunk"],
                "quote": "Gurobi optimizer",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
    })
    llm = MockLLMProvider([gen_json])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)
    req = AnswerRequest(question="What optimizer does LEMS use?", top_k=5)
    resp = run_sync(service.answer(req))
    assert resp.status == AnswerStatus.ABSTAINED
    assert resp.abstain_reason


def test_missing_or_empty_quote_rejected():
    """Claims lacking a quote or with an empty string must be rejected."""
    chunk = make_chunk()
    retriever = MockRetriever([chunk])

    # Case 1: missing quote key
    gen_json_missing = json.dumps({
        "answer": "LEMS uses Gurobi.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses Gurobi.",
                "chunk_ids": [chunk.chunk_id]
                # no "quote"
            }
        ],
        "abstained": False,
        "abstain_reason": None,
    })
    llm1 = MockLLMProvider([gen_json_missing])
    verifier1 = CitationVerifier(llm_provider=llm1)
    service1 = AnswerService(retriever=retriever, llm_provider=llm1, verifier=verifier1)
    req = AnswerRequest(question="What optimizer?", top_k=5)
    resp1 = run_sync(service1.answer(req))
    assert resp1.status == AnswerStatus.ABSTAINED
    assert resp1.abstain_reason

    # Case 2: empty quote string
    gen_json_empty = json.dumps({
        "answer": "LEMS uses Gurobi.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses Gurobi.",
                "chunk_ids": [chunk.chunk_id],
                "quote": "",
            }
        ],
        "abstained": False,
        "abstain_reason": None,
    })
    llm2 = MockLLMProvider([gen_json_empty])
    verifier2 = CitationVerifier(llm_provider=llm2)
    service2 = AnswerService(retriever=retriever, llm_provider=llm2, verifier=verifier2)
    resp2 = run_sync(service2.answer(req))
    assert resp2.status == AnswerStatus.ABSTAINED
    assert resp2.abstain_reason


def test_whitespace_and_case_normalization():
    """Quotes that differ only by whitespace or case should still be accepted."""
    chunk = make_chunk(text="LEMS delegates solving the allocation problem to the off‑the‑shelf Gurobi optimizer.")
    retriever = MockRetriever([chunk])
    gen_json = json.dumps({
        "answer": "LEMS uses the Gurobi optimizer.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "LEMS uses Gurobi optimizer.",
                "chunk_ids": [chunk.chunk_id],
                "quote": "  gurobi   OPTIMIZER ",  # extra spaces & different case
            }
        ],
        "abstained": False,
        "abstain_reason": None,
    })
    verif_json = json.dumps({"supported": True, "rationale": "OK"})
    llm = MockLLMProvider([gen_json, verif_json])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)
    req = AnswerRequest(question="Which optimizer?", top_k=5)
    resp = run_sync(service.answer(req))
    assert resp.status == AnswerStatus.ANSWERED
    assert resp.abstain_reason is None
