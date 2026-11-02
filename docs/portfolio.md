# Portfolio material — PaperTrail AI

Written 2026-11-02. Every number here comes from a run artifact in `eval/runs/`, a command output, or a test
run in this repository; nothing is rounded up, and the limits are stated in the same breath as the results.

## Project description (short form)

PaperTrail AI is a research evidence workspace for reading several papers at once. Ask a question across
indexed PDFs and get an answer in which every claim is tied to a verbatim passage on a specific page of a
specific paper — or an explicit refusal when the evidence does not support one. Retrieval is hybrid (BM25 +
FAISS dense, fused with Reciprocal Rank Fusion at k = 60, reranked with a cross-encoder); generation is
separated from verification by three independent citation checks, and disagreements between papers are
classified as `supports`, `apparent_conflict`, `contradicts`, or `unresolved` rather than flattened into "the
papers disagree". It is a working full-stack project — FastAPI backend, Next.js frontend, 401 backend tests and
155 frontend tests that run offline, benchmarks with committed run artifacts — built with the honesty constraint
front and centre: the system is designed to fail closed.

## Resume bullets (pick one or two)

- Built **PaperTrail AI**, a multi-paper literature-grounding tool (FastAPI + Next.js) where hybrid retrieval
  (BM25, FAISS, RRF fusion, cross-encoder rerank) feeds a claim-level evidence ledger with three-layer citation
  verification; Recall@5 0.88 / MRR 0.69 over 25 questions on a 5-paper, 409-chunk corpus, and the model
  abstains rather than emitting an unverified claim.
- Designed and measured a **verify-or-abstain** pipeline: a verbatim-quote precheck plus existence, substring
  and temperature-0 entailment gates, with cross-paper comparisons suppressed unless both sides back a verified
  claim — 401 backend tests and 155 frontend tests, none of which makes a live provider call.
- Shipped an **offline, reproducible evaluation harness** (4 retrieval configurations, gold-span integrity
  checker, conflict-grounding runner, human-review worksheet generator) with run artifacts and confidence
  intervals committed to the repo, and documented that its 34 gold labels remain draft because no annotation has
  yet been human-verified.

One line you should *not* use: "achieved 88% accuracy". Recall@5 of 0.88 on 25 questions, measured against
gold labels that are themselves drafts, is not an accuracy claim about the product.

## Demo walkthrough (3–5 minutes)

The full script is the "Demo walkthrough" section of the README; this is the short version, and the order
matters because it starts where a reviewer is most suspicious.

1. Dashboard: five papers, 117 pages, 409 chunks, backend health chip — the index is local and inspectable.
2. Research Library → LEMS → Inspect Chunks: show a stored table passage and a reference-list passage, so it is
   clear what was actually indexed.
3. Research Assistant, single paper, factual question: answer assembles with per-claim verification status and
   the page-26 quote behind it. (Needs a working provider key; keep a saved run open as a fallback.)
4. Same panel, a question the paper cannot answer: refusal with retrieval diagnostics, not a guess.
5. Evidence Explorer: each claim, its span, page, chunk id, and the three layer results.
6. Cross-Paper Analysis: LEMS vs GFWSVD, and the rule that a method or setting difference is
   `apparent_conflict`, not a contradiction.
7. Close on provenance: open `eval/review_worksheet.md` and say plainly — 34 gold spans, 0 human-verified, and
   here is the checklist and guide that would change that.

Step 7 is the differentiator. Most demos of this kind end at step 3.

## Five questions an interviewer will ask, and the honest answers

**"Is your retrieval benchmark statistically solid?"** No, and the README says so before the table does. n = 25
answerable questions; the reranker's Recall@5 is 0.88 with a 95% bootstrap interval of [0.74, 0.98] and MRR
[0.538, 0.827]. I keep the warning from the runner in the report rather than editing it out. What the number
supports is a directional claim — reranking beats BM25, dense and RRF-only on this corpus — and the one question
still missing Recall@5 (Q-021) is listed with its failure lists.

**"How do you know the answers are grounded?"** Three independent checks: the cited chunk must exist in the
retrieved set, the quote must be a verbatim substring of it, and a temperature-0 entailment call must confirm
the passage supports the claim. A claim that fails any layer is shown as unverified or the answer is withheld;
there is a test that a retrieved-but-uncited chunk cannot create a comparison. What I cannot claim is measured
answer correctness against human-graded ground truth, because the gold labels are still drafts.

**"You said it detects contradictions. Show me one."** I can't, and that is the honest answer. I mined all 409
chunks: 302 opposition-bearing sentences, 30 cross-document candidate pairs, none surviving the four criteria
for a genuine contradiction. The five papers largely agree. So the classification logic is proven against
explicitly labelled synthetic fixtures and the system is proven to *decline* to invent a contradiction, but
contradiction detection on real cross-paper evidence is unproven. The corpus has to grow before that dimension
can be scored, and there is a shortlisted set of candidate papers for it.

**"Why not use a vector database / a queue / Postgres?"** Because at this size the bottleneck was nowhere near
those components, and each would have added a moving part I would have to defend. Storage is versioned JSON on
disk, single-user and inspectable; retrieval runs in-process. The tradeoff is real and stated: no concurrency,
no auth, indexing happens inside the upload request. If the corpus grew another two orders of magnitude, the
chunk store and FAISS index are the first things to move.

**"What would you fix first if this were a product?"** Three things, in order. Get the 34 gold spans
human-verified, because until then no accuracy number means what people assume it means. Widen benchmark
coverage past the two papers it actually tests. And make history server-side — right now saved runs live in
`localStorage` on one browser profile, which is fine for a personal tool and not fine for a product.

## Implemented and measured versus experimental

| Capability | Status |
| --- | --- |
| PDF ingestion with section-aware chunking, extraction warnings, SHA-256 dedup | Implemented, tested offline |
| Hybrid retrieval: BM25 + FAISS dense, RRF (k=60), cross-encoder rerank | Implemented, benchmarked on 5-paper corpus |
| Claim decomposition with verbatim-quote precheck | Implemented; the precheck is a tested fail-closed path |
| Three-layer citation verification (existence, verbatim substring, temp-0 entailment) | Implemented, unit-tested |
| Zero-tolerance abstention with retrieval diagnostics | Implemented, verified live for single- and cross-paper cases |
| Cross-paper comparison ledger with four relations | Implemented; both-sides-grounding rule enforced by test |
| Apparent conflict versus contradiction separation | Implemented and unit-tested **on synthetic fixtures**, not validated on a real verified contradiction |
| Offline retrieval + conflict benchmarks with committed artifacts | Implemented and reproducible |
| Human-review worksheet and reviewer guide for gold labels | Implemented; **0 of 34 labels human-verified** |
| Contradiction detection proven on real cross-paper evidence | Not established — no verified case exists in this corpus |
| Sustained live generation independent of provider quota | Not available on the free tier; provider failures surface as explicit abstentions |
| Server-side history, multi-user storage, auth, async ingestion | Not built — single-user by design, documented as a limitation |
| Figure and plot interpretation | Not built — captions and surrounding text are indexed, graphics are not read |
| Re-ingestion of the five stored records | Deliberately not done; requires approval and judged unnecessary (see docs/limitations.md) |

Related: [../README.md](../README.md), [limitations.md](limitations.md),
[benchmark-retrieval.md](benchmark-retrieval.md), [benchmark-contradictions.md](benchmark-contradictions.md),
[supplementary-corpus-shortlist.md](supplementary-corpus-shortlist.md),
[../eval/HUMAN-REVIEW-GUIDE.md](../eval/HUMAN-REVIEW-GUIDE.md),
[../eval/LABELLING.md](../eval/LABELLING.md).
