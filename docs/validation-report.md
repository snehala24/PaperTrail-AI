# Functional validation report — research workspace release

Validated at commit `da532d8` on branch `day5-research-workspace`, 2026-11-02.
Environment: local FastAPI backend on `:8000`, Next.js dev server on `:3000`, corpus of
5 papers / 117 pages / 409 chunks at chunker `0.2.3`, Gemini model from `GEMINI_MODEL`.
Live answers were produced by real `POST /answer` calls; raw response payloads were kept
locally under `scratch/final_acceptance/` (gitignored) and are summarised below.

| Test | Result | Evidence | Notes |
| --- | --- | --- | --- |
| 1 · Single-paper factual question | **Passed** | Q: wall-clock speedup at 80% compression. `status: answered` (16.5 s). One claim, one evidence link `doc_c67573ef50cc_p26_c7` page 26, citation check `verified`. Answer quotes the 2.01× A100 / Qwen3-8B figure. | Retrieval, quotation, page reference and verification all correct; matches the researcher-drafted gold span `GOLD-001`. |
| 2 · Cross-paper comparison (GFWSVD vs LEMS) | **Passed** | Q: do the two papers describe Kronecker-factor estimation the same way? `answered` (27.7 s). Evidence on both sides: `doc_7fdb1200e823_p5_c5` (p5) and `doc_c67573ef50cc_p3_c5` (p3), both `verified`. 1 comparison, relation `unresolved`. | Each side grounded in its own paper's primary chunk; the system declined to assert consensus or disagreement it could not settle. |
| 3 · Claimed contradiction bait | **Passed** | Q framed as "GFWSVD performs worse" per the LEMS paper. `answered` (26.6 s). Evidence `doc_7fdb1200e823_p1_c4` (p1) and `doc_c67573ef50cc_p6_c5` (p6), both `verified`. 1 comparison, relation `apparent_conflict` with the evaluation-setting reason attached. | The disagreement was attributed to different evaluation settings rather than reported as a factual contradiction — the required behaviour. |
| 4 · Apparent conflict (different methodologies) | **Passed with caveat** | Q: "Does KL divergence beat MSE as the calibration metric?" First attempt: `abstained` (16.2 s), zero evidence links, both citation checks `unverified`, abstain reason = strict zero-tolerance policy. Retry: `answered` (25.3 s), evidence `doc_c67573ef50cc_p8_c6` (p8) and `doc_c67573ef50cc_p18_c1` (p18), both `verified`, 1 comparison relation `supports`. | No methodological difference was ever labelled a contradiction, and the first attempt failed closed instead of answering on unverified quotations. Caveat: the answer's relation (`supports`) differs from the AI-drafted gold expectation (`apparent_conflict`); that gold label is itself unverified, so this is an open labelling question, not a confirmed defect. |
| 5 · Honest abstention (out-of-corpus) | **Passed** | Q: total dollar compute cost of the LEMS grid search. `abstained` (11.2 s), zero evidence links, zero citation checks, no claims, no comparisons. Abstain reason names the gap: runtime and speedup metrics exist, monetary cost does not. | No invented figure and no fabricated citation. |
| 6 · Citation integrity (independent re-check) | **Passed** | Every displayed quotation in the six payloads re-checked against the stored chunk files: 7 of 7 evidence links consistent on chunk existence, document membership, page number and verbatim (whitespace/Unicode-normalised) substring match. | Script run against `data/chunks/*.json`, not against the response payloads alone. |

## Regression and browser verification

- Backend suite: **366 passed**, 0 failed, 0 skipped, 1 pre-existing `DeprecationWarning` (`backend/tests/test_exact_quote_validation.py:87`), 78 s. The increase from 359 is seven tests added during the 2026-11-02 hardening pass: two cross-paper conflict-ledger cases, four parser cases (figure-only warnings, title extraction) and two storage-resilience cases.
  *Superseded later the same day:* the final regression run at the release-preparation commits reports **401 passed**, 0 failed, 0 skipped, **no warnings summary**, 73 s (`test_exact_quote_validation.py` alone still passes 6/6). The +35 are concentrated in the rewritten review-worksheet module (19 → 36 cases); see the README "Tests and verification status" table for the current numbers.
- Frontend suite: **155 passed** across 9 files; `tsc --noEmit` clean; `next build` succeeds (route `/` 32.5 kB, First Load JS 120 kB).
- Evaluation integrity: `python -m backend.app.eval.verify_gold` resolves 34/34 gold spans inside their cited page's stored chunks, matches 25 answerable questions, keeps 5 reserved unanswerable, 0 orphan questions, 0 unmatched gold. `python -m backend.app.eval.review_worksheet --check` reports 34 annotations, 0 human-verified, 0 integrity errors.
- Browser walkthrough: Dashboard, Research Library, Research Assistant, Evidence Explorer and Cross-Paper Analysis all rendered against the live backend, including chunk-explorer search (78 chunks → 2 for "intrinsic dimension" → 0 with a no-match message for a nonsense term → 78 when cleared), document selection, loading and backend-error states, abstention, upload validation (a non-PDF and a forged `%PDF-` file were both rejected with HTTP 400 and no write to `data/`), and research history surviving a page reload. No console errors.
- Citation display was re-checked in the browser on a stored real answer: the single claim renders its page-26 verbatim quotation, chunk id and `verified` check, and the quotation was independently re-found in `data/chunks/doc_c67573ef50cc_chunks.json`. An abstained run was reopened in the same view and states "No verified evidence to display" with the failing citation named — mixed `verified`/`unverified` and `unsupported`/`unverified` check statuses are displayed as they are, never collapsed into a pass.
- Provider-blocked live behaviour: a fresh question through the real UI returned an explicit `Backend Execution Error` naming the daily quota exhaustion, with zero evidence links and zero citation checks stored. The failed run appears in history as "Request failed" and does not clear or replace previously saved answers.
- Populated cross-paper comparison rendering is covered by component tests (`frontend/tests/comparisons.test.tsx`: all four relationship labels, and an explicit case that `apparent_conflict` is never mislabelled `contradicts`). In the browser during this pass only the empty state and the relationship taxonomy were re-verified, because this profile holds no saved answer containing comparisons and generating one was provider-blocked.

## Not established by this round

- Sustained live throughput on a paid provider tier.
- Human-verified gold labels: 0 of 34, so the retrieval figures in `README.md` remain
  pipeline-validation numbers. See `eval/LABELLING.md`.
- Genuine contradiction detection: no defensible cross-paper contradiction exists in the indexed
  five-paper corpus, so the positive case remains untested. See
  [benchmark-contradictions.md](benchmark-contradictions.md).
- Reading the content of figures: figure graphics remain uninterpreted.

Retrieval at the current corpus scale is now measured — 5 papers / 409 chunks, $n=25$, hybrid
rerank Recall@5 0.8800 and MRR 0.6927 (`eval/runs/run_20261102_132940.json`); see
[benchmark-retrieval.md](benchmark-retrieval.md) for the current-versus-historical separation.
