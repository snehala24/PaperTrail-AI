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

- Backend suite: **359 passed**, 1 pre-existing `DeprecationWarning` (`backend/tests/test_exact_quote_validation.py:87`), 44.7 s.
- Frontend suite: **155 passed** across 9 files; `tsc --noEmit` clean; `next build` succeeds.
- Browser walkthrough: Dashboard, Research Library, Research Assistant, Evidence Ledger and
  Cross-Paper Comparisons all rendered against the live backend, including document
  selection, loading, empty and no-match states, abstention, backend-error handling,
  upload validation (a non-PDF file was rejected client-side without any write request),
  and research history surviving a page reload. No console errors.
- Cross-Paper Comparisons was also inspected with a stored real backend response (Test 3)
  to confirm both sides render their quoted evidence.

## Not established by this round

- Sustained live throughput on a paid provider tier.
- Human-verified gold labels: 0 of 34, so the retrieval figures in `README.md` remain
  pipeline-validation numbers. See `eval/LABELLING.md`.
- Retrieval metrics at the current 409-chunk corpus scale; the recorded run measured 209
  chunks across two papers.
- Reading the content of figures: figure graphics remain uninterpreted.
