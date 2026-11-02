# Contradiction & Apparent-Conflict Benchmark

This report answers one question honestly: **can PaperTrail AI tell a genuine contradiction apart
from a methodological difference, on this corpus?** It separates what was verified from what was
scripted, what was executed from what was blocked, and it does not present a baited or scripted case
as evidence of detection.

## Classification scheme

| Category | Meaning | What a correct system response looks like |
| --- | --- | --- |
| **A — genuine contradiction** | Two papers assert opposite values for the *same* property, under comparable conditions, at the same granularity. | `comparison_type = contradicts`, both sides cited, both quotes verbatim, reason stated. |
| **B — apparent conflict** | Statements look opposed but differ in metric, model, rank, allocation regime or evaluation conditions. | `apparent_conflict` with a `conflict_reason` naming the differing condition — never `contradicts`. |
| **C — supporting** | Two papers say compatible or mutually reinforcing things. | `supports` or no relationship asserted. |
| **D — unresolved** | Evidence is retrieved but the relationship cannot be established, or one side is missing. | `unresolved`, or an honest abstention. |

## Headline result

**No verified genuine contradiction (category A) exists in the currently indexed five-paper corpus.**
Therefore no positive A-case could be run, and none was manufactured. The live execution of the
registered B/C/D cases was **blocked today by the Gemini free-tier daily quota** (the provider limit
resets on its own day boundary), with one of four cases executing before the block and abstaining
correctly.

## Evidence 1 — corpus-wide search for a genuine contradiction

Reproducible with `scratch/contradiction_mining.py` (offline; embeddings only, zero API calls); the
candidate list is stored at `scratch/contradiction_mining/candidates.json`.

Method: split every non-references chunk into sentences, keep sentences of 60–400 characters that
carry an opposition cue (negation, comparative reversal, adversative), embed with
`all-MiniLM-L6-v2`, and take cross-document pairs at cosine ≥ 0.62. A pair is only a *candidate*
genuine contradiction if it also matches on subject and measurable property (criterion 2) and can be
read under comparable conditions (criterion 4); criteria 3–4 are decided by reading both full
sentences, which the script prints with page and chunk id.

| Measurement (2026-11-02) | Value |
| --- | --- |
| Chunks scanned | 409 across 5 papers |
| Opposition-bearing sentences | 302 |
| Cross-document pairs at cosine ≥ 0.62 | 30 |
| Highest cross-document similarity | 0.7345 |
| Pairs surviving criteria 2 + 4 (same property, comparable conditions) | **0** |

The two families of near-misses, with the exact stored text that rules them out:

**LoRA vs DoRA (the bulk of the 30).** LoRA `doc_e9a0d3128767_p1_c2` (p.1): *"LoRA performs on-par or
better than ﬁne-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer
trainable parameters, a higher training throughput, and, unlike adapters, no additional inference
latency."* DoRA `doc_f048986e4d0b_p10_c2` (p.10): *"DoRA consistently outperforms LoRA across various
fine-tuning tasks and model architectures."* These are not opposites: LoRA's claim is LoRA-vs-full
fine-tuning on encoder/early-GPT models, DoRA's is DoRA-vs-LoRA on LLaMA tasks. Both can hold
simultaneously, and DoRA's own framing (`doc_f048986e4d0b_p3_c5`, p.3 — *"This distinction between FT
and LoRA likely mirrors their respective learning capability"*) treats LoRA's numbers as correct for
LoRA. Fails criterion 4 (different model families, tasks and comparison target).

**GFWSVD vs LEMS.** GFWSVD `doc_7fdb1200e823_p1_c4` (p.1): *"GFWSVD consistently outperforms both
diagonal Fisher approximations (FWSVD …) and activation-based baselines (ASVD …, SVD-LLM …)."* LEMS
`doc_c67573ef50cc_p21_c3` (p.21): *"At rank 128, KFAC consistently achieves lower held-out Fisher
degradation than GFWSVD (e.g., ≈1.4× to 3× lower ∆F … in Table 6)."* Different metrics (perplexity /
zero-shot accuracy vs held-out ∆F) and different comparison targets (GFWSVD-vs-baselines vs
KFAC-vs-GFWSVD); the statements are compatible. Fails criterion 2 and 4.

One pair is worth flagging as an example of the opposite outcome — cross-paper **agreement**: LoRA
`doc_e9a0d3128767_p2_c1` (p.2) *"these method often fail to match the ﬁne-tuning baselines, posing a
trade-off"* and DoRA `doc_f048986e4d0b_p1_c2` (p.1) *"However, there still often exists an accuracy
gap between these methods and full fine-tuning"* (cosine 0.668). Shared opposition cues, same
proposition — category C, not A.

## Evidence 2 — registered live cases (pre-registered quotes, then executed)

Each case below was written down *before* the request, with both quotes checked character-by-character
against the stored chunk JSON (the runner refuses to ask a question whose quotes do not both resolve).
Raw responses: `scratch/contradiction_benchmark/B1.json` … `E1.json`, `summary.json`.

| Case | Category expected | Quote A (paper, page, chunk) | Quote B (paper, page, chunk) | Quotes verified | Actual result | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **B1** | B apparent conflict (LEMS's measurement of GFWSVD vs GFWSVD's self-report) | "GFWSVD degrades significantly under identical conditions" — LEMS p.6 `doc_c67573ef50cc_p6_c5` | "Our results demonstrate that GFWSVD consistently outperforms diagonal and activation-based" — GFWSVD p.9 `doc_7fdb1200e823_p9_c4` | yes | `status = error`: "Gemini daily quota exhausted (free-tier daily limit reached). Verification was not started." | **BLOCKED — provider quota** |
| **C1** | C supporting (both papers approximate the Fisher factors with a low-rank Lanczos/SVD step) | "By fitting a rank-1 approximation to" — LEMS p.3 `doc_c67573ef50cc_p3_c5` | "so we employ truncated SVD using the Lanczos method" — GFWSVD p.4 `doc_7fdb1200e823_p4_c3` | yes | `status = error`, same quota message | **BLOCKED — provider quota** |
| **D1** | D no relationship should be invented (unrelated propositions: Transformer recurrence vs LoRA inference latency) | "eschewing recurrence" — Attention p.2 `doc_bdfaa68d8984_p2_c2` | "no additional inference latency" — LoRA p.1 `doc_e9a0d3128767_p1_c2` | yes | `status = abstained`, 5.3 s: *"Cross-paper evidence incomplete: requested comparison across documents ['doc_bdfaa68d8984','doc_e9a0d3128767'], but no relevant chunks were retrieved for ['doc_bdfaa68d8984']."* | **PASSED (as an abstention check)** — the request never reaches generation, so no contradiction could be fabricated |
| **E1** | B apparent conflict (LoRA's self-report vs DoRA's framing of the FT gap) | "performs on-par or better than" — LoRA p.1 `doc_e9a0d3128767_p1_c2` | "This distinction between FT and LoRA likely mirrors their respective learning capability" — DoRA p.3 `doc_f048986e4d0b_p3_c5` | yes | `status = error`, same quota message | **BLOCKED — provider quota** |

The block is the provider's own message, surfaced by the application unchanged; the response carried
no evidence links and no comparisons, i.e. the failure degraded to an honest error rather than an
invented answer. That behaviour is itself the check the brief asked for, but it does **not** count as
evidence of contradiction detection.

### A-probe (does the pipeline escalate an apparent conflict to `contradicts`?)

Not executed today — it requires a completed generation. Registered for the next quota window: ask B1
and assert `comparison_type != "contradicts"` while both `doc_c67573ef50cc_p6_c5` and
`doc_7fdb1200e823_p9_c4` appear as verified `evidence_links`.

## Evidence 3 — what the offline conflict benchmark actually proves

`backend/app/eval/run_conflicts.py` is, by its own docstring, *"ZERO LIVE API CALLS: Fully offline and
deterministic using MockLLMProvider"*. Its generator responses are **hand-written per question id** in
`build_deterministic_mock_for_question`, including the pair it labels
`"conflict_category": "real_contradiction"` (Q-005, `doc_c67573ef50cc_p23_c3` vs
`doc_c67573ef50cc_p24_c1`).

Consequences for how its scores may be described:

- It verifies the ledger machinery end to end — retrieval, three-layer citation verification,
  conflict surfacing, artifact hygiene — against a scripted model that always says what the fixture
  says it will say. It cannot show that a real model detects a conflict it was not told about.
- Every pair it uses lives inside **one** paper (LEMS). It contains no cross-paper contradiction test
  at all.
- The `"real_contradiction"` label is an input to the fixture, not a discovered property of the corpus,
  and the same-paper pair it names is a methodological incompatibility (BitStack sign masks vs
  quantization kernels). It must not be quoted as evidence that PaperTrail detects contradictions.
- Q-030 is reported as blocked. It previously reached that state through a hardcoded question-id check
  whose stated reason — the second paper not being ingested — had become false, since GFWSVD is indexed
  and both of Q-030's gold spans verify. The block is retained because it is still the correct
  *outcome*, but for the honest reason: this runner is offline and scripted, and no deterministic mock
  response is registered for a two-paper conflict. The block reason now says so instead of claiming a
  missing paper. The evaluated/blocked partition is unchanged (4 evaluated, 1 blocked), so no score
  moved.
- Current scripted run: `eval/runs/run_20261102_140403_conflicts.json` — retrieval 4/4, citation
  verification 4/4, conflict detection 4/4, `overall_grounded_success_rate = 1.0` on the four
  scripted single-paper cases, Q-030 blocked. Read all of that as pipeline behaviour under a fixture
  that is told the answer, not as detection accuracy.

Historical artifacts: `eval/runs/run_20261028_164333_conflicts.json`,
`eval/runs/run_20261029_072737_conflicts.json` (scripted, single-paper, unchanged).

## Evidence 4 — the earlier one-sided case, stated without over-claiming

In the previous round, a cross-paper apparent-conflict question abstained on the first attempt and, on
retry, returned `answered` with relation `supports`. Its gold expectation was
`apparent_conflict`, **assistant-drafted and never human-verified**. That gold label is not ground
truth, so this case is recorded as *unresolved evidence about the model's behaviour*, not as a pass
and not as a proven instability. It is the clearest available example of why the 34 gold spans need
human sign-off before any conflict-detection accuracy number can be published.

## What would make this benchmark pass

1. Human verification of the gold spans (34 items, 0 verified) via `eval/review_worksheet.md`.
2. A corpus that actually contains a genuine contradiction — either two papers reporting opposite
   values for one metric under comparable conditions, or a paper contradicting itself. The present
   five papers do not (302 opposition-bearing sentences, 30 cross-document candidates, 0 survivors).
3. A quota window in which B1/C1/E1 can complete, plus the A-probe assertion that the pipeline does
   not escalate to `contradicts`.

Re-run when 3 is available:

```bash
.venv/Scripts/python.exe scratch/contradiction_live_run.py     # writes scratch/contradiction_benchmark/
.venv/Scripts/python.exe -m backend.app.eval.run_conflicts     # scripted offline benchmark
```

## Verdict

| Claim | Status |
| --- | --- |
| Apparent conflicts are distinguished from contradictions in unit tests (both sides grounded, reason required) | **PASSED** — `backend/tests/test_day4_cross_paper.py`, 41 tests, 2026-11-02 |
| A retrieved-but-uncited chunk can no longer create a structured comparison | **PASSED** — new test `test_conflict_with_one_uncited_side_is_noted_but_not_promoted_to_a_comparison` |
| The system abstains rather than inventing a cross-paper relationship when one paper has no evidence | **PASSED** — live case D1, 2026-11-02 |
| The system detects a genuine contradiction in this corpus | **NOT ESTABLISHED** — no verified category-A case exists (mining above), and live B/C cases were BLOCKED by provider quota |
| The offline conflict benchmark measures real model behaviour | **NOT ESTABLISHED** — it is deterministic mock-provider pipeline validation |
| Contradiction-detection accuracy can be published | **BLOCKED** — requires human-verified gold labels |
