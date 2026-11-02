# Contradiction & Apparent-Conflict Benchmark

This report answers one question honestly: **can PaperTrail AI tell a genuine contradiction apart
from a methodological difference, on this corpus?** It separates what was verified from what was
scripted, what was executed from what was blocked, and it does not present a baited or scripted case
as evidence of detection.

## Classification scheme

Five categories, because "the relationship is not established" has two different causes with two
different correct answers — one side has no grounded evidence, versus neither side does.

| Category | Meaning | What a correct system response looks like | Enforced by |
| --- | --- | --- | --- |
| **A — genuine contradiction** | Two papers assert opposite values for the *same* property, under comparable conditions, at the same granularity. | `comparison_type = contradicts`, both sides cited, both quotes verbatim, reason stated. | `test_relation_taxonomy.py::test_genuine_contradiction_category_is_reachable_when_both_sides_are_grounded` (synthetic) |
| **B — apparent conflict** | Statements look opposed but differ in metric, model, rank, allocation regime or evaluation conditions. | `apparent_conflict` with a `conflict_reason` naming the differing condition — never `contradicts`. | `test_contextual_disagreement_is_never_escalated_to_contradicts`, `test_cross_paper_genuine_apparent_conflict_retains_apparent_conflict` |
| **C — compatible findings** | Two papers say compatible or mutually reinforcing things. | `supports`, or no relationship asserted; never a disagreement note out of thin air. | `test_compatible_findings_category_is_reported_as_supports_not_as_conflict` |
| **D — one-sided evidence** | One side of the pair backs a verified claim; the other is retrieved but uncited, or absent altogether. | The reader is told the sources disagree. **No ledger edge** — a comparison naming a claim that does not exist corrupts the ledger. | `test_one_sided_evidence_is_noted_but_never_becomes_a_relationship`, `test_conflict_with_one_uncited_side_is_noted_but_not_promoted_to_a_comparison` |
| **E — insufficient evidence** | Neither side is adequately grounded, or a requested paper returned nothing. | Honest abstention: zero claims, zero evidence links, zero comparisons, and the abstention names the missing paper. | `test_insufficient_evidence_abstains_and_claims_no_relationship_at_all`, `test_unsupported_comparison_incomplete_retrieval_abstains` |

Two cross-cutting rules, tested separately: a relationship asserted **without a documented reason**
is demoted to `unresolved`, and a conflict naming a chunk that was never retrieved for this question
is dropped rather than surfaced.

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

Not executed live today — it requires a completed generation. Registered for the next quota window: ask B1
and assert `comparison_type != "contradicts"` while both `doc_c67573ef50cc_p6_c5` and
`doc_7fdb1200e823_p9_c4` appear as verified `evidence_links`.

The **offline** half of this assertion is now automated, so the escalation risk is not resting only on a
future live run: `test_contextual_disagreement_is_never_escalated_to_contradicts` drives a fixture that
names an apparent conflict and checks the stored relation set is exactly `{apparent_conflict}`, and
`test_relationship_asserted_without_a_reason_is_demoted_to_unresolved` checks that a bare `contradicts`
claim with no reason cannot reach the ledger. The scripted benchmark additionally reports
`contradicts_labels_emitted = []` on every evaluable question. What remains live-only is whether a real
model, unscripted, chooses the conservative label.

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
- The block decision is now structural rather than named: the loop consults
  `SCRIPTED_QUESTION_IDS`, the set of questions the fixture actually has responses for, and the artifact
  carries the blocked ids with their reasons in `summary.blocked_questions`. Two tests pin this in
  `backend/tests/test_retrieval.py`: one fails if either runner ever decides sample membership by a
  question id outside the scripted-fixture builder, and one fails if the registry and the builder's real
  branches ever disagree (which would either silently skip a labelled question or crash mid-run).
- Current scripted run: `eval/runs/run_20261102_150550_conflicts.json` — retrieval 4/4, citation
  verification 4/4, conflict surfacing 4/4, `overall_grounded_success_rate = 1.0` on the four
  scripted single-paper cases, Q-030 blocked. Read all of that as pipeline behaviour under a fixture
  that is told the answer, not as detection accuracy.
- The same run now also reports classification, separately: `relation_label_checked_count = 4`,
  `relation_label_match_count = 0`, `contradicts_labels_emitted = []`. Each scripted response names a
  conflicting chunk pair but supplies no `conflict_type` and no `conflict_reason`, so the pipeline's
  reason-required guard declines to assert a relation — the ledger ends up with one `unresolved`
  comparison (Q-015) and no comparison in Q-005, Q-006 and Q-017, where only one side of the pair backs
  a verified claim. A 0/4 here is the expected reading, not a regression: the dimension exists to catch
  an unearned `contradicts`, and it found none. Category A–E behaviour is asserted against explicitly
  synthetic fixtures in `backend/tests/test_relation_taxonomy.py`, whose document ids
  (`doc_00000000000a/b`) are reserved placeholders that cannot belong to an indexed paper.

Historical artifacts: `eval/runs/run_20261028_164333_conflicts.json`,
`eval/runs/run_20261029_072737_conflicts.json` (scripted, single-paper, unchanged), and
`eval/runs/run_20261102_140403_conflicts.json` (same four rates, recorded before the relation-label
dimension existed).

**Reproducibility check (2026-11-02).** Re-running the scripted conflict runner at commit `2f7cdad`
wrote [`../eval/runs/run_20261102_154550_conflicts.json`](../eval/runs/run_20261102_154550_conflicts.json):
summary and per-question records identical to `run_20261102_150550_conflicts.json`, including Q-015's
`unresolved` ledger relation and Q-030 still reported as blocked.

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
   Candidate papers to create that condition are shortlisted, with the opposing claims quoted verbatim
   from both sides and one candidate already demoted as compatible rather than conflicting, in
   [supplementary-corpus-shortlist.md](supplementary-corpus-shortlist.md) — `AWAITING USER APPROVAL`,
   nothing downloaded or ingested. That document also records a same-claim numeric discrepancy already
   visible in this corpus (the ingested PDF — file `2610.32878v1.pdf` — states "4.8 p.p." on p.1 where the current arXiv listing
   states "4.7 p.p."), which is a version discrepancy rather than a research contradiction and is not
   counted as a category-A case.
3. A quota window in which B1/C1/E1 can complete. The A-probe assertion that the pipeline does not
   escalate to `contradicts` is now automated offline (see the A-probe section above); what still needs
   the quota window is the same assertion against an unscripted model on a real cross-paper pair.

Re-run when 3 is available:

```bash
.venv/Scripts/python.exe scratch/contradiction_live_run.py     # writes scratch/contradiction_benchmark/
.venv/Scripts/python.exe -m backend.app.eval.run_conflicts     # scripted offline benchmark
```

## Verdict

| Claim | Status |
| --- | --- |
| The five evidence categories (contradiction / apparent conflict / compatible / one-sided / insufficient) produce different, correct system responses | **PASSED (offline, synthetic)** — `backend/tests/test_relation_taxonomy.py`, 8 tests, 2026-11-02. Synthetic fixtures only: this shows the decision logic separates the categories, not that real contradictions are found |
| The pipeline never escalates a contextual disagreement to `contradicts`, and never carries a reasonless relationship label | **PASSED (offline)** — `test_contextual_disagreement_is_never_escalated_to_contradicts`, `test_relationship_asserted_without_a_reason_is_demoted_to_unresolved`, and `contradicts_labels_emitted = []` in `run_20261102_150550_conflicts.json` |
| Apparent conflicts are distinguished from contradictions in unit tests (both sides grounded, reason required) | **PASSED** — `backend/tests/test_day4_cross_paper.py`, 41 tests, 2026-11-02 |
| A retrieved-but-uncited chunk can no longer create a structured comparison | **PASSED** — new test `test_conflict_with_one_uncited_side_is_noted_but_not_promoted_to_a_comparison` |
| The system abstains rather than inventing a cross-paper relationship when one paper has no evidence | **PASSED** — live case D1, 2026-11-02 |
| The system detects a genuine contradiction in this corpus | **NOT ESTABLISHED** — no verified category-A case exists (mining above), and live B/C cases were BLOCKED by provider quota. The scripted runner's relation-label match is 0/4 by design, so it contributes nothing here |
| The offline conflict benchmark measures real model behaviour | **NOT ESTABLISHED** — it is deterministic mock-provider pipeline validation |
| Contradiction-detection accuracy can be published | **BLOCKED** — requires human-verified gold labels |
