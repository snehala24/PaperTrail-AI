# Benchmark human review worksheet

Protocol version: `1.0` · Generated: 2026-11-02

An annotation counts as ground truth only after a person has opened the PDF and signed off on it. Fill one block per gold item, then copy the verdicts back into `eval/gold_evidence.jsonl` (`status`, `labelled_by`, `annotator_notes`). This worksheet is generated output — do not treat it as the dataset.

## How to complete an item

1. Open the paper at the recorded page and find the quoted span.
2. Confirm the span is verbatim, short, and answers the question.
3. Confirm the relation: `supports` must entail, `apparent_conflict` must be explained by methods, data, metric or setting differences — not by a factual disagreement.
4. Record your handle as `human:<handle>`; otherwise leave the item's draft status untouched.

## Dataset status

| Metric | Count |
| --- | --- |
| Questions | 30 |
| Gold annotations | 34 |
| Human-verified gold annotations | 0 |
| Awaiting human review | 34 |
| — of which researcher-drafted (`human_draft`) | 24 |
| — of which assistant-drafted (`ai_draft`) | 10 |
| Documents available in `data/` | 5 |

> **No annotation has been human-verified yet.** Any accuracy number quoted from `eval/runs/` is a pipeline-validation measurement against draft gold labels (24 researcher-drafted, 10 assistant-drafted), not human-graded accuracy.

## Review items

### Q-001 — What wall-clock inference speedup does LEMS achieve at 80% compression on an NVIDIA A100 GPU for Qwen3-8B?

- Type: `factual`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-001 · page 26 · `supports`

- Recorded span: “2.01× speedup”
- Annotator note: Section B.7 reports 2.01x wall-clock inference speedup at 80% compression on single A100 GPU.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p26_c7

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-002 — What truncation rank k is used during the iterative SVD weight decomposition in BitStack?

- Type: `factual`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-002 · page 22 · `supports`

- Recorded span: “truncate the result to rank k=16”
- Annotator note: Section B.5.1 explains iterative rank-16 SVD truncation.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p22_c6

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-003 — How does KFAC-SVD compare to BitStack in memory footprint and zero-shot accuracy on Llama 3-8B in Table 9?

- Type: `comparative`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-003 · page 24 · `supports`

- Recorded span: “5.07 GB”
- Annotator note: Table 9 reports KFAC-SVD memory of 5.07 GB compared to BitStack 5.19 GB.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p24_c1

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-004 · page 24 · `supports`

- Recorded span: “60.83”
- Annotator note: Table 9 reports KFAC-SVD zero-shot accuracy of 60.83 compared to BitStack 59.93.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p24_c1

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-004 — How does LEMS rank allocation compare to the uniform compression baseline in WikiText2 perplexity on Llama 3.1-70B?

- Type: `comparative`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-005 · page 22 · `supports`

- Recorded span: “reducing WikiText2 perplexity by 43%”
- Annotator note: Section B.4 reports LEMS reduces WikiText2 perplexity from 7.94 to 4.53.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p22_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-005 — Why does KFAC-SVD achieve a better memory-accuracy trade-off than BitStack despite BitStack separating binary sign masks?

- Type: `apparent_conflict`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-006 · page 23 · `contradicts`

- Recorded span: “separating out the binary mask makes combining it with quantization ineffective”
- Annotator note: Section B.5.1 explains why BitStack loses advantage under quantization.
- Conflict reason: Apparent conflict: BitStack isolates binary sign masks that cannot be compressed with standard quantization kernels, while traditional SVD allows 4-bit quantization across all components.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p23_c3

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-006 — Does KFAC-SVD consistently achieve lower perplexity than SVD-LLM across all compression regimes?

- Type: `apparent_conflict`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-007 · page 7 · `supports`

- Recorded span: “KFAC-SVD achieves approximately 15% lower perplexity than SVD-LLM under uniform allocation”
- Annotator note: Section 5.2 reports uniform allocation comparison.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p7_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-008 · page 7 · `supports`

- Recorded span: “When combined with LEMS, perplexities become tightly matched”
- Annotator note: Section 5.2 explains convergence of perplexity under LEMS.
- Conflict reason: Apparent conflict: Under uniform allocation KFAC-SVD outperforms SVD-LLM due to Fisher weighting, but under global LEMS rank allocation dominates and performance converges.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p7_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-007 — Why does simple summation of layer-wise errors systematically underestimate global degradation in compressed deep networks?

- Type: `methodological`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-009 · page 4 · `supports`

- Recorded span: “Errors introduced in early layers propagate and amplify”
- Annotator note: Section 4.2 explains non-additive error propagation across depth.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p4_c3

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-008 — What mathematical optimization formulation does LEMS use to solve the global rank allocation problem under target parameter budgets?

- Type: `methodological`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

#### GOLD-010 · page 5 · `supports`

- Recorded span: “Integer Linear Program”
- Annotator note: Section 4.2 describes global ILP formulation.
- Current label: `ai_draft` (labelled_by: assistant)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p5_c2

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-009 — What was the total financial compute cost in US dollars for all LEMS grid searches and GPU experiments?

- Type: `unanswerable`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

No gold evidence is recorded for this question. For an unanswerable query, confirm the corpus genuinely lacks the information, then write that down.

### Q-010 — What is the zero-shot BLEU score of LEMS-compressed models on WMT 2014 English-to-German machine translation?

- Type: `unanswerable`
- Question status: `ai_draft` (labelled_by: assistant)
- Target documents: doc_c67573ef50cc

No gold evidence is recorded for this question. For an unanswerable query, confirm the corpus genuinely lacks the information, then write that down.

### Q-011 — How long does the first LEMS run take on the 8B models, and how long when cached?

- Type: `factual`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-011 · page 8 · `supports`

- Recorded span: “LEMS requires 20-25 minutes for the initial run and only 3 minutes for subsequent runs”
- Annotator note: Page 8: initial runtime vs cached runtime on 8B models
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p8_c3

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-012 — Which off-the-shelf solver does LEMS use for its allocation problem?

- Type: `factual`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-012 · page 6 · `supports`

- Recorded span: “we use the Gurobi solver”
- Annotator note: Page 6: off-the-shelf solver
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p6_c2

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-013 — What limitation do the authors admit about their Fisher-based decomposition?

- Type: `factual`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-013 · page 9 · `supports`

- Recorded span: “still relies on gradient computation, limiting computational efficiency”
- Annotator note: Page 9: gradient computation efficiency limitation
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p9_c1

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-014 — How much calibration data is used to fit the SVD decompositions?

- Type: `factual`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-014 · page 6 · `supports`

- Recorded span: “we use 256 random sequences of length 2048 from WikiText2”
- Annotator note: Page 6: calibration sequences and length
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p6_c2

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-015 — Which GPU were the experiments run on? (H200 for the main experiments, A100 for the speed measurements)

- Type: `apparent_conflict`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-015 · page 6 · `supports`

- Recorded span: “All experiments are run on a single NVIDIA H200 GPU”
- Annotator note: Section 5 specifies H200 for main experiments
- Conflict reason: Apparent conflict: H200 used for main experiments, A100 used for speed measurements.
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p6_c2

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-016 · page 26 · `supports`

- Recorded span: “All measurements were conducted on a single NVIDIA A100 GPU”
- Annotator note: Appendix B.7 specifies A100 for inference speed benchmarking
- Conflict reason: Apparent conflict: H200 used for main experiments, A100 used for speed measurements.
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p26_c7

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-016 — How are the depth-penalty parameters chosen, and with how many trials?

- Type: `methodological`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-017 · page 16 · `supports`

- Recorded span: “lightweight hyperparameter search using the Tree-structured parzen estimation (TPE) algorithm”
- Annotator note: Page 16: TPE algorithm for depth-penalty
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p16_c5

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-018 · page 14 · `supports`

- Recorded span: “we limit the optimization to 10 trials”
- Annotator note: Page 14: optimization budget limited to 10 trials
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p14_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-017 — Does KL divergence beat MSE as the calibration metric? (better on accuracy, slightly worse on perplexity)

- Type: `apparent_conflict`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-019 · page 8 · `supports`

- Recorded span: “KL divergence consistently outperforms MSE in zero shot accuracy”
- Annotator note: Page 8: zero-shot accuracy advantage of KL
- Conflict reason: Apparent conflict: KL divergence outperforms MSE on zero-shot accuracy, while MSE performs slightly better on perplexity.
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p8_c6

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-020 · page 8 · `supports`

- Recorded span: “For perplexity, MSE performs slightly better than KL”
- Annotator note: Page 8: perplexity advantage of MSE
- Conflict reason: Apparent conflict: KL divergence outperforms MSE on zero-shot accuracy, while MSE performs slightly better on perplexity.
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p8_c6

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-018 — Why was KL divergence chosen as the sensitivity signal?

- Type: `methodological`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-021 · page 18 · `supports`

- Recorded span: “We select KL-divergence as it directly measures the distributional shift at the output”
- Annotator note: Page 18: sensitivity metric selection rationale
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p18_c1

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-019 — What default reference compression is used for calibration, and what is the trade-off?

- Type: `methodological`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-022 · page 8 · `supports`

- Recorded span: “we prioritize its better generative fidelity (perplexity) and adopt it as our default configuration”
- Annotator note: Page 8: default reference compression trade-off
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p8_c6

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-020 — In the Llama 3.2-3B study, did ATP or ASVD reach the lower perplexity? (ASVD, 12.06 vs 13.48)

- Type: `comparative`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-023 · page 18 · `supports`

- Recorded span: “suboptimal performance (13.48 perplexity)”
- Annotator note: Page 18: ATP perplexity
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p18_c3

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-024 · page 18 · `supports`

- Recorded span: “achieving a better perplexity of 12.06”
- Annotator note: Page 18: ASVD perplexity
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p18_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-021 — How do the authors say GFWSVD behaves under their conditions?

- Type: `comparative`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-025 · page 6 · `supports`

- Recorded span: “GFWSVD degrades significantly under identical conditions”
- Annotator note: Page 6: GFWSVD degradation
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p6_c5

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-022 — Which matters more for results: the random seed or the calibration dataset?

- Type: `comparative`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-026 · page 25 · `supports`

- Recorded span: “the calibration set has a significant impact on the performance of all methods”
- Annotator note: Page 25: calibration set impact
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p25_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-027 · page 25 · `supports`

- Recorded span: “the seed impact is low”
- Annotator note: Page 25: seed impact
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p25_c3

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-023 — Why does the ATP baseline miss layer-specific importance?

- Type: `methodological`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-028 · page 6 · `supports`

- Recorded span: “because it assumes error accumulation as a linear function”
- Annotator note: Page 6: ATP linear accumulation assumption
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p6_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-024 — Which layer types are more critical to preserve: feed-forward or attention?

- Type: `methodological`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-029 · page 18 · `supports`

- Recorded span: “feed-forward layers (up-proj, down-proj) are substantially more critical than the attention projection layers”
- Annotator note: Page 18: feed-forward vs attention layers criticality
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p18_c3

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-025 — What extensions do the authors propose as future work?

- Type: `factual`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

#### GOLD-030 · page 9 · `supports`

- Recorded span: “extending the approach to vision-language models”
- Annotator note: Page 9: future work extensions
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p9_c1

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-026 — What accuracy does LEMS reach on mixture-of-experts models? (only mentioned as future work, so it's a good abstention test)

- Type: `unanswerable`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

No gold evidence is recorded for this question. For an unanswerable query, confirm the corpus genuinely lacks the information, then write that down.

### Q-027 — What was the peak GPU memory during KFAC-SVD calibration?

- Type: `unanswerable`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

No gold evidence is recorded for this question. For an unanswerable query, confirm the corpus genuinely lacks the information, then write that down.

### Q-028 — Under what software license is the LEMS code released?

- Type: `unanswerable`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc

No gold evidence is recorded for this question. For an unanswerable query, confirm the corpus genuinely lacks the information, then write that down.

### Q-029 — Do the two papers describe the Kronecker-factor estimation the same way?

- Type: `comparative`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc, doc_7fdb1200e823

#### GOLD-031 · page 3 · `supports`

- Recorded span: “By fitting a rank-1 approximation to bF in a rearranged space using Lanczos (Lanczos, 1950), GFWSVD creates a Kronecker product A ⊗G”
- Annotator note: LEMS side: Kronecker-factor estimation via Lanczos rank-1 approximation on rearranged Fisher
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p3_c5

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-032 · page 4 · `supports`

- Recorded span: “so we employ truncated SVD using the Lanczos method (Lanczos, 1950), which avoids explicit matrix construction”
- Annotator note: GFWSVD side: Kronecker factors computed via Lanczos truncated SVD on permuted Fisher
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_7fdb1200e823 → doc_7fdb1200e823_p4_c3

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

### Q-030 — Does GFWSVD's own reported performance match the LEMS paper's claim that it degrades?

- Type: `apparent_conflict`
- Question status: `human_draft` (labelled_by: snehala24)
- Target documents: doc_c67573ef50cc, doc_7fdb1200e823

#### GOLD-033 · page 6 · `supports`

- Recorded span: “GFWSVD degrades significantly under identical conditions”
- Annotator note: LEMS side: GFWSVD degradation claim
- Conflict reason: Apparent conflict: LEMS reports GFWSVD degrades significantly under identical conditions, whereas GFWSVD reported strong performance in its own evaluation.
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_c67573ef50cc → doc_c67573ef50cc_p6_c5

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

#### GOLD-034 · page 9 · `contradicts`

- Recorded span: “Our results demonstrate that GFWSVD consistently outperforms diagonal and activation-based baselines across a wide range of compression regimes.”
- Annotator note: GFWSVD side: Reported competitive performance and outperforming baselines across compression regimes
- Conflict reason: Apparent conflict: LEMS reports GFWSVD degrades significantly under identical conditions, whereas GFWSVD reported strong performance in its own evaluation.
- Current label: `human_draft` (labelled_by: snehala24)
- Corpus check: found in doc_7fdb1200e823 → doc_7fdb1200e823_p9_c4

- [ ] Page and span are verbatim in the PDF
- [ ] The span actually answers the question
- [ ] Relation is correct (a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)
- [ ] Rejected — record why: …

```text
reviewer handle : human:
review date     :
observations    :
```

