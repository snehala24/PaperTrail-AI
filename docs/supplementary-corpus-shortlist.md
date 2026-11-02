# Supplementary corpus shortlist — candidates for a genuine cross-paper contradiction

**Status: `AWAITING USER APPROVAL`. Nothing here has been downloaded, ingested, indexed, or added to
the corpus.** The five indexed papers and their 409 chunks are unchanged. Every incoming-paper
sentence below was fetched from its arXiv abstract page on 2026-11-02 and is quoted verbatim; every
*indexed* sentence was matched against the stored chunks in `data/chunks/` and is cited by chunk id and
page. Nothing is quoted from memory.

## Why a shortlist and not a contradiction claim

[benchmark-contradictions.md](benchmark-contradictions.md) establishes that the indexed corpus contains
no verified category-A case: 302 opposition-bearing sentences and 30 cross-document candidate pairs
were mined and none survived. A contradiction cannot be manufactured by choosing a question that sounds
opposed. The only legitimate route is to add primary research that asserts the opposite of something an
indexed paper asserts, about the same measurable quantity, under comparable conditions — and then to have
the opposing spans read and signed off by a human.

Each candidate is therefore described as **a pair of claims to be tested against the full text**, never
as a confirmed contradiction. One candidate (number 3) was demoted by this process, and one (number 2)
turned out to need number 1 to mean anything — which is the point of doing the inspection before asking
for permission to ingest.

| # | Incoming paper | Indexed paper it would oppose | Assessment |
| --- | --- | --- | --- |
| 1 | Learning Rate Matters (2602.04998) | DoRA (2402.09353) | Best genuine-contradiction candidate |
| 2 | PiSSA (2404.02948) | LoRA (2106.09685) + pairs with #1 | Genuine only as a pair with #1 |
| 3 | Output-Space Allocation Costs (2606.27785) | LEMS (2610.32878) | **Demoted: compatible findings, not a conflict** |
| 4 | LoRA Learns Less and Forgets Less (2405.09673) | LoRA (2106.09685) | Best real cross-paper *apparent* conflict |
| 5 | LowRankArena (2608.26389) | GFWSVD (2505.17974), LEMS | Needs full-text tables before it counts |

## Candidate 1 — strongest case: DoRA's "consistently outperforms" vs a tuned-baseline null result

| Field | Value |
| --- | --- |
| Title | Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning |
| Authors | Yu-Ang Lee, Ching-Yun Ko, Pin-Yu Chen, Mi-Yen Yeh |
| arXiv / URL | `arXiv:2602.04998` — https://arxiv.org/abs/2602.04998 |
| Research question | Do the reported gains of LoRA variants survive a proper hyperparameter search? |
| Main finding (verbatim, abstract) | "Crucially, once learning rates are properly tuned, all methods achieve similar peak performance (within 1-2%), with only subtle rank-dependent behaviors." and "improvements reported under a single training configuration may not reflect consistent methodological advantages." |
| Indexed side it opposes (verbatim, from the stored chunk) | DoRA, `doc_f048986e4d0b_p1_c2`, p.1: "DoRA consistently outperforms LoRA on fine-tuning LLaMA, LLaVA, and VL-BART on various downstream tasks" — and `doc_f048986e4d0b_p2_c1`, p.2: "The experimental results show that DoRA consistently outperforms LoRA without sacrificing inference efficiency, such as commonsense reasoning (+3.7/+1.0 on LLaMA-7B/13B, +2.9 on LLaMA2-7B …)" |
| Conflict pair | DoRA: *consistently outperforms LoRA, +3.7 p.p. on LLaMA-7B commonsense* ↔ this paper: *the nine variants, DoRA among them, land within 1–2% of vanilla LoRA once learning rates are tuned* |
| Genuine or apparent | **Genuine in form** — same quantity (downstream task accuracy at matched rank and parameter budget), mutually exclusive assertions, and DoRA is explicitly inside the re-evaluated set. Honest caveat: "consistently" is protocol-conditional, so if DoRA's published runs were already learning-rate-optimal the pair becomes an apparent conflict about tuning budget rather than about method |
| Confidence | High |
| What settles it | Its per-method best-learning-rate peak-accuracy table for DoRA vs LoRA at matched rank and task, plus DoRA's own published learning rate — check whether the two settings overlap |
| Corpus cost | One new document; existing doc ids, chunk ids and gold spans untouched |

## Candidate 2 — the sharpest head-to-head numbers, but only meaningful beside Candidate 1

| Field | Value |
| --- | --- |
| Title | PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models |
| Authors | Fanxu Meng, Zhaohui Wang, Muhan Zhang |
| arXiv / URL | `arXiv:2404.02948` — https://arxiv.org/abs/2404.02948 |
| Research question | Does SVD-initialising the adapter (principal components trainable, residual frozen) beat LoRA's Gaussian/zero initialisation? |
| Main finding (verbatim, abstract) | "PiSSA consistently outperforms LoRA under identical experimental setups. On the GSM8K benchmark, Mistral-7B fine-tuned with PiSSA achieves an accuracy of 72.86%, surpassing LoRA's 67.7% by 5.16%." |
| Indexed side it opposes (verbatim, from the stored chunk) | LoRA, `doc_e9a0d3128767_p1_c2`, p.1: "LoRA performs on-par or better than fine- tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3" — a parity claim about full fine-tuning, **not** a ranking against PiSSA |
| Conflict pair | The real opposition is PiSSA's *+5.16 p.p. over LoRA on GSM8K under identical setups* against Candidate 1's *all methods within 1–2% of vanilla LoRA once tuned* |
| Genuine or apparent | **Genuine when paired with Candidate 1** (same benchmark family, directly incompatible magnitude claims). Against indexed LoRA alone it is **apparent**: LoRA's claim predates LLM-scale evaluation and is measured against full fine-tuning |
| Confidence | High for the Candidate 1 ↔ Candidate 2 axis; low for a LoRA-vs-PiSSA contradiction |
| What settles it | Candidate 1's PiSSA row — its GSM8K (or equivalent) accuracy under the tuned search, against PiSSA's 72.86% |
| Note | Ingesting Candidate 1 without Candidate 2 loses the paired numbers; ingesting Candidate 2 alone creates no cross-paper contradiction with this corpus |
| Also relevant | PiSSA is SVD-initialisation of adapters, so it sits in the same design space as the corpus's GFWSVD/LEMS weight-gradient-SVD line — a second, weaker axis of contact |

## Candidate 3 — demoted by inspection: compatible findings, not a contradiction

| Field | Value |
| --- | --- |
| Title | Output-Space Allocation Costs for Calibration-Guided LLM Compression: An Empirical Study |
| Authors | Qiong Tang, Xiangkun Hu, Xiangyang Liu, Yiran Chen, Yunfan Shao |
| arXiv / URL | `arXiv:2606.27785` — https://arxiv.org/abs/2606.27785 |
| Research question | Does ranking rank-allocation cost by an output-space (activation/logit) objective beat a weight-space Frobenius cost? |
| Main finding (verbatim, abstract) | "our ROCKET-ActCost achieves +0.8 percentage points higher average accuracy across 8 zero-shot benchmarks (53.1% vs 52.3%), but increases WikiText perplexity by 16% (61.46 vs 52.98)." and "The high correlation (>0.99) between weight-space and output-space errors limits allocation divergence" |
| Indexed side it was expected to oppose (verbatim, from the stored chunks) | LEMS, `doc_c67573ef50cc_p8_c6`, p.8: "KL divergence consistently outperforms MSE in zero shot accuracy; improving it up to 1.1 p.p. For perplexity, MSE performs slightly better than KL, but with small margins" — with the numbers in `doc_c67573ef50cc_p18_c1`, p.18 (Table 4) |
| Result of the comparison | **The directions agree.** LEMS already reports the same metric flip: the output-space-style objective wins on zero-shot accuracy and loses slightly on perplexity. This paper reproduces that shape at a different magnitude (+0.8 p.p. accuracy, +16% perplexity) on a different model |
| Genuine or apparent | **Neither — category C, compatible findings.** It is corroboration, not contradiction, and it must not be presented as a contradiction because both papers report an accuracy/perplexity trade-off |
| Residual value | Still the most useful ingestion target for testing whether the system can recognise *agreement across papers* (a `supports` edge with two independently grounded sides), which the current scripted benchmark can only test with synthetic fixtures |
| Confidence | High that this is not a category-A case, on the abstract-level comparison. Full text could still surface a genuine disagreement about *when* the flip appears — that would require reading LEMS §5.3.3 and this paper's Table 1 side by side |

## Candidate 4 — the clean cross-paper apparent conflict

| Field | Value |
| --- | --- |
| Title | LoRA Learns Less and Forgets Less |
| Authors | Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham |
| arXiv / URL | `arXiv:2405.09673` — https://arxiv.org/abs/2405.09673 |
| Research question | Does LoRA match full fine-tuning for code and math at LLM scale? |
| Main finding (verbatim, abstract) | "Our results show that, in the standard low-rank settings, when fine-tuning on code and math work, LoRA substantially underperforms full finetuning." and "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations" |
| Indexed side it opposes (verbatim, from the stored chunk) | LoRA, `doc_e9a0d3128767_p1_c2`, p.1: "LoRA performs on-par or better than fine- tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3" |
| Conflict pair | LoRA: *on-par or better than fine-tuning* ↔ this paper: *substantially underperforms full finetuning* |
| Genuine or apparent | **Apparent, and instructive precisely because it is apparent** — same quantity (downstream accuracy relative to full fine-tuning), opposite direction, but different regime: LoRA's claim is encoder/GPT-2-scale instruction tuning; this paper is continued pretraining of StarCoder/Magicoder-class models on ~20B tokens of code and math. The correct system behaviour is `apparent_conflict` with a reason naming the task, model family and training regime — never `contradicts` |
| Confidence | Medium |
| What settles it | A model+task overlap between the two evaluations, and the effective-rank measurement of ΔW |
| Ingestion note | Verbatim matching against `doc_e9a0d3128767_p1_c2` must survive the stored line-break artefact "fine- tuning"; the product's L2 check normalises whitespace and ligatures, and this pair is a good regression for that |

## Candidate 5 — potential, least verified

| Field | Value |
| --- | --- |
| Title | LowRankArena: A Standardized Evaluation Platform for SVD-Based LLM Compression |
| Authors | Zishan Shao, Lixun Zhang, Kangning Cui, Wenhao Wu, Jinhee Kim, Yixiao Wang, Ting Jiang, Hancheng Ye, Qinsi Wang, Fan Yang, Danyang Zhuo, Yiran Chen, Hai Li |
| arXiv / URL | `arXiv:2608.26389` — https://arxiv.org/abs/2608.26389 |
| Research question | Do published SVD-compression method rankings hold when everything is re-run under one protocol? |
| Main finding (verbatim, abstract) | "prior findings are highly conditional under standardized protocols: clear leaders and performance tiers shift across backbones and keep ratios" |
| Indexed side it opposes (verbatim, from the stored chunks) | GFWSVD, `doc_7fdb1200e823_p1_c4`, p.1: "GFWSVD consistently outperforms both diagonal Fisher approximations (FWSVD (Hsu et al., 2022)) and activation-based baselines (ASVD (Yuan et al., 2023), SVD-LLM (Wang et al., 2025c))"; LEMS, `doc_c67573ef50cc_p2_c3`, p.2 and its own "consistently outperforms existing search strategies" claim |
| Genuine or apparent | **Not established from the abstract — needs full-text inspection.** The abstract names no methods and no numbers; the pair becomes genuine only if GFWSVD or LEMS is among the audited methods and its ordering reverses at a matched keep ratio |
| Confidence | Medium. It is primary research (a new platform with released checkpoints), not a survey |
| What settles it | Its per-method, per-backbone, per-keep-ratio tables read against GFWSVD's and LEMS's own reported orderings |

## A lead found during inspection: the same claim, two numbers

While pulling the indexed side of Candidate 5 verbatim, one numeric disagreement turned up that is
already inside this corpus and needs no new paper:

| Source | Verbatim | Number |
| --- | --- | --- |
| Ingested PDF, page 1 (`doc_c67573ef50cc_p1_c2`) | "we show that LEMS consistently outperforms existing search strategies, de- livering significant zero-shot accuracy gains of up to 4.8 p.p. that generalize to model sizes of 70B parameters" | **4.8 p.p.** |
| Current arXiv listing for `2610.32878`, fetched 2026-11-02 | "delivering significant zero-shot accuracy gains of up to 4.7 p.p." | **4.7 p.p.** |

Two renderings of the *same* sentence carry different numbers. What it is and what it is not:

- It is **not** a research contradiction between two papers, and it must not be quoted as a category-A
  success. There is one claim, one author set, one quantity.
- It is most likely a **version revision**: the ingested file is `data/2610.32878v1.pdf`, and the arXiv
  page shows a later version dated 26 Oct 2026. Confirming that needs v2 downloaded and compared —
  `AWAITING USER APPROVAL`, and it does not require re-ingesting v1 or touching any existing chunk.
- A provenance gap made this harder to see than it should have been: the stored document record keeps
  `title`, `page_count` and `sha256`, but no `arxiv_id` or `version` field, so paper version comes only
  from the filename. Worth knowing before this becomes multi-user.
- Product relevance is real but narrow: a citation-anchored system that displays a quotation and its
  page is exactly the thing that would surface "4.8 on p.1 vs 4.7 on the listing" to a reader, and it is
  the kind of case that should be labelled **same-claim numeric discrepancy**, a category the current
  four-relation taxonomy does not name.

## Recommended order, if approved

Candidates **1 and 4** first: Candidate 1 is the only route here to a defensible category-A case against
an indexed paper (DoRA), and Candidate 4 is the best route to a *real* cross-paper apparent conflict,
which is what the offline benchmark currently can only simulate. **Candidate 2** is worth ingesting
alongside 1 because the paired numbers are what make 1 quantitative. **Candidate 5** should not be
ingested until its tables have been read. **Candidate 3 is not recommended as a contradiction source at
all** — it corroborates LEMS and should be labelled as such.

Two papers is the honest minimum: one paper cannot contradict a corpus of one.

## What ingestion would change

Ingestion is keyed by SHA-256 and appends, so existing `doc_…` ids, chunk ids, gold spans and benchmark
references stay valid. What *does* change: the corpus counts quoted throughout `README.md` and this
report ("5 papers / 117 pages / 409 chunks"), the retrieval candidate pool, and therefore the retrieval
benchmark numbers — the current tables would need a re-run to remain accurate, and the earlier
409-chunk run would become a historical artifact rather than the current measurement. Nothing about
adding papers changes the meaning of the existing results; it changes which corpus they describe.

## Explicitly rejected, with reasons

| Candidate | Why rejected |
| --- | --- |
| QLoRA (`arXiv:2305.14314`) — "preserving full 16-bit finetuning task performance" | Verified abstract, but it contradicts a paper that is not in this corpus. Its natural opposite, LoftQ, is also not indexed |
| LoftQ (`arXiv:2310.08659`) — "a consistent gap … between full fine-tuning and quantization plus LoRA" | Same reason: opposes QLoRA, not an indexed paper. Only contextual against LEMS, whose abstract makes no quantization-parity claim |
| `arXiv:2405.01751` | Resolved to a ferroelectrics paper, not the "QLoRA isn't all you need" work. **Unverified id — rejected** |
| LACE-SVD (`arXiv:2607.03057`), SVD-Surgeon (`arXiv:2606.23568`) | Abstracts retrieved and read; neither states a comparable metric or a directional claim → "not established from the abstract" |
| `arXiv:2609.15838` ("Per-Matrix Optimality Is Not Enough") | Real numbers (PPL 42.1 → 19.1 → 11.4; "downstream accuracy remains well below the dense model") but no matched-compression-ratio conflict with an indexed claim. Kept as an alternate |
| Multi-dimensional Safety Evaluation of LLM Compression (`arXiv:2407.04965`), TT-LoRA (`arXiv:2408.01008`) | Evaluation-only / survey / off-target quantity; not a directional primary-research claim |
| Two further ids recalled during research | Resolved to unrelated papers (metamaterials, sampling). **Rejected as unverified** |

## Approval required before

1. Downloading any of these PDFs.
2. Ingesting any of them into `data/` (new documents only; existing ids and gold spans are unaffected,
   but corpus counts and retrieval-pool size change and the retrieval benchmark should be re-run).
3. Writing new benchmark questions or gold spans about them — those would be AI-drafted and would join
   the 34 spans awaiting human review, not bypass it.
