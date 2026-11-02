# Retrieval Benchmark — Current Corpus (2026-11-02)

This page reports retrieval quality **as measured today, on the corpus that is actually indexed**.
Earlier measurements are kept in a clearly marked historical section and are never presented as
current numbers.

Artifact: [`eval/runs/run_20261102_132940.json`](../eval/runs/run_20261102_132940.json) — written by
`python -m backend.app.eval.run_retrieval`, exit status 0, fully offline (zero live API calls).

## What was measured

| Setting | Value |
| --- | --- |
| Git commit under test | `b46396b` |
| Indexed corpus | 5 papers, 117 pages, **409 chunks** (`data/`, chunker `0.2.3`) |
| Chunk size bounds | 800–1500 chars (tables up to 2500) |
| Embedding model | `all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Candidate pool into the reranker | 30 |
| Fusion | RRF, k = 60 |
| Questions in the dataset | 30 |
| Questions carrying gold evidence | 25 |
| Reserved unanswerable (no gold by design) | 5 — Q-009, Q-010, Q-026, Q-027, Q-028 |
| **Scored questions (n)** | **25** |
| Gold evidence items | 34 |

Gold coverage was re-checked against the stored chunks with `python -m backend.app.eval.verify_gold`
(exit 0): **34/34 gold spans resolve verbatim inside current chunks, 0 orphan questions, 0 unmatched
gold items**. Coverage is therefore 25/25 answerable questions.

Provenance of those labels matters for how the numbers should read: the 25 scored questions come from
20 researcher-drafted and 10 assistant-drafted questions, with 24 researcher-drafted and 10
assistant-drafted gold items, and **none of the 34 spans has been signed off against the PDF yet**
(see [`eval/LABELLING.md`](../eval/LABELLING.md)). These are pipeline-validation measurements, not
human-graded accuracy.

## Results — current corpus, all 25 scored questions

| Config | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.3400 | 0.6800 | 0.7600 | 0.5615 | 0.5215 |
| Dense | 0.4000 | 0.5800 | 0.6800 | 0.5364 | 0.4756 |
| Hybrid (RRF) | 0.4000 | 0.6800 | 0.8200 | 0.5969 | 0.5401 |
| **Hybrid + cross-encoder rerank** | **0.4600** | **0.8800** | **0.8800** | **0.6927** | **0.6718** |

95% bootstrap CIs for the shipped configuration (hybrid + rerank, n = 25):
Recall@5 **[0.74, 0.98]**, MRR **[0.538, 0.8267]**.

Split by who drafted the labels:

| Subset | n | Config | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Researcher-drafted | 17 | BM25 | 0.1471 | 0.5882 | 0.7059 | 0.4418 | 0.4224 |
| Researcher-drafted | 17 | Dense | 0.2941 | 0.5000 | 0.5882 | 0.4637 | 0.4214 |
| Researcher-drafted | 17 | Hybrid (RRF) | 0.2353 | 0.5882 | 0.7941 | 0.5053 | 0.4574 |
| Researcher-drafted | 17 | **Hybrid + rerank** | 0.4412 | **0.8824** | 0.8824 | **0.7196** | 0.6979 |
| Assistant-drafted | 8 | BM25 | 0.7500 | 0.8750 | 0.8750 | 0.8160 | 0.7321 |
| Assistant-drafted | 8 | Dense | 0.6250 | 0.7500 | 0.8750 | 0.6908 | 0.5908 |
| Assistant-drafted | 8 | Hybrid (RRF) | 0.7500 | 0.8750 | 0.8750 | 0.7917 | 0.7158 |
| Assistant-drafted | 8 | Hybrid + rerank | 0.5000 | 0.8750 | 0.8750 | 0.6354 | 0.6163 |

The researcher-drafted subset is the harder one for every config except Recall@5-with-rerank, where
both land at ≈0.88. The assistant-drafted subset is small (n = 8) and its questions were written from
the same passages the labels point at, so its higher BM25 scores should not be read as BM25 being
good.

Questions still missing their gold passage inside the top 5 with hybrid + rerank: **Q-021** only.
Wider failures, for triage: BM25 misses Q-011, Q-012, Q-013, Q-014, Q-019, Q-029; Dense misses
Q-012, Q-013, Q-018, Q-021, Q-024, Q-025; RRF misses Q-012, Q-013, Q-018, Q-021, Q-025.
Q-029 (the cross-paper Kronecker-factor question) is a BM25-only failure — rerank places its gold at
rank 2, which is the case that motivated keeping the cross-encoder in the pipeline.

## Methodology change made today (and why three run files exist)

`run_retrieval.py` carried a hardcoded exclusion of Q-029 and Q-030 in two places (the evaluation loop
and the researcher-drafted partition). Those two questions are cross-paper items that were excluded
when the second paper (GFWSVD) was not yet indexed. GFWSVD **is** indexed, both of their gold spans
verify verbatim, and the exclusion was silently shrinking n from 25 to 23.

The exclusion was removed from both places. Sequence of artifacts, all preserved:

| Artifact | n scored | Note |
| --- | --- | --- |
| `run_20261102_130202.json` | 23 | first run on the current corpus, stale exclusion still in place |
| `run_20261102_132608.json` | 23 | exclusion removed from the loop only — the per-question table showed 16 researcher rows while the aggregate still divided by 15 (the second stale filter). Kept as evidence of the inconsistency, not used for reporting. |
| `run_20261102_132940.json` | **25** | both stale filters gone. **This is the current measurement.** |

## Historical measurement (2026-10-29) — not current

Artifact: [`eval/runs/run_20261029_073045.json`](../eval/runs/run_20261029_073045.json), commit
`80e9363`, corpus **209 chunks / 3 papers**, n = 23.

| Config | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.3696 | 0.7174 | 0.8696 | 0.5976 | 0.5549 |
| Dense | 0.5000 | 0.6522 | 0.7174 | 0.6266 | 0.5604 |
| Hybrid (RRF) | 0.5652 | 0.7174 | 0.7391 | 0.7039 | 0.6306 |
| Hybrid + rerank | 0.4783 | 0.8696 | 0.9130 | 0.6827 | 0.6715 |

Historical 95% CIs for hybrid + rerank (n = 23): Recall@5 [0.6957, 1.0], MRR [0.5236, 0.8370].

### Why these two tables must not be read as a before/after experiment

Three things changed at once between 2026-10-29 and today: the corpus grew from 209 chunks over 3
papers to 409 chunks over 5 papers (Attention, LoRA and DoRA were missing then), the scored set grew
from 23 to 25 questions, and the two added questions are the hardest kind (cross-paper). With the
distractor pool nearly doubled, single-model configs lost ground — Dense MRR went 0.6266 → 0.5364 and
Dense Recall@5 went 0.6522 → 0.5800 — while hybrid + rerank held Recall@5 in the same band
(0.8696 → 0.8800) and MRR (0.6827 → 0.6927). RRF's Recall@10 rose (0.7391 → 0.8200) even though its
Recall@5 did not (0.7174 → 0.6800). That pattern is consistent with the cross-encoder absorbing the
larger corpus, but it is **not** a controlled comparison and no improvement claim is made from it.

## Reproduce

```bash
# from the repository root, with .venv active
.venv/Scripts/python.exe -m backend.app.eval.verify_gold        # gold coverage on current corpus
.venv/Scripts/python.exe -m backend.app.eval.run_retrieval      # writes a new timestamped run
```

The retrieval benchmark makes no live API calls; it loads the cached embedding and reranker models and
the stored `data/` corpus, and writes `eval/runs/run_<timestamp>.json` without touching earlier files.

**Reproducibility check (2026-11-02).** Re-running the offline suite at commit `2f7cdad` wrote
[`../eval/runs/run_20261102_154437.json`](../eval/runs/run_20261102_154437.json), whose `combined`,
`human_labeled` and `ai_draft` metric blocks and whose per-configuration failure lists are identical to
`run_20261102_132940.json` — every figure on this page reproduces exactly, not approximately.

## What this page does not establish

It does not establish answer quality, citation correctness, or contradiction detection — those are
generation-side properties measured elsewhere. It does not establish generalisation: n = 25 is far
below the 30 the runner itself warns about (`Sample size (n=25) too small for statistical
significance`), the intervals above are wide, and no gold span has been human-verified yet.
