# PaperTrail AI

**A research evidence workspace for reading several papers at once.** Ask a question across your indexed PDFs and get an answer where every sentence is tied to a quoted passage on a specific page of a specific paper — or an explicit refusal when the evidence does not support one.

PaperTrail AI is not a "chat with a PDF" wrapper. The unit it reasons over is the **claim**, not the document, and the unit it shows you is the **evidence link**, not a citation marker.

**Who it is for:** anyone who has to synthesize several papers and be accountable for the result — a researcher writing a related-work section, a graduate student checking that a review's claim is actually in the source, an engineer deciding between two methods from their published numbers. It is built as a single-user local workspace: the corpus, chunks and evaluation artifacts stay on disk where you can read them, and the product's promise is narrower than "answers about papers" — it is *auditable or explicitly refused*.

---

## The problem it addresses

Literature synthesis fails in three quiet ways: an answer sounds plausible but no passage says it; two papers disagree and a model reports a contradiction that is really a different benchmark; a paper is cited for a number that appears on a different page. Retrieval-augmented chat interfaces hide all three behind fluent prose.

PaperTrail makes each of them a visible, checkable object:

1. **Claim-to-evidence ledger.** An answer is decomposed into atomic claims. Each claim carries the exact chunks it rests on, with page numbers and verbatim quotations, so the reader can audit the reasoning rather than trust it.
2. **Three-layer citation verification.** A claim is only marked verified when (L1) the cited chunk exists in the retrieved set, (L2) the quoted span is a verbatim substring of that chunk, and (L3) an entailment check at temperature 0 confirms the passage supports the claim. Anything that does not clear all three is shown as unverified, never silently promoted.
3. **Apparent-conflict handling.** When two papers report different numbers, the relationship is classified `supports`, `apparent_conflict`, `contradicts`, or `unresolved`. Method, dataset, metric, or setting differences are labelled `apparent_conflict` with the reason attached; a relationship is never inferred from document IDs alone, and unresolved stays unresolved.
4. **Honest abstention.** Insufficient evidence, a failed provider call, or an unverifiable quotation ends in an abstention with the reason and the retrieval diagnostics that caused it — not in a generated guess.
5. **Measurable grounding.** Retrieval and conflict behaviour are evaluated against an offline benchmark with run artifacts kept in the repository, with their provenance stated next to the numbers.

---

## What you can do in the app

| Destination | What it is for |
| --- | --- |
| **Dashboard** | Corpus state at a glance: papers, pages, chunks, extraction warnings, backend health. |
| **Research Library** | Browse indexed papers, open a document's metadata and extraction warnings, and inspect every passage actually stored in the chunk explorer. Upload a PDF (up to 25 MB) to index it. |
| **Research Assistant** | Pick the papers to reason over, set the retrieval depth, ask a question, and watch the answer assemble with per-claim verification status and evidence. |
| **Evidence Ledger** | Every claim in the last answer with its quotation, page, chunk ID, verification layer results, and the reason for any abstention. |
| **Cross-Paper Comparisons** | Relationships found between claims across papers, with the quoted evidence shown for both sides of each comparison. |
| **Research history** | Runs you have completed are saved in your browser and can be reopened later; a reopened run is labelled as such rather than presented as the live answer. |

---

## Architecture

```text
[PDF upload]
     │
     ▼
[Parser — PyMuPDF]  page/section aware, records extraction warnings
     │               (figure-only pages reported as "graphics not read as data")
     ▼
[Chunker v0.2.3]    ~800–1500 char passages, section isolation,
     │              tables kept separate, references tagged and excluded by default
     ▼
[JSON store — data/]  SHA-256 dedup, version-aware re-ingestion
     │
     ▼
[Hybrid retrieval]  BM25 (rank-bm25)  +  dense FAISS (all-MiniLM-L6-v2)
     │              → Reciprocal Rank Fusion (k = 60)
     │              → Cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
     ▼
[Claim extraction — Gemini]  minimal sufficient claims, verbatim quote precheck
     │                        (a claim whose quote is not found fails closed here)
     ▼
[Citation validator]  L1 existence → L2 verbatim substring → L3 temp-0 entailment
     │
     ▼
[Conflict analyser]   supports / apparent_conflict / contradicts / unresolved,
     │                a structured comparison requires both sides to back a verified
     │                claim; a retrieved-but-uncited disagreement becomes an
     │                answer-level note, never a fabricated ledger edge
     ▼
[POST /answer]  →  Next.js workspace (ledger, comparisons, abstention, history)
```

Everything upstream of the LLM runs locally and offline; Gemini is used for claim decomposition, entailment, and conflict reasoning only.

---

## Tech stack

**Backend** — Python 3.12.10, FastAPI, Pydantic v2, PyMuPDF, `sentence-transformers` (`all-MiniLM-L6-v2`), FAISS CPU, `rank-bm25`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `google-genai` through an abstract `LLMProvider`, pytest.

**Frontend** — Next.js 14.2.24 (App Router), React 18.3.1, TypeScript 5.6, Tailwind CSS 3.4, lucide-react, Vitest 2.1.8 + Testing Library.

**Storage** — JSON files on disk under `data/` (deliberately: single-user, inspectable, no database yet). Retrieval indexes are rebuilt in-process from those files.

---

## Quickstart

Commands are Windows PowerShell; the Unix equivalents are the same without `.exe`.

### 1. Backend

```powershell
cd "papertrail Ai"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
Copy-Item .env.example .env      # then set GEMINI_API_KEY in .env
.\.venv\Scripts\uvicorn.exe backend.app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs at <http://localhost:8000/docs>; health check at <http://localhost:8000/health>.

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev                      # http://localhost:3000
```

The frontend reads the backend URL from `NEXT_PUBLIC_API_URL` (see `frontend/.env.example`, defaults to `http://localhost:8000`).

### 3. Environment variables (names only)

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Provider key for claim extraction, entailment, and conflict reasoning. Keep it in `.env`; `.env` is gitignored and is never printed by the app. |
| `GEMINI_MODEL`, `GEMINI_THINKING_LEVEL`, `GEMINI_MAX_OUTPUT_TOKENS`, `GEMINI_TIMEOUT` | Model selection and request budget. |
| `EMBEDDING_MODEL_NAME`, `RERANKER_MODEL_NAME` | Local retrieval models. |
| `STORAGE_DIR`, `EVAL_DIR` | Where documents/chunks and run artifacts live. |
| `HOST`, `PORT`, `ENVIRONMENT`, `LOG_LEVEL` | Server and logging. |
| `NEXT_PUBLIC_API_URL` | Frontend → backend base URL. |

No key, token, or credential value belongs in this repository. If a live key is ever pasted into a terminal or log, treat it as exposed and rotate it. The repository-side audit (what was scanned, what it found, and what it cannot establish) plus the manual revocation checklist the owner still has to perform is [docs/security-key-review.md](docs/security-key-review.md).

### 4. Ingesting papers

Upload through the Research Library, or:

```powershell
curl.exe -F "file=@data/2610.32878v1.pdf" http://localhost:8000/documents
```

Re-uploading an existing PDF is a no-op: upload-deduplication by SHA-256, and version-aware invalidation if the chunker changed.

---

## API surface

| Method & path | Returns |
| --- | --- |
| `GET /health` | Service, model, and corpus state. |
| `GET /documents` | Indexed documents with page counts and extraction warnings. |
| `GET /documents/{doc_id}` | One document. |
| `GET /documents/{doc_id}/chunks` | Every stored passage for that document. |
| `POST /documents` | Upload and index a PDF (≤ 25 MB), `201` with the document record. |
| `POST /search` | Hybrid retrieval with per-stage scores and ranks on each result (`bm25_score`/`bm25_rank`, `dense_score`/`dense_rank`, `fused_score`/`fused_rank`, `rerank_score`/`rerank_rank`), plus `total_candidates` and `returned_count`. |
| `POST /answer` | Grounded answer: claims, evidence links, verification results, comparisons, or an abstention with diagnostics. |

---

## Corpus indexed locally

`data/` is gitignored (papers and derived chunks stay on your machine). The workspace currently indexes:

| Paper | Pages | Chunks |
| --- | --- | --- |
| Generalized Fisher-Weighted SVD (GFWSVD) | 19 | 78 |
| Attention Is All You Need | 15 | 41 |
| Advancing SVD-based LLM Compression via Layer-Wise Error Model Search (LEMS) | 34 | 131 |
| LoRA: Low-Rank Adaptation of Large Language Models | 26 | 78 |
| DoRA: Weight-Decomposed Low-Rank Adaptation | 23 | 81 |
| **Total** | **117** | **409** |

Chunk composition: 278 text passages, 13 tables, 27 figure references, 91 reference-list chunks. All at chunker version `0.2.3`.

---

## Evaluation

### Benchmark composition and provenance — read this before quoting any number

| File | Items | Researcher-drafted (`human_draft`) | Assistant-drafted (`ai_draft`) | Human-verified |
| --- | :---: | :---: | :---: | :---: |
| `eval/questions.jsonl` | 30 | 20 | 10 | 0 |
| `eval/gold_evidence.jsonl` | 34 | 24 | 10 | 0 |

**No gold annotation has been checked against a PDF and signed off yet.** Every accuracy figure below is therefore a *pipeline-validation* measurement against draft labels — it shows the retrieval and verification stack behaves sensibly, not that it matches expert ground truth. `eval/LABELLING.md` is the protocol, [eval/HUMAN-REVIEW-GUIDE.md](eval/HUMAN-REVIEW-GUIDE.md) is the reviewer's instruction set, and `backend/app/eval/review_worksheet.py` generates `eval/review_worksheet.md`, one checklist per gold span, pre-resolved against the stored corpus.

The worksheet reports how each recorded span relates to the indexed text — 33 `exact`, 1 `case only`, 0 absent, 0 wrong-page, 0 missing-document — and flags 15 of the 34 items with a caution before anyone ticks a box. That is **structural integrity, not human validation**: every span matches text that a machine extracted, and no person has yet confirmed a single one against the PDF. Two of the flags are substantive: both `contradicts` annotations carry a `conflict_reason` that describes an *apparent* conflict, so label and reason point at different ledger relations and only a human can decide which is right.

### Retrieval results — current corpus (5 papers, 409 chunks)

Run [`eval/runs/run_20261102_132940.json`](eval/runs/run_20261102_132940.json) (commit `b46396b`), measured against the corpus the workspace indexes today: 5 papers, 117 pages, 409 chunks. Of the 30 questions, 25 are answerable and 5 are reserved for abstention testing, so n = 25 per configuration. Candidate pool 30, no human-verified gold (see the provenance table above — these are pipeline-validation figures against draft labels).

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| BM25 only | 0.3400 | 0.6800 | 0.7600 | 0.5615 | 0.5215 |
| Dense only | 0.4000 | 0.5800 | 0.6800 | 0.5364 | 0.4756 |
| Hybrid + RRF | 0.4000 | 0.6800 | 0.8200 | 0.5969 | 0.5401 |
| **Hybrid + cross-encoder rerank** | **0.4600** | **0.8800** | **0.8800** | **0.6927** | **0.6718** |

Hybrid + rerank carries Recall@5 = 0.8800 with a 95% bootstrap interval of [0.74, 0.98] and MRR [0.538, 0.827]. Split by who drafted the labels:

| Subset | Answerable n | Shipped config (hybrid + rerank) | Recall@5 | MRR |
| --- | :---: | --- | --- | --- |
| Researcher-drafted (`human_draft`) | 17 | Hybrid + rerank | 0.8824 [0.7353, 1.0] | 0.7196 |
| Assistant-drafted (`ai_draft`) | 8 | Hybrid + rerank | 0.8750 | 0.6354 |

At n = 25 and n = 17 the intervals remain wide and the subsets are not statistically separable. In the
assistant-drafted subset the reranker does **not** win: BM25 reaches the same Recall@5 (0.8750) with MRR
0.8160 versus rerank's 0.6354, and rerank's Recall@1 there is the weakest of the four configurations
(0.5000). That is eight questions, and those eight were written from the same passages their labels point
at, so it is a property of how the subset was drafted, not evidence that reranking hurts — see
[docs/benchmark-retrieval.md](docs/benchmark-retrieval.md) before reading anything into it.

Only one question misses Recall@5 under reranking (Q-021). BM25 misses six (Q-011, Q-012, Q-013, Q-014,
Q-019, Q-029), dense-only misses six (Q-012, Q-013, Q-018, Q-021, Q-024, Q-025) and hybrid + RRF misses
five (Q-012, Q-013, Q-018, Q-021, Q-025), which is what the reranker is buying. Full per-configuration
failure lists, the corpus fingerprint and the reproduce commands are in
[docs/benchmark-retrieval.md](docs/benchmark-retrieval.md).

### Retrieval results — historical (2-paper, 209-chunk corpus)

Run [`eval/runs/run_20261029_073045.json`](eval/runs/run_20261029_073045.json) (commit `80e9363`), measured on the 209-chunk two-paper corpus that existed at the time: 30 questions, 28 in scope, 23 answerable.

| Configuration | MRR | Recall@5 | nDCG@5 | Recall@10 |
| --- | --- | --- | --- | --- |
| BM25 only | 0.5976 | 0.7174 | 0.5549 | 0.8696 |
| Dense only | 0.6266 | 0.6522 | 0.5604 | 0.7174 |
| Hybrid + RRF | 0.7039 | 0.7174 | 0.6306 | 0.7391 |
| Hybrid + cross-encoder rerank | 0.6827 | 0.8696 | 0.6715 | 0.9130 |

**These two tables are not a before/after comparison.** Between them the corpus grew from 2 papers/209 chunks to 5 papers/409 chunks, the evaluated set grew from 23 to 25 questions (two hardcoded exclusions of the cross-paper questions Q-029/Q-030 were removed), and the parser's title extraction changed. Treat the upper table as the current measurement and this one as the archived run it replaced.

### Apparent-conflict results

Run [`eval/runs/run_20261102_150550_conflicts.json`](eval/runs/run_20261102_150550_conflicts.json) (and the earlier [`run_20261029_072737_conflicts.json`](eval/runs/run_20261029_072737_conflicts.json)): of 5 conflict questions, 4 are evaluable and 1 is blocked. On those 4, retrieval, citation verification, and conflict surfacing each succeeded 4/4, with `overall_grounded_success_rate = 1.0`. The 2026-10-29 run predates the verbatim-quote precheck and the both-sides-grounding filter, so it is a floor to improve on rather than a current score. The blocked case is Q-030, a cross-paper question: both papers are now indexed and both of its gold spans verify, but this runner answers every question through a scripted response and no scripted response exists for a two-paper conflict, so it is reported as blocked rather than scored.

That 4/4 measures **surfacing**, not **classification**, and the runner now says so on its own. A fifth field, reported separately instead of folded into the rates above, checks whether the relation written to the ledger is the one the case deserves:

| Dimension | Current scripted run | Reading |
| --- | --- | --- |
| Retrieval of the gold spans | 4/4 | the passages are found |
| Citation verification (3 layers) | 4/4 | every quote is verbatim and entailed |
| Conflict surfaced in the answer | 4/4 | the answer says "sources disagree" and names both chunks |
| Relation label matches the case | **0/4** | the fixture names a conflict pair without a `conflict_type` or reason, so the pipeline declines to assert any relation |
| Unearned `contradicts` labels | **0** | nothing was hardened into a contradiction |

Look at the fourth row rather than the first three. The scripted fixtures omit the reason on purpose, and the pipeline's rule — a conflict label without a documented reason is not good enough — demotes all of them, which is why the ledger ends up with one `unresolved` comparison (Q-015) and no comparison at all in the other three, where only one side of the pair backs a verified claim. The failure mode this dimension watches for is the opposite one: an unearned `contradicts` reaching the ledger. Relation classification is asserted where it can be controlled, in `backend/tests/test_relation_taxonomy.py`, against explicitly synthetic fixtures with reserved placeholder document ids.

The five evidence categories the product distinguishes, and where each is enforced:

| Category | Ledger outcome | Proven by |
| --- | --- | --- |
| Genuine contradiction | `contradicts`, both sides grounded, reason required | `test_relation_taxonomy.py` (synthetic) |
| Apparent conflict (differing setting, metric, or method) | `apparent_conflict` with the reason attached | `test_relation_taxonomy.py`, `test_day4_cross_paper.py` |
| Compatible findings | `supports`, and no disagreement note invented | `test_relation_taxonomy.py` |
| One-sided evidence | answer-level "sources disagree" note, **never** a ledger edge | `test_conflict_with_one_uncited_side_is_noted_but_not_promoted_to_a_comparison` |
| Insufficient evidence | abstention with zero claims, zero comparisons | `test_relation_taxonomy.py`, `test_unsupported_comparison_incomplete_retrieval_abstains` |

That runner is fully offline and deterministic — each question is answered through a scripted `MockLLMProvider` response — so what it establishes is that the grounding pipeline honours a conflict that has been handed to it. It does not establish that the system *finds* contradictions. Its Q-005 case is labelled `real_contradiction`, but it is a hand-written single-paper pair, not a cross-paper contradiction discovered in the corpus, and the pipeline records no relation for it. [docs/benchmark-contradictions.md](docs/benchmark-contradictions.md) records the separate search for a genuine one: across all 409 chunks, 302 opposition-bearing sentences and 30 cross-document candidate pairs were mined and **none survived the criteria for a genuine contradiction**, so no verified positive case exists in this corpus and contradiction detection is unproven rather than demonstrated.

### Reproducing

```powershell
.\.venv\Scripts\python.exe -m backend.app.eval.verify_gold      # do gold spans still match the index?
.\.venv\Scripts\python.exe -m backend.app.eval.run_retrieval    # 4 configs → eval/runs/<timestamp>.json
.\.venv\Scripts\python.exe -m backend.app.eval.run_conflicts    # conflict grounding → eval/runs/*_conflicts.json
.\.venv\Scripts\python.exe -m backend.app.eval.review_worksheet # regenerate the reviewer worksheet
.\.venv\Scripts\python.exe -m backend.app.eval.review_worksheet --check  # audit label integrity
```

`run_retrieval.py` and `run_conflicts.py` are offline: they exercise retrieval and grounding logic without sending anything to a provider.

---

## Tests and verification status

| Check | Result |
| --- | --- |
| Backend suite (`pytest backend/tests`) | **401 passed**, 0 failed, 0 skipped, 0 warnings, 71 s |
| Frontend suite (`npm run test` — Vitest) | **155 passed** across 9 files |
| Type check (`npm run typecheck`) | clean, exit 0 |
| Production build (`npm run build`) | succeeds; route `/` 32.5 kB, First Load JS 120 kB |
| Retrieval benchmark re-run (offline, commit `2f7cdad`) | `eval/runs/run_20261102_154437.json` — metric blocks and per-configuration failure lists identical to the documented `run_20261102_132940.json` |
| Conflict benchmark re-run (offline, scripted, commit `2f7cdad`) | `eval/runs/run_20261102_154550_conflicts.json` — summary and per-question records identical to `run_20261102_150550_conflicts.json`; Q-030 still `blocked`, not scored |
| Gold-span integrity (`python -m backend.app.eval.verify_gold`) | 34/34 gold spans resolve inside their cited page's stored chunks; 25 answerable questions matched, 5 reserved unanswerable, 0 orphan questions, 0 unmatched gold |
| Label integrity (`python -m backend.app.eval.review_worksheet --check`) | 34 gold annotations, 0 human-verified, 0 integrity errors |
| Browser walkthrough of all five destinations | performed against a running backend, including chunk search, document selection, loading, empty and no-match states, abstention, backend-error and upload validation; no console errors |
| Live end-to-end questions | performed for single-paper factual, multi-paper comparative, cross-paper conflict, and unanswerable prompts; see caveats below |

Backend coverage stood at 366 tests after the 2026-11-02 hardening pass (two cross-paper conflict-ledger cases, four parser cases covering figure-only warnings and title extraction, two storage-resilience cases). It is 401 now; +17 of that growth is the rewritten review-worksheet module (19 → 36 cases, covering span classification verdicts, caution generation, label integrity and the CLI), and the review-worksheet file is the only test module that changed.

The tests are offline — no test makes a live provider call, and negative tests assert that fake credentials never reach logs or responses. The six controlled live checks (single-paper, cross-paper, claimed contradiction, apparent conflict, abstention, and an independent quotation re-check) are recorded in [docs/validation-report.md](docs/validation-report.md).

---

## Known limitations

- **Figure graphics are not read.** Vector plots and raster figures are not interpreted; their captions and surrounding text are indexed and searchable. A paper whose result only exists inside a plotted curve cannot be answered from it.
- **Gold labels are unverified.** See the provenance table above; the accuracy numbers are pipeline validation, not human-graded accuracy.
- **Small benchmark.** n = 25 answerable questions overall, 17 in the researcher-drafted subset, with correspondingly wide confidence intervals.
- **No genuine contradiction has been found in this corpus.** Mining all 409 indexed chunks produced 30 cross-document candidate pairs, none of which met the criteria for a genuine contradiction ([docs/benchmark-contradictions.md](docs/benchmark-contradictions.md)). The apparent-conflict and abstention behaviour is verified; contradiction *detection* is therefore unproven rather than demonstrated, and the five-paper corpus is too narrow a base to conclude anything about it.
- **Live answers depend on a paid-tier provider for sustained use.** On a free tier, request caps will surface as explicit provider failures; the system abstains instead of answering unverified.
- **Research history is browser-local.** Saved runs live in `localStorage` on that machine and profile; there is no server-side history, sync, or export yet.
- **Single-user JSON storage.** No authentication, no concurrency control, no queue; indexing runs synchronously inside the upload request.
- **Stored titles predate the parser fix.** Four of the five indexed records already carry a correct title. One — LoRA `doc_e9a0d3128767` — stores `L RA: L -R A L L - M`, because the old largest-font heuristic collected single glyphs from a letter-spaced small-caps title. The frontend corrects it through a one-entry display table (`frontend/src/lib/display.ts`) that every title-bearing view resolves through, and falls back to the filename for an unknown garbled record rather than inventing a title. Re-ingesting is **not** recommended: the corrected extractor now yields `LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS` (all-caps, worse to read than the table's title-case value), all five records are already at chunker `0.2.3`, and the only other effect would be rewriting five indexed documents and rebuilding the FAISS index for a cosmetic field. [docs/limitations.md](docs/limitations.md) records the assessment, including the one non-cosmetic consequence (the cross-paper misattribution guard cannot see "LoRA") and why it fails safe.
- **Benchmark coverage is narrower than the corpus.** All 30 questions target LEMS (`doc_c67573ef50cc`); only Q-029 and Q-030 also involve GFWSVD. Attention, LoRA and DoRA have no gold annotations at all, so the retrieval figures describe two papers, and 32 of the 34 gold spans sit in one of them. The product handles a five-paper corpus; the *measurement* does not cover it.
- **Methodological claims are heuristic.** Section-heading detection and table/figure classification are rules tuned on the papers in this corpus, not a general PDF-layout solution.

---

## Demo walkthrough (3–5 minutes)

1. **Open the workspace** (30 s). Dashboard shows five papers, 117 pages, 409 chunks and a green backend health chip — the corpus is local and inspectable, not a hidden vector store.
2. **Show the ledger is real** (45 s). Research Library → LEMS → *Inspect Chunks*. Open the page-6 chunk whose text begins "Table 1. Comparison of LEMS against state-of-the-art search baselines" and point out that the table is stored as its own passage, while reference lists are tagged separately and excluded from answers.
3. **Ask a single-paper factual question** (60 s). Select only LEMS, set top-k 8, ask: *"What wall-clock inference speedup does LEMS report at 80% compression, and on what hardware?"* Open the Evidence Ledger: the claim is verified across existence, verbatim quotation and entailment, with the page-26 passage quoted.
4. **Show honest abstention** (45 s). Same selection, ask something the paper does not contain (e.g. a dollar compute cost). The answer is a refusal with the retrieval diagnostics that justify it — the ledger states there is nothing verified to display.
5. **Show a cross-paper comparison** (60 s). Select LEMS and GFWSVD, ask: *"Do LEMS and GFWSVD describe the Kronecker-factor estimation process in the same way?"* Cross-Paper Comparisons shows the relationship between the two claims with the quoted passage behind **each side**, classified `apparent_conflict` or `unresolved` rather than "the papers disagree".
6. **Contrast contradiction with apparent conflict** (45 s). Point at the four relation labels on the comparison card and explain the rule behind them: a methods, dataset, metric or setting difference is an `apparent_conflict` with the reason attached, and a relationship that the evidence does not settle stays `unresolved` — the system will not call two papers contradictory because their numbers differ. On the corpus indexed here no genuine cross-paper contradiction has been observed live, so this step is about the conservative classification, not about a demonstrated disagreement.
7. **Close on provenance** (30 s). Reload the page, reopen the run from research history — the view labels it as reopened from a saved run. Then open `eval/review_worksheet.md` and say plainly: 34 gold spans, 0 human-verified, and here is the checklist that would change that.

Steps 3–5 need a working `GEMINI_API_KEY` and take 20–30 s per question; steps 1–2, 6 and 7 work with the backend offline. Keep a saved run open as a fallback: if the provider refuses a request, the app shows an explicit failure, and the reopened run demonstrates the same workflow.

---

## Repository layout

```text
backend/app/
  api/          FastAPI routes: documents, search, answer, health
  claims/       claim decomposition, verbatim-quote precheck, conflict analysis
  citations/    three-layer citation verification
  core/         settings loaded from .env
  ingestion/    PyMuPDF parser + section-aware chunker (v0.2.3)
  llm/          LLMProvider interface and Gemini implementation
  retrieval/    BM25, FAISS dense, RRF fusion, cross-encoder rerank
  eval/         metrics, gold verification, retrieval/conflict runners, review worksheet
  schemas/      Pydantic v2 contracts: Document, Chunk, Claim, EvidenceLink, Answer…
frontend/
  app/, src/components/, src/lib/    workspace UI and API client
  tests/                             155 component and logic tests
eval/           questions, gold spans, labelling protocol, reviewer guide, run artifacts
docs/           architecture.md, decisions.md (ADRs), limitations.md, portfolio.md,
                benchmark-retrieval.md, benchmark-contradictions.md,
                supplementary-corpus-shortlist.md, security-key-review.md,
                validation-report.md
data/           indexed documents and chunks (gitignored)
```

## Design rules kept during development

Contracts before implementations; unsupported claims flagged rather than dropped; contradictions documented with a reason rather than asserted; no Redis/Celery/Postgres until the MVP proves it needs them; and no number in this README that does not come from a run artifact in `eval/runs/` or a command output above.

Further detail: [docs/architecture.md](docs/architecture.md), [docs/decisions.md](docs/decisions.md), [docs/limitations.md](docs/limitations.md), [docs/benchmark-retrieval.md](docs/benchmark-retrieval.md), [docs/benchmark-contradictions.md](docs/benchmark-contradictions.md), [docs/supplementary-corpus-shortlist.md](docs/supplementary-corpus-shortlist.md), [docs/security-key-review.md](docs/security-key-review.md), [docs/portfolio.md](docs/portfolio.md), [docs/validation-report.md](docs/validation-report.md), [eval/LABELLING.md](eval/LABELLING.md), [eval/HUMAN-REVIEW-GUIDE.md](eval/HUMAN-REVIEW-GUIDE.md).
