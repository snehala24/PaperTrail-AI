# PaperTrail AI

**A research evidence workspace for reading several papers at once.** Ask a question across your indexed PDFs and get an answer where every sentence is tied to a quoted passage on a specific page of a specific paper — or an explicit refusal when the evidence does not support one.

PaperTrail AI is not a "chat with a PDF" wrapper. The unit it reasons over is the **claim**, not the document, and the unit it shows you is the **evidence link**, not a citation marker.

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
     │                both sides required to be grounded in retrieved chunks
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

No key, token, or credential value belongs in this repository. If a live key is ever pasted into a terminal or log, treat it as exposed and rotate it.

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
| `POST /search` | Hybrid retrieval with per-stage scores (`bm25`, `dense`, `fused`, `rerank`). |
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

**No gold annotation has been checked against a PDF and signed off yet.** Every accuracy figure below is therefore a *pipeline-validation* measurement against draft labels — it shows the retrieval and verification stack behaves sensibly, not that it matches expert ground truth. `eval/LABELLING.md` is the protocol; `backend/app/eval/review_worksheet.py` generates `eval/review_worksheet.md`, one checklist per gold span, pre-resolved against the stored corpus.

### Retrieval results

Run [`eval/runs/run_20261029_073045.json`](eval/runs/run_20261029_073045.json) (commit `80e9363`), measured on the 209-chunk two-paper corpus that existed at the time, 30 questions, of which 28 were in scope and 23 answerable (5 unanswerable reserved for abstention testing).

| Configuration | MRR | Recall@5 | nDCG@5 | Recall@10 |
| --- | --- | --- | --- | --- |
| BM25 only | 0.5976 | 0.7174 | 0.5549 | 0.8696 |
| Dense only | 0.6266 | 0.6522 | 0.5604 | 0.7174 |
| Hybrid + RRF | 0.7039 | 0.7174 | 0.6306 | 0.7391 |
| **Hybrid + cross-encoder rerank** | 0.6827 | **0.8696** | **0.6715** | **0.9130** |

Split by who drafted the labels, the picture is honest rather than flattering:

| Subset | Answerable n | Best configuration | MRR | Recall@5 |
| --- | :---: | --- | --- | --- |
| Researcher-drafted questions | 15 | Hybrid + rerank | 0.7005 | 0.8667 [0.667, 1.0] |
| Assistant-drafted questions | 8 | Hybrid + RRF | 0.7958 | 0.8750 [0.625, 1.0] |

Sample sizes are below 30, so the 95% bootstrap intervals are wide and the subsets are not statistically separable. The recorded failure analysis shows the pattern behind it: researcher questions are phrased the way people actually ask ("how long does the first run take once cached?"), which is harder for lexical search than paraphrases generated from the paper's own vocabulary.

### Apparent-conflict results

Run [`eval/runs/run_20261029_072737_conflicts.json`](eval/runs/run_20261029_072737_conflicts.json): of 5 conflict questions, 4 were evaluable and 1 blocked pending a second paper. On those 4, retrieval, citation verification, and conflict detection each succeeded 4/4, with `overall_grounded_success_rate = 1.0`. This was measured before the verbatim-quote precheck and the both-sides-grounding filter were added, so it is a floor to improve on rather than a current score.

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
| Backend suite (`pytest backend/tests`) | **359 passed**, 1 pre-existing `DeprecationWarning`, 44.7 s |
| Frontend suite (`npm run test` — Vitest) | **155 passed** across 9 files |
| Type check (`npm run typecheck`) | clean, exit 0 |
| Production build (`npm run build`) | succeeds; route `/` 32.5 kB, First Load JS 120 kB |
| Browser walkthrough of all five destinations | performed against a running backend, including document selection, loading, empty and no-match states, abstention, backend-error and upload validation; no console errors |
| Live end-to-end questions | performed for single-paper factual, multi-paper comparative, cross-paper conflict, and unanswerable prompts; see caveats below |

The tests are offline — no test makes a live provider call, and negative tests assert that fake credentials never reach logs or responses. The six controlled live checks (single-paper, cross-paper, claimed contradiction, apparent conflict, abstention, and an independent quotation re-check) are recorded in [docs/validation-report.md](docs/validation-report.md).

---

## Known limitations

- **Figure graphics are not read.** Vector plots and raster figures are not interpreted; their captions and surrounding text are indexed and searchable. A paper whose result only exists inside a plotted curve cannot be answered from it.
- **Gold labels are unverified.** See the provenance table above; the accuracy numbers are pipeline validation, not human-graded accuracy.
- **Small benchmark.** n < 30 answerable questions per subset, with correspondingly wide confidence intervals.
- **Three indexed PDFs are not on disk.** Attention, LoRA and DoRA are indexed (chunks in `data/`) but their source files are not present locally, so human review of their gold spans needs the paper re-opened from arXiv; re-uploading the identical file is deduplicated by SHA-256 and only re-chunks if the chunker version has changed.
- **Live answers depend on a paid-tier provider for sustained use.** On a free tier, request caps will surface as explicit provider failures; the system abstains instead of answering unverified.
- **Retrieval metrics predate the current corpus.** They were measured on 209 chunks (two papers); the indexed corpus is now 409 chunks across five, so re-running is required before quoting retrieval at the current scale.
- **Research history is browser-local.** Saved runs live in `localStorage` on that machine and profile; there is no server-side history, sync, or export yet.
- **Single-user JSON storage.** No authentication, no concurrency control, no queue; indexing runs synchronously inside the upload request.
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
eval/           questions, gold spans, labelling protocol, run artifacts
docs/           architecture.md, decisions.md (ADRs), limitations.md
data/           indexed documents and chunks (gitignored)
```

## Design rules kept during development

Contracts before implementations; unsupported claims flagged rather than dropped; contradictions documented with a reason rather than asserted; no Redis/Celery/Postgres until the MVP proves it needs them; and no number in this README that does not come from a run artifact in `eval/runs/` or a command output above.

Further detail: [docs/architecture.md](docs/architecture.md), [docs/decisions.md](docs/decisions.md), [docs/limitations.md](docs/limitations.md), [eval/LABELLING.md](eval/LABELLING.md).
