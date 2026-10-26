# PaperTrail AI 📑🔬

> **Multi-Paper Research Evidence Workspace**  
> Differentiator: A claim-to-evidence ledger across multiple papers, apparent-conflict handling, and a two-step citation validator (existence, then support), backed by an empirical benchmark.

---

## 💡 Overview

PaperTrail AI is **not** a generic "chat with PDFs" wrapper. It is built for researchers who require grounded, verifiable synthesis across multiple academic papers with rigorous evidence tracking.

### Core Differentiators
1. **Claim-to-Evidence Ledger**: Every sentence generated maps directly to atomic claims linked to exact chunk IDs in source documents.
2. **Apparent-Conflict Handling**: When papers disagree, claims are flagged as *apparent conflicts* annotated with causal reasons (different datasets, metrics, sample sizes, or test conditions)—never asserting unqualified contradictions.
3. **Two-Step Citation Validator**: Validates citations in two distinct phases:
   - **Step 1 (Existence)**: Confirms the cited chunk ID exists in the corpus and retrieved context.
   - **Step 2 (Support)**: Evaluates whether the cited chunk semantically supports, weakens, contextualizes, or is insufficient for the claim.
4. **Honest Abstention**: When retrieved evidence is insufficient or unresolvable, the system explicitly abstains rather than hallucinating.
5. **Empirical Evaluation**: Validated against an offline annotated benchmark measuring retrieval metrics (Recall@k, MRR, nDCG), grounding fidelity, and conflict accuracy.

---

## 📅 Milestones & Deliverables

### Day 1: Scaffolding, Data Contracts, Ingestion & Benchmark (`day1-scaffold`)
- [x] **Repository Scaffolding**: Standardized package layout (`backend/app/{api,core,ingestion,retrieval,claims,citations,llm,eval,schemas}`, `backend/tests/`, `eval/`, `docs/`, `frontend/`), `.gitignore`, `.env.example`, and `pyproject.toml`.
- [x] **Core Data Contracts (Pydantic v2)**: `Document`, `Chunk`, `Claim`, `EvidenceLink`, `CitationCheck`, `Answer`, `HealthResponse`, `DocumentUploadResponse`.
- [x] **PDF Ingestion & Health Endpoints**: `GET /health`, `POST /documents`.
- [x] **Evaluation Benchmark Scaffold**: `eval/benchmark_schema.json`, 10 draft questions in `eval/questions.jsonl`, 10 draft gold evidence records in `eval/gold_evidence.jsonl`.
- [x] **Architecture & Design Documentation**: `docs/architecture.md`, `docs/decisions.md` (ADR 001–005), `docs/limitations.md`.

### Day 2: Retrieval Baseline (`day2-retrieval`)
- [x] **Passage-Level Chunker**:
  - Merges consecutive blocks within the same page and section into passages of ~800–1500 characters (~200–400 tokens).
  - Enforces strict section heading isolation (never merges across section boundaries).
  - Keeps tables isolated as distinct chunks with length capping (2500 characters).
  - Added `ChunkKind.REFERENCES` tagging for all content following a "References" section header. Excludes references from retrieval queries by default.
  - Reduces fragmentation from ~28 fragments/page (~963 chunks for a 34-page paper) down to ~80–200 cohesive passages.
- [x] **JSON Storage Persistence**:
  - Documents and chunks persist to `data/` as JSON (`data/` is strictly gitignored).
  - SHA-256 deduplication on upload: re-uploading an existing PDF returns the stored document without duplicate indexing.
  - Inspection endpoints: `GET /documents` and `GET /documents/{doc_id}/chunks`.
- [x] **Hybrid Retrieval & Reranking (`backend/app/retrieval`)**:
  - **BM25 Index**: Lexical search using `rank-bm25`.
  - **Dense FAISS Index**: Vector similarity search using `sentence-transformers` (`all-MiniLM-L6-v2`) and FAISS CPU (`IndexFlatIP` with L2-normalized embeddings).
  - **Reciprocal Rank Fusion (RRF)**: Merges BM25 and dense candidate rankings via $RRF(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$.
  - **Cross-Encoder Reranker**: Fine-grained query-chunk cross-attention reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - **Search Endpoint**: `POST /search` returning scored chunks with per-stage scores (`bm25`, `dense`, `fused`, `rerank`).
- [x] **Durable Gold-Label Benchmark**:
  - Redesigned gold labels to be invariant to chunker boundaries: ground-truth evidence is defined as `(doc_sha256, page, answer_span text)` with case-insensitive whitespace-normalized substring matching.
  - Updated `eval/benchmark_schema.json` and all 10 draft questions in `eval/gold_evidence.jsonl`.
- [x] **Retrieval Evaluation Runner (`backend/app/eval/run_retrieval.py`)**:
  - Evaluates 4 configurations: `bm25`, `dense`, `hybrid_rrf`, `hybrid_rerank`.
  - Computes Recall@k, Precision@k, MRR, nDCG@k broken down by question type (`FACTUAL`, `COMPARATIVE`, `APPARENT_CONFLICT`, `UNANSWERABLE`, `METHODOLOGICAL`).
  - Automatically exports timestamped provenance run artifacts to `eval/runs/<timestamp>.json`.
- [x] **Comprehensive Test Suite**: 56 tests passing (100% clean test execution across schemas, storage, chunker, parser, noise filters, boundary exits, retrieval, and metrics).
- [x] **Chunker Defect Hardening (Real 34-Page arXiv Paper Validation)**:
  - References boundary exit detection at top-level headings (`Appendix`, `Supplement`, `A. Title`) with bibliography content validation. References strictly isolated to pages 9–11 (9 chunks).
  - Multi-line heading detection within blocks (`N. Title`, `N.N. Title`, `A. Title`, `B.N.N. TITLE`) and forward section inheritance. Verified: `B.7. Inference Acceleration` cleanly identified with 2.01× speedup text; `5.2. SVD-Compression Results` isolated from 5.1.
  - Page noise filtering: running headers ($\ge 50\%$ page repetition), standalone page numbers, and rotated arXiv metadata sidebars dropped.
  - Caption integrity: Figure/Table captions preserved across sentence boundaries; fragments $< 150$ chars merged; equation debris (`| {z }`, `Shape`, `Scale`) dropped.
  - High-precision table identification: requires caption context and $> 50\%$ numeric token density.
  - Dehyphenation and symmetric whitespace/hyphen normalization in gold-span matcher.
  - Version-aware re-ingestion (`chunker_version="0.2.0"`): invalidates stale chunks on re-upload.
  - Yields 130 clean, rich chunks for the 34-page paper (100 text, 12 figure_ref, 9 table, 9 references).

---

## 📐 Architecture Pipeline

```text
[PDF Upload]
     │
     ▼
[Parser (PyMuPDF)] ──(page/section aware, extraction failure recording)
     │
     ▼
[Chunker] ──(~800-1500 char passages, section isolation, references tagged)
     │
     ▼
[JSON Persistence (data/)] ──(SHA-256 deduplication)
     │
     ▼
[Hybrid Retrieval Index]
  ├── Dense Index (FAISS CPU + sentence-transformers all-MiniLM-L6-v2)
  └── Lexical Index (rank-bm25)
     │
     ▼
[Reciprocal Rank Fusion (RRF)] ──(k=60 fusion of candidate ranks)
     │
     ▼
[Cross-Encoder Reranker] ──(cross-encoder/ms-marco-MiniLM-L-6-v2)
     │
     ▼
[POST /search API / Downstream LLM Grounding]
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, FastAPI, Pydantic v2, PyMuPDF (fitz), sentence-transformers (`all-MiniLM-L6-v2`), `cross-encoder/ms-marco-MiniLM-L-6-v2`, rank-bm25, FAISS CPU, pytest.
- **LLM**: Gemini via an abstract `LLMProvider` interface (model configured from `GEMINI_MODEL`, deferred to Day 3).
- **Frontend**: Next.js, TypeScript, Tailwind CSS (deferred to Day 4).
- **Evaluation**: Custom offline retrieval benchmark runner saving to `eval/runs/`.

---

## 🚀 Exact Execution Commands

Run all commands from the repository root:

### 1. Environment Setup
```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Run Test Suite
```powershell
# Run full automated test suite (52 tests across chunker, parser, storage, retrieval, metrics, api, schemas)
.\.venv\Scripts\pytest.exe backend/tests -v
```

### 3. Run Retrieval Evaluation Benchmark
```powershell
# Executes retrieval benchmark across all 4 configurations and saves run artifact to eval/runs/
.\.venv\Scripts\python.exe -m backend.app.eval.run_retrieval
```

### 4. Run Backend API Server
```powershell
# Start FastAPI application
.\.venv\Scripts\uvicorn.exe backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive API Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Search API: `http://localhost:8000/search`

---

## 📊 Day 2 Empirical Benchmark Results

Evaluated on `eval/questions.jsonl` (10 draft questions across 5 question types) against the benchmark corpus alongside real arXiv document chunks.  
Run Artifact: [`eval/runs/run_20261026_080938.json`](file:///eval/runs/run_20261026_080938.json)

### Overall Retrieval Metrics (10 Questions, Corpus = 190 chunks)

| Configuration | Recall@1 | Recall@5 | MRR | nDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| **BM25-only** | 0.6000 | 0.8000 | 0.7500 | 0.7508 |
| **Dense-only** (`all-MiniLM-L6-v2`) | **0.7000** | **0.8000** | **0.8000** | **0.7920** |
| **Hybrid (RRF, k=60)** | 0.6000 | 0.8000 | 0.7500 | 0.7551 |
| **Hybrid + Cross-Encoder Rerank** | **0.7000** | **0.8000** | **0.8000** | **0.7920** |

### Breakdown by Question Type (nDCG@5)

| Question Type (Count) | BM25 | Dense | Hybrid RRF | Hybrid + Rerank |
| :--- | :---: | :---: | :---: | :---: |
| **FACTUAL** (2 questions) | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **COMPARATIVE** (2 questions) | 0.4386 | 0.4599 | 0.4599 | **0.4599** |
| **APPARENT_CONFLICT** (2 questions) | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| **UNANSWERABLE** (2 questions) | 0.5000 | 0.5000 | 0.5000 | **0.5000** |
| **METHODOLOGICAL** (2 questions) | 0.6309 | **1.0000** | 0.6309 | **1.0000** |

*Key Takeaway*: Dense embedding and cross-encoder reranking achieve top performance with MRR = 0.8000 and nDCG@5 = 0.7920 across 190 corpus chunks. Cross-encoder reranking specifically rescues methodological queries where BM25 achieved only 0.6309 nDCG@5 to a perfect 1.0000. All metrics are recorded directly from `eval/runs/run_20261026_080938.json` (zero invented metrics).

---

## 📜 Development Rules
1. **Contracts First**: All schemas must be defined and tested before implementations.
2. **Transparent Grounding**: Unsupported claims are explicitly flagged, never silently dropped.
3. **Apparent Conflicts**: Contradictions are documented with reasons (dataset, metric, test setup).
4. **No Premature Complexity**: No Redis, Celery, or Postgres until MVP validation is complete.
5. **No Invented Metrics**: All numbers come strictly from empirical runs logged to `eval/runs/`.
