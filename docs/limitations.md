# Known Limitations & Boundary Conditions

## 1. Document Ingestion Limitations
- **Scanned / Image-Only PDFs**: PyMuPDF extracts programmatic text layers. PDFs consisting entirely of scanned bitmap images without embedded OCR text will generate warnings in `Document.extraction_warnings`. Optical Character Recognition (OCR) is intentionally deferred past the MVP.
- **Complex Multi-Column Math / Inline Equations**: LaTeX mathematical symbols in two-column IEEE/ACM layouts may occasionally be extracted in non-linear reading order if bounding box heuristics fail.

## 2. Chunking Heuristics
- **Section Detection**: Section headers are detected via regex heuristics (e.g. `1. Introduction`, `Abstract`, `Methodology`). Papers with idiosyncratic non-standard heading styles may have chunks grouped under generic headings.
- **Cross-Page Table Spans**: Large tables spanning multiple consecutive pages are currently partitioned per-page rather than merged into a single atomic table chunk.

## 3. Benchmark Limitations & Sample Size
- **AI-Draft Ground Truth**: All current benchmark questions and gold evidence spans in `eval/` are AI-generated drafts (`status: "ai_draft"`). They have not yet undergone expert human verification (see `eval/LABELLING.md` for human verification protocol).
- **Single-Paper Grounding**: Current empirical benchmark queries are grounded in a single 34-page arXiv paper (`2610.32878v1.pdf`). Cross-paper multi-document query evaluation is implemented in the schema and runner, but extensive multi-paper benchmarks will be expanded as additional papers are added.
- **Small Sample Size ($n=8$) & Wide Confidence Intervals**: With 8 answerable queries and 2 unanswerable queries, sample size is insufficient for formal statistical significance ($n < 30$). As reflected in the 95% bootstrap confidence intervals (e.g. Recall@5: [0.625, 1.0]), estimates have wide margins of error and serve primarily as a directional baseline and smoke test.

## 4. Scope Boundaries for Day 1 & Day 2
- **Day 1**: Contracts, ingestion, persistence, and evaluation scaffold completed.
- **Day 2**: Passage chunker v0.2.2, BM25, dense FAISS, RRF fusion, cross-encoder reranker, and evaluation runner completed.
- **Day 3 (Upcoming)**: LLMProvider abstraction, Gemini integration, answer generation, and honest abstention handling.
- **Day 4 (Upcoming)**: Claim extraction, claim-to-evidence ledgers, apparent conflict reasoning, and two-step citation validation.
- **Day 5 (Upcoming)**: Next.js frontend UI workspace.
