# Known Limitations & Boundary Conditions

## 1. Document Ingestion Limitations
- **Scanned / Image-Only PDFs**: PyMuPDF extracts programmatic text layers. PDFs consisting entirely of scanned bitmap images without embedded OCR text will generate warnings in `Document.extraction_warnings`. Optical Character Recognition (OCR) is intentionally deferred past the MVP.
- **Complex Multi-Column Math / Inline Equations**: LaTeX mathematical symbols in two-column IEEE/ACM layouts may occasionally be extracted in non-linear reading order if bounding box heuristics fail.

## 2. Chunking Heuristics
- **Section Detection**: Section headers are detected via regex heuristics (e.g. `1. Introduction`, `Abstract`, `Methodology`). Papers with idiosyncratic non-standard heading styles may have chunks grouped under generic headings.
- **Cross-Page Table Spans**: Large tables spanning multiple consecutive pages are currently partitioned per-page rather than merged into a single atomic table chunk.

## 3. Scope Boundaries for Day 1
- **Retrieval Indexing**: Dense (FAISS) and lexical (BM25) hybrid indices, as well as cross-encoder reranking, are scheduled for implementation in Day 2.
- **LLM Integration**: The `LLMProvider` abstraction and Gemini integration are scheduled for Day 3.
- **Claim Extraction & Evidence Linking**: Two-step citation validation and claim-to-evidence ledgers are scheduled for Day 4.
- **UI Frontend**: Next.js client is scheduled for Day 5.
