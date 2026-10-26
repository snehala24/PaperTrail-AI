# Architectural Decision Records (ADRs)

## ADR-001: Data Contracts Precede Implementation
- **Status**: Accepted (2026-10-26)
- **Context**: The core differentiator of PaperTrail AI is a verifiable claim-to-evidence ledger. Schema drift between retrieval, claim extraction, and citation validation will corrupt grounding and evaluation.
- **Decision**: Define all schemas in Pydantic v2 first with rigorous field validations (e.g. SHA-256 formatting, non-empty claims, mandatory conflict reasons on contradictions).
- **Consequences**: Implementation code is strictly decoupled and type-checked against frozen data contracts.

## ADR-002: PyMuPDF for Extraction with Explicit Diagnostics
- **Status**: Accepted (2026-10-26)
- **Context**: PDF parsers frequently fail silently on scanned pages, multi-column layouts, and mathematical formulas.
- **Decision**: Use PyMuPDF (fitz) to extract text blocks, preserve 1-indexed page boundaries, and track all extraction anomalies in `Document.extraction_warnings`.
- **Consequences**: The system never silently drops unparseable pages, surfacing diagnostics to users and evaluators.

## ADR-003: Apparent Conflicts Instead of Definitive Contradictions
- **Status**: Accepted (2026-10-26)
- **Context**: Scientific papers rarely contradict each other in a vacuum; differences arise due to differing benchmark versions, preprocessing, hardware, or hyperparameter choices.
- **Decision**: Label all conflicting evidence links as "apparent conflicts" requiring an explicit `conflict_reason`.
- **Consequences**: Researchers receive actionable context explaining *why* papers differ rather than an unhelpful binary flag.

## ADR-004: Two-Step Citation Validation (Existence Before Support)
- **Status**: Accepted (2026-10-26)
- **Context**: Generative models frequently hallucinate plausible-looking chunk IDs that do not exist, or reference valid chunk IDs that do not actually support the statement.
- **Decision**: Validate citations in two explicit phases:
  1. **Phase 1 (Existence)**: Verify that the cited `chunk_id` exists in the retrieved context.
  2. **Phase 2 (Support)**: Verify whether the chunk semantically supports, weakens, or is insufficient for the claim.
- **Consequences**: Allows independent measurement of citation hallucination rate (existence failures) vs citation misattribution rate (support failures).

## ADR-005: Lightweight Architecture for Initial Milestones
- **Status**: Accepted (2026-10-26)
- **Context**: Premature infrastructure (Redis, PostgreSQL, distributed queues) adds operational overhead during rapid prototyping and benchmark development.
- **Decision**: Use in-memory indices, FAISS CPU, BM25, and local JSONL logging for Day 1–5.
- **Consequences**: Fast feedback loops, instant test runs, and minimal deployment friction.

## ADR-006: Passage-Level Chunker with Section Isolation and References Tagging
- **Status**: Accepted (2026-10-26)
- **Context**: Raw paragraph extraction yielded ~28 fragments per page (~963 chunks for a 34-page arXiv paper). Fragmentary chunks lack semantic context, overflow dense indices, and degrade retrieval recall. Additionally, bibliographic references pollute retrieval results when searching for author names or prior work citations.
- **Decision**: 
  1. Merge consecutive text blocks within the same page and section into cohesive passages of ~800–1500 characters (~200–400 tokens).
  2. Enforce strict section boundaries: never merge text across detected section headers.
  3. Keep tables isolated as separate chunks and cap table text length (~2500 characters).
  4. Tag all content following a "References" heading as `ChunkKind.REFERENCES` and exclude reference chunks from search results by default (`include_references=False`).
- **Consequences**: Target chunk yield for a 34-page paper drops from ~963 fragmentary chunks to ~80–200 rich passages, providing coherent context to retrievers and rerankers while keeping references accessible on demand.

## ADR-007: Span-Based Gold Evidence Benchmarking
- **Status**: Accepted (2026-10-26)
- **Context**: Tying benchmark ground-truth labels directly to synthetic or temporary `chunk_id`s creates tight coupling between evaluation sets and specific chunking parameters. Modifying chunk sizes or boundary logic invalidates all existing gold labels.
- **Decision**: Define ground-truth gold evidence as a tuple of `(doc_sha256, page, answer_span text)`. A retrieved chunk is deemed relevant if and only if:
  1. It belongs to the designated document (matching `doc_sha256`) and page.
  2. Its text contains the `answer_span` under case-insensitive whitespace-normalized substring matching.
- **Consequences**: The evaluation benchmark is fully invariant to chunker settings, window sizes, and passage merging algorithms. Benchmarks remain durable across future ingestion iterations.

## ADR-008: Chunker Robustness, Boundary Detection, and Re-Ingestion Hygiene
- **Status**: Accepted (2026-10-26)
- **Context**: Inspecting real-world extraction outputs on a 34-page arXiv paper (`2610.32878v1.pdf`) revealed seven structural defects:
  1. *Oversized References Tagging*: Everything following the `References` header (pages 14–34 appendix) was tagged as `references`, erroneously hiding 77/135 chunks from retrieval.
  2. *Heading Detection Blindspots*: Headings appearing on subsequent lines within text blocks were missed, causing sub-section headings (e.g. `5.2`, `B.7`) to inherit previous section headers (e.g. `5.1`).
  3. *Page Artifact Noise*: Running headers, page numbers, and rotated arXiv metadata sidebars polluted chunk texts.
  4. *Caption and Formula Splitting*: Captions were split mid-sentence across chunks, and tiny formula fragments (`| {z }`, `Shape`, `Scale`) formed isolated chunks.
  5. *Table False Positives*: Chunks with text pipe delimiters (`|`) were misclassified as `table`.
  6. *Line-Break Hyphenation*: Split words (`pow-\nerful`) degraded lexical and span matching.
  7. *Stale Chunk Serving*: SHA-256 deduplication returned stale cached chunks even when chunker logic had been improved.
- **Decision**:
  1. *Dual-Signal References Boundary*: Terminate references mode upon encountering top-level appendix/supplement/glossary headings or lettered sections (`A. Title`, `B.1. Title`), and validate bibliography chunks via content checks (19xx/20xx years and citations markers like `doi`, `arXiv`, `Proceedings`, `et al.`).
  2. *Line-Aware Heading Detection*: Inspect each line for heading patterns (`N. Title`, `N.N. Title`, `A. Title`, `B.N.N. TITLE`), split blocks immediately on heading lines, and propagate section titles forward to subsequent chunks.
  3. *Noise Pruning*: Strip lines repeated across $\ge 50\%$ of pages (running headers), standalone page numbers (`^\d+$`), and rotated arXiv sidebar text during document parsing.
  4. *Caption Merge & Fragment Absorber*: Keep captions unified until terminal sentence punctuation. Absorb chunks $< 150$ characters into adjacent neighbors and drop isolated math notation fragments.
  5. *Table Density Rule*: Classify chunks as `table` only when accompanied by caption context and exhibiting $> 50\%$ numeric token density (excluding delimiter characters).
  6. *Dehyphenation & Normalized Matching*: Join line-break hyphens (`pow-\nerful` $\to$ `powerful`) when followed by lowercase characters, and apply symmetric hyphen/whitespace normalization during span evaluation.
  7. *Version-Aware Deduplication*: Store `chunker_version` (e.g. `0.2.0`) in document metadata; if stored documents have an older or missing version, re-parse and re-index.
- **Consequences**: A real 34-page paper yields 130 rich, clean chunks (100 text, 12 figure_ref, 9 table, 9 references restricted strictly to pages 9–11). Headings like `B.7. Inference Acceleration` and `5.2. SVD-Compression Results` are cleanly partitioned and inherited. Stale chunk caches are cleanly invalidated.
