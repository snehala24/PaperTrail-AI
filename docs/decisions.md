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
