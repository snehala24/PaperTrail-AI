# Benchmark Ground-Truth Labelling Protocol & Provenance

## 1. Provenance Notice & Status
> [!WARNING]
> **Draft Benchmark (Not Human-Verified)**  
> Every item in `eval/questions.jsonl` and `eval/gold_evidence.jsonl` is a **draft**, not verified ground truth. None carries `status: "human_verified"`.

| File | Items | `human_draft` (labelled_by `snehala24`) | `ai_draft` (labelled_by `assistant`) | `human_verified` |
| --- | :---: | :---: | :---: | :---: |
| `eval/questions.jsonl` | 30 | 20 | 10 | 0 |
| `eval/gold_evidence.jsonl` | 34 | 24 | 10 | 0 |

Researcher-drafted items were written by the project owner from the papers, but "written by a human" is not the same as "signed off against the PDF page by page" — that is what `human_verified` means here, and it has not happened for any item yet.

These benchmark items serve as a pipeline validation and smoke-test suite for retrieval evaluation. They must not be cited as an authoritative human-annotated benchmark until the protocol below is completed.

### Reviewing the dataset

`backend/app/eval/review_worksheet.py` renders one checklist per gold item, pre-resolved against the corpus actually stored in `data/`:

```powershell
.\.venv\Scripts\python.exe -m backend.app.eval.review_worksheet          # writes eval/review_worksheet.md
.\.venv\Scripts\python.exe -m backend.app.eval.review_worksheet --check   # audits label integrity only
```

The generated `eval/review_worksheet.md` is an output, not the dataset. Marking a checkbox there does not change any label; the verdict still has to be written back into `eval/gold_evidence.jsonl` by hand.

---

## 2. Review Protocol for Human Verification

To graduate an annotation from `human_draft` / `ai_draft` to `human_verified`:

1. **Query Clarity & Realism**:
   - Verify that the query in `eval/questions.jsonl` represents a genuine research question that a literature researcher would formulate.
   - For `unanswerable` questions, confirm that the paper truly lacks the required information (e.g. dollar compute costs, external translation benchmarks) rather than the annotator missing a section.

2. **Verbatim Text Span Verification**:
   - Open the original PDF (e.g. `2610.32878v1.pdf`).
   - Navigate to the recorded 1-based page number.
   - Confirm that `answer_span` is a concise verbatim quote from that exact page that answers or directly addresses the question.
   - Ensure the quote is short (a single sentence or key quantitative phrase, e.g. "2.01× speedup", "5.07 GB"), not whole synthesized passages.

3. **Relation & Apparent Conflict Classification**:
   - For `supports`, verify that the cited span logically entails or provides direct empirical evidence for the query answer.
   - For `apparent_conflict` / `contradicts`, verify that the underlying evidence reflects an experimental divergence (e.g. different quantization regimes, uniform vs. non-uniform allocation) and that `conflict_reason` explains the causal divergence.

4. **Graduation to Verified Status**:
   - Update `status` in `eval/questions.jsonl` and `eval/gold_evidence.jsonl` to `"human_verified"`.
   - Update `labelled_by` to the human reviewer's identifier (e.g. `"human:<handle>"`).
   - Record review date and specific observations in `annotator_notes`.

---

## 3. Multi-Document Questions
For questions spanning multiple papers:
- Use `target_doc_ids` to record all relevant document IDs.
- For each piece of cross-paper evidence, create a separate entry in `eval/gold_evidence.jsonl` pointing to the respective document's `doc_sha256` and page number, or specify `gold_chunk_ids`.
