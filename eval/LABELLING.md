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

Protocol `1.1` additionally classifies how each recorded span relates to the stored text — `exact`, `case only`, `hyphen or space repair`, `other page`, `absent`, `no document` — and prints a caution on every item where a careless tick would do damage (short spans, `contradicts` labels, duplicate spans, cross-paper items, conflict-typed questions whose spans are labelled `supports`). The current run reports 33 `exact` and 1 `case only`, with 15 of 34 items carrying at least one caution. The reviewer-facing instructions live in [HUMAN-REVIEW-GUIDE.md](HUMAN-REVIEW-GUIDE.md).

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
   - For a conflict-bearing relation, verify that the underlying evidence reflects an experimental divergence (e.g. different quantization regimes, uniform vs. non-uniform allocation) and that the conflict_reason field explains the causal divergence.
   - Schema caveat: the relation enum has no apparent_conflict member, so an apparent conflict is recorded as contradicts or weakens plus a reason. Review the label and the reason together; if the reason names a condition difference, the label itself is what needs deciding. Both contradicts items in the current dataset are in exactly that state, which the worksheet flags.

4. **Graduation to Verified Status**:
   - Update `status` in `eval/questions.jsonl` and `eval/gold_evidence.jsonl` to `"human_verified"`.
   - Update `labelled_by` to the human reviewer's identifier (e.g. `"human:<handle>"`).
   - Record review date and specific observations in `annotator_notes`.

---

## 3. Multi-Document Questions
For questions spanning multiple papers:
- Use `target_doc_ids` to record all relevant document IDs.
- For each piece of cross-paper evidence, create a separate entry in `eval/gold_evidence.jsonl` pointing to the respective document's `doc_sha256` and page number, or specify `gold_chunk_ids`.
