"""
Human review worksheet for the benchmark dataset.

The questions and gold spans in eval/ are drafted annotations, not verified ground
truth: some are researcher-drafted (``human_draft``) and some are assistant-drafted
(``ai_draft``). Until an item is signed off against the PDF, metrics computed against
them are pipeline-validation numbers, not human-verified accuracy. This tool
makes the missing step mechanical: it writes one
checklist per gold item, pre-resolved against the corpus that is actually stored in
data/, so a reviewer can confirm or reject each annotation while reading the PDF.

It never edits eval/questions.jsonl or eval/gold_evidence.jsonl, and it never promotes
a label on its own.

Usage:
    python -m backend.app.eval.review_worksheet              # write eval/review_worksheet.md
    python -m backend.app.eval.review_worksheet --check      # audit labels only
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.app.schemas.document import Document
from backend.app.schemas.eval import (
    BenchmarkQuestion,
    BenchmarkStatus,
    GoldEvidence,
    is_chunk_match,
)

REVIEW_PROTOCOL_VERSION = "1.0"

UNVERIFIED_STATUSES = {
    BenchmarkStatus.AI_DRAFT.value,
    BenchmarkStatus.DRAFT.value,
    BenchmarkStatus.HUMAN_DRAFT.value,
    BenchmarkStatus.PENDING_SECOND_PAPER.value,
}


def load_jsonl(path: Path, model):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(model(**json.loads(line)))
    return items


def resolve_gold_chunk_ids(
    gold: GoldEvidence,
    sha256_to_doc: Dict[str, Document],
    chunks_by_doc: Dict[str, list],
) -> Tuple[Optional[str], List[str]]:
    """Return (doc_id, matching chunk ids) for one gold annotation against the stored corpus."""
    doc = sha256_to_doc.get((gold.doc_sha256 or "").lower())
    if doc is None:
        return None, []
    matches = [
        chunk.chunk_id
        for chunk in chunks_by_doc.get(doc.doc_id, [])
        if is_chunk_match(chunk.page, chunk.text, gold.page, gold.answer_span)
    ]
    return doc.doc_id, matches


def is_human_verified(item) -> bool:
    return getattr(item.status, "value", item.status) == BenchmarkStatus.HUMAN_VERIFIED.value


def verification_error(item) -> Optional[str]:
    """A verified label is only trustworthy if a named human claimed it."""
    if not is_human_verified(item):
        return None
    handle = (item.labelled_by or "").strip()
    if not handle.lower().startswith("human:"):
        return (
            f"{item.gold_id if hasattr(item, 'gold_id') else item.question_id} is marked "
            f"human_verified but labelled_by={handle!r} is not a human handle"
        )
    return None


def build_worksheet(
    questions: List[BenchmarkQuestion],
    gold_items: List[GoldEvidence],
    documents: List[Document],
    chunks_by_doc: Dict[str, list],
    generated_on: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Render the reviewer worksheet and return (markdown, label_integrity_errors)."""
    sha256_to_doc = {d.sha256.lower(): d for d in documents}
    gold_by_question: Dict[str, List[GoldEvidence]] = {}
    for g in gold_items:
        gold_by_question.setdefault(g.question_id, []).append(g)

    errors: List[str] = []
    for item in list(gold_items) + list(questions):
        err = verification_error(item)
        if err:
            errors.append(err)

    verified = sum(1 for g in gold_items if is_human_verified(g))
    unverified = len(gold_items) - verified
    gold_statuses = Counter(
        getattr(g.status, "value", g.status) for g in gold_items
    )
    researcher_draft = gold_statuses.get(BenchmarkStatus.HUMAN_DRAFT.value, 0)
    ai_draft = gold_statuses.get(BenchmarkStatus.AI_DRAFT.value, 0)
    other_draft = unverified - researcher_draft - ai_draft
    draft_mix = ", ".join(
        f"{count} {label}"
        for count, label in (
            (researcher_draft, "researcher-drafted"),
            (ai_draft, "assistant-drafted"),
            (other_draft, "other unverified"),
        )
        if count
    )

    lines: List[str] = []
    lines.append("# Benchmark human review worksheet")
    lines.append("")
    lines.append(f"Protocol version: `{REVIEW_PROTOCOL_VERSION}` · Generated: {generated_on or date.today().isoformat()}")
    lines.append("")
    lines.append(
        "An annotation counts as ground truth only after a person has opened the PDF and "
        "signed off on it. Fill one block per gold item, then copy the verdicts back into "
        "`eval/gold_evidence.jsonl` (`status`, `labelled_by`, `annotator_notes`). "
        "This worksheet is generated output — do not treat it as the dataset."
    )
    lines.append("")
    lines.append("## How to complete an item")
    lines.append("")
    lines.append("1. Open the paper at the recorded page and find the quoted span.")
    lines.append("2. Confirm the span is verbatim, short, and answers the question.")
    lines.append("3. Confirm the relation: `supports` must entail, `apparent_conflict` must be explained by methods, data, metric or setting differences — not by a factual disagreement.")
    lines.append("4. Record your handle as `human:<handle>`; otherwise leave the item's draft status untouched.")
    lines.append("")
    lines.append("## Dataset status")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("| --- | --- |")
    lines.append(f"| Questions | {len(questions)} |")
    lines.append(f"| Gold annotations | {len(gold_items)} |")
    lines.append(f"| Human-verified gold annotations | {verified} |")
    lines.append(f"| Awaiting human review | {unverified} |")
    lines.append(f"| — of which researcher-drafted (`human_draft`) | {researcher_draft} |")
    lines.append(f"| — of which assistant-drafted (`ai_draft`) | {ai_draft} |")
    if other_draft:
        lines.append(f"| — of which other unverified status | {other_draft} |")
    lines.append(f"| Documents available in `data/` | {len(documents)} |")
    lines.append("")
    if verified == 0:
        lines.append(
            "> **No annotation has been human-verified yet.** Any accuracy number quoted "
            f"from `eval/runs/` is a pipeline-validation measurement against draft gold "
            f"labels ({draft_mix}), not human-graded accuracy."
        )
        lines.append("")

    lines.append("## Review items")
    lines.append("")

    for q in questions:
        q_gold = gold_by_question.get(q.question_id, [])
        lines.append(f"### {q.question_id} — {q.question}")
        lines.append("")
        lines.append(f"- Type: `{getattr(q.question_type, 'value', q.question_type)}`")
        lines.append(f"- Question status: `{getattr(q.status, 'value', q.status)}` (labelled_by: {q.labelled_by})")
        lines.append(f"- Target documents: {', '.join(q.target_doc_ids) if q.target_doc_ids else 'not recorded'}")
        lines.append("")
        if not q_gold:
            lines.append(
                "No gold evidence is recorded for this question. For an unanswerable query, "
                "confirm the corpus genuinely lacks the information, then write that down."
            )
            lines.append("")
            continue

        for g in q_gold:
            doc_id, matches = resolve_gold_chunk_ids(g, sha256_to_doc, chunks_by_doc)
            lines.append(f"#### {g.gold_id} · page {g.page} · `{getattr(g.expected_relation, 'value', g.expected_relation)}`")
            lines.append("")
            lines.append(f"- Recorded span: “{g.answer_span}”")
            lines.append(f"- Annotator note: {g.annotator_notes or '—'}")
            if g.conflict_reason:
                lines.append(f"- Conflict reason: {g.conflict_reason}")
            lines.append(
                f"- Current label: `{getattr(g.status, 'value', g.status)}` "
                f"(labelled_by: {g.labelled_by})"
            )
            if doc_id is None:
                lines.append(
                    f"- Corpus check: **no indexed document matches sha256 {g.doc_sha256[:16]}…** "
                    "— the source PDF is not in `data/`; re-upload it before reviewing."
                )
            elif not matches:
                lines.append(
                    f"- Corpus check: **the span was not found in {doc_id} on page {g.page}** "
                    "— either the page, the wording, or the chunking is wrong. Flag it."
                )
            else:
                lines.append(
                    f"- Corpus check: found in {doc_id} → {', '.join(sorted(matches)[:5])}"
                )
            lines.append("")
            lines.append("- [ ] Page and span are verbatim in the PDF")
            lines.append("- [ ] The span actually answers the question")
            lines.append(
                "- [ ] Relation is correct "
                "(a methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`)"
            )
            lines.append("- [ ] Rejected — record why: …")
            lines.append("")
            lines.append("```text")
            lines.append("reviewer handle : human:")
            lines.append("review date     :")
            lines.append("observations    :")
            lines.append("```")
            lines.append("")

    return "\n".join(lines) + "\n", errors


def _load_corpus() -> Tuple[List[Document], Dict[str, list]]:
    from backend.app.core.storage import storage

    documents = storage.list_documents()
    return documents, {d.doc_id: storage.get_chunks(d.doc_id) for d in documents}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the benchmark human-review worksheet.")
    parser.add_argument("--eval-dir", default="eval", help="Directory holding questions.jsonl and gold_evidence.jsonl")
    parser.add_argument("--out", default=None, help="Output path (default: <eval-dir>/review_worksheet.md)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only audit labels: fail if anything is marked human_verified without a human:<handle>",
    )
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    questions_path = eval_dir / "questions.jsonl"
    gold_path = eval_dir / "gold_evidence.jsonl"
    if not questions_path.exists() or not gold_path.exists():
        print(f"Missing benchmark files: {questions_path} and/or {gold_path}")
        return 1

    questions = load_jsonl(questions_path, BenchmarkQuestion)
    gold_items = load_jsonl(gold_path, GoldEvidence)
    documents, chunks_by_doc = _load_corpus()

    markdown, errors = build_worksheet(questions, gold_items, documents, chunks_by_doc)

    for err in errors:
        print(f"LABEL ERROR: {err}")

    if args.check:
        verified = sum(1 for g in gold_items if is_human_verified(g))
        print(
            f"{len(gold_items)} gold annotations, {verified} human-verified, "
            f"{len(errors)} integrity error(s)."
        )
        return 1 if errors else 0

    out_path = Path(args.out) if args.out else eval_dir / "review_worksheet.md"
    out_path.write_text(markdown, encoding="utf-8")
    unreviewed = sum(1 for g in gold_items if not is_human_verified(g))
    print(f"Wrote {out_path} ({len(gold_items)} gold items, {unreviewed} awaiting human review).")
    if errors:
        print(f"{len(errors)} label integrity error(s) reported above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
