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
import re
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
    normalize_text_span,
)

REVIEW_PROTOCOL_VERSION = "1.1"

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


def classify_span(gold: GoldEvidence, chunks_by_doc: Dict[str, list], doc_id: Optional[str]) -> Tuple[str, str]:
    """
    Say *how* a recorded span relates to the stored text, not merely whether a tolerant match exists.

    A reviewer is asked to confirm a span is verbatim. Reporting "found" for a span whose capitalisation
    differs from the source invites a tick without reading, which is how an unverified label gets
    promoted by accident. The verdicts, in decreasing order of trust:

      exact                  -- the span appears character-for-character (whitespace collapsed)
      case_only              -- only letter case differs
      hyphen_or_space_repair -- only matches after line-break hyphen repair
      other_page             -- absent on the recorded page, present elsewhere in the document
      absent                 -- not in the stored text of that document
      no_document            -- the sha256 in the annotation matches no indexed paper
    """
    span = (gold.answer_span or "").strip()
    if doc_id is None:
        return "no_document", "the source PDF is not in data/ — it must be re-uploaded before this item can be reviewed"
    if not span:
        return "absent", "no span text is recorded"

    def collapse(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    page_chunks = [c for c in chunks_by_doc.get(doc_id, []) if c.page == gold.page]
    other_chunks = [c for c in chunks_by_doc.get(doc_id, []) if c.page != gold.page]

    for group, verdict_on_match, pages in ((page_chunks, "", "page"), (other_chunks, "other_page", "another page")):
        for c in group:
            raw = collapse(c.text)
            if span in raw:
                if verdict_on_match == "other_page":
                    return "other_page", (
                        f"the span appears verbatim on page {c.page}, not page {gold.page} — "
                        "the page reference is wrong even though the wording is right"
                    )
                return "exact", "matches the stored text character for character"
            if span.lower() in raw.lower():
                if verdict_on_match == "other_page":
                    return "other_page", (
                        f"a case-folded match exists on page {c.page} rather than page {gold.page}"
                    )
                return "case_only", (
                    "letter case differs from the source, so it is not verbatim as recorded — "
                    "the reviewer must decide whether the annotation should read exactly as printed"
                )
        if any(normalize_text_span(c.text) and normalize_text_span(span) in normalize_text_span(c.text) for c in group):
            if verdict_on_match == "other_page":
                return "other_page", "only a hyphen-repaired match exists, on a different page"
            return "hyphen_or_space_repair", (
                "the span only matches after line-break hyphens are repaired — confirm the PDF really "
                "prints it as one word"
            )
    return "absent", "the span is nowhere in this document's stored text — page, wording or chunking is wrong"


def _reason_reads_as_apparent(conflict_reason: Optional[str]) -> bool:
    """
    True when the annotator's own reason frames the disagreement as condition-dependent.

    The gold schema has no `apparent_conflict` member in EvidenceRelation, so apparent conflicts are
    recorded as `contradicts`/`weakens` plus a free-text reason. When that reason says the results differ
    *because a condition differs*, the label and the prose are pointing at different ledger relations.
    """
    text = (conflict_reason or "").lower()
    if not text.strip():
        return False
    if not text.startswith(("apparent", "seemingly", "superficially")):
        cues = ("because", "whereas", "while", "under different", "different condition", "instead")
        if not any(cue in text for cue in cues):
            return False
    return any(
        cue in text
        for cue in (
            "apparent",
            "whereas",
            "while",
            "because",
            "under identical",
            "different setting",
            "different condition",
            "cannot be compared",
        )
    )


def review_cautions(
    gold: GoldEvidence,
    question: Optional[BenchmarkQuestion],
    span_verdict: str,
    duplicate_of: List[str],
) -> List[str]:
    """
    Everything about this item that a reviewer needs to know before deciding, derived from the data.

    Nothing here changes a label. It exists because a checklist that hides its own hard cases produces
    confidently wrong ground truth.
    """
    notes: List[str] = []
    span = (gold.answer_span or "").strip()
    relation = getattr(gold.expected_relation, "value", gold.expected_relation)

    if span_verdict == "exact":
        pass
    elif span_verdict == "no_document":
        notes.append("source document is not indexed — this item cannot be reviewed yet")
    elif span_verdict == "absent":
        notes.append("recorded span not found in the stored text of this document")
    elif span_verdict == "other_page":
        notes.append("page reference disagrees with where the span actually occurs")
    elif span_verdict == "case_only":
        notes.append("recorded span differs from the source in letter case — not verbatim as written")
    elif span_verdict == "hyphen_or_space_repair":
        notes.append("recorded span needs hyphen/whitespace repair to match the source")

    if len(span) <= 12:
        notes.append(
            f"span is only {len(span)} characters — confirm it is the right cell/phrase and not a "
            "coincidental substring elsewhere on the page"
        )
    if relation == "contradicts":
        notes.append(
            "expected relation is `contradicts` — the hardest label in this set. Two statements contradict "
            "only if they assert opposite values for the same property under comparable conditions; "
            "otherwise the correct label is `apparent_conflict` or `unresolved`"
        )
    if relation in ("contradicts", "weakens") and not (gold.conflict_reason or "").strip():
        notes.append(
            "a conflict-bearing relation recorded without a conflict_reason — the ledger needs the "
            "differing condition in writing before it can label this `apparent_conflict`"
        )
    if relation == "contradicts" and _reason_reads_as_apparent(gold.conflict_reason):
        notes.append(
            "expected relation is `contradicts` but the recorded conflict_reason describes an apparent "
            "conflict (it says the two results differ because of a condition, not because both cannot be "
            "true) — these two fields disagree, and only the reviewer can say which one is right"
        )
    if duplicate_of:
        notes.append(
            f"identical span is also recorded by {', '.join(duplicate_of)} — one reading decides both, "
            "so record the same verdict in each block"
        )
    if question is not None:
        q_type = getattr(question.question_type, "value", question.question_type)
        if q_type == "unanswerable":
            notes.append("this question is an abstention probe; a gold span here is itself the finding — it should not exist")
        if len(question.target_doc_ids or []) >= 2:
            notes.append(
                "cross-paper question — verify that this span belongs to the document recorded here, "
                "and that a companion span exists for the other paper"
            )
        if q_type == "apparent_conflict" and relation == "supports":
            notes.append(
                "the question asks about a conflict but this span is labelled `supports`: that can be "
                "correct (the span grounds one side of the answer), and the reviewer should say which reading applies"
            )
    return notes


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
    lines.append(
        "4. Note the schema: `expected_relation` has no `apparent_conflict` member, so an apparent conflict is "
        "recorded as `contradicts` or `weakens` plus a `conflict_reason`. Judge the label and the reason together."
    )
    lines.append("5. Record your handle as `human:<handle>`; otherwise leave the item's draft status untouched.")
    lines.append(
        "6. A verdict of \"the span is right but the label is wrong\" is still a review finding — write it in "
        "`observations`; do not silently edit the JSONL."
    )
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

    q_by_id = {q.question_id: q for q in questions}
    span_keys: Dict[Tuple[str, int, str], List[str]] = {}
    for g in gold_items:
        span_keys.setdefault(((g.doc_sha256 or "").lower(), g.page, (g.answer_span or "").strip().lower()), []).append(g.gold_id)

    verdict_by_gold: Dict[str, Tuple[str, str]] = {}
    cautions_by_gold: Dict[str, List[str]] = {}
    for g in gold_items:
        doc_id, _matches = resolve_gold_chunk_ids(g, sha256_to_doc, chunks_by_doc)
        verdict, explanation = classify_span(g, chunks_by_doc, doc_id)
        verdict_by_gold[g.gold_id] = (verdict, explanation, doc_id)
        duplicates = [gid for gid in span_keys[((g.doc_sha256 or "").lower(), g.page, (g.answer_span or "").strip().lower())] if gid != g.gold_id]
        cautions_by_gold[g.gold_id] = review_cautions(g, q_by_id.get(g.question_id), verdict, duplicates)

    lines.append("## Where to start")
    lines.append("")
    lines.append(
        "Read [HUMAN-REVIEW-GUIDE.md](HUMAN-REVIEW-GUIDE.md) first: it explains what each relation label "
        "means, how to check a page reference, and what a reviewer is committing to. "
        "The items below are the ones where a careless tick does the most damage."
    )
    lines.append("")
    hard = [g for g in gold_items if cautions_by_gold[g.gold_id]]
    if hard:
        lines.append("| Gold item | Why it needs care |")
        lines.append("| --- | --- |")
        for g in hard:
            lines.append(f"| {g.gold_id} ({g.question_id}, p.{g.page}) | {'; '.join(cautions_by_gold[g.gold_id])} |")
        lines.append("")
        lines.append(
            f"{len(hard)} of {len(gold_items)} items carry at least one caution. "
            "A caution is a reading instruction, not a defect verdict: several are cross-paper or "
            "short-span items that are perfectly annotated and simply need the reviewer to look carefully."
        )
    else:
        lines.append("No item carried a caution in this generation.")
    lines.append("")
    exact = sum(1 for v in verdict_by_gold.values() if v[0] == "exact")
    lines.append(
        f"Against the stored corpus: {exact} of {len(gold_items)} spans match character for character; "
        f"{len(gold_items) - exact} need a closer look for the reason shown in their block."
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
            verdict, explanation, doc_id = verdict_by_gold[g.gold_id]
            matches = [
                chunk.chunk_id
                for chunk in chunks_by_doc.get(doc_id or "", [])
                if is_chunk_match(chunk.page, chunk.text, g.page, g.answer_span)
            ]
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
                    f"- Corpus check: **{verdict.replace('_', ' ')}** — "
                    f"no indexed document matches sha256 {(g.doc_sha256 or '')[:16]}… ({explanation})"
                )
            elif matches:
                lines.append(
                    f"- Corpus check: **{verdict.replace('_', ' ')}** in {doc_id} → {', '.join(sorted(matches)[:5])} — {explanation}"
                )
            else:
                lines.append(
                    f"- Corpus check: **{verdict.replace('_', ' ')}** — {explanation}. "
                    f"Recorded source: {doc_id}, page {g.page}."
                )
            for caution in cautions_by_gold[g.gold_id]:
                lines.append(f"- ⚠ {caution}")
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
