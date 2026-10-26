"""
CLI tool to verify benchmark gold evidence against persisted corpus in data/.
Checks whether gold answer spans or chunk IDs match chunks in the current ingested corpus.

Usage:
    python -m backend.app.eval.verify_gold
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from backend.app.core.storage import storage
from backend.app.schemas.eval import (
    BenchmarkQuestion,
    BenchmarkQuestionType,
    GoldEvidence,
    is_chunk_match,
)


def verify_benchmark_gold() -> int:
    eval_dir = Path("eval")
    questions_file = eval_dir / "questions.jsonl"
    gold_file = eval_dir / "gold_evidence.jsonl"

    if not questions_file.exists() or not gold_file.exists():
        print(f"Error: questions file ({questions_file}) or gold evidence ({gold_file}) not found.")
        return 1

    questions: List[BenchmarkQuestion] = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(BenchmarkQuestion(**json.loads(line)))

    gold_items: List[GoldEvidence] = []
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gold_items.append(GoldEvidence(**json.loads(line)))

    all_docs = storage.list_documents()
    sha256_to_doc = {d.sha256.lower(): d for d in all_docs}
    all_chunks = storage.get_all_chunks()
    chunks_by_id = {c.chunk_id: c for c in all_chunks}

    print("=" * 60)
    print("PaperTrail AI — Benchmark Gold Verification")
    print(f"Loaded {len(questions)} questions and {len(gold_items)} gold evidence items.")
    print(f"Corpus: {len(all_docs)} documents, {len(all_chunks)} chunks in data/.\n")

    # Group gold items by question_id
    gold_by_q: Dict[str, List[GoldEvidence]] = {}
    for g in gold_items:
        gold_by_q.setdefault(g.question_id, []).append(g)

    # Check each gold item
    gold_matches: Dict[str, List[str]] = {}
    unmatched_gold: List[GoldEvidence] = []

    for g in gold_items:
        matching_cids: List[str] = []

        # Check explicit chunk IDs if provided
        if g.gold_chunk_ids:
            for cid in g.gold_chunk_ids:
                if cid in chunks_by_id:
                    matching_cids.append(cid)

        # Match by (doc_sha256, page, answer_span)
        target_doc = sha256_to_doc.get(g.doc_sha256.lower())
        if target_doc:
            doc_chunks = storage.get_chunks(target_doc.doc_id)
            for chunk in doc_chunks:
                if chunk.page == g.page and is_chunk_match(chunk.page, chunk.text, g.page, g.answer_span):
                    matching_cids.append(chunk.chunk_id)

        matching_cids = list(dict.fromkeys(matching_cids))
        gold_matches[g.gold_id] = matching_cids
        if not matching_cids:
            unmatched_gold.append(g)

    # Question-level analysis
    answerable_count = 0
    unanswerable_count = 0
    orphan_questions: List[BenchmarkQuestion] = []
    matched_questions: List[Tuple[BenchmarkQuestion, List[str]]] = []

    for q in questions:
        q_golds = gold_by_q.get(q.question_id, [])
        all_q_matches: Set[str] = set()
        for g in q_golds:
            all_q_matches.update(gold_matches.get(g.gold_id, []))

        if q.question_type == BenchmarkQuestionType.UNANSWERABLE:
            unanswerable_count += 1
            print(f"  [UNANSWERABLE] {q.question_id}: '{q.question[:65]}...' (Expected 0 matches: correctly reserved for Day 3 abstention)")
        elif len(all_q_matches) > 0:
            answerable_count += 1
            matched_questions.append((q, sorted(list(all_q_matches))))
            print(f"  [VERIFIED]     {q.question_id}: '{q.question[:65]}...' -> Matched {len(all_q_matches)} chunk(s): {sorted(list(all_q_matches))}")
        else:
            orphan_questions.append(q)
            print(f"  [ORPHAN]       {q.question_id}: '{q.question[:65]}...' -> ZERO matching chunks in corpus!")

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY:")
    print(f"  Total Questions:        {len(questions)}")
    print(f"  Answerable (Matched):   {answerable_count}")
    print(f"  Unanswerable (Reserved): {unanswerable_count}")
    print(f"  Orphan Questions:       {len(orphan_questions)}")
    print(f"  Total Gold Items:       {len(gold_items)}")
    print(f"  Unmatched Gold Items:   {len(unmatched_gold)}")
    print("=" * 60)

    if unmatched_gold:
        print("\nUnmatched Gold Evidence Items:")
        for ug in unmatched_gold:
            print(f"  - {ug.gold_id} (Q: {ug.question_id}, p.{ug.page}): \"{ug.answer_span}\"")

    if orphan_questions:
        print(f"\nWARNING: {len(orphan_questions)} answerable question(s) have 0 matching chunks!")
        return 1

    print("\nSUCCESS: All answerable questions have at least one verifiable matching chunk in the corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(verify_benchmark_gold())
