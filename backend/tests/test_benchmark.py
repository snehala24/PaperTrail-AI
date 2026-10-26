import json
from pathlib import Path
from backend.app.schemas import (
    BenchmarkQuestion,
    BenchmarkQuestionType,
    BenchmarkStatus,
    GoldEvidence,
    EvidenceRelation,
)

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"


def test_benchmark_schema_file_exists():
    schema_path = EVAL_DIR / "benchmark_schema.json"
    assert schema_path.exists()
    content = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "definitions" in content
    assert "BenchmarkQuestion" in content["definitions"]
    assert "GoldEvidence" in content["definitions"]


def test_eval_questions_loading():
    questions_path = EVAL_DIR / "questions.jsonl"
    assert questions_path.exists()

    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q = BenchmarkQuestion(**item)
                questions.append(q)

    assert len(questions) == 10
    # All initial questions must be draft status
    assert all(q.status == BenchmarkStatus.DRAFT for q in questions)

    # Must cover required categories
    categories = {q.question_type for q in questions}
    assert BenchmarkQuestionType.FACTUAL in categories
    assert BenchmarkQuestionType.COMPARATIVE in categories
    assert BenchmarkQuestionType.APPARENT_CONFLICT in categories
    assert BenchmarkQuestionType.UNANSWERABLE in categories
    assert BenchmarkQuestionType.METHODOLOGICAL in categories


def test_eval_gold_evidence_loading():
    gold_path = EVAL_DIR / "gold_evidence.jsonl"
    assert gold_path.exists()

    gold_entries = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                g = GoldEvidence(**item)
                gold_entries.append(g)

    assert len(gold_entries) == 10
    assert all(g.status == BenchmarkStatus.DRAFT for g in gold_entries)

    # Rule 3: Whenever relation is contradicts or weakens, conflict reason must be provided
    for g in gold_entries:
        if g.expected_relation in (EvidenceRelation.CONTRADICTS, EvidenceRelation.WEAKENS):
            assert g.conflict_reason is not None and len(g.conflict_reason) > 0
