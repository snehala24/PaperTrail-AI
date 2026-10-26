import json
from pathlib import Path
from backend.app.schemas import (
    BenchmarkQuestion,
    BenchmarkQuestionType,
    BenchmarkStatus,
    GoldEvidence,
    EvidenceRelation,
)
from backend.app.schemas.eval import is_chunk_match

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

    assert len(questions) == 30
    ai_questions = [q for q in questions if q.status == BenchmarkStatus.AI_DRAFT]
    human_questions = [
        q for q in questions
        if q.status in (BenchmarkStatus.HUMAN_DRAFT, BenchmarkStatus.HUMAN_VERIFIED, BenchmarkStatus.PENDING_SECOND_PAPER)
    ]
    assert len(ai_questions) == 10
    assert all(q.labelled_by == "assistant" for q in ai_questions)

    assert len(human_questions) == 20
    assert all(q.labelled_by == "snehala24" for q in human_questions)

    pending_questions = [q for q in questions if q.status == BenchmarkStatus.PENDING_SECOND_PAPER]
    assert len(pending_questions) == 2
    assert {q.question_id for q in pending_questions} == {"Q-029", "Q-030"}

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

    assert len(gold_entries) == 34
    ai_gold = [g for g in gold_entries if g.status == BenchmarkStatus.AI_DRAFT]
    human_gold = [
        g for g in gold_entries
        if g.status in (BenchmarkStatus.HUMAN_DRAFT, BenchmarkStatus.HUMAN_VERIFIED, BenchmarkStatus.PENDING_SECOND_PAPER)
    ]
    assert len(ai_gold) == 10
    assert all(g.labelled_by == "assistant" for g in ai_gold)

    assert len(human_gold) == 24
    assert all(g.labelled_by == "snehala24" for g in human_gold)

    pending_gold = [g for g in gold_entries if g.status == BenchmarkStatus.PENDING_SECOND_PAPER]
    assert len(pending_gold) == 4
    assert {g.gold_id for g in pending_gold} == {"GOLD-031", "GOLD-032", "GOLD-033", "GOLD-034"}

    for g in gold_entries:
        if not ("TODO" in g.doc_sha256 or g.answer_span.startswith("TODO")):
            assert len(g.doc_sha256) == 64
        assert g.page >= 1
        assert len(g.answer_span) > 0
        if g.expected_relation in (EvidenceRelation.CONTRADICTS, EvidenceRelation.WEAKENS):
            assert g.conflict_reason is not None and len(g.conflict_reason) > 0


def test_is_chunk_match_helper():
    chunk_page = 3
    chunk_text = "We evaluate BERT-Base with L=12, H=768, A=12, and Total Parameters=110M across downstream tasks."
    gold_span = "Total Parameters=110M"

    # Match on same page and text span
    assert is_chunk_match(chunk_page, chunk_text, 3, gold_span) is True
    # Case insensitivity
    assert is_chunk_match(chunk_page, chunk_text, 3, "total parameters=110m") is True
    # Different page must not match
    assert is_chunk_match(chunk_page, chunk_text, 4, gold_span) is False
    # Missing span must not match
    assert is_chunk_match(chunk_page, chunk_text, 3, "Total Parameters=340M") is False
