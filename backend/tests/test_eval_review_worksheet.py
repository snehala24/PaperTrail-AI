"""
Offline tests for the benchmark human-review worksheet.

No live API calls, no corpus dependency: synthetic documents and chunks only.
"""

import json
from pathlib import Path

import pytest

from backend.app.eval.review_worksheet import (
    build_worksheet,
    main,
    resolve_gold_chunk_ids,
    verification_error,
)
from backend.app.schemas.claim import EvidenceRelation
from backend.app.schemas.document import Chunk, Document
from backend.app.schemas.eval import (
    BenchmarkQuestion,
    BenchmarkQuestionType,
    BenchmarkStatus,
    GoldEvidence,
)

DOC_SHA = "c67573ef50ccf236151da342e53ba1319560146f4ccdd104e6cb41ad24439922"


def _document() -> Document:
    return Document(
        doc_id="doc_c67573ef50cc",
        title="GFWSVD",
        filename="2610.32878v1.pdf",
        sha256=DOC_SHA,
        page_count=30,
        extraction_warnings=[],
    )


def _chunk(text: str = "The method achieves a 2.01x speedup at 80% compression.") -> Chunk:
    return Chunk(
        chunk_id="doc_c67573ef50cc_p26_c3",
        doc_id="doc_c67573ef50cc",
        page=26,
        text=text,
        kind="text",
    )


def _question(status: BenchmarkStatus = BenchmarkStatus.AI_DRAFT) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id="Q-001",
        question="What speedup does the method report at 80% compression?",
        question_type=BenchmarkQuestionType.FACTUAL,
        status=status,
        labelled_by="assistant",
        target_doc_ids=["doc_c67573ef50cc"],
    )


def _gold(
    status: BenchmarkStatus = BenchmarkStatus.AI_DRAFT,
    labelled_by: str = "assistant",
    answer_span: str = "2.01x speedup",
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
    sha: str = DOC_SHA,
) -> GoldEvidence:
    return GoldEvidence(
        gold_id="GOLD-001",
        question_id="Q-001",
        doc_sha256=sha,
        page=26,
        answer_span=answer_span,
        expected_relation=relation,
        status=status,
        labelled_by=labelled_by,
        annotator_notes="Section B.7 reports the wall-clock speedup.",
    )


def _corpus():
    doc = _document()
    return [doc], {doc.doc_id: [_chunk()]}


class TestBuildWorksheet:
    def test_renders_every_item_with_its_verbatim_span_and_checkboxes(self):
        questions, gold = [_question()], [_gold()]
        markdown, errors = build_worksheet(questions, gold, *_corpus())

        assert errors == []
        assert "# Benchmark human review worksheet" in markdown
        assert "Q-001" in markdown
        assert "GOLD-001" in markdown
        assert "2.01x speedup" in markdown
        assert "page 26" in markdown
        assert "- [ ] Page and span are verbatim in the PDF" in markdown
        assert "reviewer handle : human:" in markdown

    def test_reports_where_the_annotation_was_found_in_the_stored_corpus(self):
        markdown, _ = build_worksheet([_question()], [_gold()], *_corpus())
        assert "Corpus check: found in doc_c67573ef50cc" in markdown
        assert "doc_c67573ef50cc_p26_c3" in markdown

    def test_flags_a_span_that_the_indexed_chunks_do_not_contain(self):
        gold = _gold(answer_span="a 4.7x speedup on TPU v5")
        markdown, _ = build_worksheet([_question()], [gold], *_corpus())
        assert "the span was not found in doc_c67573ef50cc on page 26" in markdown

    def test_flags_a_document_that_is_not_in_the_corpus(self):
        gold = _gold(sha="0" * 64)
        markdown, _ = build_worksheet([_question()], [gold], *_corpus())
        assert "no indexed document matches sha256" in markdown
        assert "re-upload it before reviewing" in markdown

    def test_states_that_nothing_is_human_verified_while_every_label_is_a_draft(self):
        markdown, _ = build_worksheet([_question()], [_gold()], *_corpus())
        assert "**No annotation has been human-verified yet.**" in markdown
        assert "| Human-verified gold annotations | 0 |" in markdown
        assert "against draft gold labels (1 assistant-drafted)" in markdown

    def test_the_provenance_note_counts_researcher_and_assistant_drafts_separately(self):
        gold = [
            _gold(),
            _gold(
                status=BenchmarkStatus.HUMAN_DRAFT,
                labelled_by="snehala24",
            ),
        ]
        markdown, _ = build_worksheet([_question()], gold, *_corpus())
        assert "| — of which researcher-drafted (`human_draft`) | 1 |" in markdown
        assert "| — of which assistant-drafted (`ai_draft`) | 1 |" in markdown
        assert "against draft gold labels (1 researcher-drafted, 1 assistant-drafted)" in markdown
        assert "AI-drafted gold labels" not in markdown

    def test_method_difference_is_asked_about_as_apparent_conflict(self):
        gold = _gold(
            relation=EvidenceRelation.WEAKENS,
        )
        gold.conflict_reason = "Uniform versus non-uniform budget allocation."
        markdown, _ = build_worksheet([_question()], [gold], *_corpus())
        assert "Conflict reason: Uniform versus non-uniform budget allocation." in markdown
        assert "methods/data/metric/setting difference is `apparent_conflict`, not `contradicts`" in markdown

    def test_question_without_gold_evidence_is_surfaced_for_review(self):
        markdown, _ = build_worksheet([_question()], [], *_corpus())
        assert "No gold evidence is recorded for this question" in markdown

    def test_does_not_modify_the_dataset_it_reads(self):
        gold = _gold()
        question = _question()
        build_worksheet([question], [gold], *_corpus())
        assert gold.status is BenchmarkStatus.AI_DRAFT
        assert gold.labelled_by == "assistant"
        assert question.status is BenchmarkStatus.AI_DRAFT


class TestResolveGoldChunkIds:
    def test_returns_document_and_matching_chunks(self):
        doc_id, matches = resolve_gold_chunk_ids(_gold(), {DOC_SHA: _document()}, {"doc_c67573ef50cc": [_chunk()]})
        assert doc_id == "doc_c67573ef50cc"
        assert matches == ["doc_c67573ef50cc_p26_c3"]

    def test_unknown_document_resolves_to_nothing(self):
        doc_id, matches = resolve_gold_chunk_ids(_gold(sha="f" * 64), {}, {})
        assert doc_id is None
        assert matches == []


class TestLabelIntegrity:
    def test_draft_labels_are_clean(self):
        assert verification_error(_gold()) is None

    def test_verified_label_without_a_human_handle_is_an_error(self):
        err = verification_error(_gold(status=BenchmarkStatus.HUMAN_VERIFIED, labelled_by="assistant"))
        assert err is not None
        assert "GOLD-001" in err
        assert "not a human handle" in err

    def test_verified_label_with_a_human_handle_is_accepted(self):
        assert verification_error(
            _gold(status=BenchmarkStatus.HUMAN_VERIFIED, labelled_by="human:reviewer")
        ) is None

    def test_build_worksheet_collects_bad_verified_labels(self):
        _, errors = build_worksheet(
            [_question()],
            [_gold(status=BenchmarkStatus.HUMAN_VERIFIED)],
            *_corpus(),
        )
        assert len(errors) == 1


class TestCli:
    def _write_dataset(self, tmp_path: Path, gold_lines):
        (tmp_path / "questions.jsonl").write_text(
            json.dumps(_question().model_dump(mode="json")) + "\n", encoding="utf-8"
        )
        (tmp_path / "gold_evidence.jsonl").write_text(
            "\n".join(json.dumps(g.model_dump(mode="json")) for g in gold_lines) + "\n",
            encoding="utf-8",
        )

    def test_writes_worksheet_without_touching_the_dataset(self, tmp_path, monkeypatch, capsys):
        self._write_dataset(tmp_path, [_gold()])
        monkeypatch.setattr("backend.app.eval.review_worksheet._load_corpus", lambda: _corpus())

        before = (tmp_path / "gold_evidence.jsonl").read_text(encoding="utf-8")
        assert main(["--eval-dir", str(tmp_path)]) == 0

        out = tmp_path / "review_worksheet.md"
        assert out.exists()
        assert "GOLD-001" in out.read_text(encoding="utf-8")
        assert (tmp_path / "gold_evidence.jsonl").read_text(encoding="utf-8") == before
        assert "awaiting human review" in capsys.readouterr().out

    def test_check_passes_for_an_honest_draft_dataset(self, tmp_path, monkeypatch):
        self._write_dataset(tmp_path, [_gold()])
        monkeypatch.setattr("backend.app.eval.review_worksheet._load_corpus", lambda: _corpus())
        assert main(["--eval-dir", str(tmp_path), "--check"]) == 0

    def test_check_fails_when_a_draft_is_marked_verified_by_a_non_human(self, tmp_path, monkeypatch, capsys):
        self._write_dataset(tmp_path, [_gold(status=BenchmarkStatus.HUMAN_VERIFIED)])
        monkeypatch.setattr("backend.app.eval.review_worksheet._load_corpus", lambda: _corpus())
        assert main(["--eval-dir", str(tmp_path), "--check"]) == 1
        assert "LABEL ERROR" in capsys.readouterr().out

    def test_missing_files_are_reported_rather_than_crashing(self, tmp_path):
        assert main(["--eval-dir", str(tmp_path)]) == 1
