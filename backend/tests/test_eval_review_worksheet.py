"""
Offline tests for the benchmark human-review worksheet.

No live API calls, no corpus dependency: synthetic documents and chunks only.
"""

import json
from pathlib import Path

import pytest

from backend.app.eval.review_worksheet import (
    build_worksheet,
    classify_span,
    main,
    resolve_gold_chunk_ids,
    review_cautions,
    verification_error,
)
from backend.app.schemas.claim import EvidenceRelation
from backend.app.schemas.document import Chunk, Document
from backend.app.schemas.eval import (
    BenchmarkQuestion,
    BenchmarkQuestionType,
    BenchmarkStatus,
    GoldEvidence,
    is_chunk_match,
)

DOC_SHA = "c67573ef50ccf236151da342e53ba1319560146f4ccdd104e6cb41ad24439922"
DOC_ID = "doc_c67573ef50cc"


def _document() -> Document:
    return Document(
        doc_id=DOC_ID,
        title="GFWSVD",
        filename="2610.32878v1.pdf",
        sha256=DOC_SHA,
        page_count=30,
        extraction_warnings=[],
    )


def _chunk(text: str = "The method achieves a 2.01x speedup at 80% compression.", page: int = 26) -> Chunk:
    return Chunk(
        chunk_id=f"{DOC_ID}_p{page}_c3",
        doc_id=DOC_ID,
        page=page,
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
        target_doc_ids=[DOC_ID],
    )


def _gold(
    status: BenchmarkStatus = BenchmarkStatus.AI_DRAFT,
    labelled_by: str = "assistant",
    answer_span: str = "2.01x speedup",
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
    sha: str = DOC_SHA,
    page: int = 26,
) -> GoldEvidence:
    return GoldEvidence(
        gold_id="GOLD-001",
        question_id="Q-001",
        doc_sha256=sha,
        page=page,
        answer_span=answer_span,
        expected_relation=relation,
        status=status,
        labelled_by=labelled_by,
        annotator_notes="Section B.7 reports the wall-clock speedup.",
    )


def _corpus(*chunks: Chunk):
    doc = _document()
    return [doc], {doc.doc_id: list(chunks) or [_chunk()]}


def _chunks(*chunks: Chunk):
    return {DOC_ID: list(chunks)}


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
        assert f"Corpus check: **exact** in {DOC_ID}" in markdown
        assert "matches the stored text character for character" in markdown
        assert f"{DOC_ID}_p26_c3" in markdown

    def test_flags_a_span_that_the_indexed_chunks_do_not_contain(self):
        gold = _gold(answer_span="a 4.7x speedup on TPU v5")
        markdown, _ = build_worksheet([_question()], [gold], *_corpus())
        assert "Corpus check: **absent**" in markdown
        assert "the span is nowhere in this document's stored text" in markdown
        assert f"Recorded source: {DOC_ID}, page 26." in markdown

    def test_flags_a_document_that_is_not_in_the_corpus(self):
        gold = _gold(sha="0" * 64)
        markdown, _ = build_worksheet([_question()], [gold], *_corpus())
        assert "no indexed document matches sha256" in markdown
        assert "it must be re-uploaded before this item can be reviewed" in markdown
        assert "source document is not indexed" in markdown

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
        assert "no `apparent_conflict` member" in markdown

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


class TestSpanClassification:
    """
    The worksheet used to answer only "found / not found". That hid the one case a reviewer must not
    tick blindly: a span that matches only after letter case or line-break hyphens are repaired.
    """

    def test_verbatim_span_is_classified_exact(self):
        verdict, explanation = classify_span(_gold(), _chunks(_chunk()), DOC_ID)
        assert verdict == "exact"
        assert "character for character" in explanation

    def test_letter_case_deviation_is_reported_as_case_only_not_as_a_match(self):
        gold = _gold(answer_span="2.01X SPEEDUP")
        verdict, explanation = classify_span(gold, _chunks(_chunk()), DOC_ID)
        assert verdict == "case_only"
        assert "not verbatim as recorded" in explanation

    def test_span_that_only_matches_after_linebreak_hyphen_repair_is_named_as_such(self):
        chunks = _chunks(_chunk("The method achieves a pe- nitwise improvement."))
        gold = _gold(answer_span="penitwise improvement")
        verdict, explanation = classify_span(gold, chunks, DOC_ID)
        assert verdict == "hyphen_or_space_repair"
        assert "line-break hyphens" in explanation
        # the tolerant matcher still says "found" — the worksheet must not stop there
        assert is_chunk_match(26, chunks[DOC_ID][0].text, gold.page, gold.answer_span) is True

    def test_span_present_on_another_page_is_reported_as_a_wrong_page_reference(self):
        chunks = _chunks(
            _chunk("Unrelated text on the recorded page."),
            _chunk("A different page holds the 2.01x speedup claim.", page=11),
        )
        verdict, explanation = classify_span(_gold(), chunks, DOC_ID)
        assert verdict == "other_page"
        assert "page 11" in explanation
        assert "page reference is wrong" in explanation

    def test_absent_span_and_unindexed_document_get_their_own_verdicts(self):
        _, absent = classify_span(
            _gold(answer_span="a number that is not printed"), _chunks(_chunk()), DOC_ID
        )
        assert absent.startswith("the span is nowhere")
        verdict, explanation = classify_span(_gold(), {}, None)
        assert verdict == "no_document"
        assert "not in data/" in explanation

    def test_the_worksheet_renders_the_verdict_and_links_the_review_guide(self):
        gold = _gold(answer_span="INTEGER LINEAR PROGRAM", page=5)
        markdown, _ = build_worksheet(
            [_question()],
            [gold],
            *_corpus(_chunk("Integer linear program (ILP) is solved.", page=5)),
        )
        assert "Corpus check: **case only**" in markdown
        assert "[HUMAN-REVIEW-GUIDE.md](HUMAN-REVIEW-GUIDE.md)" in markdown
        assert "## Where to start" in markdown
        assert "A caution is a reading instruction, not a defect verdict" in markdown


class TestReviewCautions:
    def _cautions(self, gold, question=None, verdict="exact", duplicates=None):
        return review_cautions(gold, question, verdict, duplicates or [])

    def test_short_span_is_flagged_as_a_possible_coincidence(self):
        notes = self._cautions(_gold(answer_span="5.07 GB"))
        assert any("coincidental substring" in n for n in notes)

    def test_contradicts_label_is_flagged_as_the_hardest_call(self):
        gold = _gold(relation=EvidenceRelation.CONTRADICTS)
        gold.conflict_reason = "Uniform versus non-uniform budget allocation."
        notes = self._cautions(gold)
        assert any("hardest label" in n for n in notes)
        assert not any("without a conflict_reason" in n for n in notes)

    def test_conflict_relation_recorded_without_a_reason_is_flagged(self):
        gold = _gold(relation=EvidenceRelation.WEAKENS)
        gold.conflict_reason = None
        notes = self._cautions(gold)
        assert any("without a conflict_reason" in n for n in notes)

    def test_a_contradicts_label_whose_reason_describes_an_apparent_conflict_is_surfaced(self):
        gold = _gold(relation=EvidenceRelation.CONTRADICTS)
        gold.conflict_reason = (
            "Apparent conflict: LEMS reports degradation, whereas this paper reported strong performance."
        )
        notes = self._cautions(gold)
        assert any("these two fields disagree" in n for n in notes)

    def test_a_contradicts_label_with_a_factual_reason_is_not_double_flagged(self):
        gold = _gold(relation=EvidenceRelation.CONTRADICTS)
        gold.conflict_reason = "Both papers report the same metric on the same model and split, at 12.0% and 4.0%."
        notes = self._cautions(gold)
        assert not any("these two fields disagree" in n for n in notes)

    def test_shared_span_points_the_reviewer_at_twin_items(self):
        notes = self._cautions(_gold(), duplicates=["GOLD-033"])
        assert any("GOLD-033" in n and "same verdict in each block" in n for n in notes)

    def test_cross_paper_question_asks_for_the_companion_span(self):
        question = _question()
        question.target_doc_ids = [DOC_ID, "doc_7fdb1200e823"]
        notes = self._cautions(_gold(), question)
        assert any("companion span" in n for n in notes)

    def test_gold_span_attached_to_an_abstention_probe_is_flagged_as_itself_the_finding(self):
        question = _question()
        question.question_type = BenchmarkQuestionType.UNANSWERABLE
        notes = self._cautions(_gold(), question)
        assert any("abstention probe" in n for n in notes)

    def test_conflict_question_supported_by_one_side_is_explained_not_called_a_defect(self):
        question = _question()
        question.question_type = BenchmarkQuestionType.APPARENT_CONFLICT
        notes = self._cautions(_gold(relation=EvidenceRelation.SUPPORTS), question)
        assert any("can be correct" in n for n in notes)

    def test_caution_for_a_span_verdict_matches_the_classification(self):
        notes = self._cautions(_gold(), verdict="other_page")
        assert any("page reference disagrees" in n for n in notes)
        assert not any("character for character" in n for n in notes)

    def test_a_clean_item_carries_no_caution(self):
        assert self._cautions(_gold(answer_span="2.01x speedup at 80% compression")) == []


class TestResolveGoldChunkIds:
    def test_returns_document_and_matching_chunks(self):
        doc_id, matches = resolve_gold_chunk_ids(_gold(), {DOC_SHA: _document()}, _chunks(_chunk()))
        assert doc_id == DOC_ID
        assert matches == [f"{DOC_ID}_p26_c3"]

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
