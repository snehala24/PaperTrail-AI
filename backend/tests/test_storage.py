import pytest
from pathlib import Path
from backend.app.core.storage import DocumentStorage
from backend.app.schemas import Document, Chunk, ChunkKind

SAMPLE_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture
def temp_storage(tmp_path):
    return DocumentStorage(base_dir=tmp_path)


def test_save_and_get_document(temp_storage):
    doc = Document(
        doc_id="doc-001",
        title="Test Document",
        filename="test.pdf",
        sha256=SAMPLE_SHA,
        page_count=2,
    )
    chunks = [
        Chunk(chunk_id="doc-001_p1_c1", doc_id="doc-001", page=1, text="Chunk 1 text", kind=ChunkKind.TEXT),
        Chunk(chunk_id="doc-001_p2_c1", doc_id="doc-001", page=2, text="Chunk 2 text", kind=ChunkKind.TEXT),
    ]

    saved_doc = temp_storage.save_document(doc, chunks)
    assert saved_doc.doc_id == "doc-001"

    # Retrieve document
    loaded_doc = temp_storage.get_document("doc-001")
    assert loaded_doc is not None
    assert loaded_doc.title == "Test Document"
    assert loaded_doc.sha256 == SAMPLE_SHA

    # Retrieve chunks
    loaded_chunks = temp_storage.get_chunks("doc-001")
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0].chunk_id == "doc-001_p1_c1"


def test_deduplication_by_sha256(temp_storage):
    doc = Document(
        doc_id="doc-original",
        title="Original",
        filename="first.pdf",
        sha256=SAMPLE_SHA,
        page_count=1,
    )
    chunks = [Chunk(chunk_id="doc-original_p1_c1", doc_id="doc-original", page=1, text="Text", kind=ChunkKind.TEXT)]
    temp_storage.save_document(doc, chunks)

    # Lookup by sha256
    found_doc = temp_storage.get_document_by_sha256(SAMPLE_SHA)
    assert found_doc is not None
    assert found_doc.doc_id == "doc-original"

    # Saving again with same sha256 should return existing without re-saving under new id
    doc_duplicate = Document(
        doc_id="doc-duplicate",
        title="Duplicate",
        filename="second.pdf",
        sha256=SAMPLE_SHA,
        page_count=1,
    )
    res_doc = temp_storage.save_document(doc_duplicate, chunks)
    assert res_doc.doc_id == "doc-original"  # Preserved original doc_id


def test_list_documents(temp_storage):
    doc1 = Document(doc_id="doc-1", title="Doc 1", filename="1.pdf", sha256="1" * 64, page_count=1)
    doc2 = Document(doc_id="doc-2", title="Doc 2", filename="2.pdf", sha256="2" * 64, page_count=1)

    temp_storage.save_document(doc1, [])
    temp_storage.save_document(doc2, [])

    all_docs = temp_storage.list_documents()
    assert len(all_docs) == 2
    ids = {d.doc_id for d in all_docs}
    assert ids == {"doc-1", "doc-2"}


def test_unreadable_records_are_skipped_but_named_in_the_log(temp_storage, caplog):
    """
    A damaged JSON record must not take the whole workspace down, but it also must not disappear
    quietly: without a log line an operator cannot tell a smaller corpus from a broken one.
    """
    good = Document(doc_id="doc-good", title="Good", filename="good.pdf", sha256="a" * 64, page_count=1)
    temp_storage.save_document(good, [
        Chunk(chunk_id="doc-good_p1_c1", doc_id="doc-good", page=1, text="Usable text", kind=ChunkKind.TEXT)
    ])

    (temp_storage.docs_dir / "doc-broken.json").write_text("{ this is not json", encoding="utf-8")
    (temp_storage.chunks_dir / "doc_broken_chunks.json").write_text("{ this is not json", encoding="utf-8")

    with caplog.at_level("WARNING", logger="backend.app.core.storage"):
        docs = temp_storage.list_documents()
        chunks = temp_storage.get_all_chunks()

    assert [d.doc_id for d in docs] == ["doc-good"]
    assert [c.chunk_id for c in chunks] == ["doc-good_p1_c1"]

    warning_text = " ".join(r.getMessage() for r in caplog.records)
    assert "doc-broken.json" in warning_text
    assert "doc_broken_chunks.json" in warning_text


def test_corrupt_dedup_index_is_reported_instead_of_silently_emptied(tmp_path, caplog):
    """
    Losing the SHA-256 index without a trace makes every future upload look new, so the load has to say so.
    """
    (tmp_path / "sha256_index.json").write_text("not json at all", encoding="utf-8")

    with caplog.at_level("WARNING", logger="backend.app.core.storage"):
        storage = DocumentStorage(base_dir=tmp_path)

    assert storage._sha256_map == {}
    assert "SHA-256 dedup index" in caplog.text
