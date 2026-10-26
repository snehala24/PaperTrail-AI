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
