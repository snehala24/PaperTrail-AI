import os

# Set Windows compatibility and offline environment flags for all tests
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Preload faiss before torch/transformers to prevent OpenMP duplicate runtime initialization conflict on Windows
import faiss
import pytest


@pytest.fixture(autouse=True)
def isolate_test_storage(tmp_path, monkeypatch):
    """Ensure all automated tests run against an isolated temporary data storage directory."""
    from backend.app.core.storage import DocumentStorage
    import backend.app.core.storage as storage_mod
    import backend.app.api.documents as docs_mod

    temp_store = DocumentStorage(base_dir=tmp_path / "test_data")
    monkeypatch.setattr(storage_mod, "storage", temp_store)
    monkeypatch.setattr(docs_mod, "storage", temp_store)
    yield temp_store
