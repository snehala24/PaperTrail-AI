import os
import sys
import types
import numpy as np

# Set Windows compatibility and offline environment flags for all tests
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Preload faiss before torch/transformers to prevent OpenMP duplicate runtime initialization conflict on Windows.
# Provide a lightweight stub if the real FAISS library cannot be imported (common on Windows CI).
try:
    import faiss  # noqa: F401
except Exception:  # pragma: no cover
    # Create a minimal fake FAISS module with the functions/classes used in the code.
    fake_faiss = types.SimpleNamespace()

    def _normalize_L2(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return x / norm

    fake_faiss.normalize_L2 = _normalize_L2

    class _IndexFlatIP:
        def __init__(self, dim: int):
            self.dim = dim
            self.embeddings = np.empty((0, dim), dtype=np.float32)

        def add(self, emb: np.ndarray) -> None:
            self.embeddings = np.vstack([self.embeddings, emb.astype(np.float32)])

        def search(self, query_emb: np.ndarray, k: int):
            # query_emb shape (1, dim)
            if self.embeddings.size == 0:
                scores = np.zeros((1, k), dtype=np.float32)
                indices = np.full((1, k), -1, dtype=np.int64)
                return scores, indices
            # compute inner product (cosine similarity after normalization)
            scores_vec = np.dot(self.embeddings, query_emb.T).flatten()
            top_idx = np.argsort(-scores_vec)[:k]
            top_scores = scores_vec[top_idx]
            if len(top_idx) < k:
                pad_len = k - len(top_idx)
                top_idx = np.concatenate([top_idx, np.full(pad_len, -1, dtype=np.int64)])
                # Match native FAISS: missing slots use a very low score, not 0.0.
                top_scores = np.concatenate(
                    [top_scores, np.full(pad_len, np.float32("-inf"), dtype=np.float32)]
                )
            scores = top_scores.reshape(1, k)
            indices = top_idx.reshape(1, k)
            return scores, indices

    fake_faiss.IndexFlatIP = _IndexFlatIP
    sys.modules["faiss"] = fake_faiss

# Ensure a valid import now that the stub is in place.
import faiss  # noqa: E402
import pytest


def assert_fail_closed_abstention(resp, *reason_needles: str) -> None:
    """Shared contract: unsupported or invalid evidence must abstain without exposing claims."""
    from backend.app.schemas import AnswerStatus

    assert resp.status == AnswerStatus.ABSTAINED
    assert resp.evidence_links == []
    assert resp.abstain_reason
    assert "abstain" in (resp.answer or "").lower()
    reason = resp.abstain_reason.lower()
    needles = reason_needles or (
        "quote",
        "chunk",
        "verbatim",
        "unsupported",
        "zero-tolerance",
        "invalid",
        "missing",
    )
    assert any(n.lower() in reason for n in needles), resp.abstain_reason

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
