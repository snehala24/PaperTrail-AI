import json
from pathlib import Path
from typing import Dict, List, Optional
from backend.app.schemas import Document, Chunk


class DocumentStorage:
    """
    JSON-based file storage for Documents and Chunks.
    Persists data in data/ (gitignored) and maintains SHA-256 deduplication indexing.
    """

    def __init__(self, base_dir: Path | str = "./data"):
        self.base_dir = Path(base_dir)
        self.docs_dir = self.base_dir / "documents"
        self.chunks_dir = self.base_dir / "chunks"
        self.index_file = self.base_dir / "sha256_index.json"

        # Ensure directory structures exist
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self._sha256_map: Dict[str, str] = self._load_sha256_index()

    def _load_sha256_index(self) -> Dict[str, str]:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_sha256_index(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self._sha256_map, f, indent=2)

    def get_document_by_sha256(self, sha256: str) -> Optional[Document]:
        """Check if a document with this SHA-256 has already been ingested."""
        doc_id = self._sha256_map.get(sha256.lower())
        if doc_id:
            return self.get_document(doc_id)
        return None

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Load a Document by doc_id."""
        doc_path = self.docs_dir / f"{doc_id}.json"
        if not doc_path.exists():
            return None
        with open(doc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return Document(**data)

    def get_chunks(self, doc_id: str) -> List[Chunk]:
        """Load granular chunks for a given doc_id."""
        chunks_path = self.chunks_dir / f"{doc_id}_chunks.json"
        if not chunks_path.exists():
            return []
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [Chunk(**c) for c in data]

    def list_documents(self) -> List[Document]:
        """List all ingested Documents."""
        docs: List[Document] = []
        for file in self.docs_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    docs.append(Document(**json.load(f)))
            except Exception:
                continue
        return docs

    def get_all_chunks(self) -> List[Chunk]:
        """Load all chunks across all ingested documents."""
        all_chunks: List[Chunk] = []
        for file in self.chunks_dir.glob("*_chunks.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_chunks.extend(Chunk(**c) for c in data)
            except Exception:
                continue
        return all_chunks

    def save_document(self, doc: Document, chunks: List[Chunk]) -> Document:
        """
        Save Document and Chunks to disk.
        If a document with the same SHA-256 exists, return existing document (deduplication).
        """
        existing = self.get_document_by_sha256(doc.sha256)
        if existing:
            return existing

        doc_path = self.docs_dir / f"{doc.doc_id}.json"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(indent=2))

        chunks_path = self.chunks_dir / f"{doc.doc_id}_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            f.write(json.dumps([c.model_dump() for c in chunks], indent=2))

        self._sha256_map[doc.sha256.lower()] = doc.doc_id
        self._save_sha256_index()

        return doc


# Global default instance
storage = DocumentStorage()
