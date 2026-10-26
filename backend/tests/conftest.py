import os

# Set Windows compatibility and offline environment flags for all tests
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Preload faiss before torch/transformers to prevent OpenMP duplicate runtime initialization conflict on Windows
import faiss
