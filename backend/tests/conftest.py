import os

# Set Windows compatibility and offline environment flags for all tests
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Preload faiss before torch/transformers to prevent OpenMP duplicate runtime initialization conflict on Windows
import faiss
