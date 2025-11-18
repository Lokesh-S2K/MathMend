"""
modules/indexer_faiss.py
-------------------------
Manages FAISS indexing, saving/loading, and querying for structure-aware retrieval.

Features:
- Supports multiple index types (FlatL2, HNSW, IVF)
- Stores vector normalization, ID mappings, and metadata
- Allows incremental addition of new embeddings
- Provides convenience wrappers for query and bulk retrieval
"""

from __future__ import annotations
import os
import json
import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional


try:
    import faiss.contrib.torch_utils  # noqa: F401
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class FaissIndexer:
    """
    Wrapper around FAISS index for structure vector retrieval.

    Usage:
        >>> indexer = FaissIndexer(dim=512, index_type="Flat", normalize=True)
        >>> indexer.build(vectors, ids)
        >>> indexer.save("output/faiss.index", "output/faiss_id_map.json")

        >>> indexer = FaissIndexer.load("output/faiss.index", "output/faiss_id_map.json")
        >>> scores, retrieved_ids = indexer.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "Flat",
        normalize: bool = True,
        use_gpu: bool = False
    ):
        self.dim = dim
        self.index_type = index_type
        self.normalize = normalize
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.index = None
        self.id_map: Dict[int, Dict] = {}

    # ----------------------------------------------------------------------
    # Index Creation
    # ----------------------------------------------------------------------
    def _make_index(self) -> faiss.Index:
        """Creates the FAISS index based on type."""
        if self.index_type.lower() in ["flat", "indexflatl2"]:
            index = faiss.IndexFlatL2(self.dim)
        elif self.index_type.lower() == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32)
            index.hnsw.efSearch = 64
        elif self.index_type.lower() == "ivf":
            nlist = min(100, max(10, int(self.dim / 2)))
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        if self.use_gpu:
            print("[FAISS] Moving index to GPU ...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    # ----------------------------------------------------------------------
    # Build & Add
    # ----------------------------------------------------------------------
    def build(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata_list: Optional[List[Dict]] = None
    ):
        """
        Build and populate a FAISS index.

        Args:
            vectors: (N, D) numpy array
            ids: list of dataset IDs corresponding to each vector
            metadata_list: optional list of metadata dicts for each ID
        """
        assert len(vectors) == len(ids), "Vectors and IDs must match in length."
        vectors = np.array(vectors).astype("float32")

        if self.normalize:
            faiss.normalize_L2(vectors)

        print(f"[FAISS] Building index type={self.index_type}, dim={self.dim}")
        self.index = self._make_index()

        if isinstance(self.index, faiss.IndexIVFFlat):
            print("[FAISS] Training IVF index...")
            self.index.train(vectors)

        self.index.add(vectors)
        print(f"[FAISS] Added {len(vectors)} vectors.")

        # Build ID map
        for i, idx in enumerate(ids):
            self.id_map[i] = {
                "id": idx,
                "metadata": metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            }

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Query the FAISS index.

        Args:
            query_vec: (D,) numpy array
            top_k: number of neighbors to retrieve

        Returns:
            scores: (top_k,) distances
            results: list of dicts [{id, metadata}, ...]
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")

        query_vec = np.array(query_vec).astype("float32").reshape(1, -1)
        if self.normalize:
            faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i in indices[0]:
            if i == -1:
                continue
            entry = self.id_map.get(int(i), {"id": None, "metadata": {}})
            results.append(entry)

        return distances[0], results

    # ----------------------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------------------
    def save(self, index_path: str, idmap_path: str):
        """Saves FAISS index and ID map."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if self.use_gpu:
            self.index = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(self.index, index_path)
        with open(idmap_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2)
        print(f"[FAISS] Saved index to {index_path}")
        print(f"[FAISS] Saved ID map to {idmap_path}")

    @classmethod
    def load(cls, index_path: str, idmap_path: str, use_gpu: bool = False):
        """Loads a FAISS index + ID map."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")

        index = faiss.read_index(index_path)
        if use_gpu and GPU_AVAILABLE:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        with open(idmap_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        dim = index.d
        obj = cls(dim=dim, use_gpu=use_gpu)
        obj.index = index
        obj.id_map = {int(k): v for k, v in id_map.items()}
        print(f"[FAISS] Loaded index ({dim} dims) and ID map with {len(obj.id_map)} entries.")
        return obj

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def add_vectors(self, vectors: np.ndarray, ids: List[str], metadata_list: Optional[List[Dict]] = None):
        """Add more vectors incrementally."""
        assert self.index is not None, "Index must be built first."
        vectors = np.array(vectors).astype("float32")
        if self.normalize:
            faiss.normalize_L2(vectors)

        start_idx = len(self.id_map)
        self.index.add(vectors)

        for i, idx in enumerate(ids):
            self.id_map[start_idx + i] = {
                "id": idx,
                "metadata": metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            }
        print(f"[FAISS] Added {len(vectors)} new vectors (total: {len(self.id_map)})")

    def size(self) -> int:
        return len(self.id_map)
