import os
import numpy as np
import faiss
import logging
from typing import List
import threading

logger = logging.getLogger(__name__)

def _profile_dir(assets_path: str) -> str:
    """Return path to the single FAISS store directory."""
    return os.path.join(assets_path, "faiss_store")

class FaissStore:
    def __init__(self, assets_path: str, rescore_multiplier: int = 20):
        self.profile_path = _profile_dir(assets_path)
        self.rescore_multiplier = rescore_multiplier

        if not os.path.isdir(self.profile_path):
            raise FileNotFoundError(f"FAISS store directory not found: {self.profile_path}")

        faiss_path = os.path.join(self.profile_path, "embeddings.faiss")
        id_map_path = os.path.join(self.profile_path, "id_map.npy")
        self.signature_path = os.path.join(self.profile_path, "embedding_signature.json")

        logger.info("Loading FAISS index (binary + int8 rescoring)...")

        # Load the binary index and the separate int8 rescore vectors
        self.base_index = faiss.read_index_binary(faiss_path)
        rescore_path = os.path.join(self.profile_path, "rescore_vectors.npy")
        self.rescore_vectors = np.load(rescore_path, mmap_mode="r")

        self.index = self.base_index
        # Memory-map id_map to avoid loading entire array into RAM at import
        self.id_map = np.load(id_map_path, allow_pickle=False, mmap_mode="r")
        self._search_lock = threading.Lock()
        self._id_to_index = None  # lazily constructed reverse map
    
    @staticmethod
    def _float_to_binary_packbits(vectors: np.ndarray) -> np.ndarray:
        """Helper used in tests to validate binary packing logic."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return np.packbits((vectors >= 0).astype(np.uint8), axis=-1)
    
    def get_gpu_status(self) -> dict:
        """GPU acceleration is not applicable for binary index; always CPU."""
        return {"gpu_requested": False, "gpu_available": False, "gpu_active": False}
    
    def close(self):
        """No-op for CPU-only binary index"""
        return None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()

    def search(self, vectors: np.ndarray, k: int) -> List[List[str]]:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n_queries = vectors.shape[0]
        all_results = []

        # Stage 1: Fast binary search for a large number of candidates
        k_rescore = k * self.rescore_multiplier
        q_bin = np.packbits(np.where(vectors >= 0, 1, 0), axis=-1)
        # Guard FAISS calls with a lock for thread-safety across profiles/builds
        with self._search_lock:
            _, initial_indices = self.index.search(q_bin, k_rescore)

        for query_idx in range(n_queries):
            cand_indices = initial_indices[query_idx]
            cand_indices = cand_indices[cand_indices != -1]
            if len(cand_indices) == 0:
                all_results.append([])
                continue

            # Stage 2: Precise rescoring with int8 vectors
            cand_vecs_int8 = self.rescore_vectors[cand_indices]
            query_vector_fp32 = vectors[query_idx]

            cand_vecs_fp32 = cand_vecs_int8.astype(np.float32) / 127.0
            scores = np.dot(cand_vecs_fp32, query_vector_fp32.T).squeeze()

            final_idx_indices = np.argsort(-scores)[:k]
            final_indices = cand_indices[final_idx_indices]
            all_results.append([str(self.id_map[i]) for i in final_indices])
                
        return all_results

    def _ensure_reverse_map(self):
        """Build a reverse map from ID string -> index into rescore/id_map arrays."""
        if self._id_to_index is None:
            # Convert to Python str to ensure consistent keys
            mapping = {}
            for i in range(self.id_map.shape[0]):
                try:
                    k = str(self.id_map[i])
                    mapping[k] = i
                except Exception:
                    continue
            self._id_to_index = mapping

    def score_ids(self, query_vector: np.ndarray, ids: List[str]) -> dict:
        """Compute cosine-like scores between a single query vector and candidate IDs.

        Uses the precomputed int8 rescore vectors; no term re-embedding.
        Returns a dict {id: score} for the provided ids that exist in the index.
        """
        if query_vector.ndim != 1:
            # Flatten any 2D of shape (1, d)
            query_vector = query_vector.reshape(-1)
        self._ensure_reverse_map()
        if not ids:
            return {}
        # Collect available indices
        idxs = []
        keep_ids = []
        for _id in ids:
            j = self._id_to_index.get(str(_id)) if self._id_to_index else None
            if j is not None:
                idxs.append(j)
                keep_ids.append(str(_id))
        if not idxs:
            return {}
        cand_vecs_int8 = self.rescore_vectors[idxs]
        cand_vecs_fp32 = cand_vecs_int8.astype(np.float32) / 127.0
        # Dot product equals cosine similarity when both sides are normalized
        scores = cand_vecs_fp32 @ query_vector.astype(np.float32)
        return {k: float(v) for k, v in zip(keep_ids, scores.tolist())}
