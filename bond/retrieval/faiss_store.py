import os
import numpy as np
import faiss
import logging
from typing import List

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
        self.id_map = np.load(id_map_path, allow_pickle=False)
    
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
