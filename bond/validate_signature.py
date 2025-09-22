import json
import os
import numpy as np
from typing import Callable
from .logger import logger

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def validate_embedding_signature(signature_path: str, embed_fn: Callable[[list], list], threshold: float = 0.999):
    if not os.path.exists(signature_path):
        raise FileNotFoundError(f"Embedding signature not found at {signature_path}. Please run the build script.")
    with open(signature_path, "r", encoding="utf-8") as f:
        sig = json.load(f)
    
    anchor_text = sig["anchor_text"]
    anchor_vec = np.array(sig["anchor_vector"], dtype=np.float32)
    test_vec = np.array(embed_fn([anchor_text])[0], dtype=np.float32)
    
    sim = cosine(anchor_vec, test_vec)
    if sim < threshold:
        raise RuntimeError(
            f"Embedding mismatch detected! The loaded model '{sig.get('model_id', 'unknown')}' "
            f"does not match the index signature. Cosine similarity was {sim:.6f} (threshold: {threshold}). "
            "Ensure you are using the correct model or rebuild the index with `bond-build-faiss`."
        )
    logger.info("Embedding signature validated successfully.")
    return True
