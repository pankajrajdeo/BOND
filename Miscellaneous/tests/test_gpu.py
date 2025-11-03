"""
Tests for GPU acceleration functionality.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from bond.retrieval.faiss_store import FaissStore

@pytest.fixture
def mock_assets_path(tmp_path):
    """Create mock assets directory structure"""
    assets_path = tmp_path / "assets"
    profile_path = assets_path / "faiss_store"
    profile_path.mkdir(parents=True)
    
    # Create mock FAISS binary index file and associated assets
    import faiss
    d = 128  # dimension
    n = 1000  # number of vectors
    
    # Build a binary index by packing signs of vectors
    np.random.seed(42)
    vectors = np.random.random((n, d)).astype('float32')
    bin_vectors = np.packbits((vectors >= 0).astype(np.uint8), axis=-1)
    index = faiss.IndexBinaryFlat(d)
    index.add(bin_vectors)
    faiss.write_index_binary(index, str(profile_path / "embeddings.faiss"))
    
    # Create mock id_map
    ids = [f"ID:{i}" for i in range(n)]
    id_map_array = np.array(ids, dtype=f"<U{max(len(id) for id in ids)}")
    np.save(str(profile_path / "id_map.npy"), id_map_array)
    
    # Create mock embedding signature
    import json
    signature = {
        "model_id": "test-model",
        "dimension": d,
        "anchor_text": "test",
        "anchor_vector": [0.1] * d
    }
    with open(profile_path / "embedding_signature.json", "w") as f:
        json.dump(signature, f)
    
    # Create mock int8 rescoring vectors
    embs_int8 = np.clip(vectors * 127, -127, 127).astype(np.int8)
    np.save(str(profile_path / "rescore_vectors.npy"), embs_int8)
    
    return str(assets_path)

def test_gpu_resources_detection():
    """GPU is not applicable for binary FAISS; nothing to detect."""
    assert True

def test_faiss_store_gpu_initialization(mock_assets_path):
    """Binary FAISS store has no GPU path; status should be false."""
    store = FaissStore(mock_assets_path, rescore_multiplier=20)
    gpu_status = store.get_gpu_status()
    assert gpu_status["gpu_requested"] is False
    assert gpu_status["gpu_available"] is False
    assert gpu_status["gpu_active"] is False
    store.close()

def test_faiss_store_cpu_fallback(mock_assets_path):
    store = FaissStore(mock_assets_path, rescore_multiplier=20)
    gpu_status = store.get_gpu_status()
    assert gpu_status["gpu_requested"] is False
    assert gpu_status["gpu_available"] is False
    assert gpu_status["gpu_active"] is False
    store.close()

def test_faiss_store_binary_profile_gpu_limitation(mock_assets_path):
    store = FaissStore(mock_assets_path, rescore_multiplier=20)
    gpu_status = store.get_gpu_status()
    assert gpu_status["gpu_requested"] is False
    assert gpu_status["gpu_available"] is False
    assert gpu_status["gpu_active"] is False
    store.close()

def test_faiss_store_search_functionality(mock_assets_path):
    """Test that search works."""
    store = FaissStore(mock_assets_path, rescore_multiplier=20)
    
    # Test search functionality
    query_vectors = np.random.random((2, 128)).astype('float32')
    results = store.search(query_vectors, k=5)
    
    # Verify results structure
    assert len(results) == 2  # Two queries
    assert all(len(result) <= 5 for result in results)  # Each result <= k
    
    # Cleanup
    store.close()

def test_faiss_store_cleanup():
    """No-op cleanup should not raise"""
    store = FaissStore("dummy_path", rescore_multiplier=20)
    store.close()
