import pytest
import numpy as np
from bond.validate_signature import cosine, validate_embedding_signature

def test_cosine_similarity():
    # Test with identical vectors
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert cosine(a, b) == pytest.approx(1.0)
    
    # Test with orthogonal vectors
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine(a, b) == pytest.approx(0.0)
    
    # Test with opposite vectors
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine(a, b) == pytest.approx(-1.0)

def test_validate_embedding_signature_missing_file():
    def mock_embed_fn(texts):
        return [[0.1, 0.2, 0.3]]
    
    with pytest.raises(FileNotFoundError):
        validate_embedding_signature("nonexistent.json", mock_embed_fn)
