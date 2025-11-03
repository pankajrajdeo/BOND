import pytest
import numpy as np
from bond.retrieval.faiss_store import FaissStore

def test_binary_packbits_consistency():
    """Test that binary packbits conversion is consistent and correct"""
    
    # Test with known vectors
    test_vectors = np.array([
        [1.0, -1.0, 0.5, -0.5],  # Mixed signs
        [0.0, 0.0, 0.0, 0.0],    # All zeros
        [1.0, 1.0, 1.0, 1.0],    # All positive
        [-1.0, -1.0, -1.0, -1.0] # All negative
    ], dtype=np.float32)
    
    # Convert to binary
    binary_result = FaissStore._float_to_binary_packbits(test_vectors)
    
    # Check shape: 4 vectors, 4 dimensions = 4 bits per vector = 1 byte per vector
    assert binary_result.shape == (4, 1)
    assert binary_result.dtype == np.uint8
    
    # Check first vector: [1.0, -1.0, 0.5, -0.5] -> [1, 0, 1, 0] -> 0b1010 = 10
    assert binary_result[0, 0] == 10  # 0b1010
    
    # Check second vector: [0.0, 0.0, 0.0, 0.0] -> [0, 0, 0, 0] -> 0b0000 = 0
    assert binary_result[1, 0] == 0   # 0b0000
    
    # Check third vector: [1.0, 1.0, 1.0, 1.0] -> [1, 1, 1, 1] -> 0b1111 = 15
    assert binary_result[2, 0] == 15  # 0b1111
    
    # Check fourth vector: [-1.0, -1.0, -1.0, -1.0] -> [0, 0, 0, 0] -> 0b0000 = 0
    assert binary_result[3, 0] == 0   # 0b0000

def test_binary_packbits_edge_cases():
    """Test edge cases for binary conversion"""
    
    # Test with single vector
    single_vector = np.array([[1.0, -1.0]], dtype=np.float32)
    result = FaissStore._float_to_binary_packbits(single_vector)
    assert result.shape == (1, 1)
    assert result[0, 0] == 2  # 0b10
    
    # Test with odd number of dimensions
    odd_vector = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
    result = FaissStore._float_to_binary_packbits(odd_vector)
    assert result.shape == (1, 1)
    # 3 bits: [1, 0, 1] -> 0b101 = 5
    assert result[0, 0] == 5
    
    # Test with empty array
    empty_vector = np.array([], dtype=np.float32).reshape(0, 4)
    result = FaissStore._float_to_binary_packbits(empty_vector)
    assert result.shape == (0, 1)

def test_binary_packbits_reversibility():
    """Test that the conversion process is mathematically sound"""
    
    # Create test vectors
    original = np.array([
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    ], dtype=np.float32)
    
    # Convert to binary
    binary = FaissStore._float_to_binary_packbits(original)
    
    # Verify binary representation
    expected_bits = [1, 0, 1, 0, 1, 0, 1, 0]  # [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    expected_byte = sum(bit << (7-i) for i, bit in enumerate(expected_bits))
    
    assert binary[0, 0] == expected_byte
