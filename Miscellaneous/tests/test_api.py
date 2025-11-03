"""
Tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from bond.server import app
from bond.models import QueryResponse

client = TestClient(app)

@pytest.fixture
def mock_matcher():
    """Mock BondMatcher for testing"""
    with patch('bond.server.get_matcher') as mock_get:
        mock_matcher = MagicMock()
        mock_get.return_value = mock_matcher
        yield mock_matcher

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "BOND API"
    assert "timestamp" in data
    assert "version" in data

def test_query_endpoint_missing_auth(mock_matcher):
    """Test query endpoint requires authentication when API key is set"""
    with patch.dict('os.environ', {'BOND_API_KEY': 'test-key'}):
        response = client.post("/query", json={"query": "test"})
        assert response.status_code == 401
        assert "Missing Authorization header" in response.json()["detail"]

def test_query_endpoint_with_auth(mock_matcher):
    """Test query endpoint with valid authentication"""
    # Mock the query response
    mock_matcher.query.return_value = {
        "results": [
            {
                "id": "test:1",
                "label": "Test Term",
                "source": "test",
                "definition": "A test term",
                "fusion_score": 0.8,
                "confidence": 0.9
            }
        ],
        "chosen": {
            "id": "test:1",
            "label": "Test Term",
            "source": "test",
            "definition": "A test term",
            "fusion_score": 0.8,
            "confidence": 0.9
        }
    }
    
    with patch.dict('os.environ', {'BOND_API_KEY': 'test-key'}):
        response = client.post(
            "/query", 
            json={"query": "test"},
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
    
    # Verify the mock was called
    mock_matcher.query.assert_called_once()

def test_batch_query_endpoint(mock_matcher):
    """Test batch query endpoint"""
    # Mock the batch query response
    mock_matcher.batch_query.return_value = [
        {
            "results": [{"id": "test:1", "label": "Test", "source": "test", "definition": "Test", "fusion_score": 0.8, "confidence": 0.9}],
            "chosen": {"id": "test:1", "label": "Test", "source": "test", "definition": "Test", "fusion_score": 0.8, "confidence": 0.9}
        }
    ]
    
    with patch.dict('os.environ', {'BOND_API_KEY': 'test-key'}):
        response = client.post(
            "/batch_query",
            json={"items": [{"query": "test"}]},
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
    
    # Verify the mock was called
    mock_matcher.batch_query.assert_called_once()

def test_config_endpoint(mock_matcher):
    """Test config endpoint"""
    # Mock the config
    mock_matcher.cfg.model_dump.return_value = {
        "assets_path": "assets",
        "embed_model": "st:test",
        "expansion_llm_model": "openai/gpt-4o-mini",
        "disambiguation_llm_model": "openai/gpt-4o-mini",
        "enable_expansion": True,
        "n_expansions": 3,
        "topk_final": 10,
        "rrf_k": 60.0
    }
    
    with patch.dict('os.environ', {'BOND_API_KEY': 'test-key'}):
        response = client.get(
            "/config",
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
    
    # Verify the mock was called
    mock_matcher.cfg.model_dump.assert_called_once()

def test_ontologies_endpoint(mock_matcher):
    """Test ontologies endpoint"""
    # Mock the ontologies
    mock_matcher.get_available_ontologies.return_value = ["cl", "mondo"]
    # Mock the new connection pattern
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [1000]
    mock_matcher.get_connection.return_value.cursor.return_value = mock_cursor
    
    with patch.dict('os.environ', {'BOND_API_KEY': 'test-key'}):
        response = client.get(
            "/ontologies",
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
    
    # Verify the mock was called
    mock_matcher.get_available_ontologies.assert_called_once()

def test_cache_endpoints(mock_matcher):
    """Test cache management endpoints"""
    # Cache is disabled; endpoints return zeros
    
    with patch.dict('os.environ', {'BOND_API_KEY': 'test-key'}):
        # Test cache stats
        response = client.get(
            "/cache/stats",
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        assert response.json()["cache_size"] == 0
        
        # Test cache clear
        response = client.post(
            "/cache/clear",
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Cache disabled"

def test_gpu_status_endpoint(mock_matcher):
    """Test GPU status endpoint"""
    # GPU not applicable for binary index
    mock_matcher.faiss.get_gpu_status.return_value = {
        "gpu_requested": False,
        "gpu_available": False,
        "gpu_active": False
    }
    
    with patch.dict('os.environ', {'BOND_API_KEY': 'test-key'}):
        response = client.get(
            "/gpu-status",
            headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["gpu_requested"] is False
        assert data["gpu_available"] is False

def test_error_handling():
    """Test error handling for invalid requests"""
    # Test invalid JSON
    response = client.post("/query", data="invalid json")
    assert response.status_code == 422
    
    # Test missing required fields
    response = client.post("/query", json={})
    assert response.status_code == 422
    assert "query" in response.json()["detail"][0]["loc"]
