"""
Tests for HTTP provider adapters in bond/providers.py
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from bond.providers import resolve_embeddings, ChatLLM, _dot_get

class TestDotGet:
    """Test the _dot_get helper function"""
    
    def test_simple_path(self):
        data = {"a": {"b": "value"}}
        assert _dot_get(data, "a.b") == "value"
    
    def test_array_access(self):
        data = {"items": [{"id": 1}, {"id": 2}]}
        assert _dot_get(data, "items[0].id") == 1
        assert _dot_get(data, "items[1].id") == 2
    
    def test_missing_path(self):
        data = {"a": "b"}
        assert _dot_get(data, "a.b.c") is None
        assert _dot_get(data, "missing", default="default") == "default"
    
    def test_empty_path(self):
        data = {"a": "b"}
        assert _dot_get(data, "") == data

class TestHTTPEmbeddings:
    """Test HTTP embedding provider"""
    
    @patch.dict(os.environ, {
        "BOND_HTTP_EMBED_URL": "https://test.com/embed",
        "BOND_HTTP_EMBED_METHOD": "POST",
        "BOND_HTTP_EMBED_HEADERS": '{"Authorization":"Bearer test"}',
        "BOND_HTTP_EMBED_INPUT_KEY": "texts",
        "BOND_HTTP_EMBED_OUTPUT_PATH": "data.embeddings"
    })
    @patch('bond.providers.requests.request')
    def test_http_embedding_success(self, mock_request):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        embed_fn = resolve_embeddings("http:test")
        result = embed_fn(["hello", "world"])
        
        # Verify request was made correctly
        mock_request.assert_called_once_with(
            "POST", "https://test.com/embed",
            headers={"Authorization": "Bearer test"},
            json={"texts": ["hello", "world"]},
            timeout=60
        )
        
        # Verify result is normalized
        assert len(result) == 2
        assert len(result[0]) == 3
        # Check normalization (should be unit vectors)
        assert abs(sum(x*x for x in result[0]) - 1.0) < 1e-6
    
    @patch.dict(os.environ, {
        "BOND_HTTP_EMBED_URL": "https://test.com/embed"
    })
    def test_http_embedding_missing_url(self):
        with pytest.raises(ValueError, match="BOND_HTTP_EMBED_URL is required"):
            resolve_embeddings("http:test")
    
    @patch.dict(os.environ, {
        "BOND_HTTP_EMBED_URL": "https://test.com/embed",
        "BOND_HTTP_EMBED_OUTPUT_PATH": "data.embeddings"
    })
    @patch('bond.providers.requests.request')
    def test_http_embedding_invalid_response(self, mock_request):
        # Mock response with invalid embedding path
        mock_response = MagicMock()
        mock_response.json.return_value = {"wrong": "path"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        embed_fn = resolve_embeddings("http:test")
        with pytest.raises(RuntimeError, match="not found or invalid"):
            embed_fn(["hello"])

class TestHTTPLLM:
    """Test HTTP LLM provider"""
    
    @patch.dict(os.environ, {
        "BOND_HTTP_LLM_URL": "https://test.com/chat",
        "BOND_HTTP_LLM_METHOD": "POST",
        "BOND_HTTP_LLM_HEADERS": '{"X-API-Key":"test123"}',
        "BOND_HTTP_LLM_INPUT_KEY": "prompt",
        "BOND_HTTP_LLM_OUTPUT_PATH": "response.text"
    })
    @patch('bond.providers.requests.request')
    def test_http_llm_success(self, mock_request):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": {
                "text": "This is the answer"
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        llm = ChatLLM("http:test")
        result = llm.text("What is 2+2?")
        
        # Verify request was made correctly
        mock_request.assert_called_once_with(
            "POST", "https://test.com/chat",
            headers={"X-API-Key": "test123"},
            json={"prompt": "What is 2+2?", "temperature": 0.0, "max_tokens": 512},
            timeout=60
        )
        
        assert result == "This is the answer"
    
    @patch.dict(os.environ, {
        "BOND_HTTP_LLM_URL": "https://test.com/chat"
    })
    def test_http_llm_missing_url(self):
        with pytest.raises(ValueError, match="BOND_HTTP_LLM_URL is required"):
            ChatLLM("http:test")
    
    @patch.dict(os.environ, {
        "BOND_HTTP_LLM_URL": "https://test.com/chat",
        "BOND_HTTP_LLM_OUTPUT_PATH": "response.text"
    })
    @patch('bond.providers.requests.request')
    def test_http_llm_invalid_response(self, mock_request):
        # Mock response with invalid output path
        mock_response = MagicMock()
        mock_response.json.return_value = {"wrong": "path"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response
        
        llm = ChatLLM("http:test")
        with pytest.raises(RuntimeError, match="not found or not a string"):
            llm.text("test")

class TestProviderAliases:
    """Test provider convenience aliases"""
    
    def test_ollama_alias(self):
        # provider alias routes to litellm
        fn = resolve_embeddings("ollama:test-model")
        assert callable(fn)

if __name__ == "__main__":
    pytest.main([__file__])
