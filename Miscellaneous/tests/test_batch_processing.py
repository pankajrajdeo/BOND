import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from bond.pipeline import BondMatcher
from bond.config import BondSettings

@patch('bond.pipeline.sqlite3.connect')
@patch('bond.pipeline.FaissStore')
@patch('bond.pipeline.resolve_embeddings')
@patch('bond.pipeline.ChatLLM')
@patch('bond.pipeline.validate_embedding_signature')
def test_true_batch_query_calls_dependencies_correctly(mock_validate, mock_llm, mock_resolve, mock_faiss_store, mock_connect):
    """
    Test that the refactored batch_query method calls its dependencies
    with batched inputs instead of in a loop.
    """
    # --- Setup Mocks ---
    settings = BondSettings(enable_expansion=False, expansion_llm_model="openai/gpt-4o-mini", disambiguation_llm_model="openai/gpt-4o-mini") # Disable expansion for simpler testing
    
    # Mock embedding function to return predictable vectors
    mock_embed_fn = MagicMock(return_value=[[0.1]*16, [0.2]*16, [0.3]*16])
    mock_resolve.return_value = mock_embed_fn

    # Mock FAISS store and its search method
    mock_faiss_instance = MagicMock()
    # ✅ FIX: Mock the new FaissStore.search() method returning List[List[str]]
    mock_faiss_instance.search.return_value = [["ID:0","ID:1","ID:2"], ["ID:3","ID:4","ID:5"]]
    mock_faiss_store.return_value = mock_faiss_instance
    
    # Mock the DB connection cursor
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [] # No metadata needed for this test
    mock_connect.return_value.cursor.return_value = mock_cursor

    # --- Test Execution ---
    with patch('bond.pipeline.search_exact') as mock_search_exact, \
         patch('bond.pipeline.search_bm25') as mock_search_bm25:
        
        # Mock return values for retrieval functions
        mock_search_exact.return_value = [{'id': 'E:1', 'label': 't-cell'}]
        mock_search_bm25.return_value = [{'id': 'B:1'}]

        matcher = BondMatcher(settings)
        items = [
            {"query": "T-cell", "field_name": "cell type"},
            {"query": "cancer", "field_name": "disease"}
        ]
        matcher.batch_query(items)

        # --- Assertions ---
        # 1. Assert embedding function was called ONCE with all unique queries
        mock_embed_fn.assert_called_once()
        # The argument should be a list of all unique queries
        assert set(mock_embed_fn.call_args[0][0]) == {"T-cell", "cancer"}

        # 2. Assert FAISS search was called ONCE with a batch of vectors
        mock_faiss_instance.search.assert_called_once()
        # The first argument should be a numpy array of shape (num_queries, dim)
        faiss_input_vectors = mock_faiss_instance.search.call_args[0][0]
        assert isinstance(faiss_input_vectors, np.ndarray)
        assert faiss_input_vectors.shape[0] == 2  # number of unique queries in the batch

        # 3. Assert exact search was called ONCE with a list of normalized queries
        mock_search_exact.assert_called_once()
        # The second argument should be a list of normalized query strings
        assert set(mock_search_exact.call_args[1]['q_norms']) == {"t-cell", "cancer"}
        
        # 4. Assert BM25 was called twice (once per query, as it's not batched)
        assert mock_search_bm25.call_count == 2

def test_batch_query_with_expansions():
    """Test batch query with query expansion enabled"""
    
    with patch('bond.pipeline.sqlite3.connect'), \
         patch('bond.pipeline.FaissStore'), \
         patch('bond.pipeline.resolve_embeddings'), \
         patch('bond.pipeline.ChatLLM'), \
         patch('bond.pipeline.validate_embedding_signature'):

        settings = BondSettings(enable_expansion=True, n_expansions=2, expansion_llm_model="openai/gpt-4o-mini", disambiguation_llm_model="openai/gpt-4o-mini")
        matcher = BondMatcher(settings)

        # Mock expansion LLM to return expansions
        matcher.expansion_llm = MagicMock()
        matcher.expansion_llm.text.return_value = '{"expansions": ["T lymphocyte", "Tcell"], "context_terms": []}'

        # Mock embedding function
        mock_embed_fn = MagicMock(return_value=[[0.1]*16, [0.2]*16, [0.3]*16, [0.4]*16])
        mock_resolve.return_value = mock_embed_fn

        # Mock FAISS
        mock_faiss_instance = MagicMock()
        mock_faiss_instance.search.return_value = [["ID:0","ID:1","ID:2"], ["ID:3","ID:4","ID:5"]]
        mock_faiss_store.return_value = mock_faiss_instance

        # Mock DB
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_connect.return_value.cursor.return_value = mock_cursor

        with patch('bond.pipeline.search_exact') as mock_search_exact, \
             patch('bond.pipeline.search_bm25') as mock_search_bm25:
            
            mock_search_exact.return_value = [{'id': 'E:1', 'label': 't-cell'}]
            mock_search_bm25.return_value = [{'id': 'B:1'}]

            matcher = BondMatcher(settings)
            items = [
                {"query": "T-cell", "field_name": "cell type"},
                {"query": "cancer", "field_name": "disease"}
            ]
            
            results = matcher.batch_query(items)

            # Verify LLM was called for expansions
            assert matcher.llm.text.call_count == 2

            # Verify results
            assert len(results) == 2

def test_batch_query_empty_list():
    """Test batch query with empty input"""
    
    with patch('bond.pipeline.sqlite3.connect'), \
         patch('bond.pipeline.FaissStore'), \
         patch('bond.pipeline.resolve_embeddings'), \
         patch('bond.pipeline.ChatLLM'), \
         patch('bond.pipeline.validate_embedding_signature'):

        settings = BondSettings(expansion_llm_model="openai/gpt-4o-mini", disambiguation_llm_model="openai/gpt-4o-mini")
        matcher = BondMatcher(settings)

        results = matcher.batch_query([])

        # Should return empty list
        assert results == []

def test_batch_query_single_item():
    """Test batch query with single item (edge case)"""
    
    with patch('bond.pipeline.sqlite3.connect'), \
         patch('bond.pipeline.FaissStore'), \
         patch('bond.pipeline.resolve_embeddings'), \
         patch('bond.pipeline.ChatLLM'), \
         patch('bond.pipeline.validate_embedding_signature'):

        settings = BondSettings(expansion_llm_model="openai/gpt-4o-mini", disambiguation_llm_model="openai/gpt-4o-mini")
        matcher = BondMatcher(settings)

        # Mock FAISS
        mock_faiss_instance = MagicMock()
        mock_faiss_instance.search.return_value = [["ID:0","ID:1","ID:2"]]
        mock_faiss_store.return_value = mock_faiss_instance

        # Mock DB
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_connect.return_value.cursor.return_value = mock_cursor

        with patch('bond.pipeline.search_exact') as mock_search_exact, \
             patch('bond.pipeline.search_bm25') as mock_search_bm25:
            
            mock_search_exact.return_value = [{'id': 'E:1', 'label': 't-cell'}]
            mock_search_bm25.return_value = [{'id': 'B:1'}]

            items = [{"query": "T-cell", "field_name": "cell type"}]

            results = matcher.batch_query(items)

            # Verify results
            assert len(results) == 1

def test_batch_query_error_handling():
    """Test that batch query handles errors gracefully"""
    
    with patch('bond.pipeline.sqlite3.connect'), \
         patch('bond.pipeline.FaissStore'), \
         patch('bond.pipeline.resolve_embeddings'), \
         patch('bond.pipeline.ChatLLM'), \
         patch('bond.pipeline.validate_embedding_signature'):

        settings = BondSettings(expansion_llm_model="openai/gpt-4o-mini", disambiguation_llm_model="openai/gpt-4o-mini")
        matcher = BondMatcher(settings)

        # Mock FAISS to raise an error
        mock_faiss_instance = MagicMock()
        mock_faiss_instance.search.side_effect = Exception("FAISS error")
        mock_faiss_store.return_value = mock_faiss_instance

        # Mock DB
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_connect.return_value.cursor.return_value = mock_cursor

        items = [
            {"query": "T-cell", "field_name": "cell type"},
            {"query": "cancer", "field_name": "disease"}
        ]

        results = matcher.batch_query(items)

        # Should return error results for both items
        assert len(results) == 2
        assert all("error" in result for result in results)
        assert "FAISS error" in results[0]["error"]
