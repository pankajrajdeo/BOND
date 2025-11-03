import pytest
import tempfile
import os
import sqlite3
from bond.retrieval.bm25_sqlite import search_exact, search_bm25

def test_ontology_filtering_exact_search():
    """Test that exact search respects ontology filtering"""
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Create tables
        cur.execute("""
            CREATE TABLE terms (
                id TEXT PRIMARY KEY, label TEXT, def TEXT, norm_label TEXT, source TEXT
            )""")
        cur.execute("""
            CREATE VIRTUAL TABLE terms_fts USING fts5(label, def, content='terms', content_rowid='rowid')
        """)
        
        # Insert test data with different sources
        test_data = [
            ("T1", "T-cell", "A type of lymphocyte", "t-cell", "cl"),
            ("T2", "T-cell", "Another T-cell term", "t-cell", "mondo"),
            ("T3", "B-cell", "A different lymphocyte", "b-cell", "cl"),
            ("T4", "T-cell receptor", "Receptor on T-cells", "t-cell receptor", "uberon")
        ]
        
        for item in test_data:
            cur.execute("INSERT INTO terms(id,label,def,norm_label,source) VALUES (?,?,?,?,?)", item)
        
        # Populate FTS
        cur.execute("INSERT INTO terms_fts(rowid, label, def) SELECT rowid, label, def FROM terms")
        conn.commit()
        
        # Test without filter - should return all T-cell terms
        results_no_filter = search_exact(conn, "terms", "t-cell", 10)
        assert len(results_no_filter) == 2  # T1, T2 (T4 has norm_label "t-cell receptor")
        
        # Test with cl filter - should return only cl terms
        results_cl_filter = search_exact(conn, "terms", "t-cell", 10, sources=["cl"])
        assert len(results_cl_filter) == 1
        assert results_cl_filter[0]["id"] == "T1"
        
        # Test with mondo filter - should return only mondo terms
        results_mondo_filter = search_exact(conn, "terms", "t-cell", 10, sources=["mondo"])
        assert len(results_mondo_filter) == 1
        assert results_mondo_filter[0]["id"] == "T2"
        
        # Test with multiple sources filter
        results_multi_filter = search_exact(conn, "terms", "t-cell", 10, sources=["cl", "mondo"])
        assert len(results_multi_filter) == 2
        source_ids = {r["id"] for r in results_multi_filter}
        assert source_ids == {"T1", "T2"}
        
    finally:
        conn.close()
        os.unlink(db_path)

def test_ontology_filtering_bm25_search():
    """Test that BM25 search respects ontology filtering"""
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE terms (
                id TEXT PRIMARY KEY, label TEXT, def TEXT, norm_label TEXT, source TEXT
            )""")
        cur.execute("""
            CREATE VIRTUAL TABLE terms_fts USING fts5(label, def, content='terms', content_rowid='rowid')
        """)
        
        # Insert test data
        test_data = [
            ("T1", "T-cell", "A type of lymphocyte", "t-cell", "cl"),
            ("T2", "T-cell", "Another T-cell term", "t-cell", "mondo"),
            ("T3", "B-cell", "A different lymphocyte", "b-cell", "cl"),
            ("T4", "T-cell receptor", "Receptor on T-cells", "t-cell receptor", "uberon")
        ]
        
        for item in test_data:
            cur.execute("INSERT INTO terms(id,label,def,norm_label,source) VALUES (?,?,?,?,?)", item)
        
        cur.execute("INSERT INTO terms_fts(rowid, label, def) SELECT rowid, label, def FROM terms")
        conn.commit()
        
        # Test without filter
        results_no_filter = search_bm25(conn, "terms", "terms_fts", "T-cell", 10)
        assert len(results_no_filter) >= 2  # Should find T1 and T2
        
        # Test with cl filter
        results_cl_filter = search_bm25(conn, "terms", "terms_fts", "T-cell", 10, sources=["cl"])
        assert len(results_cl_filter) == 1
        assert results_cl_filter[0]["id"] == "T1"
        
        # Test with multiple sources
        results_multi_filter = search_bm25(conn, "terms", "terms_fts", "T-cell", 10, sources=["cl", "mondo"])
        assert len(results_multi_filter) == 2
        
    finally:
        conn.close()
        os.unlink(db_path)

def test_ontology_filtering_edge_cases():
    """Test edge cases for ontology filtering"""
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE terms (
                id TEXT PRIMARY KEY, label TEXT, def TEXT, norm_label TEXT, source TEXT
            )""")
        cur.execute("""
            CREATE VIRTUAL TABLE terms_fts USING fts5(label, def, content='terms', content_rowid='rowid')
        """)
        
        cur.execute("INSERT INTO terms(id,label,def,norm_label,source) VALUES (?,?,?,?,?)", 
                   ("T1", "Test", "Definition", "test", "cl"))
        cur.execute("INSERT INTO terms_fts(rowid, label, def) SELECT rowid, label, def FROM terms")
        conn.commit()
        
        # Test with empty sources list (should behave like no filter)
        results_empty = search_exact(conn, "terms", "test", 10, sources=[])
        assert len(results_empty) == 1
        
        # Test with None sources (should behave like no filter)
        results_none = search_exact(conn, "terms", "test", 10, sources=None)
        assert len(results_none) == 1
        
        # Test with non-existent source (should return empty)
        results_bad_source = search_exact(conn, "terms", "test", 10, sources=["nonexistent"])
        assert len(results_bad_source) == 0
        
    finally:
        conn.close()
        os.unlink(db_path)
