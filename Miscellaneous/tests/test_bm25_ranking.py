import pytest
import sqlite3
import tempfile
import os
from bond.retrieval.bm25_sqlite import search_bm25

def test_bm25_ranking():
    """Test that BM25 ranking works correctly and changes with query"""
    
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
        
        # Insert test data
        test_data = [
            ("T1", "T-cell", "A type of lymphocyte", "t-cell", "CL"),
            ("T2", "T lymphocyte", "Another name for T-cell", "t lymphocyte", "CL"),
            ("T3", "B-cell", "A different lymphocyte", "b-cell", "CL"),
            ("T4", "T-cell receptor", "Receptor on T-cells", "t-cell receptor", "CL")
        ]
        
        for item in test_data:
            cur.execute("INSERT INTO terms(id,label,def,norm_label,source) VALUES (?,?,?,?,?)", item)
        
        # Populate FTS
        cur.execute("INSERT INTO terms_fts(rowid, label, def) SELECT rowid, label, def FROM terms")
        conn.commit()
        
        # Test search with different queries
        results1 = search_bm25(conn, "terms", "terms_fts", "T-cell", 3)
        results2 = search_bm25(conn, "terms", "terms_fts", "lymphocyte", 3)
        
        # Verify results are different for different queries
        assert len(results1) > 0
        assert len(results2) > 0
        
        # Verify T-cell related terms appear in first query
        t_cell_ids = [r["id"] for r in results1]
        assert "T1" in t_cell_ids or "T2" in t_cell_ids
        
        # Verify lymphocyte appears in second query
        lymph_ids = [r["id"] for r in results2]
        assert any(id in lymph_ids for id in ["T1", "T2", "T3"])
        
    finally:
        conn.close()
        os.unlink(db_path)

def test_bm25_returns_metadata():
    """Test that BM25 search returns all expected metadata fields"""
    
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
                   ("T1", "Test", "Definition", "test", "TEST"))
        cur.execute("INSERT INTO terms_fts(rowid, label, def) SELECT rowid, label, def FROM terms")
        conn.commit()
        
        results = search_bm25(conn, "terms", "terms_fts", "Test", 1)
        
        assert len(results) == 1
        result = results[0]
        assert "id" in result
        assert "label" in result
        assert "def" in result
        assert "source" in result
        
    finally:
        conn.close()
        os.unlink(db_path)
