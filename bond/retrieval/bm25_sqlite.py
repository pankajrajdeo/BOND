import sqlite3
from typing import List, Dict, Optional

def _escape_fts_phrase(s: str) -> str:
    return s.replace('"', '""')

def search_exact(
    conn: sqlite3.Connection,
    table_terms: str,
    table_fts: str,
    q_norms: List[str],
    k: int,
    sources: Optional[List[str]] = None,
) -> List[Dict]:
    """Exact match via FTS over the provided FTS table (label + synonyms_*).

    Args:
        conn: Open SQLite connection
        table_terms: Name of the base terms table (content source for FTS)
        table_fts: Name of the FTS virtual table (should mirror text columns)
        q_norms: Normalized exact phrases to match
        k: Max results per phrase (deduplicated overall)
        sources: Optional ontology_id filter
    Returns:
        List of dicts with keys: {id, label}
    """
    cur = conn.cursor()
    if isinstance(q_norms, str):
        q_norms = [q_norms]
    if not q_norms:
        return []

    clauses = []
    for q in q_norms:
        p = _escape_fts_phrase(q)
        clauses.append(
            f"(label:\"{p}\" OR synonyms_exact:\"{p}\" OR synonyms_related:\"{p}\" OR synonyms_broad:\"{p}\" OR synonyms_narrow:\"{p}\")"
        )
    match_expr = " OR ".join(clauses)

    base = (
        f"SELECT t.curie, t.label, t.ontology_id FROM {table_fts} "
        f"JOIN {table_terms} t ON t.rowid = {table_fts}.rowid "
        f"WHERE {table_fts} MATCH ?"
    )
    params: List = [match_expr]
    if sources:
        placeholders = ",".join("?" for _ in sources)
        base += f" AND t.ontology_id IN ({placeholders})"
        params.extend(sources)

    try:
        base += f" ORDER BY bm25({table_fts}) ASC LIMIT ?"
        params.append(k * len(q_norms))
        rows = cur.execute(base, params).fetchall()
    except sqlite3.OperationalError:
        rows = cur.execute(base + " LIMIT ?", params + [k * len(q_norms)]).fetchall()

    out = []
    seen = set()
    for curie, label, _src in rows:
        if curie not in seen:
            seen.add(curie)
            out.append({"id": curie, "label": label})
    return out[: k * len(q_norms)]

def search_bm25(conn: sqlite3.Connection, table_terms: str, table_fts: str, q: str, k: int, sources: Optional[List[str]] = None) -> List[Dict]:
    """BM25 over ontology_terms_fts only."""
    cur = conn.cursor()
    match_param = '"' + q.replace('"', '""') + '"'
    base_query = (
        f"SELECT t.curie, t.label, t.definition, t.ontology_id "
        f"FROM {table_fts} JOIN {table_terms} t ON t.rowid = {table_fts}.rowid "
        f"WHERE {table_fts} MATCH ?"
    )
    params = [match_param]
    if sources:
        placeholders = ",".join("?" for _ in sources)
        base_query += f" AND t.ontology_id IN ({placeholders})"
        params.extend(sources)

    try:
        base_query += f" ORDER BY bm25({table_fts}) ASC LIMIT ?"
        params.append(k)
        cur.execute(base_query, params)
    except sqlite3.OperationalError:
        base_query = base_query.replace(f"ORDER BY bm25({table_fts}) ASC", "ORDER BY length(t.label) ASC")
        cur.execute(base_query, params)

    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0],
            "label": row[1],
            "def": row[2],
            "source": row[3],
        })
    return results
"""SQLite FTS retrieval helpers for BOND.

This module exposes two retrieval channels over the text index:
  - search_exact: phrase-style matching against label and synonym columns
  - search_bm25: BM25-ranked retrieval over the same FTS virtual table

Both functions accept the base terms table and the FTS table names, and support
optional ontology_id scoping via the `sources` parameter.
"""
