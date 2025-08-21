import sqlite3
from typing import List, Dict, Optional

def _normalize(s: str) -> str:
    return " ".join(s.lower().split())

def search_exact(conn: sqlite3.Connection, table_terms: str, q_norms: List[str], k: int, sources: Optional[List[str]] = None) -> List[Dict]:
    """
    Performs an exact match search. Can handle a single query (as str) or a batch (as list).
    """
    cur = conn.cursor()

    # Ensure q_norms is a list for consistent processing
    if isinstance(q_norms, str):
        q_norms = [q_norms]

    if not q_norms:
        return []

    # Create placeholders for the IN clause
    norm_placeholders = ",".join("?" for _ in q_norms)

    # Match against normalized label
    label_query = f"SELECT id, label FROM {table_terms} WHERE norm_label IN ({norm_placeholders})"
    params: List = list(q_norms)
    if sources:
        source_placeholders = ",".join("?" for _ in sources)
        label_query += f" AND source IN ({source_placeholders})"
        params.extend(sources)

    # Match against ALL synonym types (exact, related, broad, generic).
    # Preferred: use JSON1 (json_each). Fallback: string search for '"<syn>"' inside the JSON string.
    use_json1 = True
    try:
        # Quick probe to see if JSON1 is available on this connection
        cur.execute("SELECT json('[]')")
    except sqlite3.OperationalError:
        use_json1 = False

    if use_json1:
        # Search across all synonym columns
        syn_query = (
            f"SELECT t.id, t.label FROM {table_terms} t WHERE "
            f"EXISTS (SELECT 1 FROM json_each(t.syn_exact) je WHERE lower(trim(je.value)) IN ({norm_placeholders})) OR "
            f"EXISTS (SELECT 1 FROM json_each(t.syn_related) je WHERE lower(trim(je.value)) IN ({norm_placeholders})) OR "
            f"EXISTS (SELECT 1 FROM json_each(t.syn_broad) je WHERE lower(trim(je.value)) IN ({norm_placeholders})) OR "
            f"EXISTS (SELECT 1 FROM json_each(t.syn_generic) je WHERE lower(trim(je.value)) IN ({norm_placeholders}))"
        )
        syn_params: List = list(q_norms) * 4  # 4 copies for the 4 synonym types
        if sources:
            source_placeholders = ",".join("?" for _ in sources)
            syn_query += f" AND t.source IN ({source_placeholders})"
            syn_params.extend(sources)
    else:
        # Fallback without JSON1: search for quoted synonym token within ALL synonym JSON columns
        syn_conditions = []
        for _ in q_norms:
            syn_conditions.extend([
                "instr(lower(COALESCE(t.syn_exact,'')), ?) > 0",
                "instr(lower(COALESCE(t.syn_related,'')), ?) > 0", 
                "instr(lower(COALESCE(t.syn_broad,'')), ?) > 0",
                "instr(lower(COALESCE(t.syn_generic,'')), ?) > 0"
            ])
        syn_query = f"SELECT t.id, t.label FROM {table_terms} t WHERE ({' OR '.join(syn_conditions)})"
        syn_params: List = []
        for q in q_norms:
            quoted_q = '"' + q + '"'
            syn_params.extend([quoted_q, quoted_q, quoted_q, quoted_q])  # 4 copies for 4 synonym types
        if sources:
            source_placeholders = ",".join("?" for _ in sources)
            syn_query += f" AND t.source IN ({source_placeholders})"
            syn_params.extend(sources)

    # Union label and synonym matches, de-duplicate, and order deterministically
    # Execute separately and merge in Python to avoid UNION syntax issues across SQLite builds
    rows = []
    try:
        rows.extend(cur.execute(label_query, params).fetchall())
    except sqlite3.OperationalError:
        pass
    try:
        rows.extend(cur.execute(syn_query, syn_params).fetchall())
    except sqlite3.OperationalError:
        pass

    # De-duplicate by id, prefer shortest label
    seen = {}
    for r in rows:
        rid, rlabel = r[0], r[1]
        if rid not in seen or len(rlabel or "") < len(seen[rid] or ""):
            seen[rid] = rlabel

    merged = [{"id": rid, "label": lbl} for rid, lbl in seen.items()]
    merged.sort(key=lambda x: (len(x["label"] or ""), x["id"]))
    return merged[: k * len(q_norms)]

def search_bm25(conn: sqlite3.Connection, table_terms: str, table_fts: str, q: str, k: int, sources: Optional[List[str]] = None) -> List[Dict]:
    cur = conn.cursor()
    
    # Use parameterized MATCH with a quoted phrase to avoid token parsing issues (e.g., hyphens)
    # Use parameterized phrase query for MATCH; surround with quotes to keep multi-word phrases together
    match_param = '"' + q.replace('"', '""') + '"'
    base_query = f"""SELECT t.id, t.label, t.def, t.source
                      FROM {table_fts}
                      JOIN {table_terms} t ON t.rowid = {table_fts}.rowid
                     WHERE {table_fts} MATCH ?"""
    params = [match_param]
    
    if sources:
        placeholders = ",".join("?" for _ in sources)
        base_query += f" AND t.source IN ({placeholders})"
        params.extend(sources)
        
    # Try to use bm25() ranking function, fallback to simple LIMIT if not available
    try:
        base_query += f" ORDER BY bm25({table_fts}) ASC LIMIT ?"
        params.append(k)
        cur.execute(base_query, params)
    except sqlite3.OperationalError:
        # Fallback for SQLite builds without bm25() function
        base_query = base_query.replace(f"ORDER BY bm25({table_fts}) ASC", "ORDER BY length(t.label) ASC")
        cur.execute(base_query, params)
    
    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0],
            "label": row[1], 
            "def": row[2],
            "source": row[3]
        })
    return results
