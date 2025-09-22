from typing import Dict, List
import sqlite3


def compute_graph_neighbors(
    conn: sqlite3.Connection,
    table_terms: str,
    seed_ids: List[str],
    depth: int,
) -> Dict[str, int]:
    """Return neighbor node -> distance for graph expansion within ontology terms.

    Ensures neighbors exist in the terms table to avoid unresolvable IDs.
    """
    if depth <= 0 or not seed_ids:
        return {}
    cur = conn.cursor()
    neighbors: Dict[str, int] = {}
    seen = set(seed_ids)
    frontier = set(seed_ids)

    # Check edges table exists
    try:
        has_edges = bool(cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ontology_edges'").fetchone())
    except Exception:
        has_edges = False
    if not has_edges:
        return {}

    for d in range(1, depth + 1):
        if not frontier:
            break
        qmarks = ",".join("?" for _ in frontier)
        try:
            rows_p = cur.execute(
                f"SELECT DISTINCT e.target_curie FROM ontology_edges e JOIN {table_terms} t ON t.curie = e.target_curie WHERE e.source_curie IN ({qmarks})",
                list(frontier),
            ).fetchall()
            rows_c = cur.execute(
                f"SELECT DISTINCT e.source_curie FROM ontology_edges e JOIN {table_terms} t ON t.curie = e.source_curie WHERE e.target_curie IN ({qmarks})",
                list(frontier),
            ).fetchall()
        except Exception:
            break
        new_nodes = [r[0] for r in rows_p] + [r[0] for r in rows_c]
        next_frontier = []
        for nid in new_nodes:
            if nid and nid not in seen:
                seen.add(nid)
                neighbors[nid] = d
                next_frontier.append(nid)
        frontier = set(next_frontier)

    return neighbors

