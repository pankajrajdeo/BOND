#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from pathlib import Path


def connect_db(assets_path: str) -> sqlite3.Connection:
    db_path = Path(assets_path) / "ontology.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_stats(conn: sqlite3.Connection, limit: int | None = None) -> dict:
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total_terms FROM terms")
    total_terms = int(cur.fetchone()[0])

    cur.execute("SELECT COUNT(DISTINCT source) AS num_sources FROM terms")
    num_sources = int(cur.fetchone()[0])

    cur.execute(
        """
        SELECT source, COUNT(*) as n
        FROM terms
        GROUP BY source
        ORDER BY n DESC, source ASC
        """
    )
    by_source_all = [dict(row) for row in cur.fetchall()]
    by_source = by_source_all[:limit] if limit else by_source_all

    # Source status / versions if available
    versions = []
    try:
        cur.execute(
            """
            SELECT source,
                   COALESCE(last_updated_authoritative, '') AS last_updated_authoritative,
                   COALESCE(last_seen_via_import, '')      AS last_seen_via_import,
                   COALESCE(data_version, '')              AS data_version
            FROM source_status
            ORDER BY source
            """
        )
        versions = [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        versions = []

    return {
        "total_terms": total_terms,
        "num_sources": num_sources,
        "by_source": by_source,
        "versions": versions,
    }


def print_human(stats: dict) -> None:
    print(f"Total terms: {stats['total_terms']}")
    print(f"Sources:     {stats['num_sources']}")
    print("")
    print("Terms by source (desc):")
    for row in stats["by_source"]:
        print(f"- {row['source']}: {row['n']}")
    if stats.get("versions"):
        print("")
        print("Source versions/status:")
        for v in stats["versions"]:
            print(
                f"- {v['source']}: version='{v['data_version']}' auth='{v['last_updated_authoritative']}' import='{v['last_seen_via_import']}'"
            )


def main():
    ap = argparse.ArgumentParser(description="Show ontology statistics from the BOND SQLite database")
    ap.add_argument("--assets", default=os.getenv("BOND_ASSETS_PATH", "assets"), help="Path to assets directory (default: $BOND_ASSETS_PATH or ./assets)")
    ap.add_argument("--limit", type=int, default=20, help="Show only top-N sources by count (default: 20; 0 = all)")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of human-readable text")
    args = ap.parse_args()

    conn = connect_db(args.assets)
    try:
        limit = None if args.limit == 0 else args.limit
        stats = get_stats(conn, limit=limit)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_human(stats)


if __name__ == "__main__":
    main()


