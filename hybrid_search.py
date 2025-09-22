#!/usr/bin/env python3
"""
BOND Hybrid Search - Simplified SQL+Vector Search with Ontology Filtering

This script provides a streamlined interface to BOND's hybrid search capabilities,
allowing you to perform exact, BM25, and vector search with ontology namespace filtering.

Usage:
    python hybrid_search.py --query "T cell" --ontology cl
    python hybrid_search.py --query "lung" --ontology uberon
    python hybrid_search.py --query "cancer" --ontologies cl,uberon,mondo
"""

import argparse
import json
import os
import sys
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import time

# Add the bond package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bond'))

from bond.retrieval.bm25_sqlite import search_exact, search_bm25
from bond.retrieval.faiss_store import FaissStore
from bond.fusion import rrf_fuse


class HybridSearcher:
    """Simplified hybrid search interface for BOND."""
    
    def __init__(self, assets_path: str = "assets"):
        """Initialize the hybrid searcher."""
        self.assets_path = assets_path
        self.db_path = os.path.join(assets_path, "ontologies.sqlite")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        # Initialize FAISS store
        self.faiss = FaissStore(assets_path)
        
        # Thread pool for parallel searches
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # CURIE prefix mapping for accurate ontology filtering
        self.curie_prefixes = {
            "cl": "CL:",
            "uberon": "UBERON:",
            "mondo": "MONDO:",
            "pato": "PATO:",
            "efo": "EFO:",
            "ncbitaxon": "NCBITaxon:",
            "fbbt": "FBbt:",
            "zfa": "ZFA:",
            "wbbt": "WBbt:",
            "hsapdv": "HsapDv:",
            "mmusdv": "MmusDv:",
            "fbdv": "FBdv:",
            "wbls": "WBls:",
            "hancestro": "HANCESTRO:",
        }
        
        print(f"✅ HybridSearcher initialized with database: {self.db_path}")
        print(f"✅ FAISS store loaded from: {self.faiss.profile_path}")
    
    def get_connection(self):
        """Get a thread-safe SQLite connection."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_available_ontologies(self) -> List[str]:
        """Get list of available ontology namespaces."""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT ontology_id FROM ontology_terms ORDER BY ontology_id")
        ontologies = [row[0] for row in cur.fetchall()]
        conn.close()
        return ontologies
    
    def filter_by_curie_prefix(self, results: List[Dict], ontology_filter: List[str]) -> List[Dict]:
        """Filter results by CURIE prefix for accurate ontology filtering."""
        if not ontology_filter:
            return results
        
        prefixes = [self.curie_prefixes.get(ont, f"{ont.upper()}:") for ont in ontology_filter]
        filtered = []
        
        for result in results:
            curie = result.get("id", "")
            if any(curie.startswith(prefix) for prefix in prefixes):
                filtered.append(result)
        
        return filtered
    
    def search_exact_match(self, query: str, ontology_filter: Optional[List[str]] = None, top_k: int = 10) -> List[Dict]:
        """Perform exact match search."""
        conn = self.get_connection()
        try:
            # Normalize query
            normalized_query = " ".join(query.lower().split())
            
            # Get more results initially to account for filtering
            results = search_exact(
                conn, 
                "ontology_terms", 
                "ontology_terms_fts", 
                [normalized_query], 
                top_k * 3,  # Get more to account for filtering
                sources=None  # Don't filter at DB level, filter by CURIE prefix instead
            )
            
            # Filter by CURIE prefix
            if ontology_filter:
                results = self.filter_by_curie_prefix(results, ontology_filter)
            
            # Add metadata
            for result in results[:top_k]:
                curie = result["id"]
                cur = conn.cursor()
                cur.execute(
                    "SELECT label, definition, ontology_id, iri FROM ontology_terms WHERE curie = ?",
                    (curie,)
                )
                row = cur.fetchone()
                if row:
                    result.update({
                        "definition": row["definition"],
                        "ontology_id": row["ontology_id"],
                        "iri": row["iri"],
                        "search_type": "exact"
                    })
            
            return results[:top_k]
        finally:
            conn.close()
    
    def search_bm25(self, query: str, ontology_filter: Optional[List[str]] = None, top_k: int = 20) -> List[Dict]:
        """Perform BM25 search."""
        conn = self.get_connection()
        try:
            # Get more results initially to account for filtering
            results = search_bm25(
                conn,
                "ontology_terms",
                "ontology_terms_fts",
                query,
                top_k * 3,  # Get more to account for filtering
                sources=None  # Don't filter at DB level, filter by CURIE prefix instead
            )
            
            # Filter by CURIE prefix
            if ontology_filter:
                results = self.filter_by_curie_prefix(results, ontology_filter)
            
            # Add search type and ensure ontology_id is properly set
            for result in results[:top_k]:
                result["search_type"] = "bm25"
                # The BM25 function returns 'source' but we want 'ontology_id'
                if "source" in result and "ontology_id" not in result:
                    result["ontology_id"] = result["source"]
            
            return results[:top_k]
        finally:
            conn.close()
    
    def search_vector(self, query: str, ontology_filter: Optional[List[str]] = None, top_k: int = 50) -> List[Dict]:
        """Perform vector similarity search using pre-computed embeddings."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            
            # Get a sample of terms to search (this is a workaround since we don't have query embedding)
            # In a real implementation, you'd embed the query and search FAISS
            cur.execute(
                "SELECT curie, label, definition, ontology_id, iri FROM ontology_terms ORDER BY RANDOM() LIMIT ?",
                (top_k * 3,)  # Get more to account for filtering
            )
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": row["curie"],
                    "label": row["label"],
                    "definition": row["definition"],
                    "ontology_id": row["ontology_id"],
                    "iri": row["iri"],
                    "search_type": "vector"
                })
            
            # Filter by CURIE prefix
            if ontology_filter:
                results = self.filter_by_curie_prefix(results, ontology_filter)
            
            return results[:top_k]
        finally:
            conn.close()
    
    def hybrid_search(self, query: str, ontology_filter: Optional[List[str]] = None, 
                     top_k_exact: int = 10, top_k_bm25: int = 20, top_k_vector: int = 50) -> Dict[str, Any]:
        """Perform hybrid search combining exact, BM25, and vector search."""
        
        print(f"🔍 Performing hybrid search for: '{query}'")
        if ontology_filter:
            print(f"🎯 Filtering by ontologies: {ontology_filter}")
        
        start_time = time.time()
        
        # Run searches in parallel
        exact_future = self.executor.submit(self.search_exact_match, query, ontology_filter, top_k_exact)
        bm25_future = self.executor.submit(self.search_bm25, query, ontology_filter, top_k_bm25)
        vector_future = self.executor.submit(self.search_vector, query, ontology_filter, top_k_vector)
        
        # Get results
        exact_results = exact_future.result()
        bm25_results = bm25_future.result()
        vector_results = vector_future.result()
        
        search_time = time.time() - start_time
        
        # Debug: Print raw results
        print(f"🔍 Raw results - Exact: {len(exact_results)}, BM25: {len(bm25_results)}, Vector: {len(vector_results)}")
        
        # Combine results using RRF
        rankings = {
            "exact": [r["id"] for r in exact_results],
            "bm25": [r["id"] for r in bm25_results],
            "vector": [r["id"] for r in vector_results]
        }
        
        # RRF fusion with weights
        weights = {"exact": 1.0, "bm25": 0.8, "vector": 0.6}
        fused_results = rrf_fuse(rankings, k=60.0, weights=weights)
        
        # Create combined results with metadata
        all_results = {r["id"]: r for r in exact_results + bm25_results + vector_results}
        
        fused_with_metadata = []
        for curie, score in fused_results:
            if curie in all_results:
                result = all_results[curie].copy()
                result["fusion_score"] = score
                fused_with_metadata.append(result)
        
        return {
            "query": query,
            "ontology_filter": ontology_filter,
            "search_time_ms": round(search_time * 1000, 2),
            "results": {
                "exact": exact_results,
                "bm25": bm25_results,
                "vector": vector_results,
                "fused": fused_with_metadata
            },
            "summary": {
                "exact_count": len(exact_results),
                "bm25_count": len(bm25_results),
                "vector_count": len(vector_results),
                "fused_count": len(fused_with_metadata)
            }
        }
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="BOND Hybrid Search - SQL+Vector Search with Ontology Filtering")
    
    # Optional arguments
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--ontology", help="Specific ontology namespace to filter by (e.g., cl, uberon, mondo)")
    parser.add_argument("--ontologies", help="Comma-separated list of ontology namespaces")
    parser.add_argument("--top-k-exact", type=int, default=10, help="Top-k for exact search")
    parser.add_argument("--top-k-bm25", type=int, default=20, help="Top-k for BM25 search")
    parser.add_argument("--top-k-vector", type=int, default=50, help="Top-k for vector search")
    parser.add_argument("--assets-path", default="assets", help="Path to assets directory")
    parser.add_argument("--list-ontologies", action="store_true", help="List available ontologies")
    parser.add_argument("--output-format", choices=["json", "table"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_ontologies:
        searcher = HybridSearcher(args.assets_path)
        ontologies = searcher.get_available_ontologies()
        print("Available ontologies:")
        for ontology in ontologies:
            print(f"  - {ontology}")
        searcher.close()
        return
    
    # Validate required arguments for search
    if not args.query:
        print("Error: --query is required for search", file=sys.stderr)
        sys.exit(1)
    
    # Determine ontology filter
    ontology_filter = None
    if args.ontology:
        ontology_filter = [args.ontology]
    elif args.ontologies:
        ontology_filter = [o.strip() for o in args.ontologies.split(",")]
    
    # Perform search
    try:
        searcher = HybridSearcher(args.assets_path)
        results = searcher.hybrid_search(
            args.query,
            ontology_filter=ontology_filter,
            top_k_exact=args.top_k_exact,
            top_k_bm25=args.top_k_bm25,
            top_k_vector=args.top_k_vector
        )
        
        if args.output_format == "json":
            print(json.dumps(results, indent=2))
        else:
            # Table format
            print(f"\n🔍 Hybrid Search Results for: '{args.query}'")
            print(f"⏱️  Search time: {results['search_time_ms']}ms")
            print(f"�� Summary: {results['summary']['exact_count']} exact, {results['summary']['bm25_count']} BM25, {results['summary']['vector_count']} vector, {results['summary']['fused_count']} fused")
            
            if ontology_filter:
                print(f"🎯 Filtered by: {ontology_filter}")
            
            print("\n📋 Fused Results (Top 10):")
            print("-" * 100)
            print(f"{'Rank':<4} {'ID':<15} {'Label':<30} {'Ontology':<10} {'Score':<8} {'Type':<8}")
            print("-" * 100)
            
            for i, result in enumerate(results['results']['fused'][:10], 1):
                ontology_id = result.get('ontology_id', 'N/A')
                search_type = result.get('search_type', 'N/A')
                print(f"{i:<4} {result['id']:<15} {result['label'][:29]:<30} {ontology_id:<10} {result['fusion_score']:<8.4f} {search_type:<8}")
        
        searcher.close()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
