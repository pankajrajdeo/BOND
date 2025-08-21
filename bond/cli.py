import argparse
import json
from .runtime_env import configure_runtime
import os
from .config import BondSettings
from .pipeline import BondMatcher

def main():
    # Configure runtime before importing heavy libs used downstream
    configure_runtime()
    # Default to WARNING logs for CLI unless user overrides
    os.environ.setdefault("BOND_LOG_LEVEL", "WARNING")
    p = argparse.ArgumentParser(description="BOND: Biomedical Ontology Normalization and Disambiguation")
    p.add_argument("--query", required=True, help="The term to harmonize.")
    p.add_argument("--field_name", default=None, help="Context: the name of the data field.")
    p.add_argument("--dataset_description", default=None, help="Context: a description of the dataset.")
    p.add_argument("--embed", default=None, help="Override embedding model (auto-detected: st:/litellm/http)")
    p.add_argument("--n_expansions", type=int, default=None, help="Number of query expansions to generate")
    p.add_argument("--topk_final", type=int, default=None, help="Number of final results to return")
    p.add_argument("--num_choices", type=int, default=None, help="Also return N alternatives in addition to the chosen")
    p.add_argument("--topk_exact", type=int, default=None, help="Override top-k for exact match retrieval")
    p.add_argument("--topk_bm25", type=int, default=None, help="Override top-k for BM25 retrieval")
    p.add_argument("--topk_dense", type=int, default=None, help="Override top-k for vector retrieval")
    p.add_argument("--rrf_k", type=float, default=None, help="Override RRF k parameter")
    p.add_argument("--return_trace", action="store_true", help="Include detailed trace information in output")
    p.add_argument("--verbose", action="store_true", help="Print expanded queries and top-k candidates to stderr")
    p.add_argument("--restrict_to_ontologies", nargs="+", default=None, help="Restrict search to specific ontology sources (e.g., mondo cl)")
    args = p.parse_args()

    cfg = BondSettings()
    if args.embed:
        cfg.embed_model = args.embed
    
    matcher = BondMatcher(cfg)
    result = matcher.query(
        args.query,
        field_name=args.field_name,
        dataset_description=args.dataset_description,
        n_expansions=args.n_expansions,
        topk_final=args.topk_final,
        num_choices=args.num_choices,
        topk_exact=args.topk_exact,
        topk_bm25=args.topk_bm25,
        topk_dense=args.topk_dense,
        rrf_k=args.rrf_k,
        return_trace=args.return_trace or args.verbose,
        restrict_to_ontologies=args.restrict_to_ontologies,
    )
    if args.verbose and "trace" in result:
        import sys
        trace = result["trace"]
        print("\n[verbose] expansions:", file=sys.stderr)
        for e in trace.get("expansions", []):
            print(f"  - {e}", file=sys.stderr)
        print("[verbose] fusion top-k:", file=sys.stderr)
        for fid, score in trace.get("fusion", []):
            print(f"  - {fid} (score={score:.6f})", file=sys.stderr)

    # Output formatting per new spec
    def _chosen_clean(ch):
        if not ch:
            return None
        ordered = {
            "id": ch.get("id"),
            "label": ch.get("label"),
            "definition": ch.get("definition"),
            "source": ch.get("source"),
            "iri": ch.get("iri"),
            "synonyms_exact": ch.get("synonyms_exact"),
            "synonyms_related": ch.get("synonyms_related"),
            "synonyms_broad": ch.get("synonyms_broad"),
            "synonyms_generic": ch.get("synonyms_generic"),
            "alt_ids": ch.get("alt_ids"),
            "xrefs": ch.get("xrefs"),
            "namespace": ch.get("namespace"),
            "subsets": ch.get("subsets"),
            "comments": ch.get("comments"),
            "parents_is_a": ch.get("parents_is_a"),
            "abstracts": ch.get("abstracts"),
        }
        return ordered

    chosen = _chosen_clean(result.get("chosen"))
    if args.verbose or args.return_trace:
        out = {
            "expansions": result.get("trace", {}).get("expansions", []),
            "context_terms": result.get("trace", {}).get("context_terms", []),
            "fusion_top_k": result.get("trace", {}).get("fusion", []),
            "trace": result.get("trace", {}),
            "results": result.get("results", []),
            "disambiguation": {
                **(chosen or {}),
                "reason": (result.get("chosen") or {}).get("reason"),
                "retrieval_confidence": (result.get("chosen") or {}).get("retrieval_confidence"),
                "llm_confidence": (result.get("chosen") or {}).get("llm_confidence"),
            },
            "alternatives": result.get("alternatives", []),
        }
    else:
        out = {
            **(chosen or {}),
            "reason": (result.get("chosen") or {}).get("reason"),
            "retrieval_confidence": (result.get("chosen") or {}).get("retrieval_confidence"),
            "llm_confidence": (result.get("chosen") or {}).get("llm_confidence"),
        }
        if args.num_choices and args.num_choices > 0:
            out = {
                **out,
                "alternatives": result.get("alternatives", [])
            }
    print(json.dumps(out, indent=2))
