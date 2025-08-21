#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

def try_import_rdflib():
    try:
        import rdflib  # type: ignore
        from rdflib import Graph
        return Graph
    except Exception:
        return None

def audit_with_pronto(path: str) -> Dict[str, Any]:
    from pronto import Ontology  # lazy import
    stats = {
        "file": os.path.basename(path),
        "source_name": os.path.splitext(os.path.basename(path))[0],
        "terms_total": 0,
        "terms_with_label": 0,
        "terms_with_definition": 0,
        "terms_with_syn_exact": 0,
        "terms_with_syn_related": 0,
        "terms_with_syn_broad": 0,
        "terms_with_http_id": 0,
        "notes": [],
    }
    onto = Ontology(path)
    for term in onto.terms():
        if getattr(term, "obsolete", False):
            continue
        stats["terms_total"] += 1

        label = getattr(term, "name", None)
        if label:
            stats["terms_with_label"] += 1

        # pronto maps textual def to .definition
        definition = getattr(term, "definition", None)
        if definition and str(definition).strip():
            stats["terms_with_definition"] += 1

        # synonyms by scope
        has_exact = False
        has_related = False
        has_broad = False
        for syn in getattr(term, "synonyms", []) or []:
            text = getattr(syn, "description", None) or getattr(syn, "name", None)
            if not text:
                continue
            scope = None
            try:
                scope = str(getattr(syn, "scope", None) or "").upper()
            except Exception:
                scope = None
            if scope == "EXACT":
                has_exact = True
            elif scope == "RELATED":
                has_related = True
            elif scope == "BROAD":
                has_broad = True
        if has_exact:
            stats["terms_with_syn_exact"] += 1
        if has_related:
            stats["terms_with_syn_related"] += 1
        if has_broad:
            stats["terms_with_syn_broad"] += 1

        tid = getattr(term, "id", "") or ""
        if tid.startswith("http://") or tid.startswith("https://"):
            stats["terms_with_http_id"] += 1

    return stats

def audit_with_rdflib(path: str) -> Optional[Dict[str, Any]]:
    Graph = try_import_rdflib()
    if Graph is None:
        return None
    g = Graph()
    g.parse(path)

    # Known predicates (match by localname to avoid full prefix headaches)
    def is_pred(p, local: str) -> bool:
        ps = str(p)
        return ps.endswith(local)

    subjects_exact = set()
    subjects_related = set()
    subjects_broad = set()
    subjects_label = set()
    subjects_def = set()

    for s, p, o in g.triples((None, None, None)):
        if is_pred(p, "hasExactSynonym") or is_pred(p, "hasExactSynonym#"):
            subjects_exact.add(s)
        elif is_pred(p, "hasRelatedSynonym"):
            subjects_related.add(s)
        elif is_pred(p, "hasBroadSynonym"):
            subjects_broad.add(s)
        elif is_pred(p, "prefLabel") or is_pred(p, "label"):
            subjects_label.add(s)
        elif is_pred(p, "IAO_0000115") or is_pred(p, "definition") or is_pred(p, "Definition"):
            subjects_def.add(s)
        # RadLex-specific extras
        elif str(p).endswith("Preferred_name"):
            subjects_label.add(s)
        elif str(p).endswith("Synonym_English") or str(p).endswith("Synonym_German"):
            subjects_related.add(s)

    # Estimate term count by counting unique subjects that have any label or definition
    subjects_any = set().union(subjects_label, subjects_def, subjects_exact, subjects_related, subjects_broad)
    return {
        "rdflib_terms_estimate": len(subjects_any),
        "rdflib_with_label": len(subjects_label),
        "rdflib_with_definition": len(subjects_def),
        "rdflib_with_syn_exact": len(subjects_exact),
        "rdflib_with_syn_related": len(subjects_related),
        "rdflib_with_syn_broad": len(subjects_broad),
    }

def main():
    ap = argparse.ArgumentParser(description="Audit OWL contents for labels/definitions/synonyms")
    ap.add_argument("--dir", default="data", help="Directory containing OWL/OBO files")
    ap.add_argument("--json", action="store_true", help="Print JSON summary")
    args = ap.parse_args()

    files = []
    if os.path.isdir(args.dir):
        for fn in sorted(os.listdir(args.dir)):
            if fn.lower().endswith((".owl", ".obo")):
                files.append(os.path.join(args.dir, fn))
    else:
        print(f"Directory not found: {args.dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for path in files:
        try:
            pstats = audit_with_pronto(path)
        except Exception as e:
            pstats = {
                "file": os.path.basename(path),
                "source_name": os.path.splitext(os.path.basename(path))[0],
                "error": f"pronto parse failed: {e}",
            }
        try:
            rstats = audit_with_rdflib(path)
            if rstats:
                pstats.update(rstats)
        except Exception as e:
            pstats.setdefault("notes", []).append(f"rdflib scan failed: {e}")
        results.append(pstats)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # tabular text
        cols = [
            "file", "terms_total", "terms_with_label", "terms_with_definition",
            "terms_with_syn_exact", "terms_with_syn_related", "terms_with_syn_broad", "terms_with_http_id",
            "rdflib_terms_estimate", "rdflib_with_label", "rdflib_with_definition",
            "rdflib_with_syn_exact", "rdflib_with_syn_related", "rdflib_with_syn_broad",
        ]
        header = "\t".join(cols)
        print(header)
        for r in results:
            row = [str(r.get(c, "")) for c in cols]
            print("\t".join(row))

if __name__ == "__main__":
    main()


