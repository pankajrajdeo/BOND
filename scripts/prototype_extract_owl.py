#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Any, List, Tuple
import random
import re

# Lazy imports to avoid heavy deps on load
def _import_pronto():
    from pronto import Ontology
    return Ontology

def _import_rdflib():
    try:
        from rdflib import Graph
        return Graph
    except Exception:
        return None

def _normalize_http_to_curie_and_source(http_id: str, default_record_id: str, default_source: str) -> Tuple[str, str]:
    try:
        iri = http_id
        lower_iri = iri.lower()
        if lower_iri.startswith("http://purl.obolibrary.org/obo/") or lower_iri.startswith("https://purl.obolibrary.org/obo/"):
            tail = iri.rstrip("/").split("/")[-1]
            if "_" in tail:
                prefix, local = tail.split("_", 1)
                return f"{prefix}:{local}", prefix.lower()
            return default_record_id, default_source
        if "purl.org/sig/ont/fma/" in lower_iri:
            last = iri.rstrip("/").split("/")[-1]
            up = last.upper()
            if up.startswith("FMA") and up[3:].isdigit():
                return f"FMA:{up[3:]}", "fma"
            return default_record_id, "fma"
        if "//www.radlex.org/rid/" in lower_iri:
            last = iri.rstrip("/").split("/")[-1]
            up = last.upper()
            if up.startswith("RID") and up[3:].isdigit():
                return f"RID:{up[3:]}", "radlex"
            return default_record_id, "radlex"
        if lower_iri.startswith("http://identifiers.org/") or lower_iri.startswith("https://identifiers.org/"):
            parts = iri.strip("/").split("/")
            if len(parts) >= 2:
                key = parts[-2].lower()
                acc = parts[-1]
                mapping = {
                    "hgnc": "HGNC",
                    "ncbitaxon": "NCBITaxon",
                    "omim": "OMIM",
                    "orphanet": "ORPHA",
                    "uniprot": "UniProt",
                    "ensembl": None,
                }
                if key in mapping and mapping[key] is not None:
                    return f"{mapping[key]}:{acc}", mapping[key].lower()
                if ":" in acc:
                    pref = acc.split(":", 1)[0]
                    return acc, pref.lower()
            return default_record_id, default_source
        if "www.ebi.ac.uk" in lower_iri:
            last = iri.rstrip("/").split("/")[-1]
            if last.startswith("EFO_") and last[4:].isdigit():
                return f"EFO:{last[4:]}", "efo"
            return default_record_id, default_source
        if "www.orpha.net" in lower_iri:
            last = iri.rstrip("/").split("/")[-1]
            if last.startswith("Orphanet_") and last.split("_")[-1].isdigit():
                return f"ORPHA:{last.split('_')[-1]}", "orphanet"
            return default_record_id, "orphanet"
        if "www.bioassayontology.org" in lower_iri:
            last = iri.rstrip("/").split("/")[-1]
            if last.startswith("BAO_") and last[4:].isdigit():
                return f"BAO:{last[4:]}", "bao"
            return default_record_id, "bao"
        if "www.geneontology.org" in lower_iri:
            last = iri.rstrip("/").split("/")[-1]
            if last.startswith("GO_") and last[3:].isdigit():
                return f"GO:{last[3:]}", "go"
            return default_record_id, default_source
        if "dbpedia.org/resource/" in lower_iri:
            last = iri.rstrip("/").split("/")[-1]
            return f"DBPEDIA:{last}", "dbpedia"
        return default_record_id, default_source
    except Exception:
        return default_record_id, default_source

def _build_rdflib_maps(path: str, source_name: str) -> Dict[str, Dict[str, Any]]:
    Graph = _import_rdflib()
    if Graph is None:
        return {}
    g = Graph()
    g.parse(path)
    ann: Dict[str, Dict[str, Any]] = {}
    RADLEX = source_name.lower() == "radlex"

    def ensure(s):
        ent = ann.setdefault(
            str(s),
            {
                "labels": set(),
                "pref": None,
                "syn_exact": set(),
                "syn_related": set(),
                "syn_broad": set(),
                "defs": set(),
                "parents_is_a": set(),
            },
        )
        return ent

    from rdflib import RDFS, URIRef
    for s, p, o in g.triples((None, None, None)):
        try:
            ps = str(p)
            ps_lower = ps.lower()
            ent = ensure(s)
            # Parents (is_a)
            if p == RDFS.subClassOf and isinstance(o, URIRef):
                ent["parents_is_a"].add(str(o))
                continue
            # Labels
            if ps_lower.endswith("preflabel") or ps_lower.endswith("label"):
                ent["labels"].add(str(o))
            # Definitions (IAO:0000115 + fallbacks)
            elif ps_lower.endswith("iao_0000115") or ps_lower.endswith("/definition") or ps_lower.endswith("definition"):
                ent["defs"].add(str(o))
            # Synonyms (oboInOwl variants)
            elif "hasexactsynonym" in ps_lower:
                ent["syn_exact"].add(str(o))
            elif "hasrelatedsynonym" in ps_lower:
                ent["syn_related"].add(str(o))
            elif "hasbroadsynonym" in ps_lower:
                ent["syn_broad"].add(str(o))
            # Generic synonyms: any other predicate containing 'synonym'
            elif "synonym" in ps_lower:
                ent.setdefault("syn_generic", set()).add(str(o))
            # RadLex specific
            elif RADLEX and (ps_lower.endswith("preferred_name") or ps_lower.endswith("preferred name") or ps_lower.endswith("/preferred_name") or "preferred_name" in ps_lower):
                ent["pref"] = str(o)
            elif RADLEX and (ps_lower.endswith("synonym_english") or ps_lower.endswith("synonym_german") or ps_lower.endswith("related_modality") or ps_lower.endswith("/synonym") or ps_lower.endswith("synonym")):
                ent["syn_related"].add(str(o))
            # DBpedia abstracts
            if "dbpedia.org/ontology/abstract" in ps_lower:
                ent.setdefault("abstracts", []).append(str(o))
            # Xrefs
            if ps_lower.endswith("hasdbxref") or ps_lower.endswith("database_cross_reference"):
                ent.setdefault("xrefs", set()).add(str(o))
            # Alternate IDs
            if ps_lower.endswith("hasalternativeid") or ps_lower.endswith("alternative_id"):
                ent.setdefault("alt_ids", set()).add(str(o))
            # Namespace / ontology name (e.g., GO namespace)
            if ps_lower.endswith("hasobonamespace"):
                ent.setdefault("namespace", set()).add(str(o))
            # Subsets
            if ps_lower.endswith("insubset") or ps_lower.endswith("in_subset"):
                ent.setdefault("subsets", set()).add(str(o))
            # Comments
            if ps_lower.endswith("comment"):
                ent.setdefault("comments", set()).add(str(o))
        except Exception:
            continue
    # Build auxiliary key map for subjects by tail for resilient matching (especially RadLex)
    aux: Dict[str, Dict[str, Any]] = {}
    for subj, info in ann.items():
        if not isinstance(info, dict):
            continue
        tail = str(subj).rstrip('/').split('/')[-1].lower()
        aux[tail] = info
        # RadLex variants: RID####, RID_####
        m = re.match(r"rid[_-]?(\d+)$", tail, flags=re.IGNORECASE)
        if m:
            num = m.group(1)
            aux[f"rid{num}"] = info
            aux[f"rid_{num}"] = info
    ann["__aux__"] = aux
    return ann

def _extract_samples(path: str, limit: int = 500) -> Dict[str, Any]:
    Ontology = _import_pronto()
    source_name = os.path.splitext(os.path.basename(path))[0]
    # Sanitize creation date patterns like 'creation: 16MAY2017' if present
    def _sanitize(p: str) -> str:
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            return p
        pattern = r"(?i)creation(?:[_\-\s]?date)?\s*[:=]\s*(\d{1,2})([A-Za-z]{3})(\d{4})"
        def repl(m):
            day, mon, year = m.group(1), m.group(2), m.group(3)
            mon_map = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06','jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12'}
            mm = mon_map.get(mon.lower(), '01')
            return f"{year}-{mm}-{day.zfill(2)}"
        new = re.sub(pattern, repl, text)
        if new == text:
            return p
        tmp = p + ".san.owl"
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(new)
        return tmp
    spath = _sanitize(path)
    onto = Ontology(spath)
    rdflib_ann = _build_rdflib_maps(spath, source_name)
    # Collect all non-obsolete terms and sample randomly across the ontology
    all_terms = [t for t in onto.terms() if not getattr(t, "obsolete", False)]
    if len(all_terms) > limit:
        sampled_terms = random.sample(all_terms, k=limit)
    else:
        sampled_terms = all_terms

    out: List[Dict[str, Any]] = []
    for term in sampled_terms:

        label = getattr(term, "name", None) or ""
        tid = getattr(term, "id", "") or ""
        iri = None
        record_id = tid
        curie_prefix = source_name.lower()
        if tid.startswith("http://") or tid.startswith("https://"):
            iri = tid
            record_id, curie_prefix = _normalize_http_to_curie_and_source(tid, record_id, curie_prefix)
        elif ":" in tid:
            pref, local = tid.split(":", 1)
            iri = f"http://purl.obolibrary.org/obo/{pref}_{local}"
            curie_prefix = pref.lower()
        else:
            iri = None

        syn_exact: List[str] = []
        syn_related: List[str] = []
        syn_broad: List[str] = []
        syn_generic: List[str] = []
        alt_ids: List[str] = []
        xrefs: List[str] = []
        namespace: List[str] = []
        subsets: List[str] = []
        comments: List[str] = []
        abstracts: List[str] = []
        # pronto synonyms
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
                syn_exact.append(text)
            elif scope == "RELATED":
                syn_related.append(text)
            elif scope == "BROAD":
                syn_broad.append(text)

        # rdflib augmentation
        if rdflib_ann:
            iri_self = None
            if tid.startswith("http://") or tid.startswith("https://"):
                iri_self = tid
            elif ":" in tid:
                pref, local = tid.split(":", 1)
                if source_name.lower() == "radlex" and pref.upper() == "RID":
                    iri_self = f"http://www.radlex.org/RID/{pref}{local}"
                else:
                    iri_self = f"http://purl.obolibrary.org/obo/{pref}_{local}"
            info = None
            if iri_self and iri_self in rdflib_ann:
                info = rdflib_ann[iri_self]
            elif iri_self:
                aux = rdflib_ann.get("__aux__") if isinstance(rdflib_ann, dict) else None
                if isinstance(aux, dict):
                    tail = iri_self.rstrip('/').split('/')[-1].lower()
                    info = aux.get(tail)
            if info is not None:
                if (not label) or label.upper().startswith("RID"):
                    if info.get("pref"):
                        label = info["pref"]
                    else:
                        for lb in info.get("labels", set()):
                            if not lb.upper().startswith("RID"):
                                label = lb
                                break
                # definition fallback
                rdefs = list(info.get("defs", set()))
                definition = getattr(term, "definition", None)
                definition = str(definition) if definition else None
                if (not definition) and rdefs:
                    definition = rdefs[0]
                # synonyms merge
                for s in info.get("syn_exact", set()):
                    if s not in syn_exact:
                        syn_exact.append(s)
                for s in info.get("syn_related", set()):
                    if s not in syn_related:
                        syn_related.append(s)
                for s in info.get("syn_broad", set()):
                    if s not in syn_broad:
                        syn_broad.append(s)
                for s in info.get("syn_generic", set()):
                    if s not in syn_generic:
                        syn_generic.append(s)
                # extras
                for s in info.get("alt_ids", set()):
                    if s not in alt_ids:
                        alt_ids.append(s)
                for s in info.get("xrefs", set()):
                    if s not in xrefs:
                        xrefs.append(s)
                for s in info.get("namespace", set()):
                    if s not in namespace:
                        namespace.append(s)
                for s in info.get("subsets", set()):
                    if s not in subsets:
                        subsets.append(s)
                for s in info.get("comments", set()):
                    if s not in comments:
                        comments.append(s)
                for s in info.get("abstracts", []):
                    if s not in abstracts:
                        abstracts.append(s)
            else:
                definition = getattr(term, "definition", None)
                definition = str(definition) if definition else None
        else:
            definition = getattr(term, "definition", None)
            definition = str(definition) if definition else None

        out.append({
            "id": record_id,
            "label": label,
            "definition": definition,
            "source": curie_prefix,
            "iri": iri,
            "synonyms_exact": syn_exact or [],
            "synonyms_related": syn_related or [],
            "synonyms_broad": syn_broad or [],
            "synonyms_generic": syn_generic or [],
            "alt_ids": alt_ids or [],
            "xrefs": xrefs or [],
            "namespace": namespace or [],
            "subsets": subsets or [],
            "comments": comments or [],
            "abstracts": abstracts or [],
        })

    return {"source": source_name.lower(), "count": len(out), "samples": out}

def main():
    ap = argparse.ArgumentParser(description="Prototype extractor to preview parsing strategy")
    ap.add_argument("--dir", default="data", help="Directory containing OWL/OBO files")
    ap.add_argument("--out", default="assets/preview_owl_samples.json", help="Output JSON path")
    ap.add_argument("--limit", type=int, default=500, help="Samples per file (random)")
    args = ap.parse_args()

    files = []
    for fn in sorted(os.listdir(args.dir)):
        if fn.lower().endswith((".owl", ".obo")):
            files.append(os.path.join(args.dir, fn))

    results: List[Dict[str, Any]] = []
    for p in files:
        try:
            res = _extract_samples(p, args.limit)
            results.append(res)
        except Exception as e:
            results.append({"source": os.path.basename(p), "error": str(e)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote preview samples to {args.out}")

if __name__ == "__main__":
    main()


