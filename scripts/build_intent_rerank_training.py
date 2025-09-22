#!/usr/bin/env python3
"""
Builds training data for an intent-based semantic reranker from CELLxGENE metadata.

Inputs (defaults assume repo layout under Miscellaneous/CellxGene_Benchmark/benchmark_data):
  - harmonized_data_sample.csv           (organism / tissue by dataset_id)
  - 03b_llm_generated_column_mappings.json (author/ontology columns per dataset)
  - metadata_schemas/                    (raw per-dataset CSVs)

Output:
  - intent_rerank_training.jsonl         (one example per row with query/positive/candidates)

Notes:
  - Uses BOND in retrieval_only mode; no LLM calls.
  - Reuses the existing FAISS store and SQLite to generate hard negatives.
  - Does not re-embed any ontology terms.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from bond.pipeline import BondMatcher
from bond.config import BondSettings
import sqlite3


FIELDS = [
    "assay",
    "cell_type",
    "development_stage",
    "disease",
    "self_reported_ethnicity",
    "sex",
    "tissue",
]


def load_context(harmonized_csv: Path) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(harmonized_csv, low_memory=False)
    # Use first non-null organism/tissue per dataset
    cols = ["dataset_id", "organism", "tissue"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Harmonized CSV missing columns: {missing}")
    g = (
        df[cols]
        .dropna(subset=["dataset_id"])
        .groupby("dataset_id", as_index=False)
        .first()
        .set_index("dataset_id")
        .to_dict("index")
    )
    return {str(k): {"organism": v.get("organism"), "tissue": v.get("tissue")} for k, v in g.items()}


def build_intent_text(query: str, field_name: str, organism: str | None, tissue: str | None, expansions: List[str] | None, context_terms: List[str] | None) -> str:
    parts: List[str] = []
    if field_name:
        parts.append(f"field={field_name}")
    if organism:
        parts.append(f"organism={organism}")
    if tissue:
        parts.append(f"tissue={tissue}")
    parts.append(f"query={query}")
    if expansions:
        alts = [x for x in expansions[:4] if x and x != query]
        if alts:
            parts.append("alts=" + " | ".join(alts))
    if context_terms:
        parts.append("ctx=" + ", ".join(context_terms[:5]))
    return " ; ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-root", default="Miscellaneous/CellxGene_Benchmark/benchmark_data", help="Root directory containing harmonized CSV, mappings, and metadata_schemas")
    p.add_argument("--mappings", default=None, help="Path to 03b_llm_generated_column_mappings.json (defaults under benchmark-root)")
    p.add_argument("--harmonized", default=None, help="Path to harmonized_data_sample.csv (defaults under benchmark-root)")
    p.add_argument("--schemas", default=None, help="Path to metadata_schemas directory (defaults under benchmark-root)")
    p.add_argument("--output", default=None, help="Output JSONL path (defaults under benchmark-root)")
    p.add_argument("--topk", type=int, default=20, help="Top-k candidates to retain per example")
    p.add_argument("--skip-obsolete", action="store_true", default=True, help="Skip rows where positive ID is obsolete (per local DB)")
    p.add_argument("--include-candidate-metadata", action="store_true", help="Include label/definition/synonyms for candidate IDs as well")
    args = p.parse_args()

    root = Path(args.benchmark_root)
    mappings_path = Path(args.mappings) if args.mappings else (root / "03b_llm_generated_column_mappings.json")
    harmonized_path = Path(args.harmonized) if args.harmonized else (root / "harmonized_data_sample.csv")
    schemas_dir = Path(args.schemas) if args.schemas else (root / "metadata_schemas")
    out_path = Path(args.output) if args.output else (root / "intent_rerank_training.jsonl")

    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings not found: {mappings_path}")
    if not harmonized_path.exists():
        raise FileNotFoundError(f"Harmonized CSV not found: {harmonized_path}")
    if not schemas_dir.exists():
        raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")

    with open(mappings_path, "r") as f:
        mappings: Dict[str, Dict] = json.load(f)
    context = load_context(harmonized_path)

    # Initialize BOND in retrieval-only mode
    cfg = BondSettings(retrieval_only=True)
    matcher = BondMatcher(cfg)

    # SQLite connection for metadata hydration
    conn = sqlite3.connect(f"file:{matcher.db_path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    def hydrate(ids: List[str]) -> Dict[str, Dict]:
        if not ids:
            return {}
        qmarks = ",".join("?" for _ in ids)
        cur = conn.cursor()
        cur.execute(
            f"SELECT curie, label, definition, ontology_id, synonyms_exact, synonyms_related, synonyms_broad, is_obsolete, term_doc FROM {matcher.cfg.table_terms} WHERE curie IN ({qmarks})",
            ids,
        )
        out = {}
        for row in cur.fetchall():
            def _split(x):
                if not x:
                    return None
                vals = [t.strip() for t in str(x).split("|") if t.strip()]
                return vals or None
            out[row["curie"]] = {
                "label": row["label"],
                "definition": row["definition"],
                "ontology": row["ontology_id"],
                "synonyms_exact": _split(row["synonyms_exact"]),
                "synonyms_related": _split(row["synonyms_related"]),
                "synonyms_broad": _split(row["synonyms_broad"]),
                "is_obsolete": bool(row["is_obsolete"]),
                "term_doc": row["term_doc"],
            }
        return out

    n_written = 0
    seen = set()  # dedupe by (field, organism, tissue, query, positive)
    with out_path.open("w") as outf:
        for ds_id, mp in tqdm(mappings.items(), desc="Datasets"):
            schema_csv = schemas_dir / f"{ds_id}_metadata.csv"
            if not schema_csv.exists():
                continue
            df = pd.read_csv(schema_csv, low_memory=False, on_bad_lines="skip")
            org = (context.get(ds_id) or {}).get("organism")
            tis = (context.get(ds_id) or {}).get("tissue")
            for field in tqdm(FIELDS, desc=f"Fields ({ds_id})", leave=False):
                rec = mp.get(field, {}) or {}
                a_col = rec.get("author_term_column")
                o_col = rec.get("ontology_id_column")
                if not a_col or not o_col or a_col not in df.columns or o_col not in df.columns:
                    continue
                sub = df[[a_col, o_col]].dropna()
                # Filter out blanks/unknowns
                BAD = {"", "na", "n/a", "nan", "none", "null", "unknown", "unspecified", "unassigned"}
                def _bad(s: str) -> bool:
                    s = str(s).strip().lower()
                    return (s in BAD) or (len(s) == 0)
                for _, row in tqdm(sub.iterrows(), desc=f"Rows ({ds_id}/{field})", total=len(sub), leave=False):
                    q = str(row[a_col]).strip()
                    pos = str(row[o_col]).strip()
                    if _bad(q) or _bad(pos) or ":" not in pos:
                        continue
                    key = (field, org or "", tis or "", q.lower(), pos)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Skip obsolete positives if requested
                    pos_meta = hydrate([pos]).get(pos)
                    if args.skip_obsolete and pos_meta and pos_meta.get("is_obsolete"):
                        continue
                    if not q or not pos or ":" not in pos:
                        continue
                    # Run retrieval-only BOND to get candidates and context terms
                    try:
                        out = matcher.query(q, field, org or "", tis or "", topk_final=args.topk, return_trace=True, exact_only=False)
                    except Exception as e:
                        # Skip failed queries but log occasionally
                        if n_written % 100 == 0:
                            print(f"⚠️  Skipped query '{q[:30]}...' due to: {str(e)[:50]}...")
                        continue
                    trace = out.get("trace") or {}
                    cand = trace.get("candidates") or {}
                    # Merge candidate IDs, preserve rough ordering by channels
                    all_ids: List[str] = []
                    for ch in ["exact", "bm25", "dense", "bm25_ctx", "dense_full"]:
                        for i in cand.get(ch, []) or []:
                            if i not in all_ids:
                                all_ids.append(i)
                    # Ensure gold present
                    if pos not in all_ids:
                        all_ids = [pos] + all_ids
                    # Build intent text mirroring matcher
                    expansions = trace.get("expansions") or []
                    ctx_terms = trace.get("context_terms") or []
                    intent = build_intent_text(q, field, org, tis, expansions, ctx_terms)
                    # Hydrate positive and optionally candidate metadata
                    pos_meta = pos_meta or {}
                    cand_meta = hydrate(all_ids[: args.topk]) if args.include_candidate_metadata else None
                    rec_out = {
                        "dataset_id": ds_id,
                        "field": field,
                        "organism": org,
                        "tissue": tis,
                        "query": q,
                        "positive_id": pos,
                        "candidates": all_ids[: args.topk],
                        "intent": intent,
                        "context_terms": ctx_terms,
                        "positive_meta": pos_meta,
                        **({"candidate_meta": cand_meta} if cand_meta is not None else {}),
                    }
                    outf.write(json.dumps(rec_out) + "\n")
                    n_written += 1

    matcher.close()
    conn.close()
    print(f"✅ Successfully wrote {n_written} examples to {out_path}")
    print(f"📊 Processed {len(seen)} unique query/positive pairs")


if __name__ == "__main__":
    main()
