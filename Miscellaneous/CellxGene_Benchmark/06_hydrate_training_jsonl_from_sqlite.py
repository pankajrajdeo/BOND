#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrate compiled benchmark pairs with ontology metadata (no extra filtering),
produce ML-ready JSONL splits and a biologist-friendly CSV table.

WHAT THIS DOES
--------------
1) Reads the compiled CSV from the previous step
   (compiled_training_data_raw_fixed.csv).
2) For each ontology_id:
   - loads original term (label/def/synonyms/is_obsolete)
   - loads replaced_by / consider
   - resolves obsolete chains via replaced_by (resolution_path)
   - loads resolved term metadata
3) Writes:
   a) Three JSONL files: *_train.jsonl, *_dev.jsonl, *_test.jsonl
      - FLATTENED fields (no giant "mapping" object)
      - Arrays preserved for synonyms & resolution_path
   b) One publication table CSV: benchmark_readable_table.csv
      - Clear column order
      - Short, readable previews for synonyms (no walls of text)

IMPORTANT
---------
- NO new filtering is performed here. This script trusts the previous compiler.
- If a term is missing in SQLite (e.g., an ontology you didn't load), we
  still write the row with empty metadata and match_status="missing".

OUTPUT FILES
------------
JSONL base:
  /Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/\
Miscellaneous/CellxGene_Benchmark/benchmark_data/bond_czi_benchmark_data_hydrated_{split}.jsonl

CSV (publication-friendly):
  /Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/\
Miscellaneous/CellxGene_Benchmark/benchmark_data/benchmark_readable_table.csv
"""

import os
import json
import sqlite3
import random
from typing import List, Dict, Tuple, Optional

import pandas as pd
from tqdm.auto import tqdm

# -----------------------
# CONFIG
# -----------------------
INPUT_CSV  = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/compiled_training_data_raw_fixed.csv"
SQLITE_DB  = "/Users/rajlq7/Downloads/Terms/BOND/assets/ontologies.sqlite"
OUTPUT_BASE = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/bond_czi_benchmark_data_hydrated"

# Publication/table output (for HF viewer & papers)
READABLE_TABLE_CSV = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/benchmark_readable_table.csv"

# Synonym preview caps for the CSV (JSONL keeps full arrays)
SYN_PREVIEW_CAP_EXACT = 15
SYN_PREVIEW_CAP_OTHER = 15

# -----------------------
# HELPERS (formatting only)
# -----------------------
def _norm_text(x: Optional[str]) -> str:
    if x is None:
        return ''
    if not isinstance(x, str):
        x = str(x)
    return ' '.join(x.strip().lower().split())

def _normalize_syn_list_str(s: Optional[str]) -> List[str]:
    """
    Input is a '|' separated string from the DB; output is a unique-preserving list.
    """
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split("|") if p and p.strip()]
    seen, out = set(), []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out

def s(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return ''
    return str(x)

def f(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None

def i(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0
        return int(x)
    except Exception:
        return 0

def _ensure_array_format(value):
    """
    Ensure array fields are properly formatted for Hugging Face datasets.
    - Convert None to empty list
    - Ensure all elements are strings
    - Remove any None elements from arrays
    """
    if value is None:
        return []
    if isinstance(value, list):
        # Filter out None values and ensure all elements are strings
        return [str(item) for item in value if item is not None]
    return []

# -----------------------
# SQLITE access
# -----------------------
def _fetch_term(conn: sqlite3.Connection, curie: str):
    if not curie:
        return None
    row = conn.execute(
        """
        SELECT curie,label,definition,
               synonyms_exact,
               synonyms_narrow,
               synonyms_broad,
               synonyms_related,
               COALESCE(is_obsolete,0) AS is_obsolete
        FROM ontology_terms WHERE curie=?
        """,
        (curie,)
    ).fetchone()
    if not row:
        return None
    curie, label, definition, s_ex, s_na, s_br, s_re, is_obs = row
    return {
        "curie": curie,
        "label": label or "",
        "definition": definition or "",
        "synonyms": {
            "exact": _normalize_syn_list_str(s_ex),
            "narrow": _normalize_syn_list_str(s_na),
            "broad": _normalize_syn_list_str(s_br),
            "related": _normalize_syn_list_str(s_re),
        },
        "is_obsolete": 1 if (is_obs or is_obs == 1) else 0,
    }

def _fetch_replacements(conn: sqlite3.Connection, curie: str) -> Tuple[List[str], List[str]]:
    if not curie:
        return [], []
    rows = conn.execute(
        "SELECT replacement_curie, relation FROM term_replacement WHERE curie=?",
        (curie,)
    ).fetchall()
    replaced_by, consider = [], []
    for rep, rel in rows:
        if rel == 'replaced_by':
            replaced_by.append(rep)
        elif rel == 'consider':
            consider.append(rep)
    # stable order & dedup
    replaced_by = list(dict.fromkeys(replaced_by))
    consider    = list(dict.fromkeys(consider))
    return replaced_by, consider

def _resolve_curie(conn: sqlite3.Connection, curie: str, max_hops: int = 5) -> Tuple[str, List[str]]:
    """
    Follow replaced_by pointers up to max_hops, picking the first replacement
    that exists in ontology_terms. Returns (resolved_curie, resolution_path).
    """
    if not curie:
        return "", []
    path = [curie]
    visited = set([curie])
    current = curie
    hops = 0
    while hops < max_hops:
        t = _fetch_term(conn, current)
        if not t or not t.get("is_obsolete"):
            break
        repl, _ = _fetch_replacements(conn, current)
        next_cur = None
        for cand in repl:
            if _fetch_term(conn, cand):
                next_cur = cand
                break
        if not next_cur or next_cur in visited:
            break
        path.append(next_cur)
        visited.add(next_cur)
        current = next_cur
        hops += 1
    return current, path

# -----------------------
# MAIN
# -----------------------
def main():
    # Load compiled CSV from previous step
    df = pd.read_csv(INPUT_CSV)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows from {INPUT_CSV}")

    # Split by anchor (no filtering)
    key_series = (
        df['field_type'].astype(str).str.lower() + '|' +
        df['author_term'].map(_norm_text) + '|' +
        df['tissue'].map(_norm_text) + '|' +
        df['organism'].map(_norm_text)
    )
    unique_keys = key_series.dropna().unique().tolist()
    rnd = random.Random(7)
    rnd.shuffle(unique_keys)
    n = len(unique_keys)
    n_train = int(0.90 * n)
    n_dev = int(0.05 * n)
    train_keys = set(unique_keys[:n_train])
    dev_keys   = set(unique_keys[n_train:n_train+n_dev])
    test_keys  = set(unique_keys[n_train+n_dev:])
    def assign_split(k: str) -> str:
        if k in train_keys: return 'train'
        if k in dev_keys:   return 'dev'
        return 'test'
    df['__split_key'] = key_series
    df['split'] = df['__split_key'].map(assign_split)
    print(f"Split anchors -> train:{len(train_keys)} dev:{len(dev_keys)} test:{len(test_keys)}")

    # Open DB
    conn = sqlite3.connect(SQLITE_DB)

    # Prep JSONL writers
    out_dir = os.path.dirname(OUTPUT_BASE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    paths = {
        'train': OUTPUT_BASE + '_train.jsonl',
        'dev':   OUTPUT_BASE + '_dev.jsonl',
        'test':  OUTPUT_BASE + '_test.jsonl',
    }
    files = {name: open(p, 'w', encoding='utf-8') for name, p in paths.items()}

    # Collect rows for publication table CSV
    pub_rows: List[Dict] = []

    try:
        for _, r in tqdm(df.iterrows(), total=len(df), desc='Hydrating'):
            ont_id = s(r.get('ontology_id')).strip() or ""

            # Fetch original
            original = _fetch_term(conn, ont_id) if ont_id else None
            rb, cons = _fetch_replacements(conn, ont_id) if ont_id else ([], [])
            # Resolve chain
            resolved_id, res_path = _resolve_curie(conn, ont_id, max_hops=5) if ont_id else ("", [])
            resolved = _fetch_term(conn, resolved_id) if resolved_id else None

            # Match status
            if not ont_id:
                match_status = "missing"
            elif not original:
                # ID present but not found in DB
                match_status = "missing"
            elif resolved_id and resolved_id != ont_id:
                match_status = "replaced"
            else:
                match_status = "unchanged"

            # Flattened JSONL record (arrays preserved)
            rec = {
                # Dataset context
                'dataset_id':                     s(r.get('dataset_id')),
                'dataset_title':                  s(r.get('dataset_title')),
                'collection_name':                s(r.get('collection_name')),
                # Anchor context
                'field_type':                     s(r.get('field_type')),
                'organism':                       s(r.get('organism')),
                'tissue':                         s(r.get('tissue')),
                'author_term':                    s(r.get('author_term')),
                # Author→Ontology mapping (original)
                'original_ontology_id':           ont_id,
                'original_is_obsolete':           int(original.get('is_obsolete')) if original else 0,
                'original_label':                 original.get('label', "") if original else "",
                'original_definition':            original.get('definition', "") if original else "",
                'original_synonyms_exact':        _ensure_array_format(original.get('synonyms', {}).get('exact', []) if original else []),
                'original_synonyms_narrow':       _ensure_array_format(original.get('synonyms', {}).get('narrow', []) if original else []),
                'original_synonyms_broad':        _ensure_array_format(original.get('synonyms', {}).get('broad', []) if original else []),
                'original_synonyms_related':      _ensure_array_format(original.get('synonyms', {}).get('related', []) if original else []),
                'original_replaced_by':           _ensure_array_format(rb or []),
                'original_consider':              _ensure_array_format(cons or []),
                # Resolved mapping
                'resolved_ontology_id':           resolved_id or "",
                'resolved_label':                 resolved.get('label', "") if resolved else "",
                'resolved_definition':            resolved.get('definition', "") if resolved else "",
                'resolved_synonyms_exact':        _ensure_array_format(resolved.get('synonyms', {}).get('exact', []) if resolved else []),
                'resolved_synonyms_narrow':       _ensure_array_format(resolved.get('synonyms', {}).get('narrow', []) if resolved else []),
                'resolved_synonyms_broad':        _ensure_array_format(resolved.get('synonyms', {}).get('broad', []) if resolved else []),
                'resolved_synonyms_related':      _ensure_array_format(resolved.get('synonyms', {}).get('related', []) if resolved else []),
                'resolution_path':                _ensure_array_format(res_path or ([] if not ont_id else [ont_id])),
                'match_status':                   match_status,  # "unchanged" | "replaced" | "missing"
                # Evidence & bookkeeping
                'support_dataset_count':          i(r.get('support_dataset_count')),
                'support_row_count':              i(r.get('support_row_count')),
                'llm_predicted_author_column':    s(r.get('llm_predicted_author_column')),
                'author_confidence':              f(r.get('author_confidence')),
                'ontology_confidence':            f(r.get('ontology_confidence')),
                'split':                          s(r.get('split') or 'train'),
            }

            # JSONL write
            files[rec['split']].write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Publication row (short previews for synonyms)
            def preview(lst: List[str], cap: int) -> str:
                if not lst:
                    return ""
                cut = lst[:cap]
                s_join = "; ".join(cut)
                if len(lst) > cap:
                    s_join += f" …(+{len(lst)-cap})"
                return s_join

            pub_rows.append({
                # Order designed for HF viewers & papers
                "dataset_id":                  rec["dataset_id"],
                "dataset_title":               rec["dataset_title"],
                "collection_name":             rec["collection_name"],
                "field_type":                  rec["field_type"],
                "organism":                    rec["organism"],
                "tissue":                      rec["tissue"],
                "author_term":                 rec["author_term"],
                "original_ontology_id":        rec["original_ontology_id"],
                "original_label":              rec["original_label"],
                "original_is_obsolete":        rec["original_is_obsolete"],
                "resolved_ontology_id":        rec["resolved_ontology_id"],
                "resolved_label":              rec["resolved_label"],
                "match_status":                rec["match_status"],
                "resolution_path":             " → ".join(rec["resolution_path"]) if rec["resolution_path"] else "",
                "original_replaced_by":        "; ".join(rec["original_replaced_by"]) if rec["original_replaced_by"] else "",
                "original_consider":           "; ".join(rec["original_consider"]) if rec["original_consider"] else "",
                "original_definition":         rec["original_definition"],
                "resolved_definition":         rec["resolved_definition"],
                "original_synonyms_exact_preview":   preview(rec["original_synonyms_exact"],  SYN_PREVIEW_CAP_EXACT),
                "original_synonyms_other_preview":   preview(
                    rec["original_synonyms_narrow"] + rec["original_synonyms_broad"] + rec["original_synonyms_related"],
                    SYN_PREVIEW_CAP_OTHER),
                "resolved_synonyms_exact_preview":   preview(rec["resolved_synonyms_exact"],  SYN_PREVIEW_CAP_EXACT),
                "resolved_synonyms_other_preview":   preview(
                    rec["resolved_synonyms_narrow"] + rec["resolved_synonyms_broad"] + rec["resolved_synonyms_related"],
                    SYN_PREVIEW_CAP_OTHER),
                "support_dataset_count":       rec["support_dataset_count"],
                "support_row_count":           rec["support_row_count"],
                "llm_predicted_author_column": rec["llm_predicted_author_column"],
                "author_confidence":           rec["author_confidence"],
                "ontology_confidence":         rec["ontology_confidence"],
                "split":                       rec["split"],
            })

    finally:
        for file_handle in files.values():
            try:
                file_handle.close()
            except Exception:
                pass
        conn.close()

    # Publication CSV with a clear, stable column order
    pub_df = pd.DataFrame(pub_rows)

    pub_columns_order = [
        # Provenance
        "dataset_id", "dataset_title", "collection_name",
        # Anchor
        "field_type", "organism", "tissue", "author_term",
        # Ontology (original → resolved)
        "original_ontology_id", "original_label", "original_is_obsolete",
        "resolved_ontology_id", "resolved_label", "match_status", "resolution_path",
        "original_replaced_by", "original_consider",
        # Definitions & quick synonym previews (for human readers)
        "original_definition", "resolved_definition",
        "original_synonyms_exact_preview", "original_synonyms_other_preview",
        "resolved_synonyms_exact_preview", "resolved_synonyms_other_preview",
        # Evidence & split
        "support_dataset_count", "support_row_count",
        "llm_predicted_author_column", "author_confidence", "ontology_confidence",
        "split",
    ]
    # Keep only columns that exist (defensive) and in the desired order
    pub_df = pub_df[[c for c in pub_columns_order if c in pub_df.columns]]

    os.makedirs(os.path.dirname(READABLE_TABLE_CSV), exist_ok=True)
    pub_df.to_csv(READABLE_TABLE_CSV, index=False)

    # Summary
    counts = {name: sum(1 for _ in open(p, 'r', encoding='utf-8')) for name, p in paths.items()}
    total_out = sum(counts.values())
    print("✅ Wrote JSONL splits:")
    print(f"  - train: {paths['train']} ({counts['train']} rows)")
    print(f"  - dev:   {paths['dev']} ({counts['dev']} rows)")
    print(f"  - test:  {paths['test']} ({counts['test']} rows)")
    print(f"Total output rows: {total_out} (input rows: {total_rows})")
    print(f"✅ Publication table CSV: {READABLE_TABLE_CSV} ({len(pub_df)} rows)")

if __name__ == '__main__':
    main()