#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-safe compiler for CELLxGENE author ↔ ontology pairs.

Key changes vs. previous version:
- Explode multi-IDs (disease / human ethnicity) INSIDE each worker.
- Aggregate inside worker to unique anchors with per-dataset support_row_count.
- Vectorized global prefix filtering (no row-wise .apply).
- No second ontology re-validation pass after explode (already split to single IDs).
"""

import os
import json
import re
import unicodedata
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
from functools import partial

# -----------------------------
# 1) Configuration
# -----------------------------

MAPPING_FILE_PATH = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/llm_generated_column_mappings_merged.json"
METADATA_DIR_PATH = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/metadata_schemas"
MANIFEST_FILE_PATH = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/all_datasets_manifest.csv"
OUTPUT_FILE_PATH = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/compiled_training_data_raw_fixed.csv"

FILTER_COUNTS_PATH = os.path.join(os.path.dirname(OUTPUT_FILE_PATH), "compiled_filter_counts.csv")
FILTER_SAMPLES_PATH = os.path.join(os.path.dirname(OUTPUT_FILE_PATH), "compiled_filter_samples.csv")

CATEGORIES = [
    "assay", "cell_type", "development_stage", "disease",
    "self_reported_ethnicity", "sex", "tissue"
]

# Feature flags
ENABLE_MAJORITY_RESOLUTION = False
INCLUDE_ALTERNATIVES = True
ALTERNATIVE_LIMIT = 2            # lower default helps control size; grouping removes dupes anyway
GLOBAL_DEDUP_MODE = "anchor_ontology"  # 'anchor' or 'anchor_ontology'

MAX_SAMPLES_PER_REASON = 500
MAX_PROCESSES = min(8, os.cpu_count() or 1)

# -----------------------------
# 2) Normalization & Filters
# -----------------------------

_NA_LIKE = {"na", "n/a", "null", "none", "nan", "[na]", "[none]"}

def _norm_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = unicodedata.normalize("NFKC", x).strip()
    return x if x else None

def _normalize_text_for_keys(x: Any) -> str:
    t = _norm_text(x)
    if t is None:
        return ""
    return re.sub(r"\s+", " ", t).lower()

_num_re = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")
_date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _is_pure_number(s: str) -> bool:
    return bool(_num_re.fullmatch(s))

def _is_date_like(s: str) -> bool:
    return bool(_date_re.fullmatch(s))

def _is_na_like(s: str) -> bool:
    return s.lower() in _NA_LIKE

# Accept CURIEs (prefix:) or Cellosaurus CVCL_
ONTOLOGY_ID_COLON_OR_CVCL = re.compile(r"^(?:[A-Za-z][A-Za-z0-9_]*:|CVCL_)")

def _valid_or_multi_ontology_id(raw: Any, field_type: Optional[str] = None) -> bool:
    s = _norm_text(raw)
    if not s:
        return False
    if field_type in {"disease", "self_reported_ethnicity"} and "||" in s:
        toks = [t.strip() for t in s.split("||") if t.strip()]
        return any(ONTOLOGY_ID_COLON_OR_CVCL.match(t) for t in toks)
    if field_type == "tissue" and s.startswith("CVCL_"):
        return True
    return bool(ONTOLOGY_ID_COLON_OR_CVCL.match(s))

def _organism_key(organism_val: Optional[str]) -> str:
    o = (_norm_text(organism_val) or "").lower()
    if "sapiens" in o or o == "human": return "human"
    if "mus musculus" in o or o == "mouse": return "mouse"
    if "danio rerio" in o or "zebrafish" in o: return "zebrafish"
    if "drosophila" in o or "fly" in o: return "fly"
    if "caenorhabditis elegans" in o or "c. elegans" in o or "celegans" in o: return "celegans"
    return "other"

_ALLOWED_SEX = {"PATO:0000383", "PATO:0000384", "PATO:0001340"}

def _allowed_prefixes(field_type: str, org_key: str) -> set:
    ft = field_type
    if ft == "tissue":
        return {"UBERON:", "CVCL_", "WBbt:", "ZFA:", "FBbt:"}
    if ft == "cell_type":
        ok = {"CL:"}
        if org_key == "celegans": ok |= {"WBbt:"}
        elif org_key == "zebrafish": ok |= {"ZFA:"}
        elif org_key == "fly": ok |= {"FBbt:"}
        return ok
    if ft == "development_stage":
        mapping = {"human":"HsapDv:", "mouse":"MmusDv:", "zebrafish":"ZFS:", "fly":"FBdv:", "celegans":"WBls:"}
        return {mapping[org_key]} if org_key in mapping else {"UBERON:"}
    if ft == "disease":
        return {"MONDO:", "PATO:"}
    if ft == "assay":
        return {"EFO:"}
    if ft == "sex":
        return {"PATO:"}
    if ft == "self_reported_ethnicity":
        return {"HANCESTRO:", "AfPO:"} if org_key == "human" else set()
    return set()

def _prefix(s: str) -> str:
    if s.startswith("CVCL_"): return "CVCL_"
    if ":" in s: return s.split(":", 1)[0] + ":"
    return ""

# -----------------------------
# 3) Filter audit utils
# -----------------------------

def _init_report() -> Dict[str, Any]:
    return {
        "placeholder_author": 0,
        "invalid_ontology_format": 0,
        "kept_rows": 0,
        "samples": []
    }

def _maybe_sample(samples: List[Dict[str, Any]], reason: str, row: Dict[str, Any]):
    if len([s for s in samples if s.get("reason") == reason]) < MAX_SAMPLES_PER_REASON:
        samples.append({"reason": reason, **row})

# -----------------------------
# 4) Worker: extract & AGGREGATE per dataset
# -----------------------------

def _explode_multi_ids_local(df: pd.DataFrame) -> pd.DataFrame:
    """Split disease and human ethnicity '||' lists BEFORE grouping."""
    if df.empty:
        return df

    out = df

    # Disease
    mask_dz = out['field_type'].eq('disease')
    if mask_dz.any():
        dz = out.loc[mask_dz].copy()
        dz['ontology_id'] = dz['ontology_id'].astype(str)
        dz['ontology_id'] = dz['ontology_id'].str.split(r"\s*\|\|\s*")
        dz = dz.explode('ontology_id')
        dz['ontology_id'] = dz['ontology_id'].map(lambda s: _norm_text(s) or "")
        out = pd.concat([out.loc[~mask_dz], dz], ignore_index=True)

    # Ethnicity (human only)
    mask_eth = out['field_type'].eq('self_reported_ethnicity')
    if mask_eth.any():
        eth = out.loc[mask_eth].copy()
        eth['org_key'] = eth['organism'].map(_organism_key)
        eth_h = eth.loc[eth['org_key'].eq('human')].copy()
        eth_o = eth.loc[~eth['org_key'].eq('human')].copy()

        if len(eth_h):
            eth_h['ontology_id'] = eth_h['ontology_id'].astype(str)
            eth_h['ontology_id'] = eth_h['ontology_id'].str.split(r"\s*\|\|\s*")
            eth_h = eth_h.explode('ontology_id')
            eth_h['ontology_id'] = eth_h['ontology_id'].map(lambda s: _norm_text(s) or "")

        eth_out = pd.concat([eth_h.drop(columns=['org_key'], errors='ignore'),
                             eth_o.drop(columns=['org_key'], errors='ignore')], ignore_index=True)
        out = pd.concat([out.loc[~mask_eth], eth_out], ignore_index=True)

    return out

def process_dataset(
    dataset_item,
    metadata_dir: str,
    context_lookup: Dict[str, Dict[str, Any]],
    include_alternatives: bool = True,
    alternative_limit: int = 2
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dataset_id, mapping = dataset_item
    report = _init_report()

    metadata_filename = f"{dataset_id}_metadata.csv.gz"
    metadata_filepath = os.path.join(metadata_dir, metadata_filename)
    if not os.path.exists(metadata_filepath):
        return pd.DataFrame(), report

    try:
        try:
            df_meta = pd.read_csv(metadata_filepath, compression='gzip', low_memory=False, on_bad_lines="skip")
        except Exception:
            df_meta = pd.read_csv(metadata_filepath, compression='gzip', low_memory=False, on_bad_lines="skip", engine='python')

        have_tissue = 'tissue' in df_meta.columns
        have_org = 'organism' in df_meta.columns

        # Minimal context DF (no copy of whole df)
        ctx = pd.DataFrame({
            'row_idx': df_meta.index,
            'tissue': df_meta['tissue'] if have_tissue else pd.Series([None]*len(df_meta)),
            'organism': df_meta['organism'] if have_org else pd.Series([None]*len(df_meta)),
        })

        context = context_lookup.get(dataset_id, {})
        out_rows = []

        for field_type in CATEGORIES:
            field_map = mapping.get(field_type, {})
            author_col = field_map.get("author_term_column")
            ontology_col = field_map.get("ontology_id_column")
            if not (author_col and ontology_col and ontology_col in df_meta.columns):
                continue

            cand_authors = []
            if author_col in df_meta.columns:
                cand_authors.append(author_col)
            if include_alternatives:
                for c in (field_map.get("alternative_author_term_columns") or [])[:int(alternative_limit)]:
                    if c in df_meta.columns and c not in cand_authors:
                        cand_authors.append(c)

            # Collect frames per candidate column (very small, 3 columns)
            small_frames = []
            for a_col in cand_authors:
                if a_col == ontology_col:
                    continue
                tmp = pd.DataFrame({
                    'row_idx': df_meta.index,
                    'author_term': df_meta[a_col],
                    'ontology_id': df_meta[ontology_col],
                })

                # author keep/drop with audit
                tmp['author_term'] = tmp['author_term'].map(_norm_text)
                def _keep_author(s: Optional[str]) -> bool:
                    if s is None: return False
                    if _is_na_like(s) or _is_pure_number(s) or _is_date_like(s): return False
                    return True
                keep_mask = tmp['author_term'].map(_keep_author)
                report["placeholder_author"] += int((~keep_mask).sum())
                tmp = tmp.loc[keep_mask]

                # ontology basic validation (allow multi string for disease/ethnicity)
                val_mask = tmp['ontology_id'].map(lambda v: _valid_or_multi_ontology_id(v, field_type=field_type))
                report["invalid_ontology_format"] += int((~val_mask).sum())
                tmp = tmp.loc[val_mask]

                if tmp.empty:
                    continue

                # Join per-row context now (only 3 cols join)
                tmp = tmp.merge(ctx, on='row_idx', how='left')

                # Dataset fallbacks if missing
                if not have_tissue:
                    ds_t = _norm_text(context.get("tissue"))
                    tmp['tissue'] = ds_t if (ds_t and (';' not in ds_t)) else None
                if not have_org:
                    ds_o = _norm_text(context.get("organisms"))
                    tmp['organism'] = ds_o if (ds_o and (',' not in ds_o)) else None

                tmp['field_type'] = field_type
                tmp['llm_predicted_author_column'] = author_col
                tmp['author_confidence'] = field_map.get("author_confidence")
                tmp['ontology_confidence'] = field_map.get("ontology_confidence")
                small_frames.append(tmp[['author_term','ontology_id','tissue','organism',
                                         'field_type','llm_predicted_author_column',
                                         'author_confidence','ontology_confidence']])

            if small_frames:
                ft_block = pd.concat(small_frames, ignore_index=True)
                # Split multi IDs for this dataset/field
                ft_block = _explode_multi_ids_local(ft_block)

                # Normalize keys for grouping
                ft_block['author_term_norm'] = ft_block['author_term'].map(_normalize_text_for_keys)
                ft_block['tissue_norm'] = ft_block['tissue'].map(_normalize_text_for_keys)
                ft_block['organism_norm'] = ft_block['organism'].map(_normalize_text_for_keys)

                # Group INSIDE WORKER to collapse row-level duplication
                gcols = ['field_type','author_term_norm','tissue_norm','organism_norm','ontology_id']
                agg = (ft_block
                       .groupby(gcols, as_index=False)
                       .agg(
                           author_term=('author_term','first'),
                           tissue=('tissue','first'),
                           organism=('organism','first'),
                           llm_predicted_author_column=('llm_predicted_author_column','first'),
                           author_confidence=('author_confidence','max'),
                           ontology_confidence=('ontology_confidence','max'),
                           support_row_count=('ontology_id','size')
                       ))

                # Attach dataset metadata once
                agg['dataset_id'] = dataset_id
                agg['dataset_title'] = context.get("dataset_title", "N/A")
                agg['collection_name'] = context.get("collection_name", "N/A")

                out_rows.append(agg)

        if not out_rows:
            return pd.DataFrame(), report

        merged_small = pd.concat(out_rows, ignore_index=True)
        report["kept_rows"] += int(merged_small['support_row_count'].sum())

        # Return only compact, aggregated rows
        return merged_small, report

    except Exception as e:
        print(f"Error processing {dataset_id}: {e}")
        return pd.DataFrame(), report

# -----------------------------
# 5) Vectorized global filters
# -----------------------------

def _vectorized_allowed_mask(df: pd.DataFrame) -> pd.Series:
    """Compute allowed-prefix mask without row-wise apply."""
    s = df['ontology_id'].astype(str)
    ft = df['field_type'].astype(str)

    org_key = df['organism'].map(_organism_key)

    # Initialize False mask
    allowed = pd.Series(False, index=df.index)

    # sex
    m = ft.eq('sex')
    if m.any():
        s_m = s[m]
        allowed.loc[m] = s_m.str.startswith('PATO:') & s_m.isin(list(_ALLOWED_SEX))

    # disease
    m = ft.eq('disease')
    if m.any():
        s_m = s[m]
        allowed.loc[m] = s_m.str.startswith('MONDO:') | s_m.eq('PATO:0000461')

    # assay
    m = ft.eq('assay')
    if m.any():
        allowed.loc[m] = s[m].str.startswith('EFO:')

    # self_reported_ethnicity (human)
    m = ft.eq('self_reported_ethnicity') & org_key.eq('human')
    if m.any():
        s_m = s[m]
        allowed.loc[m] = s_m.str.startswith('HANCESTRO:') | s_m.str.startswith('AfPO:')

    # tissue
    m = ft.eq('tissue')
    if m.any():
        s_m = s[m]
        ok = (s_m.str.startswith('UBERON:') |
              s_m.str.startswith('CVCL_')  |
              s_m.str.startswith('WBbt:')  |
              s_m.str.startswith('ZFA:')   |
              s_m.str.startswith('FBbt:'))
        allowed.loc[m] = ok

    # development_stage by organism
    m = ft.eq('development_stage')
    if m.any():
        idx = df.index[m]
        s_m = s.loc[idx]
        org_m = org_key.loc[idx]
        cond = (
            (org_m.eq('human') & s_m.str.startswith('HsapDv:')) |
            (org_m.eq('mouse') & s_m.str.startswith('MmusDv:')) |
            (org_m.eq('zebrafish') & s_m.str.startswith('ZFS:')) |
            (org_m.eq('fly') & s_m.str.startswith('FBdv:')) |
            (org_m.eq('celegans') & s_m.str.startswith('WBls:')) |
            (org_m.eq('other') & s_m.str.startswith('UBERON:'))
        )
        allowed.loc[idx] = cond

    # cell_type (+ organism allowances)
    m = ft.eq('cell_type')
    if m.any():
        idx = df.index[m]
        s_m = s.loc[idx]
        org_m = org_key.loc[idx]
        base = s_m.str.startswith('CL:')
        extra = (
            (org_m.eq('celegans') & s_m.str.startswith('WBbt:')) |
            (org_m.eq('zebrafish') & s_m.str.startswith('ZFA:')) |
            (org_m.eq('fly') & s_m.str.startswith('FBbt:'))
        )
        allowed.loc[idx] = base | extra

    return allowed

# -----------------------------
# 6) Main Orchestration
# -----------------------------

def main():
    print("--- Starting Raw Data Consolidation (Parallel & Memory-Safe) ---")

    print(f"Loading column mappings from: {MAPPING_FILE_PATH}")
    with open(MAPPING_FILE_PATH, 'r') as f:
        all_mappings = json.load(f)

    print(f"Loading dataset manifest for context from: {MANIFEST_FILE_PATH}")
    manifest_df = pd.read_csv(MANIFEST_FILE_PATH)
    context_lookup = manifest_df.set_index('dataset_id').to_dict('index')

    datasets_to_process = list(all_mappings.items())
    print(f"Found {len(datasets_to_process)} datasets to process...")
    print(f"Using {MAX_PROCESSES} parallel processes...")

    worker_func = partial(
        process_dataset,
        metadata_dir=METADATA_DIR_PATH,
        context_lookup=context_lookup,
        include_alternatives=INCLUDE_ALTERNATIVES,
        alternative_limit=ALTERNATIVE_LIMIT,
    )

    agg_report = {
        "placeholder_author": 0,
        "invalid_ontology_format": 0,
        "kept_rows": 0,
        "disallowed_prefix": 0,
        "filter_samples": []
    }

    compact_frames: List[pd.DataFrame] = []
    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        for df_res, rep in tqdm(pool.imap_unordered(worker_func, datasets_to_process),
                                total=len(datasets_to_process), desc="Processing datasets"):
            if isinstance(df_res, pd.DataFrame) and not df_res.empty:
                compact_frames.append(df_res)
            agg_report["placeholder_author"] += rep.get("placeholder_author", 0)
            agg_report["invalid_ontology_format"] += rep.get("invalid_ontology_format", 0)
            agg_report["kept_rows"] += rep.get("kept_rows", 0)
            for s in rep.get("samples", []):
                if len(agg_report["filter_samples"]) < MAX_SAMPLES_PER_REASON * 10:
                    agg_report["filter_samples"].append(s)

    if not compact_frames:
        print("No data was extracted. Please check file paths and JSON mapping file.")
        return

    print("\nConsolidating compact results...")
    final_df = pd.concat(compact_frames, ignore_index=True)

    # Vectorized global ontology-prefix policy
    allowed_mask = _vectorized_allowed_mask(final_df)
    disallowed = final_df.loc[~allowed_mask]
    if not disallowed.empty:
        for _, r in disallowed.head(MAX_SAMPLES_PER_REASON).iterrows():
            _maybe_sample(agg_report["filter_samples"], "disallowed_prefix", {
                "dataset_id": r.get("dataset_id"),
                "field_type": r.get("field_type"),
                "author_term": r.get("author_term"),
                "ontology_id": r.get("ontology_id"),
                "tissue": r.get("tissue"),
                "organism": r.get("organism"),
            })
    agg_report["disallowed_prefix"] += int((~allowed_mask).sum())
    final_df = final_df.loc[allowed_mask].copy()

    # Compute GLOBAL support stats (sum rows, nunique datasets) for key
    # Note: author_term_norm / tissue_norm / organism_norm already exist in per-dataset frames
    support_key = ['field_type','author_term_norm','tissue_norm','organism_norm','ontology_id']
    global_stats = (
        final_df.groupby(support_key, as_index=False)
                .agg(
                    support_row_count=('support_row_count','sum'),
                    support_dataset_count=('dataset_id','nunique')
                )
    )

    # Merge global stats back to per-dataset rows
    final_df = final_df.merge(global_stats, on=support_key, suffixes=('', '_global'))

    # Optional final de-dup
    dedup_removed = 0
    dedup_label = "none"
    if ENABLE_MAJORITY_RESOLUTION:
        # Keep the most frequent ontology per anchor context (global)
        grp_cols = ['field_type','author_term_norm','tissue_norm','organism_norm','ontology_id']
        counts = final_df.groupby(grp_cols, as_index=False)['support_row_count_global'].sum()
        counts_sorted = counts.sort_values(
            by=['field_type','author_term_norm','tissue_norm','organism_norm','support_row_count_global','ontology_id'],
            ascending=[True, True, True, True, False, True]
        )
        winners = counts_sorted.drop_duplicates(subset=['field_type','author_term_norm','tissue_norm','organism_norm'], keep='first')
        key_cols = ['field_type','author_term_norm','tissue_norm','organism_norm']
        final_df = final_df.merge(
            winners[key_cols + ['ontology_id']].rename(columns={'ontology_id':'winner_ontology_id'}),
            on=key_cols, how='inner'
        )
        pre = len(final_df)
        final_df = final_df[final_df['ontology_id'] == final_df['winner_ontology_id']]
        final_df = final_df.drop_duplicates(subset=key_cols + ['dataset_id'])
        dedup_removed = pre - len(final_df)
        dedup_label = 'anchor (after majority)'
        final_df = final_df.drop(columns=['winner_ontology_id'])
    else:
        if GLOBAL_DEDUP_MODE == 'anchor':
            pre = len(final_df)
            final_df = final_df.drop_duplicates(subset=['field_type','author_term_norm','tissue_norm','organism_norm','dataset_id'])
            dedup_removed = pre - len(final_df)
            dedup_label = 'anchor'
        elif GLOBAL_DEDUP_MODE == 'anchor_ontology':
            pre = len(final_df)
            final_df = final_df.drop_duplicates(subset=support_key + ['dataset_id'])
            dedup_removed = pre - len(final_df)
            dedup_label = 'anchor+ontology'

    # Assemble publishable columns (keep global supports)
    final_columns = [
        "dataset_id", "dataset_title", "collection_name",
        "field_type", "organism", "tissue",
        "author_term", "ontology_id",
        "support_row_count_global", "support_dataset_count",
        "llm_predicted_author_column", "author_confidence", "ontology_confidence",
    ]
    final_rename = {
        "support_row_count_global": "support_row_count",
    }
    final_df = final_df[final_columns].rename(columns=final_rename)

    # Save outputs
    print(f"\nFinal rows: {len(final_df)}")
    print(f"De-duplicated ({dedup_label}) rows removed: {dedup_removed}")
    print(f"Saving compiled data to: {OUTPUT_FILE_PATH}")
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)

    counts_rows = [
        {"metric": "worker_placeholder_author_dropped", "value": agg_report["placeholder_author"]},
        {"metric": "worker_invalid_ontology_format_dropped", "value": agg_report["invalid_ontology_format"]},
        {"metric": "worker_kept_rows_sum", "value": agg_report["kept_rows"]},
        {"metric": "global_disallowed_prefix_dropped", "value": agg_report["disallowed_prefix"]},
        {"metric": "final_output_rows", "value": len(final_df)},
    ]
    pd.DataFrame(counts_rows, columns=["metric","value"]).to_csv(FILTER_COUNTS_PATH, index=False)

    if agg_report["filter_samples"]:
        pd.DataFrame(agg_report["filter_samples"], columns=[
            "reason","dataset_id","field_type","author_term","ontology_id","tissue","organism"
        ]).to_csv(FILTER_SAMPLES_PATH, index=False)

    print("\n--- Process Complete ---")
    print(f"Filter counts saved to: {FILTER_COUNTS_PATH}")
    if agg_report["filter_samples"]:
        print(f"Filter samples saved to: {FILTER_SAMPLES_PATH}")
    print("--------------------------")
    print("Category breakdown of extracted rows:")
    print(final_df['field_type'].value_counts())

if __name__ == "__main__":
    main()
