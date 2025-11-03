"""
CELLxGENE Census – Dataset Manifest & Coverage Overview
-------------------------------------------------------

Produces a lightweight, dataset-level manifest and coverage summary across the
entire Census snapshot (no H5AD downloads, no per-cell rows). Use this to
explore distribution and decide which dataset_ids to target next.

Outputs (written under --outdir, default: Miscellaneous/CellxGene_Benchmark/benchmark_data):
- all_datasets_manifest.csv
  One row per dataset_id with dataset metadata, organism flags, tissue names,
  and boolean coverage flags per ontology-mapped field.

- coverage_summary.csv
  Aggregate counts across all datasets and per-field coverage.
"""

import os
import argparse
import pandas as pd
import cellxgene_census
from typing import Dict, List, Set
from tqdm import tqdm


FIELDS = {
    "assay": "assay_ontology_term_id",
    "cell_type": "cell_type_ontology_term_id",
    "development_stage": "development_stage_ontology_term_id",
    "disease": "disease_ontology_term_id",
    "sex": "sex_ontology_term_id",
    "self_reported_ethnicity": "self_reported_ethnicity_ontology_term_id",
    "tissue": "tissue_ontology_term_id",
}


def _ids_for(org: str, census) -> Set[str]:
    df = cellxgene_census.get_obs(census, org, column_names=["dataset_id"])
    return set(df["dataset_id"].unique().tolist())


def _coverage_ids_for_column(org: str, col: str, census) -> Set[str]:
    try:
        df = cellxgene_census.get_obs(
            census,
            org,
            value_filter=f"{col} is not null",
            column_names=["dataset_id"],
        )
        return set(df["dataset_id"].unique().tolist())
    except Exception:
        try:
            df = cellxgene_census.get_obs(census, org, column_names=["dataset_id", col])
            df = df[df[col].notna()]
            return set(df["dataset_id"].unique().tolist())
        except Exception:
            return set()


def _tissue_names_for(org: str, census) -> pd.DataFrame:
    """Return dataset_id → semicolon-delimited tissue names."""
    try:
        df = cellxgene_census.get_obs(
            census,
            org,
            column_names=["dataset_id", "tissue"],
        )
        grouped = (
            df.dropna()
              .groupby("dataset_id")["tissue"]
              .apply(lambda x: ";".join(sorted(set(x.astype(str)))))
              .reset_index()
        )
        return grouped
    except Exception:
        return pd.DataFrame(columns=["dataset_id", "tissue"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default=os.path.join("Miscellaneous", "CellxGene_Benchmark", "benchmark_data"),
        help="Output directory for CSVs",
    )
    ap.add_argument(
        "--census-version",
        default=os.getenv("CENSUS_VERSION", "2025-01-30"),
        help="Pinned Census snapshot version (override via env CENSUS_VERSION)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Using Census version: {args.census_version}")

    census = cellxgene_census.open_soma(census_version=args.census_version)
    try:
        # Datasets table (titles, collections, cell counts)
        datasets_df = (
            census["census_info"]["datasets"].read().concat().to_pandas()
        )
        wanted_cols = [
            "dataset_id",
            "dataset_title",
            "collection_name",
            "collection_doi",
            "citation",
            "dataset_total_cell_count",
        ]
        for c in wanted_cols:
            if c not in datasets_df.columns:
                datasets_df[c] = None

        # Enumerate organisms
        try:
            organism_keys = sorted(list(census["census_data"].keys()))
        except Exception:
            organism_keys = []
        print("🔍 Scanning organism coverage" +
              (f" for: {', '.join(organism_keys)}" if organism_keys else "..."))
        org_to_ids: Dict[str, Set[str]] = {}
        for org in tqdm(organism_keys, desc="Organisms", unit="org"):
            try:
                org_to_ids[org] = _ids_for(org, census)
            except Exception:
                org_to_ids[org] = set()

        # Field coverage
        coverage: Dict[str, Set[str]] = {}
        for field, col in tqdm(FIELDS.items(), desc="Processing fields", unit="field"):
            union_ids: Set[str] = set()
            for org in organism_keys:
                try:
                    union_ids |= _coverage_ids_for_column(org, col, census)
                except Exception:
                    pass
            coverage[field] = union_ids

        # Build manifest
        print("📊 Building manifest...")
        manifest = datasets_df[wanted_cols].copy()

        # Organism flags + combined organisms string
        def _sanitize_org(o: str) -> str:
            return "".join(ch if ch.isalnum() else "_" for ch in o)
        for org in organism_keys:
            coln = f"organism_{_sanitize_org(org)}"
            ids = org_to_ids.get(org, set())
            manifest[coln] = manifest["dataset_id"].apply(lambda x, s=ids: x in s)
        def _list_orgs(ds_id: str) -> str:
            present = [o for o in organism_keys if ds_id in (org_to_ids.get(o) or set())]
            return ",".join(present)
        manifest["organisms"] = manifest["dataset_id"].apply(_list_orgs)

        # Add coverage booleans
        for field in FIELDS.keys():
            manifest[f"has_{field}"] = manifest["dataset_id"].apply(
                lambda x, s=coverage[field]: x in s
            )

        # Tissue names (new!)
        tissue_frames = []
        for org in organism_keys:
            tissue_frames.append(_tissue_names_for(org, census))
        if tissue_frames:
            tissue_df = pd.concat(tissue_frames).drop_duplicates()
            manifest = manifest.merge(tissue_df, on="dataset_id", how="left")
        else:
            manifest["tissue"] = None

        # Save manifest
        manifest_path = os.path.join(args.outdir, "all_datasets_manifest.csv")
        manifest.to_csv(manifest_path, index=False)
        print(f"✅ Wrote manifest: {manifest_path} ({len(manifest)} datasets)")

        # Coverage summary
        print("📈 Generating coverage summary...")
        rows = []
        rows.append({"metric": "total_datasets", "value": len(manifest)})
        for org in organism_keys:
            coln = f"organism_{_sanitize_org(org)}"
            if coln in manifest.columns:
                rows.append({"metric": f"{coln}_count", "value": int(manifest[coln].sum())})
        for field in FIELDS.keys():
            rows.append({
                "metric": f"datasets_with_{field}",
                "value": int(manifest[f"has_{field}"].sum()),
            })
        summary = pd.DataFrame(rows, columns=["metric", "value"])
        summary_path = os.path.join(args.outdir, "coverage_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"✅ Wrote summary:  {summary_path}")

    finally:
        census.close()


if __name__ == "__main__":
    main()
