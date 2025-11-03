"""
Analyze CELLxGENE manifest and produce coverage stats + publication-ready figures.

Inputs:
  --manifest PATH   (all_datasets_manifest.csv)
  --summary  PATH   (coverage_summary.csv)
  --outdir   DIR    (where to write stats and figures)

Outputs in outdir:
  stats_overview.json
  coverage_by_field.csv
  coverage_by_organism.csv
  field_cocoverage.csv (pairwise co-coverage counts)
  collection_counts.csv (top collections by dataset count)

  figures/
    datasets_per_organism.{png,pdf}
    datasets_with_field.{png,pdf}
    field_coverage_by_organism.{png,pdf}
    cell_count_hist.{png,pdf}
    top_collections.{png,pdf}
"""

import os
import json
import argparse
import pandas as pd

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _savefig(fig, path_base: str):
    fig.tight_layout()
    fig.savefig(path_base + ".png", dpi=300)
    try:
        fig.savefig(path_base + ".pdf")
    except Exception:
        pass
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    _ensure_dir(args.outdir)
    figs_dir = os.path.join(args.outdir, "figures")
    _ensure_dir(figs_dir)

    manifest = pd.read_csv(args.manifest)
    summary = pd.read_csv(args.summary)

    # Identify organism flag columns dynamically
    org_cols = [c for c in manifest.columns if c.startswith("organism_")]
    field_cols = [c for c in manifest.columns if c.startswith("has_")]

    # Coverage by organism
    cov_org = []
    for c in org_cols:
        cov_org.append({"organism": c.removeprefix("organism_"), "datasets": int(manifest[c].sum())})
    coverage_by_organism = pd.DataFrame(cov_org).sort_values("datasets", ascending=False)
    coverage_by_organism.to_csv(os.path.join(args.outdir, "coverage_by_organism.csv"), index=False)

    # Coverage by field
    cov_field = []
    for c in field_cols:
        cov_field.append({"field": c.removeprefix("has_"), "datasets_with_field": int(manifest[c].sum())})
    coverage_by_field = pd.DataFrame(cov_field).sort_values("datasets_with_field", ascending=False)
    coverage_by_field.to_csv(os.path.join(args.outdir, "coverage_by_field.csv"), index=False)

    # Field co-coverage matrix (pairwise counts)
    import itertools
    co_rows = []
    for a, b in itertools.combinations(field_cols, 2):
        cnt = int(((manifest[a] == True) & (manifest[b] == True)).sum())
        co_rows.append({"field_a": a.removeprefix("has_"), "field_b": b.removeprefix("has_"), "datasets": cnt})
    field_cocov = pd.DataFrame(co_rows)
    field_cocov.to_csv(os.path.join(args.outdir, "field_cocoverage.csv"), index=False)

    # Collection counts
    col_counts = (
        manifest["collection_name"].fillna("Unknown").value_counts().reset_index(name="datasets").rename(columns={"index": "collection_name"})
    )
    col_counts.to_csv(os.path.join(args.outdir, "collection_counts.csv"), index=False)

    # Stats overview
    stats = {
        "total_datasets": int(len(manifest)),
        "organism_columns": org_cols,
        "field_columns": field_cols,
        "min_cells_per_dataset": float(manifest.get("dataset_total_cell_count", pd.Series()).min()) if "dataset_total_cell_count" in manifest.columns else None,
        "max_cells_per_dataset": float(manifest.get("dataset_total_cell_count", pd.Series()).max()) if "dataset_total_cell_count" in manifest.columns else None,
        "median_cells_per_dataset": float(manifest.get("dataset_total_cell_count", pd.Series()).median()) if "dataset_total_cell_count" in manifest.columns else None,
    }
    with open(os.path.join(args.outdir, "stats_overview.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Figures
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # Datasets per organism
    if not coverage_by_organism.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(coverage_by_organism, x="organism", y="datasets", color="#4C78A8", ax=ax)
        ax.set_xlabel("Organism")
        ax.set_ylabel("# Datasets")
        ax.set_title("Datasets per Organism")
        ax.tick_params(axis='x', rotation=30)
        _savefig(fig, os.path.join(figs_dir, "datasets_per_organism"))

    # Datasets with field
    if not coverage_by_field.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(coverage_by_field, x="field", y="datasets_with_field", color="#F58518", ax=ax)
        ax.set_xlabel("Field")
        ax.set_ylabel("# Datasets with Field Present")
        ax.set_title("Field Coverage Across Datasets")
        ax.tick_params(axis='x', rotation=30)
        _savefig(fig, os.path.join(figs_dir, "datasets_with_field"))

    # Field coverage by organism (stacked)
    if org_cols and field_cols:
        # Build a long DF of counts per (organism, field)
        rows = []
        for org_c in org_cols:
            ds_ids = set(manifest.loc[manifest[org_c] == True, "dataset_id"].tolist())
            for f_c in field_cols:
                cnt = int(manifest.loc[manifest[f_c] == True].loc[manifest["dataset_id"].isin(ds_ids)].shape[0])
                rows.append({"organism": org_c.removeprefix("organism_"), "field": f_c.removeprefix("has_"), "datasets": cnt})
        long_df = pd.DataFrame(rows)
        if not long_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(long_df, x="field", y="datasets", hue="organism", ax=ax)
            ax.set_xlabel("Field")
            ax.set_ylabel("# Datasets")
            ax.set_title("Field Coverage by Organism")
            ax.tick_params(axis='x', rotation=30)
            _savefig(fig, os.path.join(figs_dir, "field_coverage_by_organism"))

    # Cell count histogram (if available)
    if "dataset_total_cell_count" in manifest.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = manifest["dataset_total_cell_count"].dropna()
        if not vals.empty:
            sns.histplot(vals, bins=30, ax=ax, color="#72B7B2")
            ax.set_xlabel("Total Cells per Dataset")
            ax.set_ylabel("# Datasets")
            ax.set_title("Distribution of Dataset Cell Counts")
            _savefig(fig, os.path.join(figs_dir, "cell_count_hist"))

    # Top collections by dataset count
    topN = col_counts.head(20)
    if not topN.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(topN, y="collection_name", x="datasets", color="#54A24B", ax=ax)
        ax.set_xlabel("# Datasets")
        ax.set_ylabel("Collection")
        ax.set_title("Top Collections by Dataset Count (Top 20)")
        _savefig(fig, os.path.join(figs_dir, "top_collections"))

    print(f"Wrote analysis artifacts to: {args.outdir}")

if __name__ == "__main__":
    main()

