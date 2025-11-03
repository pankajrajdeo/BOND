import os
import json
import sqlite3
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional

# --- Configuration ---
ASSETS_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/assets")
INPUT_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/bond_benchmark_test.csv")
DB_PATH = ASSETS_PATH / "ontologies.sqlite"
TABLE_TERMS = "ontology_terms"
TABLE_TERMS_FTS = "ontology_terms_fts"
MAX_K = 5 # We will always fetch top 5 to calculate accuracy at 1, 3, and 5.

# --- Standalone Functions (Copied from BOND codebase) ---

# From bond/schema_policies.py
FIELD_TO_ONTOLOGIES = {
    "cell_type": ["cl", "fbbt", "zfa", "wbbt"], "tissue": ["uberon", "fbbt", "zfa", "wbbt"],
    "disease": ["mondo", "pato"], "development_stage": ["hsapdv", "mmusdv", "fbdv", "wbls", "zfa"],
    "sex": ["pato"], "self_reported_ethnicity": ["hancestro"], "assay": ["efo"], "organism": ["ncbitaxon"],
}
SUPPORTED_ORGANISMS = {
    "Homo sapiens": "NCBITaxon:9606", "Mus musculus": "NCBITaxon:10090",
    "Danio rerio": "NCBITaxon:7955", "Drosophila melanogaster": "NCBITaxon:7227",
    "Caenorhabditis elegans": "NCBITaxon:6239",
}
SPECIES_ROUTING = {
    "NCBITaxon:9606": {"cell_type": ["cl"], "development_stage": ["hsapdv"], "tissue": ["uberon"]},
    "NCBITaxon:10090": {"cell_type": ["cl"], "development_stage": ["mmusdv"], "tissue": ["uberon"]},
    "NCBITaxon:7955": {"cell_type": ["cl", "zfa"], "development_stage": ["zfa"], "tissue": ["uberon", "zfa"]},
    "NCBITaxon:7227": {"cell_type": ["cl", "fbbt"], "development_stage": ["fbdv"], "tissue": ["uberon", "fbbt"]},
    "NCBITaxon:6239": {"cell_type": ["cl", "wbbt"], "development_stage": ["wbls"], "tissue": ["uberon", "wbbt"]},
}

def _canonical_taxon(organism: Optional[str]) -> Optional[str]:
    return SUPPORTED_ORGANISMS.get(organism)

def allowed_ontologies_for(field_name: Optional[str], organism: Optional[str]) -> Optional[List[str]]:
    if not field_name: return None
    field = field_name.strip().lower()
    base = FIELD_TO_ONTOLOGIES.get(field)
    if not base: raise ValueError(f"Unsupported field: {field}")
    tax = _canonical_taxon(organism)
    if tax and tax in SPECIES_ROUTING and field in SPECIES_ROUTING[tax]:
        return SPECIES_ROUTING[tax][field]
    return base

# From bond/retrieval/bm25_sqlite.py
def search_bm25(conn: sqlite3.Connection, table_terms: str, table_fts: str, q: str, k: int, sources: Optional[List[str]] = None) -> List[Dict]:
    """BM25 over ontology_terms_fts only."""
    cur = conn.cursor()
    match_param = '"' + q.replace('"', '""') + '"'
    base_query = (
        f"SELECT t.curie, t.label FROM {table_fts} "
        f"JOIN {table_terms} t ON t.rowid = {table_fts}.rowid WHERE {table_fts} MATCH ?"
    )
    params = [match_param]
    if sources:
        placeholders = ",".join("?" for _ in sources)
        base_query += f" AND t.ontology_id IN ({placeholders})"
        params.extend(sources)
    try:
        base_query += f" ORDER BY bm25({table_fts}) ASC LIMIT ?"
        params.append(k)
        cur.execute(base_query, params)
    except sqlite3.OperationalError:
        base_query = base_query.replace(f"ORDER BY bm25({table_fts}) ASC", "")
        cur.execute(base_query + " LIMIT ?", params)
    
    return [{"id": row[0], "label": row[1]} for row in cur.fetchall()]

# --- Evaluation Logic ---

def calculate_accuracy(df_results: pd.DataFrame):
    """Calculates and prints accuracy@1, @3, and @5."""
    def is_correct(row, k):
        # Extract just the IDs from the list of prediction dictionaries
        predicted_ids = [pred['id'] for pred in row['predictions'][:k]]
        return row['target_ontology_id'] in predicted_ids

    total = len(df_results)
    acc1 = df_results.apply(lambda row: is_correct(row, 1), axis=1).sum() / total
    acc3 = df_results.apply(lambda row: is_correct(row, 3), axis=1).sum() / total
    acc5 = df_results.apply(lambda row: is_correct(row, 5), axis=1).sum() / total
    
    print("\n--- BM25 Only Evaluation Results ---")
    print(f"Accuracy@1: {acc1:.4f}")
    print(f"Accuracy@3: {acc3:.4f}")
    print(f"Accuracy@5: {acc5:.4f}")
    print("------------------------------------")

def run_bm25_evaluation():
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return
    if not INPUT_CSV_PATH.exists():
        print(f"ERROR: Input CSV not found at {INPUT_CSV_PATH}")
        return

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    df_test = pd.read_csv(INPUT_CSV_PATH)
    
    # Add BM25 prediction columns to the original dataframe
    bm25_predictions = []
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Running BM25 Baseline"):
        try:
            ontology_filter = allowed_ontologies_for(row['field'], row['organism'])
            bm25_hits = search_bm25(conn, TABLE_TERMS, TABLE_TERMS_FTS, row['source_term'], MAX_K, sources=ontology_filter)
            bm25_predictions.append(bm25_hits)
        except Exception as e:
            print(f"Warning: Error processing term '{row['source_term']}': {e}")
            bm25_predictions.append([])
            
    conn.close()
    
    # Calculate accuracy for evaluation
    df_eval = pd.DataFrame({
        'source_term': df_test['source_term'],
        'target_ontology_id': df_test['target_ontology_id'],
        'predictions': bm25_predictions
    })
    calculate_accuracy(df_eval)
    
    # Add BM25 prediction columns to original dataframe
    # Top-1 predictions
    df_test['bm25_top1_predicted_id'] = [pred[0]['id'] if len(pred) >= 1 else None for pred in bm25_predictions]
    df_test['bm25_top1_predicted_label'] = [pred[0]['label'] if len(pred) >= 1 else None for pred in bm25_predictions]
    
    # Top-3 predictions
    for i in range(3):
        df_test[f'bm25_top3_rank{i+1}_id'] = [pred[i]['id'] if len(pred) > i else None for pred in bm25_predictions]
        df_test[f'bm25_top3_rank{i+1}_label'] = [pred[i]['label'] if len(pred) > i else None for pred in bm25_predictions]
    
    # Top-5 predictions
    for i in range(5):
        df_test[f'bm25_top5_rank{i+1}_id'] = [pred[i]['id'] if len(pred) > i else None for pred in bm25_predictions]
        df_test[f'bm25_top5_rank{i+1}_label'] = [pred[i]['label'] if len(pred) > i else None for pred in bm25_predictions]
    
    # Save the enhanced CSV with original structure + BM25 predictions
    output_path = "/Users/rajlq7/Downloads/Terms/BOND/evals/BM25-Only/bm25_baseline_predictions.csv"
    df_test.to_csv(output_path, index=False)
    print(f"BM25 baseline predictions appended and saved to {output_path}")
    print(f"Added {2 + 6 + 10} BM25 prediction columns to original {len(df_test.columns) - 18} columns")

if __name__ == "__main__":
    run_bm25_evaluation()

