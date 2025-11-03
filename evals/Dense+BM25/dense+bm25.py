import os
import json
import sqlite3
import pandas as pd
import numpy as np
import faiss
import requests
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
ASSETS_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/assets")
INPUT_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/bond_benchmark_test.csv")
DB_PATH = ASSETS_PATH / "ontologies.sqlite"
FAISS_STORE_PATH = ASSETS_PATH / "faiss_store"
TABLE_TERMS = "ontology_terms"
TABLE_TERMS_FTS = "ontology_terms_fts"
MAX_K = 5
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Standalone Functions (Copied from BOND codebase) ---

# From bond/schema_policies.py (condensed)
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
def _canonical_taxon(organism: Optional[str]) -> Optional[str]: return SUPPORTED_ORGANISMS.get(organism)
def allowed_ontologies_for(field_name: Optional[str], organism: Optional[str]) -> Optional[List[str]]:
    if not field_name: return None
    field = field_name.strip().lower()
    base = FIELD_TO_ONTOLOGIES.get(field)
    if not base: raise ValueError(f"Unsupported field: {field}")
    tax = _canonical_taxon(organism)
    if tax and tax in SPECIES_ROUTING and field in SPECIES_ROUTING[tax]:
        return SPECIES_ROUTING[tax][field]
    return base

# From bond/providers.py
def _norm(vectors: List[List[float]]) -> List[List[float]]:
    arr = np.asarray(vectors, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    return arr.tolist()

def resolve_embeddings(spec: str) -> Callable[[List[str]], List[List[float]]]:
    """Resolve embedding model using same logic as BOND (supports Ollama via direct API)."""
    if spec.startswith("st:"):
        # Local SentenceTransformer
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(spec[len("st:"):])
        def embed(texts: List[str]) -> List[List[float]]:
            embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return _norm(embeddings.tolist())
        return embed
    
    elif spec.startswith("ollama:") or spec.startswith("litellm:ollama/") or spec.startswith("ollama/"):
        # Ollama via direct API (same as BOND main app)
        import requests
        import json
        
        # Handle different ollama prefixes
        if spec.startswith("ollama:"):
            ollama_model = spec[len("ollama:"):]
        elif spec.startswith("litellm:ollama/"):
            ollama_model = spec[len("litellm:ollama/"):]
        elif spec.startswith("ollama/"):
            ollama_model = spec[len("ollama/"):]
        else:
            ollama_model = spec
            
        # Use OLLAMA_API_BASE environment variable
        api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        
        def embed(texts: List[str]) -> List[List[float]]:
            vectors = []
            for text in texts:
                try:
                    response = requests.post(
                        f"{api_base}/api/embeddings",
                        json={"model": ollama_model, "prompt": text},
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    vectors.append(data["embedding"])
                except Exception as e:
                    raise RuntimeError(f"Ollama embedding failed for text '{text[:50]}...': {e}")
            return _norm(vectors)
        return embed
    
    elif "/" in spec and not spec.startswith("st:"):
        # Auto-detect LiteLLM model (contains '/')
        import litellm
        from litellm import embedding as llm_embedding
        
        # Use appropriate API base for ollama models
        api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434") if spec.startswith("ollama/") else None
        
        def embed(texts: List[str]) -> List[List[float]]:
            kwargs = {"model": spec, "input": texts}
            if api_base:
                kwargs["api_base"] = api_base
            resp = llm_embedding(**kwargs)
            vectors = [item["embedding"] for item in resp["data"]]
            return _norm(vectors)
        return embed
    
    else:
        # Default to SentenceTransformer
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(spec)
        def embed(texts: List[str]) -> List[List[float]]:
            embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return _norm(embeddings.tolist())
        return embed

# From bond/retrieval/faiss_store.py
class FaissStore:
    def __init__(self, store_path: Path, rescore_multiplier: int = 20):
        self.index = faiss.read_index_binary(str(store_path / "embeddings.faiss"))
        self.id_map = np.load(store_path / "id_map.npy", allow_pickle=False)
        self.rescore_vectors = np.load(store_path / "rescore_vectors.npy", mmap_mode="r")
        self.rescore_multiplier = rescore_multiplier

    def search(self, vectors: np.ndarray, k: int) -> List[List[str]]:
        if vectors.ndim == 1: vectors = vectors.reshape(1, -1)
        k_rescore = k * self.rescore_multiplier
        q_bin = np.packbits(np.where(vectors >= 0, 1, 0), axis=-1)
        _, initial_indices = self.index.search(q_bin, k_rescore)
        all_results = []
        for i in range(vectors.shape[0]):
            cand_indices = initial_indices[i]
            cand_indices = cand_indices[cand_indices != -1]
            if len(cand_indices) == 0:
                all_results.append([])
                continue
            cand_vecs_fp32 = self.rescore_vectors[cand_indices].astype(np.float32) / 127.0
            scores = np.dot(cand_vecs_fp32, vectors[i].T).squeeze()
            final_idx = cand_indices[np.argsort(-scores)[:k]]
            all_results.append([str(self.id_map[j]) for j in final_idx])
        return all_results

# From bond/retrieval/bm25_sqlite.py
def search_bm25(conn: sqlite3.Connection, table_terms: str, table_fts: str, q: str, k: int, sources: Optional[List[str]] = None) -> List[Dict]:
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
    base_query += f" ORDER BY bm25({table_fts}) ASC LIMIT ?"
    params.append(k)
    cur.execute(base_query, params)
    return [{"id": row[0], "label": row[1]} for row in cur.fetchall()]

# From bond/fusion.py
def rrf_fuse(rankings: Dict[str, List[str]], k: float = 60.0, weights: Dict[str, float] | None = None) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    if weights is None: weights = {src: 1.0 for src in rankings.keys()}
    for src, ids in rankings.items():
        weight = weights.get(src, 1.0)
        for rank, id_ in enumerate(ids, start=1):
            scores[id_] += weight / (k + rank)
    fused = list(scores.items())
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused

def hydrate_ids(conn: sqlite3.Connection, ids: List[str]) -> List[Dict[str, str]]:
    if not ids: return []
    qmarks = ",".join("?" for _ in ids)
    cur = conn.cursor()
    cur.execute(f"SELECT curie, label FROM {TABLE_TERMS} WHERE curie IN ({qmarks})", ids)
    id_to_label = {row[0]: row[1] for row in cur.fetchall()}
    return [{"id": i, "label": id_to_label.get(i, "N/A")} for i in ids]

# --- Evaluation Logic ---
def calculate_accuracy(df_results: pd.DataFrame):
    def is_correct(row, k):
        predicted_ids = [pred['id'] for pred in row['predictions'][:k]]
        return row['target_ontology_id'] in predicted_ids
    total = len(df_results)
    acc1 = df_results.apply(lambda row: is_correct(row, 1), axis=1).sum() / total
    acc3 = df_results.apply(lambda row: is_correct(row, 3), axis=1).sum() / total
    acc5 = df_results.apply(lambda row: is_correct(row, 5), axis=1).sum() / total
    print("\n--- BM25 + Dense Hybrid Evaluation Results ---")
    print(f"Accuracy@1: {acc1:.4f}")
    print(f"Accuracy@3: {acc3:.4f}")
    print(f"Accuracy@5: {acc5:.4f}")
    print("--------------------------------------------")

def run_hybrid_evaluation():
    if not FAISS_STORE_PATH.exists():
        print(f"ERROR: FAISS store not found at {FAISS_STORE_PATH}")
        return
    if not INPUT_CSV_PATH.exists():
        print(f"ERROR: Input CSV not found at {INPUT_CSV_PATH}")
        return

    # Initialize components using same embedding model as BOND
    # Use the BOND_EMBED_MODEL from environment, fallback to signature file
    embed_model_spec = os.getenv("BOND_EMBED_MODEL")
    if not embed_model_spec:
        with open(FAISS_STORE_PATH / "embedding_signature.json", "r") as f:
            embed_model_name = json.load(f)["model_id"]
        embed_model_spec = f"st:{embed_model_name}"
    
    print(f"Using embedding model: {embed_model_spec}")
    embed_fn = resolve_embeddings(embed_model_spec)
    faiss_store = FaissStore(FAISS_STORE_PATH)
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    
    df_test = pd.read_csv(INPUT_CSV_PATH)
    
    # Add Hybrid prediction columns to the original dataframe
    hybrid_predictions = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Running Hybrid Baseline"):
        try:
            ontology_filter = allowed_ontologies_for(row['field'], row['organism'])
            
            bm25_hits = search_bm25(conn, TABLE_TERMS, TABLE_TERMS_FTS, row['source_term'], 20, sources=ontology_filter)
            bm25_ids = [hit['id'] for hit in bm25_hits]

            query_vector = np.array(embed_fn([row['source_term']]), dtype=np.float32)
            dense_ids_raw = faiss_store.search(query_vector, k=50)[0]
            if ontology_filter and dense_ids_raw:
                qmarks = ",".join("?" for _ in dense_ids_raw)
                cur = conn.cursor()
                cur.execute(
                    f"SELECT curie FROM {TABLE_TERMS} WHERE curie IN ({qmarks}) AND ontology_id IN ({','.join('?' for _ in ontology_filter)})",
                    dense_ids_raw + ontology_filter)
                valid_ids = {r[0] for r in cur.fetchall()}
                dense_ids = [id_ for id_ in dense_ids_raw if id_ in valid_ids]
            else:
                dense_ids = dense_ids_raw

            rankings = {"bm25": bm25_ids, "dense": dense_ids}
            fused_results = rrf_fuse(rankings, weights={"bm25": 0.8, "dense": 0.6})
            predicted_ids = [res[0] for res in fused_results[:MAX_K]]
            
            hydrated_hits = hydrate_ids(conn, predicted_ids)
            hybrid_predictions.append(hydrated_hits)
        except Exception as e:
            print(f"Warning: Error processing term '{row['source_term']}': {e}")
            hybrid_predictions.append([])
            
    conn.close()
    
    # Calculate accuracy for evaluation
    df_eval = pd.DataFrame({
        'source_term': df_test['source_term'],
        'target_ontology_id': df_test['target_ontology_id'],
        'predictions': hybrid_predictions
    })
    calculate_accuracy(df_eval)
    
    # Add Hybrid (Dense+BM25) prediction columns to original dataframe
    # Top-1 predictions
    df_test['hybrid_top1_predicted_id'] = [pred[0]['id'] if len(pred) >= 1 else None for pred in hybrid_predictions]
    df_test['hybrid_top1_predicted_label'] = [pred[0]['label'] if len(pred) >= 1 else None for pred in hybrid_predictions]
    
    # Top-3 predictions
    for i in range(3):
        df_test[f'hybrid_top3_rank{i+1}_id'] = [pred[i]['id'] if len(pred) > i else None for pred in hybrid_predictions]
        df_test[f'hybrid_top3_rank{i+1}_label'] = [pred[i]['label'] if len(pred) > i else None for pred in hybrid_predictions]
    
    # Top-5 predictions
    for i in range(5):
        df_test[f'hybrid_top5_rank{i+1}_id'] = [pred[i]['id'] if len(pred) > i else None for pred in hybrid_predictions]
        df_test[f'hybrid_top5_rank{i+1}_label'] = [pred[i]['label'] if len(pred) > i else None for pred in hybrid_predictions]
    
    # Save the enhanced CSV with original structure + Hybrid predictions
    output_path = "/Users/rajlq7/Downloads/Terms/BOND/evals/Dense+BM25/dense+bm25_baseline_predictions.csv"
    df_test.to_csv(output_path, index=False)
    print(f"Hybrid baseline predictions appended and saved to {output_path}")
    print(f"Added {2 + 6 + 10} Hybrid prediction columns to original {len(df_test.columns) - 18} columns")

if __name__ == "__main__":
    run_hybrid_evaluation()

