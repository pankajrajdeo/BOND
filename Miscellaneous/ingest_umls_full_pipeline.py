#!/usr/bin/env python3
"""
A complete, one-shot script to ingest UMLS data from a PostgreSQL database
into the BOND SQLite index and then generate and append the necessary FAISS embeddings.

This version uses a single, continuous progress bar for the embedding process.
"""

import os
import sys
import json
import sqlite3
import psycopg2
import numpy as np
import faiss
import io
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
from collections import defaultdict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass

# --- BEGIN USER CONFIGURATION ---
POSTGRES_DB = "umls_db"
POSTGRES_USER = "umls_user"
POSTGRES_PASSWORD = "Pankyaa@0598820"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
# --- END USER CONFIGURATION ---

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bond.config import BondSettings
from bond.providers import resolve_embeddings
from bond.logger import logger

TTY_TO_SYNONYM_TYPE = {
    'SY': 'syn_exact', 'SYN': 'syn_exact', 'IS': 'syn_exact',
    'RT': 'syn_related', 'RL': 'syn_related', 'RQ': 'syn_related',
    'BT': 'syn_broad', 'RB': 'syn_broad',
    'AB': 'syn_generic', 'EP': 'syn_generic',
}

def connect_to_postgres():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST, port=POSTGRES_PORT
        )
        logger.info("✅ Successfully connected to PostgreSQL UMLS database.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Could not connect to PostgreSQL: {e}")
        sys.exit(1)

def fetch_umls_data(pg_conn):
    """Fetches and processes UMLS data from PostgreSQL."""
    umls_concepts = defaultdict(lambda: {
        'label': None, 'definition': None, 'syn_exact': set(),
        'syn_related': set(), 'syn_broad': set(), 'syn_generic': set(),
        'parents_is_a': set(),
    })
    with pg_conn.cursor() as cur:
        logger.info("Fetching all English terms from MRCONSO...")
        query_terms = "SELECT CUI, STR, TTY, ISPREF, TS FROM mrconso WHERE LAT = 'ENG' AND SUPPRESS = 'N';"
        cur.execute(query_terms)
        for cui, term_string, tty, ispref, ts in tqdm(cur.fetchall(), desc="Processing Terms"):
            if ispref == 'Y' and ts == 'P':
                umls_concepts[cui]['label'] = term_string
            else:
                syn_type = TTY_TO_SYNONYM_TYPE.get(tty, 'syn_generic')
                umls_concepts[cui][syn_type].add(term_string)
        logger.info("Fetching definitions from MRDEF...")
        query_defs = "SELECT CUI, DEF FROM mrdef WHERE SAB != 'MSH' AND SUPPRESS = 'N';"
        cur.execute(query_defs)
        for cui, definition in tqdm(cur.fetchall(), desc="Processing Definitions"):
            if cui in umls_concepts and not umls_concepts[cui]['definition']:
                umls_concepts[cui]['definition'] = definition
        logger.info("Fetching parent relationships from MRREL...")
        query_parents = "SELECT CUI1, CUI2 FROM mrrel WHERE REL = 'PAR' AND SUPPRESS = 'N';"
        cur.execute(query_parents)
        for child_cui, parent_cui in tqdm(cur.fetchall(), desc="Processing Relationships"):
            if child_cui in umls_concepts:
                umls_concepts[child_cui]['parents_is_a'].add(parent_cui)
    return umls_concepts

def ingest_into_sqlite(sqlite_conn, umls_data):
    """Inserts the processed UMLS data into the BOND SQLite database."""
    cur = sqlite_conn.cursor()
    logger.info(f"Ingesting {len(umls_data)} UMLS concepts into the BOND database...")
    insert_query = """
        INSERT INTO terms (id, label, definition, source, syn_exact, syn_related, syn_broad, syn_generic, parents_is_a, def, norm_label, ingested_via)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            label=excluded.label, definition=excluded.definition, syn_exact=excluded.syn_exact,
            syn_related=excluded.syn_related, syn_broad=excluded.syn_broad, syn_generic=excluded.syn_generic,
            parents_is_a=excluded.parents_is_a, def=excluded.def, norm_label=excluded.norm_label;
    """
    records_to_insert = []
    for cui, data in umls_data.items():
        if not data['label']: continue
        full_text_parts = [data['label'], data['definition']] + list(data['syn_exact']) + list(data['syn_related']) + list(data['syn_broad']) + list(data['syn_generic'])
        full_text = " ".join(filter(None, full_text_parts))
        norm_label = " ".join((data['label'] or "").lower().split())
        records_to_insert.append((
            cui, data['label'], data['definition'], 'umls',
            json.dumps(list(data['syn_exact'])), json.dumps(list(data['syn_related'])),
            json.dumps(list(data['syn_broad'])), json.dumps(list(data['syn_generic'])),
            json.dumps(list(data['parents_is_a'])), full_text, norm_label, 'umls_ingestor_script'
        ))
    cur.executemany(insert_query, tqdm(records_to_insert, desc="Inserting Records"))
    sqlite_conn.commit()
    logger.info("✅ Successfully ingested UMLS data into SQLite.")


def generate_and_append_embeddings(cfg: BondSettings):
    """
    Generates embeddings for new terms and appends them to the FAISS index,
    showing a single continuous progress bar.
    """
    logger.info("--- Starting FAISS Embedding Generation for New Terms ---")
    
    sqlite_conn = sqlite3.connect(f"file:{cfg.sqlite_path}?mode=ro", uri=True)
    cur = sqlite_conn.cursor()

    store_path = os.path.join(cfg.assets_path, "faiss_store")
    os.makedirs(store_path, exist_ok=True)
    id_map_path = os.path.join(store_path, "id_map.npy")
    rescore_path = os.path.join(store_path, "rescore_vectors.npy")
    faiss_path = os.path.join(store_path, "embeddings.faiss")

    try:
        existing_ids = set(np.load(id_map_path))
        logger.info(f"Found {len(existing_ids)} existing vectors in FAISS index.")
    except FileNotFoundError:
        logger.warning("FAISS index not found. A new index will be created.")
        existing_ids = set()

    cur.execute("SELECT id FROM terms")
    all_db_ids = {row[0] for row in cur.fetchall()}
    new_ids = list(all_db_ids - existing_ids)

    if not new_ids:
        logger.info("✅ FAISS index is already synchronized. No new embeddings to generate.")
        sqlite_conn.close()
        return

    logger.info(f"Found {len(new_ids)} new terms to embed. Fetching text from database...")
    
    # Fetch all texts first to set up the single progress bar
    all_texts_to_embed = []
    all_ids_to_embed = []
    sql_batch_size = 900 # Stay under the 999 variable limit
    for i in tqdm(range(0, len(new_ids), sql_batch_size), desc="Fetching Texts"):
        id_chunk = new_ids[i:i + sql_batch_size]
        qmarks = ",".join("?" for _ in id_chunk)
        cur.execute(f"SELECT id, label, def FROM terms WHERE id IN ({qmarks})", id_chunk)
        rows = cur.fetchall()
        for r in rows:
            all_ids_to_embed.append(r[0])
            all_texts_to_embed.append((r[1] or "") + " " + (r[2] or ""))
    
    sqlite_conn.close()

    embed_fn = resolve_embeddings(cfg.embed_model)
    
    def embed_resilient(batch_texts: list, batch_ids: list):
        """Embed a batch, recursively bisecting on failure to isolate malformed items."""
        try:
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                out = embed_fn(batch_texts)
            if not isinstance(out, list) or len(out) != len(batch_texts):
                raise RuntimeError("Embedding provider returned unexpected shape")
            return out
        except Exception as e:
            if len(batch_texts) == 1:
                logger.error(f"Skipping malformed text for id={batch_ids[0]} due to error: {e}")
                return [None]
            mid = len(batch_texts) // 2
            left = embed_resilient(batch_texts[:mid], batch_ids[:mid])
            right = embed_resilient(batch_texts[mid:], batch_ids[mid:])
            return left + right

    model_batch_size = int(os.getenv("BOND_EMB_BATCH", "64"))
    save_chunk_size = 50000  # Save progress every 50,000 embeddings

    temp_embeddings = []
    temp_ids = []

    with tqdm(total=len(all_texts_to_embed), desc="Generating Embeddings") as pbar:
        for i in range(0, len(all_texts_to_embed), model_batch_size):
            batch_texts = all_texts_to_embed[i:i + model_batch_size]
            batch_ids = all_ids_to_embed[i:i + model_batch_size]
            
            batch_embeddings = embed_resilient(batch_texts, batch_ids)
            
            for k, emb in enumerate(batch_embeddings):
                if emb is not None:
                    temp_embeddings.append(emb)
                    temp_ids.append(batch_ids[k])
            
            pbar.update(len(batch_texts))

            # Save progress periodically
            if len(temp_embeddings) >= save_chunk_size:
                pbar.set_postfix_str("Saving progress...")
                if os.path.exists(faiss_path):
                    index = faiss.read_index_binary(faiss_path)
                else:
                    dimension_bits = len(temp_embeddings[0])
                    index = faiss.IndexBinaryFlat(dimension_bits)
                
                new_fp32 = np.array(temp_embeddings, dtype=np.float32)
                new_binary = np.packbits(np.where(new_fp32 >= 0, 1, 0), axis=-1)
                index.add(new_binary)

                new_rescore = np.clip(new_fp32 * 127, -127, 127).astype(np.int8)
                
                if os.path.exists(id_map_path):
                    id_map = np.load(id_map_path)
                    rescore_vectors = np.load(rescore_path)
                    final_ids = np.concatenate([id_map, np.array(temp_ids)])
                    final_rescore_vectors = np.concatenate([rescore_vectors, new_rescore])
                else:
                    final_ids = np.array(temp_ids)
                    final_rescore_vectors = new_rescore

                faiss.write_index_binary(index, faiss_path)
                np.save(id_map_path, final_ids)
                np.save(rescore_path, final_rescore_vectors)
                
                temp_embeddings, temp_ids = [], [] # Reset temps
                pbar.set_postfix_str("")

    # Final save for any remaining embeddings
    if temp_embeddings:
        pbar.set_postfix_str("Saving final batch...")
        index = faiss.read_index_binary(faiss_path)
        new_fp32 = np.array(temp_embeddings, dtype=np.float32)
        new_binary = np.packbits(np.where(new_fp32 >= 0, 1, 0), axis=-1)
        index.add(new_binary)
        new_rescore = np.clip(new_fp32 * 127, -127, 127).astype(np.int8)
        id_map = np.load(id_map_path)
        rescore_vectors = np.load(rescore_path)
        final_ids = np.concatenate([id_map, np.array(temp_ids)])
        final_rescore_vectors = np.concatenate([rescore_vectors, new_rescore])
        faiss.write_index_binary(index, faiss_path)
        np.save(id_map_path, final_ids)
        np.save(rescore_path, final_rescore_vectors)

    logger.info("✅ Successfully updated FAISS index with all new UMLS embeddings.")

def main():
    """Main execution function."""
    cfg = BondSettings()
    
    pg_conn = connect_to_postgres()
    sqlite_conn = sqlite3.connect(cfg.sqlite_path)
    if sqlite_conn.execute("SELECT COUNT(*) FROM terms WHERE source='umls'").fetchone()[0] == 0:
        logger.info("No UMLS data found in SQLite. Starting full ingestion...")
        umls_data = fetch_umls_data(pg_conn)
        ingest_into_sqlite(sqlite_conn, umls_data)
    else:
        logger.info("UMLS data already exists in SQLite. Skipping ingestion.")
    pg_conn.close()
    sqlite_conn.close()
    
    generate_and_append_embeddings(cfg)
    
    logger.info("\n--- UMLS Ingestion and Embedding Process Complete ---")

if __name__ == "__main__":
    main()