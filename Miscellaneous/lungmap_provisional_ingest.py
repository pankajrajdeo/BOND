#!/usr/bin/env python3
"""
A script to intelligently ingest the Lung Cell Nomenclature XLSX file.
It creates an enriched copy of existing Cell Ontology terms and adds new
provisional terms, all under a new 'cl_provisional' source, while linking
them to their parents using IRIs.
"""

import os
import sys
import json
import sqlite3
import pandas as pd # <-- Use pandas to read Excel files
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bond.config import BondSettings
from bond.logger import logger

# --- Configuration ---
# Correct path to the XLSX file
XLSX_FILE_PATH = "/Users/rajlq7/Downloads/Terms/BOND/done/Lung-nomenclature.xlsx"
CUSTOM_SOURCE_NAME = "cl_provisional"
CUSTOM_ID_PREFIX = "LCNP"
ENRICHED_ID_PREFIX = "LCN" # Prefix for copied/enriched terms

def build_term_cache(sqlite_conn):
    """
    Queries the database to create caches for linking and copying.
    """
    logger.info("Building caches of existing terms for linking and copying...")
    cur = sqlite_conn.cursor()
    cur.execute("SELECT * FROM terms")
    
    label_to_iri = {}
    id_to_record = {}
    
    col_names = [description[0] for description in cur.description]
    
    for row in cur.fetchall():
        record = dict(zip(col_names, row))
        label = record.get('label')
        iri = record.get('iri')
        term_id = record.get('id')

        if label and iri:
            label_to_iri[label] = iri
        if term_id:
            id_to_record[term_id] = record
            
    logger.info(f"Caches created. {len(label_to_iri)} labels mapped to IRIs; {len(id_to_record)} records cached.")
    return label_to_iri, id_to_record

def get_last_id_number(sqlite_conn, prefix):
    """Finds the highest existing number for a given ID prefix."""
    cur = sqlite_conn.cursor()
    cur.execute("SELECT id FROM terms WHERE id LIKE ?", (f"{prefix}:%",))
    max_num = 0
    for row in cur.fetchall():
        try:
            num = int(row[0].split(':')[1])
            if num > max_num:
                max_num = num
        except (ValueError, IndexError):
            continue
    return max_num

def process_nomenclature_file(id_cache):
    """Reads the XLSX file using pandas and processes the data."""
    if not os.path.exists(XLSX_FILE_PATH):
        logger.error(f"File not found: {XLSX_FILE_PATH}.")
        sys.exit(1)

    records_to_insert = []
    unmapped_terms = []

    # *** FIX IS HERE: Read the Excel file using pandas ***
    df = pd.read_excel(XLSX_FILE_PATH)
    # Replace NaN values with empty strings for easier processing
    df = df.fillna('')

    for _, row in df.iterrows():
        lcn_name = str(row.get('LCN Cell Name', '')).strip()
        if not lcn_name:
            continue

        synonyms = [s.strip() for s in str(row.get('Synonyms', '')).split(',') if s.strip()]
        all_new_synonyms = set([lcn_name] + synonyms)
        ontology_id = str(row.get('Cell Ontology ID', '')).strip()

        if ontology_id and ontology_id in id_cache:
            original_record = id_cache[ontology_id].copy()
            original_record['id'] = f"{ENRICHED_ID_PREFIX}-{ontology_id}"
            original_record['source'] = CUSTOM_SOURCE_NAME
            
            existing_syn_list = json.loads(original_record.get('syn_generic') or '[]')
            updated_syn_set = set(existing_syn_list) | all_new_synonyms
            original_record['syn_generic'] = json.dumps(list(updated_syn_set))
            
            new_def_parts = [original_record.get('def')] + list(all_new_synonyms)
            original_record['def'] = " ".join(filter(None, new_def_parts))
            
            records_to_insert.append(original_record)
        else:
            unmapped_terms.append({
                "label": lcn_name,
                "synonyms": synonyms,
                "parent_label": str(row.get('Cell Parent', '')).strip()
            })
            
    return records_to_insert, unmapped_terms

def create_new_terms(sqlite_conn, unmapped_terms, parent_cache):
    """Creates new records for unmapped provisional terms, linking them to parents."""
    if not unmapped_terms:
        logger.info("No unmapped provisional terms to create.")
        return []

    logger.info(f"Found {len(unmapped_terms)} new unmapped terms to create and link.")
    id_counter = get_last_id_number(sqlite_conn, CUSTOM_ID_PREFIX) + 1
    
    new_records = []
    for term_data in unmapped_terms:
        label = term_data["label"]
        synonyms = term_data["synonyms"]
        parent_label = term_data["parent_label"]

        parent_iri = parent_cache.get(parent_label)
        parents_is_a_list = [parent_iri] if parent_iri else []
        if not parent_iri and parent_label:
            logger.warning(f"Could not find IRI for parent '{parent_label}' of new term '{label}'.")

        new_id = f"{CUSTOM_ID_PREFIX}:{str(id_counter).zfill(7)}"
        id_counter += 1

        full_text = " ".join([label] + synonyms)
        norm_label = " ".join(label.lower().split())

        new_records.append({
            'id': new_id, 'label': label, 'def': full_text, 'norm_label': norm_label,
            'source': CUSTOM_SOURCE_NAME, 'iri': None, 'definition': None,
            'syn_exact': '[]', 'syn_related': '[]', 'syn_broad': '[]',
            'syn_generic': json.dumps(synonyms), 'alt_ids': '[]', 'xrefs': '[]',
            'namespace': '[]', 'subsets': '[]', 'comments': '[]',
            'parents_is_a': json.dumps(parents_is_a_list), 'abstracts': '[]',
            'ingested_via': 'provisional_lung_ingestor', 'provenance_rank': 0, 'updated_at': None
        })
        
    return new_records

def main():
    """Main execution function."""
    cfg = BondSettings()
    sqlite_conn = sqlite3.connect(cfg.sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row

    label_cache, id_cache = build_term_cache(sqlite_conn)
    
    enriched_records, unmapped_terms = process_nomenclature_file(id_cache)
    new_records = create_new_terms(sqlite_conn, unmapped_terms, label_cache)
    
    all_records_to_insert = enriched_records + new_records
    
    if not all_records_to_insert:
        logger.info("No new data to ingest. Database is up to date.")
        sqlite_conn.close()
        return

    logger.info(f"Preparing to insert/replace a total of {len(all_records_to_insert)} records into the database.")
    
    cols = all_records_to_insert[0].keys()
    
    insert_query = f"""
        INSERT OR REPLACE INTO terms ({', '.join(cols)})
        VALUES ({', '.join('?' for _ in cols)})
    """
    
    insert_data = [tuple(rec[col] for col in cols) for rec in all_records_to_insert]
    
    cur = sqlite_conn.cursor()
    cur.executemany(insert_query, tqdm(insert_data, desc="Ingesting Data"))
    sqlite_conn.commit()
    sqlite_conn.close()

    logger.info("\n--- Lung Nomenclature Ingestion Complete ---")
    logger.info("The data has been intelligently added to your SQLite database.")
    logger.info("➡️ Your final step is to update the embeddings for the modified and new data.")
    print("\nRun the following command in your terminal:")
    print("\n    bond-build-index\n")

if __name__ == "__main__":
    main()