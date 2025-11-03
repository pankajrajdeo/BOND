# ============================================================
# LLM Workflow to Generate Dual-Target Column Mappings
# Deterministic, robust, and supports alternatives for edge cases
# ============================================================

import os
import re
import json
import getpass
import pandas as pd
from typing import Optional, List, Dict
from tqdm.auto import tqdm

# LangChain and Pydantic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# -------------------------------
# 1. Setup and Configuration
# -------------------------------

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

BASE_PROJECT_PATH = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/"
HARMONIZED_DATA_PATH = os.path.join(BASE_PROJECT_PATH, 'all_datasets_manifest.csv')
SCHEMA_ARCHIVE_PATH = os.path.join(BASE_PROJECT_PATH, 'metadata_schemas/')
FINAL_MAPPING_PATH = os.path.join(BASE_PROJECT_PATH, 'llm_generated_column_mappings_merged.json')

MODEL_NAME = "gpt-5-mini"
#LLM_TEMPERATURE = 0
LLM_SEED = 7
CANDIDATE_LIMIT = 3
MAX_HEADERS_FOR_PROMPT = 200
ROWS_PREVIEW = 5

# -------------------------------
# 2. Schema (strict with alternatives)
# -------------------------------

class FieldMapping(BaseModel):
    """Primary + optional alternatives for one standard category."""
    author_term_column: Optional[str] = Field(None)
    ontology_id_column: Optional[str] = Field(None)
    alternative_author_term_columns: Optional[List[str]] = Field(default=None)
    alternative_ontology_id_columns: Optional[List[str]] = Field(default=None)
    author_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    ontology_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    class Config:
        extra = "forbid"

class AuthorColumnMapping(BaseModel):
    assay: FieldMapping
    cell_type: FieldMapping
    development_stage: FieldMapping
    disease: FieldMapping
    self_reported_ethnicity: FieldMapping
    sex: FieldMapping
    tissue: FieldMapping

    class Config:
        extra = "forbid"

# -------------------------------
# 3. LLM Initialization
# -------------------------------

llm = ChatOpenAI(
    model=MODEL_NAME,
    #temperature=LLM_TEMPERATURE,
    max_retries=2,
    timeout=60,
    model_kwargs={"seed": LLM_SEED, "response_format": {"type": "json_object"}},
)

llm_with_structure = llm.with_structured_output(AuthorColumnMapping)

# -------------------------------
# 4. Prompt
# -------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are an expert bioinformatician specializing in CellxGene metadata curation.

Your goal is to select, for each category, exactly one best candidate when possible:
  (A) the author's free-text descriptive column ("author terms")
  (B) the corresponding ontology ID column

Categories:
- assay
- cell_type
- development_stage
- disease
- self_reported_ethnicity
- sex
- tissue

Key distinctions (read values, not just headers):
- "Author terms" are typically raw, inconsistent strings created by submitters. They often include typos, abbreviations, cluster labels or codes, lab-specific names, mixed casing, parentheses, slashes, numbers, or generic words like "normal", "unknown", "NA". Examples: "Rod", "ASPC 2", "CD4_Tcell_1 (naive)", "fibro_7", "cluster_11", "myeloid", "neuron/NeuN+", "n/a".
- Harmonized standard labels are clean ontology names or close paraphrases that align 1:1 with an ontology ID column. They look like canonical noun phrases (e.g., "retinal rod cell", "myofibroblast cell", "visceral fat"), typically paired with a *_ontology_term_id column.
- The standard column names are reserved by CellxGene, so if the column name is a standard column name (cell_type, assay, development_stage, disease, self_reported_ethnicity, sex, tissue), then it is a harmonized label and not an author term.
- Ontology ID values are CURIEs or codes, commonly matching prefixes like:
  CL:, UBERON:, EFO:, MONDO:, DOID:, HPO:, PATO:, HANCESTRO:, HsapDv:, MmusDv:, NCBITaxon:, CHEBI:, etc.

Strong heuristics:
1) Columns named *_ontology_term_id with CURIE-like values are (B) ontology ID columns.
2) If a column X has a paired X_ontology_term_id and X's values look like clean ontology labels,
   then X is a harmonized label (NOT an author term). Do NOT pick X as (A) unless its values clearly look raw/idiosyncratic.
3) Prefer columns explicitly hinting at author provenance when present (e.g., "author_*",
   "mapped_reference_annotation", "manual_annotation", "cluster_annotation", "free_annotation",
   "original_*", "celltype_original", "broad_cell_type", "fine_cell_type", "reported_*").
4) When multiple plausible author columns exist, pick the one whose values show the most "author-like" signals:
   high lexical variety; presence of digits/underscores/hyphens/slashes/parentheses; short codes; mixed case; lab jargon.
5) If no convincing author column exists (e.g., only a clean label paired with *_ontology_term_id),
   set (A) to null for that category.

Forbidden mistakes (be careful):
- Do NOT select the harmonized label column (e.g., "cell_type") as (A) when its values look canonical and it is paired with "cell_type_ontology_term_id".
- Do NOT assume columns starting with "author_" are always (A) — still verify the values look author-like.
- Do NOT infer (A) from header name alone. Always inspect the first {ROWS_PREVIEW} row values.

Decision process (apply per category):
1) Enumerate candidate (B) columns: prefer *_ontology_term_id with CURIE-like values.
2) For (A), inspect all non-ID columns whose header or context suggests relevance.
   Score each candidate by author-like value patterns vs. harmonized-label patterns.
3) Choose one primary (A) and (B) when possible.
4) If uncertain or multiple candidates look valid, include up to {CANDIDATE_LIMIT} alternatives for each side.
5) If no suitable column exists for (A) or (B), leave that field null.

Use ONLY column names that appear in **All Headers**.
Treat **Value Samples** as hints for variety; they are not exhaustive or row-aligned.
Return valid JSON ONLY that matches the strict schema. No extra text.
"""
        ),
        (
            "human",
            """Map the columns for this dataset:

## Context
- Dataset ID: {dataset_id}
- Dataset Title: {dataset_title}
- Collection: {collection_name}
- Organism: {organism}
- Primary Tissue (Harmonized): {tissue}

## Heuristic Candidates
{heuristic_candidates}

## All Headers (first {max_headers} shown)
{all_headers}

## First {rows_preview} Rows
```csv
{metadata_head}
```"""
        ),
    ]
)

chain = prompt | llm_with_structure

# -------------------------------
# 5. Helpers
# -------------------------------

AUTHOR_PATTERNS = {
    "assay": ["assay", "technology"],
    "cell_type": ["cell_type", "annotation"],
    "development_stage": ["stage", "age", "development"],
    "disease": ["disease", "condition", "phenotype"],
    "self_reported_ethnicity": ["ethnicity", "ancestry"],
    "sex": ["sex", "gender"],
    "tissue": ["tissue", "organ", "site"],
}
ONTO_PATTERNS = {
    "assay": ["assay_ontology", "_ontology_term_id"],
    "cell_type": ["cell_type_ontology", "_ontology_term_id"],
    "development_stage": ["development_stage_ontology", "_ontology_term_id"],
    "disease": ["disease_ontology", "_ontology_term_id"],
    "self_reported_ethnicity": ["ethnicity_ontology", "_ontology_term_id"],
    "sex": ["sex_ontology", "_ontology_term_id"],
    "tissue": ["tissue_ontology", "_ontology_term_id"],
}

def _contains_any(name: str, needles: List[str]) -> bool:
    nm = name.lower()
    return any(k in nm for k in needles)

def heuristic_candidates(headers: List[str]) -> Dict:
    out = {}
    for key in AUTHOR_PATTERNS:
        authors = [h for h in headers if _contains_any(h, AUTHOR_PATTERNS[key])]
        onts = [h for h in headers if _contains_any(h, ONTO_PATTERNS[key])]
        out[key] = {"likely_author": authors[:CANDIDATE_LIMIT],
                    "likely_ontology": onts[:CANDIDATE_LIMIT]}
    return out

def canonicalize(col: Optional[str], headers: List[str]) -> Optional[str]:
    if not col:
        return None
    lower_map = {h.lower(): h for h in headers}
    return lower_map.get(col.lower(), col if col in headers else None)

def sanitize_mapping(mapping: Dict, headers: List[str]) -> Dict:
    """Ensure outputs exist in headers; promote alternatives if needed."""
    header_set = set(headers)
    lower_map = {h.lower(): h for h in headers}

    def fix(col: Optional[str]) -> Optional[str]:
        if not col:
            return None
        return lower_map.get(col.lower(), col if col in header_set else None)

    def fix_list(cols: Optional[List[str]], primary: Optional[str]) -> Optional[List[str]]:
        if not cols:
            return None
        dedup = []
        for c in cols:
            canon = fix(c)
            if canon and canon != primary and canon not in dedup:
                dedup.append(canon)
        return dedup[:CANDIDATE_LIMIT] if dedup else None

    for cat, fm in mapping.items():
        a = fix(fm.get("author_term_column"))
        o = fix(fm.get("ontology_id_column"))
        alt_a = fix_list(fm.get("alternative_author_term_columns"), a)
        alt_o = fix_list(fm.get("alternative_ontology_id_columns"), o)

        if a is None and alt_a:
            a, alt_a = alt_a[0], alt_a[1:]
        if o is None and alt_o:
            o, alt_o = alt_o[0], alt_o[1:]

        fm["author_term_column"] = a
        fm["ontology_id_column"] = o
        fm["alternative_author_term_columns"] = alt_a
        fm["alternative_ontology_id_columns"] = alt_o
    return mapping

def nz(x, default="N/A"):
    try:
        return x if pd.notna(x) else default
    except Exception:
        return default

# -------------------------------
# 6. Main Processing
# -------------------------------

print("Loading harmonized data...")
harmonized_df = pd.read_csv(HARMONIZED_DATA_PATH, low_memory=False)
context_lookup = harmonized_df.drop_duplicates('dataset_id').set_index('dataset_id').to_dict('index')

if os.path.exists(FINAL_MAPPING_PATH):
    print("Loading existing mapping file...")
    with open(FINAL_MAPPING_PATH, 'r') as f:
        all_mappings = json.load(f)
else:
    all_mappings = {}

metadata_files = sorted(f for f in os.listdir(SCHEMA_ARCHIVE_PATH) if f.endswith('_metadata.csv') or f.endswith('_metadata.csv.gz'))
print(f"Found {len(metadata_files)} metadata files.")

# Calculate which datasets need processing
datasets_to_process = []
datasets_to_skip = []
for filename in metadata_files:
    dataset_id = filename.replace("_metadata.csv", "").replace(".gz", "")
    if dataset_id in all_mappings:
        datasets_to_skip.append(dataset_id)
    else:
        datasets_to_process.append(dataset_id)

print(f"Datasets already in mapping file: {len(datasets_to_skip)}")
print(f"Datasets to process: {len(datasets_to_process)}")

if not datasets_to_process:
    print("No new datasets to process. All datasets are already mapped.")
    exit(0)

progress_bar = tqdm(datasets_to_process, desc="Generating Mappings")

for dataset_id in progress_bar:
    # Try compressed file first, then uncompressed
    filename = f"{dataset_id}_metadata.csv.gz"
    filepath = os.path.join(SCHEMA_ARCHIVE_PATH, filename)
    
    if not os.path.exists(filepath):
        filename = f"{dataset_id}_metadata.csv"
        filepath = os.path.join(SCHEMA_ARCHIVE_PATH, filename)
    
    if not os.path.exists(filepath):
        tqdm.write(f"Warning: No metadata file found for {dataset_id}")
        continue

    try:
        context = context_lookup.get(dataset_id, {})
        # Handle both compressed and uncompressed files
        if filename.endswith('.gz'):
            df_full = pd.read_csv(filepath, compression='gzip', low_memory=False, on_bad_lines="skip")
        else:
            df_full = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")
        
        all_headers = df_full.columns.tolist()
        headers_for_prompt = all_headers[:MAX_HEADERS_FOR_PROMPT]
        metadata_head_str = df_full.head(ROWS_PREVIEW).to_csv(index=False)
        heur = heuristic_candidates(all_headers)

        mapping_result = chain.invoke({
            "dataset_id": dataset_id,
            "dataset_title": nz(context.get("dataset_title")),
            "collection_name": nz(context.get("collection_name")),
            "organism": nz(context.get("organisms")),
            "tissue": nz(context.get("tissue")),
            "heuristic_candidates": json.dumps(heur, indent=2),
            "all_headers": headers_for_prompt,
            "max_headers": MAX_HEADERS_FOR_PROMPT,
            "rows_preview": ROWS_PREVIEW,
            "metadata_head": metadata_head_str,
        })

        dump = mapping_result.model_dump() if hasattr(mapping_result, "model_dump") else mapping_result.dict()
        dump = sanitize_mapping(dump, all_headers)
        all_mappings[dataset_id] = dump

        # Update the JSON file incrementally after each mapping
        with open(FINAL_MAPPING_PATH, 'w') as f:
            json.dump(all_mappings, f, indent=4)

        tqdm.write(f"✓ {dataset_id} saved")
    except Exception as e:
        tqdm.write(f"Error processing {dataset_id}: {e}")

# -------------------------------
# 7. Summary
# -------------------------------

print(f"\n✅ Complete. Mappings saved to {FINAL_MAPPING_PATH}")

def coverage_stats(mappings: Dict[str, Dict]) -> Dict:
    cats = ["assay", "cell_type", "development_stage", "disease",
            "self_reported_ethnicity", "sex", "tissue"]
    stats = {c: {"both":0,"only_author":0,"only_ontology":0,"none":0,"has_alternatives":0} for c in cats}
    for mp in mappings.values():
        for c in cats:
            fm = mp.get(c, {})
            a, o = fm.get("author_term_column"), fm.get("ontology_id_column")
            alt_a, alt_o = fm.get("alternative_author_term_columns") or [], fm.get("alternative_ontology_id_columns") or []
            if a and o: stats[c]["both"] += 1
            elif a and not o: stats[c]["only_author"] += 1
            elif o and not a: stats[c]["only_ontology"] += 1
            else: stats[c]["none"] += 1
            if alt_a or alt_o: stats[c]["has_alternatives"] += 1
    return stats

if all_mappings:
    stats = coverage_stats(all_mappings)
    print("\nCoverage summary:")
    for cat, st in stats.items():
        print(f"  {cat:24s} -> both={st['both']}, only_author={st['only_author']}, "
              f"only_ontology={st['only_ontology']}, none={st['none']}, "
              f"has_alternatives={st['has_alternatives']}")
    preview = {k: all_mappings[k] for k in list(all_mappings)[:5]}
    print("\nPreview of first 5 mappings:")
    print(json.dumps(preview, indent=4))
