import os, json, time, math
from pathlib import Path
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
load_dotenv()  # expects OPENAI_API_KEY
MODEL = "gpt-4o"
INPUT_CSV = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/bond_benchmark_test.csv")
BATCH_INPUT_JSONL = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/OpenAI-Only/openai_batch_input.jsonl")
OUTPUT_JSONL = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/OpenAI-Only/lopenai_batch_output_merged.jsonl")
JOB_METADATA = {"job": "bond_ontology_gpt4o"}
POLL_SECS = 20

# -----------------------------
# Ontology Routing (from BOND schema_policies.py)
# -----------------------------
FIELD_TO_ONTOLOGIES = {
    "cell_type": ["cl", "fbbt", "zfa", "wbbt"], 
    "tissue": ["uberon", "fbbt", "zfa", "wbbt"],
    "disease": ["mondo", "pato"], 
    "development_stage": ["hsapdv", "mmusdv", "fbdv", "wbls", "zfa"],
    "sex": ["pato"], 
    "self_reported_ethnicity": ["hancestro"], 
    "assay": ["efo"], 
    "organism": ["ncbitaxon"],
}

SUPPORTED_ORGANISMS = {
    "Homo sapiens": "NCBITaxon:9606", 
    "Mus musculus": "NCBITaxon:10090",
    "Danio rerio": "NCBITaxon:7955", 
    "Drosophila melanogaster": "NCBITaxon:7227",
    "Caenorhabditis elegans": "NCBITaxon:6239",
}

SPECIES_ROUTING = {
    "NCBITaxon:9606": {"cell_type": ["cl"], "development_stage": ["hsapdv"], "tissue": ["uberon"]},
    "NCBITaxon:10090": {"cell_type": ["cl"], "development_stage": ["mmusdv"], "tissue": ["uberon"]},
    "NCBITaxon:7955": {"cell_type": ["cl", "zfa"], "development_stage": ["zfa"], "tissue": ["uberon", "zfa"]},
    "NCBITaxon:7227": {"cell_type": ["cl", "fbbt"], "development_stage": ["fbdv"], "tissue": ["uberon", "fbbt"]},
    "NCBITaxon:6239": {"cell_type": ["cl", "wbbt"], "development_stage": ["wbls"], "tissue": ["uberon", "wbbt"]},
}

ONTOLOGY_DESCRIPTIONS = {
    "cl": "Cell Ontology (CL) - for cell types",
    "uberon": "Uber Anatomy Ontology (UBERON) - for anatomical structures and tissues",
    "mondo": "Monarch Disease Ontology (MONDO) - for diseases",
    "pato": "Phenotype And Trait Ontology (PATO) - for phenotypes, traits, sex",
    "hsapdv": "Human Developmental Stages (HsapDv) - for human development stages",
    "mmusdv": "Mouse Developmental Stages (MmusDv) - for mouse development stages",
    "fbdv": "FlyBase Developmental Ontology (FBdv) - for fly development stages",
    "wbls": "WormBase Life Stage (WBls) - for worm development stages", 
    "zfa": "Zebrafish Anatomy and Development (ZFA) - for zebrafish anatomy and development",
    "fbbt": "FlyBase Anatomy Ontology (FBbt) - for fly anatomy",
    "wbbt": "WormBase Anatomy Ontology (WBbt) - for worm anatomy",
    "hancestro": "Human Ancestry Ontology (HANCESTRO) - for ancestry and ethnicity",
    "efo": "Experimental Factor Ontology (EFO) - for experimental assays and factors",
    "ncbitaxon": "NCBI Taxonomy - for organism classification"
}

def _canonical_taxon(organism: Optional[str]) -> Optional[str]:
    return SUPPORTED_ORGANISMS.get(organism)

def allowed_ontologies_for(field_name: Optional[str], organism: Optional[str]) -> Optional[List[str]]:
    if not field_name: 
        return None
    field = field_name.strip().lower()
    base = FIELD_TO_ONTOLOGIES.get(field)
    if not base: 
        raise ValueError(f"Unsupported field: {field}")
    tax = _canonical_taxon(organism)
    if tax and tax in SPECIES_ROUTING and field in SPECIES_ROUTING[tax]:
        return SPECIES_ROUTING[tax][field]
    return base

def get_ontology_guidance(field: str, organism: str) -> str:
    """Generate field and organism-specific ontology guidance for the prompt."""
    allowed_onts = allowed_ontologies_for(field, organism)
    if not allowed_onts:
        return ""
    
    guidance_parts = [f"\nIMPORTANT: For {field} in {organism}, restrict predictions to these ontologies:"]
    for ont in allowed_onts:
        desc = ONTOLOGY_DESCRIPTIONS.get(ont, f"{ont.upper()} ontology")
        guidance_parts.append(f"- {ont.upper()}: {desc}")
    
    guidance_parts.append(f"\nOnly return CURIEs from these {len(allowed_onts)} ontologies: {', '.join(ont.upper() for ont in allowed_onts)}")
    guidance_parts.append("If the term doesn't fit these ontologies, return null values.")
    
    return "\n".join(guidance_parts)

# -----------------------------
# Accuracy Calculation (same as other baselines)
# -----------------------------
def calculate_accuracy(predictions_list, target_ids):
    """Calculate accuracy@1 for OpenAI predictions."""
    total = len(predictions_list)
    if total == 0:
        print("\n--- OpenAI Evaluation Results ---")
        print("No predictions to evaluate")
        print("--------------------------------")
        return
    
    correct_count = 0
    for pred, target in zip(predictions_list, target_ids):
        # OpenAI only gives us top-1 prediction
        predicted_id = pred.get('id') if pred else None
        if predicted_id and predicted_id == target:
            correct_count += 1
    
    accuracy = correct_count / total
    print("\n--- OpenAI Evaluation Results ---")
    print(f"Accuracy@1: {accuracy:.4f}")
    print(f"Correct: {correct_count}/{total}")
    print("--------------------------------")
    
    return accuracy

# -----------------------------
# System guidance (shared across all requests)
# Keep this LONG and IDENTICAL across lines to maximize automatic prompt caching.
# Tip: include several few-shot examples so this shared prefix is >1,024 tokens.
# -----------------------------
SYSTEM_GUIDANCE = """You are a meticulous biomedical ontology curator. Output STRICT JSON only.
Your task: map the given biological term to the single most accurate ontology CURIE and label.
Follow organism and field-specific ontology restrictions. If no confident match within allowed ontologies, return nulls.
Keep 'reason' short; 'confidence' 0..1.

Return exactly:
{
  "ontology_id": "CL:0000123" | null,
  "ontology_label": "some label" | null,
  "reason": "one sentence",
  "confidence": 0.0-1.0
}

Examples:
Input:
- Term: "lung"
- Field: "tissue"
- Organism: "Homo sapiens"
- Tissue Context: "lung"
(Allowed: UBERON only for human tissues)
Output:
{"ontology_id":"UBERON:0002048","ontology_label":"lung","reason":"Direct match to Uberon lung organ.","confidence":1.0}

Input:
- Term: "TAM"
- Field: "cell_type"
- Organism: "Homo sapiens"
- Tissue Context: "lung adenocarcinoma"
(Allowed: CL only for human cell types)
Output:
{"ontology_id":"CL:0001057","ontology_label":"tumor-associated macrophage","reason":"TAM abbreviation maps to CL term.","confidence":0.9}

Input:
- Term: "E14"
- Field: "development_stage"
- Organism: "Mus musculus"
- Tissue Context: "embryo"
(Allowed: MmusDv only for mouse development)
Output:
{"ontology_id":"MMUSDV:0000014","ontology_label":"embryonic day 14","reason":"Mouse-specific developmental stage.","confidence":1.0}

Input:
- Term: "fibro-myo-endothelial"
- Field: "cell_type"
- Organism: "Homo sapiens"
- Tissue Context: "heart"
(Allowed: CL only for human cell types)
Output:
{"ontology_id":null,"ontology_label":null,"reason":"Composite label; no single CL term.","confidence":0.2}
""".strip()

USER_PROMPT_TEMPLATE = """Analyze and return ONLY a JSON object with keys:
ontology_id, ontology_label, reason, confidence.

Input:
- Term: "{source_term}"
- Field: "{field}"
- Organism: "{organism}"
- Tissue Context: "{tissue}"
"""

def build_batch_input():
    df = pd.read_csv(INPUT_CSV)
    with BATCH_INPUT_JSONL.open("w") as f:
        for i, row in df.iterrows():
            # Generate dynamic ontology restrictions for this specific field/organism combination
            ontology_guidance = get_ontology_guidance(row["field"], row["organism"])
            
            # Create field and organism-specific system guidance
            dynamic_system_guidance = SYSTEM_GUIDANCE + ontology_guidance
            
            prompt = USER_PROMPT_TEMPLATE.format(
                source_term=row["source_term"],
                field=row["field"],
                organism=row["organism"],
                tissue=row.get("tissue", "N/A"),
            )
            
            # Use Chat Completions with JSON mode for robust machine-parsable output
            body = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": dynamic_system_guidance},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 300,
                "response_format": {"type": "json_object"},  # force valid JSON
            }
            line = {
                "custom_id": f"row_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line) + "\n")
    return df

def main():
    client = OpenAI()

    # 1) Build JSONL with one request per line (single model per file)
    df = build_batch_input()
    print(f"Prepared {len(df)} requests at {BATCH_INPUT_JSONL}")

    # 2) Upload file for Batch
    up = client.files.create(file=BATCH_INPUT_JSONL.open("rb"), purpose="batch")
    print("Uploaded file:", up.id)

    # 3) Create batch job (24h window)
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=JOB_METADATA,
    )
    print("Batch created:", batch.id, "| status:", batch.status)

    # 4) Poll until terminal state
    term_states = {"completed", "failed", "cancelled", "expired"}
    with tqdm(total=100, desc="Polling batch") as pbar:
        while batch.status not in term_states:
            time.sleep(POLL_SECS)
            batch = client.batches.retrieve(batch.id)
            rc = batch.request_counts
            if rc:
                # Handle Pydantic model attributes
                done = (getattr(rc, "completed", 0) + getattr(rc, "failed", 0))
                total = getattr(rc, "total", 0) or 1
            else:
                done = 0
                total = 1
            pct = (done / total) * 100
            pbar.n = pct
            pbar.set_postfix_str(f"status={batch.status} {done}/{total}")
            pbar.refresh()
        pbar.n = 100
        pbar.set_postfix_str(f"status={batch.status}")
        pbar.refresh()
    if batch.status != "completed":
        print("Batch ended with status:", batch.status)

    # 5) Retrieve outputs (and errors if any)
    outputs = []
    if batch.output_file_id:
        out_text = client.files.content(batch.output_file_id).text
        outputs.extend(out_text.strip().splitlines())
    if batch.error_file_id:
        err_text = client.files.content(batch.error_file_id).text
        for line in err_text.strip().splitlines():
            outputs.append(line)

    # 6) Merge back to original rows and write both JSONL and CSV
    #    Each output line has: {"custom_id": "...","response":{"status_code":200,"body":{...}}, "error": null}
    by_id = {}
    for line in outputs:
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        by_id[cid] = obj

    # Prepare predictions for CSV
    openai_predictions = []
    
    # overwrite previous JSONL
    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()

    with OUTPUT_JSONL.open("w") as f:
        for i in range(len(df)):
            cid = f"row_{i}"
            base = df.iloc[i].to_dict()
            
            # Initialize prediction values
            pred_id = None
            pred_label = None
            pred_reason = "No result"
            pred_confidence = 0.0
            batch_error = None
            
            rec = by_id.get(cid)
            if rec and not rec.get("error"):
                body = (rec.get("response") or {}).get("body") or {}
                # Chat Completions JSON mode → choices[0].message.content is a JSON string
                try:
                    content = body["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    pred_id = parsed.get("ontology_id")
                    pred_label = parsed.get("ontology_label")
                    pred_reason = parsed.get("reason", "No reason provided")
                    pred_confidence = parsed.get("confidence", 0.0)
                except Exception as e:
                    pred_reason = f"Parse error: {e}"
            elif rec and rec.get("error"):
                batch_error = rec["error"].get("message", "Unknown batch error")
                pred_reason = f"Batch error: {batch_error}"
            
            # Store for CSV
            openai_predictions.append({
                'id': pred_id,
                'label': pred_label,
                'reason': pred_reason,
                'confidence': pred_confidence
            })
            
            # Update base for JSONL
            base.update(
                predicted_ontology_id=pred_id,
                predicted_ontology_label=pred_label,
                predicted_reason=pred_reason,
                predicted_confidence=pred_confidence,
                batch_error=batch_error,
            )
            f.write(json.dumps(base) + "\n")

    # 7) Calculate accuracy
    target_ids = df['target_ontology_id'].tolist()
    calculate_accuracy(openai_predictions, target_ids)
    
    # 8) Create enhanced CSV with OpenAI predictions appended
    # Add OpenAI prediction columns to original dataframe
    df['openai_top1_predicted_id'] = [pred['id'] for pred in openai_predictions]
    df['openai_top1_predicted_label'] = [pred['label'] for pred in openai_predictions]
    df['openai_predicted_reason'] = [pred['reason'] for pred in openai_predictions]
    df['openai_predicted_confidence'] = [pred['confidence'] for pred in openai_predictions]
    
    # Save enhanced CSV
    csv_output_path = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/OpenAI-Only/openai_baseline_predictions.csv")
    df.to_csv(csv_output_path, index=False)
    
    print(f"\n✅ Done. Results saved to:")
    print(f"   📄 JSONL: {OUTPUT_JSONL}")
    print(f"   📊 CSV: {csv_output_path}")
    print(f"   📈 Added 4 OpenAI prediction columns to original {len(df.columns) - 4} columns")

if __name__ == "__main__":
    main()
