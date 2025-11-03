import os
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
load_dotenv()  # loads ANTHROPIC_API_KEY
MODEL_NAME = "claude-sonnet-4-20250514"  # pick a concrete snapshot for stability
INPUT_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/remaining_rows.csv")
OUTPUT_FILE_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/anthropic_baseline_predictions_remaining.jsonl")
BATCH_SIZE = 100                     # operational chunking (not an Anthropic hard limit)
POLLING_INTERVAL_SECONDS = 30
SAVE_BATCH_RESULTS = True               # Save individual batch results to pending directory
PENDING_DIR = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/pending")

# -----------------------------
# Tool definition (JSON Schema)
# -----------------------------
TOOL_SCHEMA = {
    "name": "predict_ontology",
    "description": "Predicts the single best ontology term for a given biological entity.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ontology_id": {
                "type": ["string", "null"],
                "description": "The single best-matching ontology CURIE (e.g., 'CL:0000123'). Must be null if no confident match exists."
            },
            "ontology_label": {
                "type": ["string", "null"],
                "description": "The human-readable label for the chosen ontology term (e.g., 'macrophage'). Must be null if no confident match exists."
            },
            "reason": {
                "type": "string",
                "description": "A brief, one-sentence justification for the choice, referencing the input term and context."
            },
            "confidence": {
                "type": "number",
                "description": "A confidence score from 0.0 to 1.0 representing the model's certainty."
            }
        },
        "required": ["ontology_id", "ontology_label", "reason", "confidence"]
    }
}

# -----------------------------
# Static guidance (cacheable)
# -----------------------------
STATIC_GUIDANCE = """
You are a meticulous biomedical ontology curator. Your sole task is to map the given biological term to the single most accurate ontology CURIE and its standard label based on the provided context. You must respond by using the `predict_ontology` tool.

Example 1: Clear Match
Input:
- Term: "lung"
- Field: "tissue"
- Organism: "Homo sapiens"
- Tissue Context: "lung"
Tool Call:
{"ontology_id": "UBERON:0002048", "ontology_label": "lung", "reason": "The term 'lung' directly corresponds to the Uberon ontology term for the lung organ.", "confidence": 1.0}

Example 2: Context-Dependent Abbreviation
Input:
- Term: "TAM"
- Field: "cell_type"
- Organism: "Homo sapiens"
- Tissue Context: "lung adenocarcinoma"
Tool Call:
{"ontology_id": "CL:0001057", "ontology_label": "tumor-associated macrophage", "reason": "The abbreviation 'TAM' in a cancer tissue context refers to 'tumor-associated macrophage', which corresponds to this Cell Ontology ID.", "confidence": 0.9}

Example 3: No Confident Match
Input:
- Term: "fibro-myo-endothelial"
- Field: "cell_type"
- Organism: "Homo sapiens"
- Tissue Context: "heart"
Tool Call:
{"ontology_id": null, "ontology_label": null, "reason": "The term is a composite description of multiple cell types and does not map to a single ontology term.", "confidence": 0.2}
""".strip()

# -----------------------------
# Per-row prompt template
# -----------------------------
PROMPT_TEMPLATE = """\
Analyze the following input and respond ONLY by calling the `predict_ontology` tool with your answer.

Input:
- Term: "{source_term}"
- Field: "{field}"
- Organism: "{organism}"
- Tissue Context: "{tissue}"
""".strip()


def run_anthropic_batch_evaluation():
    # 1) Init client & load data
    try:
        client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY
        print("Anthropic client initialized.")
        df_test = pd.read_csv(INPUT_CSV_PATH)
        print(f"Loaded {len(df_test)} test samples from {INPUT_CSV_PATH}")
        
        # Ensure pending directory exists
        PENDING_DIR.mkdir(exist_ok=True)
        print(f"Pending directory: {PENDING_DIR}")
        
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_CSV_PATH}")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize client or load data. Details: {e}")
        return

    # 2) Build requests
    all_requests = []
    for index, row in df_test.iterrows():
        prompt = PROMPT_TEMPLATE.format(
            source_term=row["source_term"],
            field=row["field"],
            organism=row["organism"],
            tissue=row.get("tissue", "N/A"),
        )

        params = {
            "model": MODEL_NAME,
            "max_tokens": 256,
            "temperature": 0.0,
            # Cache your static guidance (and implicitly the tools before it)
            "system": [
                {
                    "type": "text",
                    "text": STATIC_GUIDANCE,
                    "cache_control": {"type": "ephemeral"}  # ttl defaults to 5m; you can set "ttl": "1h"
                }
            ],
            "tools": [TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": "predict_ontology"},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,  # string shorthand for a single text block
                }
            ],
        }

        all_requests.append(
            {
                "custom_id": f"row_{index}",
                "params": params,
            }
        )

    print(f"Prepared {len(all_requests)} API requests.")

    # Reset output file
    if OUTPUT_FILE_PATH.exists():
        OUTPUT_FILE_PATH.unlink()
        print(f"Removed existing results file: {OUTPUT_FILE_PATH}")

    # 3) Process in separate batches of exactly BATCH_SIZE
    num_batches = (len(all_requests) + BATCH_SIZE - 1) // BATCH_SIZE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n🚀 Starting batch processing: {num_batches} batches of up to {BATCH_SIZE} requests each")
    print(f"📊 Total requests to process: {len(all_requests)}")
    
    # Track all submitted batches
    submitted_batches = []
    
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_requests))
        request_chunk = all_requests[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"📦 BATCH {i+1}/{num_batches}")
        print(f"📊 Rows {start_idx} to {end_idx-1} ({len(request_chunk)} requests)")
        print(f"{'='*60}")

        try:
            # 4) Submit batch (don't wait)
            print(f"🚀 Submitting batch {i+1}...")
            batch = client.messages.batches.create(requests=request_chunk)
            batch_info = {
                'batch_id': batch.id,
                'batch_number': i+1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'request_count': len(request_chunk),
                'submitted_at': datetime.now().isoformat(),
                'status': batch.processing_status
            }
            submitted_batches.append(batch_info)
            
            print(f"✅ Batch {i+1} submitted successfully!")
            print(f"   🆔 Batch ID: {batch.id}")
            print(f"   📊 Status: {batch.processing_status}")
            print(f"   📝 Requests: {len(request_chunk)}")
            
            # Save batch info for tracking
            batch_info_file = PENDING_DIR / f"batch_{i+1:03d}_{timestamp}_info.json"
            with open(batch_info_file, 'w') as f:
                json.dump(batch_info, f, indent=2)
            print(f"   💾 Batch info saved to: {batch_info_file.name}")
            
            # Small delay between submissions
            if i < num_batches - 1:  # Don't delay after the last batch
                print(f"   ⏳ Waiting 2 seconds before next submission...")
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ ERROR submitting batch {i+1}: {e}")
            continue
    
    # Save summary of all submitted batches
    summary_file = PENDING_DIR / f"batch_summary_{timestamp}.json"
    summary = {
        'timestamp': timestamp,
        'total_requests': len(all_requests),
        'total_batches': len(submitted_batches),
        'batch_size': BATCH_SIZE,
        'input_file': str(INPUT_CSV_PATH),
        'batches': submitted_batches
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎉 ALL BATCHES SUBMITTED!")
    print(f"{'='*60}")
    print(f"📊 Summary:")
    print(f"   • Total batches submitted: {len(submitted_batches)}")
    print(f"   • Total requests: {len(all_requests)}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Summary saved to: {summary_file.name}")
    print(f"\n💡 Next steps:")
    print(f"   1. Monitor batch progress in Anthropic Console")
    print(f"   2. Use the batch IDs above to check status")
    print(f"   3. Download results when batches complete")
    print(f"   4. Run analyze_and_deduplicate.py to process results")
    
    # Print all batch IDs for easy reference
    print(f"\n🆔 Batch IDs for reference:")
    for batch_info in submitted_batches:
        print(f"   Batch {batch_info['batch_number']:2d}: {batch_info['batch_id']}")
    
    return submitted_batches




if __name__ == "__main__":
    run_anthropic_batch_evaluation()
