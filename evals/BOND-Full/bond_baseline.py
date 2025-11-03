
#!/usr/bin/env python3
"""
BOND Full Pipeline Evaluation Script with Complete Pipeline Dissection

This script evaluates the complete BOND framework including:
1. Query expansion (LLM-powered)
2. Multi-channel retrieval (Exact + BM25 + Dense)
3. Reciprocal Rank Fusion (RRF)
4. Context-aware re-ranking
5. Graph expansion (when triggered)
6. LLM disambiguation

ENHANCED FEATURES - COMPLETE PIPELINE VISIBILITY:
- Query expansion terms and context generation
- Candidate counts from each retrieval channel
- RRF fusion results (top-10 before graph)
- Graph expansion execution status and results
- Final disambiguation candidates provided to LLM
- Confidence scores and reasoning at each stage

OUTPUT: Original CSV + 42 BOND columns:
- 18 prediction columns (top-1, top-3, top-5 IDs/labels)
- 4 confidence/reasoning columns (fusion, retrieval, LLM confidence + reason)
- 11 pipeline dissection columns (expansion, retrieval, fusion, graph, disambiguation)
- 9 retrieval confidence columns for each ranked prediction

This enables complete analysis of where the pipeline succeeds or fails.
"""

import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Add BOND to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bond"))

from bond import BondMatcher
from bond.config import BondSettings

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()

# Configuration: Use sample for optimization, full dataset for final evaluation
USE_SAMPLE = True  # Set to False for full dataset evaluation
SAMPLE_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/BOND-Full/bond_sample_200.csv")
FULL_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/bond_benchmark_test.csv")

INPUT_CSV_PATH = SAMPLE_CSV_PATH if USE_SAMPLE else FULL_CSV_PATH
OUTPUT_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/BOND-Full/bond_baseline_predictions_sample.csv") if USE_SAMPLE else Path("/Users/rajlq7/Downloads/Terms/BOND/evals/BOND-Full/bond_baseline_predictions.csv")
MAX_K = 5  # For accuracy@1, @3, @5 calculation

# -----------------------------
# Accuracy Calculation (same as other baselines)
# -----------------------------
def calculate_accuracy(df_results: pd.DataFrame):
    """Calculate accuracy@1, @3, and @5 for BOND predictions."""
    def is_correct(row, k):
        predicted_ids = [pred['id'] for pred in row['predictions'][:k]]
        return row['target_ontology_id'] in predicted_ids
    
    total = len(df_results)
    if total == 0:
        print("\n--- BOND Full Pipeline Evaluation Results ---")
        print("No predictions to evaluate")
        print("--------------------------------------------")
        return
    
    acc1 = df_results.apply(lambda row: is_correct(row, 1), axis=1).sum() / total
    acc3 = df_results.apply(lambda row: is_correct(row, 3), axis=1).sum() / total
    acc5 = df_results.apply(lambda row: is_correct(row, 5), axis=1).sum() / total
    
    print("\n--- BOND Full Pipeline Evaluation Results ---")
    print(f"Accuracy@1: {acc1:.4f}")
    print(f"Accuracy@3: {acc3:.4f}")
    print(f"Accuracy@5: {acc5:.4f}")
    print("---------------------------------------------")
    
    return acc1, acc3, acc5

def run_bond_evaluation():
    """Run BOND evaluation on the benchmark dataset."""
    
    if not INPUT_CSV_PATH.exists():
        print(f"ERROR: Input CSV not found at {INPUT_CSV_PATH}")
        return
    
    mode = "SAMPLE (197 queries)" if USE_SAMPLE else "FULL DATASET (1,968 queries)"
    print(f"🚀 Starting BOND Full Pipeline Evaluation - {mode}")
    print(f"📄 Input: {INPUT_CSV_PATH}")
    print(f"📊 Output: {OUTPUT_CSV_PATH}")
    
    if USE_SAMPLE:
        print("🎯 OPTIMIZATION MODE: Using stratified sample for faster iteration")
        print("   • Change USE_SAMPLE = False for full evaluation")
    else:
        print("🎯 PRODUCTION MODE: Running full dataset evaluation")
    
    # Load benchmark dataset
    df_test = pd.read_csv(INPUT_CSV_PATH)
    print(f"📈 Loaded {len(df_test)} test samples")
    
    # Initialize BOND with default settings
    print("\n🔧 Initializing BOND matcher...")
    try:
        bond_settings = BondSettings()
        print(f"   • Embedding model: {bond_settings.embed_model}")
        print(f"   • Expansion LLM: {bond_settings.expansion_llm_model}")
        print(f"   • Disambiguation LLM: {bond_settings.disambiguation_llm_model}")
        print(f"   • Assets path: {bond_settings.assets_path}")
        
        matcher = BondMatcher(bond_settings)
        print("✅ BOND matcher initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize BOND matcher: {e}")
        return
    
    # Add BOND prediction columns to the original dataframe
    bond_predictions = []
    bond_confidence_data = []
    
    print(f"\n🔍 Processing {len(df_test)} queries...")
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Running BOND Pipeline"):
        try:
            # Query BOND with full pipeline - enable trace for confidence data
            result = matcher.query(
                query=row["source_term"],
                field_name=row["field"],
                organism=row["organism"],
                tissue=row.get("tissue", "N/A"),
                topk_final=MAX_K,  # Get top 5 for accuracy calculation
                return_trace=True,  # Enable to get confidence metrics
            )
            
            # Handle abstain case (BOND returned None)
            if result is None:
                print(f"Info: BOND abstained on '{row['source_term']}' - no valid candidates")
                bond_predictions.append([])
                # Populate with abstain info
                abstain_info = {
                    # Basic confidence
                    "fusion_score": None,
                    "retrieval_confidence": None, 
                    "llm_confidence": None,
                    "llm_reason": "BOND abstained - no valid candidates",
                    # Pipeline dissection
                    "original_query": row["source_term"],
                    "expansion_terms": [],
                    "context_terms": [],
                    "candidates_exact": 0,
                    "candidates_bm25": 0,
                    "candidates_dense": 0,
                    "fusion_top10": [],
                    "graph_executed": False,
                    "fusion_graph_top10": [],
                    "disambiguation_candidates": [],
                    "ontology_filter": []
                }
                bond_confidence_data.append(abstain_info)
                continue
            
            # Extract predictions from BOND result - use ranked results list
            predictions = []
            
            # BOND returns results ranked by fusion score, use this directly
            if "results" in result and result["results"]:
                for i, res in enumerate(result["results"][:MAX_K]):
                    predictions.append({
                        "id": res.get("id"),
                        "label": res.get("label"),
                        "retrieval_confidence": res.get("retrieval_confidence", None)
                    })
            
            # If no results, add the chosen result as fallback
            elif "chosen" in result and result["chosen"]:
                chosen = result["chosen"]
                predictions.append({
                    "id": chosen.get("id"),
                    "label": chosen.get("label"),
                    "retrieval_confidence": chosen.get("retrieval_confidence", None)
                })
            
            bond_predictions.append(predictions)
            
            # Extract comprehensive pipeline trace information
            chosen = result.get("chosen", {}) or {}  # Handle None case
            trace = result.get("trace", {}) or {}    # Handle None case
            
            # Basic confidence info
            confidence_info = {
                "fusion_score": chosen.get("fusion_score", None),
                "retrieval_confidence": chosen.get("retrieval_confidence", None),
                "llm_confidence": chosen.get("llm_confidence", None),
                "llm_reason": chosen.get("reason", None)
            }
            
            # Pipeline dissection info
            pipeline_info = {
                # Query expansion
                "original_query": trace.get("query", None),
                "expansion_terms": trace.get("expansions", []) or [],
                "context_terms": trace.get("context_terms", []) or [],
                
                # Retrieval channels
                "candidates_exact": len(trace.get("candidates", {}).get("exact", []) or []),
                "candidates_bm25": len(trace.get("candidates", {}).get("bm25", []) or []), 
                "candidates_dense": len(trace.get("candidates", {}).get("dense", []) or []),
                
                # RRF fusion results (before graph)
                "fusion_top10": trace.get("fusion", [])[:10] if trace.get("fusion") else [],
                
                # Graph expansion
                "graph_executed": bool(trace.get("fusion_graph")),
                "fusion_graph_top10": trace.get("fusion_graph", [])[:10] if trace.get("fusion_graph") else [],
                
                # Disambiguation input
                "disambiguation_candidates": trace.get("disambiguation", {}).get("candidates", []) if trace.get("disambiguation") else [],
                
                # Ontology filtering
                "ontology_filter": trace.get("ontology_filter", []) or []
            }
            
            # Combine all info
            all_info = {**confidence_info, **pipeline_info}
            bond_confidence_data.append(all_info)
            
        except Exception as e:
            print(f"Warning: Error processing term '{row['source_term']}': {e}")
            bond_predictions.append([])
            # Error case - populate with None values for all fields
            error_info = {
                # Basic confidence
                "fusion_score": None,
                "retrieval_confidence": None, 
                "llm_confidence": None,
                "llm_reason": None,
                # Pipeline dissection
                "original_query": None,
                "expansion_terms": [],
                "context_terms": [],
                "candidates_exact": None,
                "candidates_bm25": None,
                "candidates_dense": None,
                "fusion_top10": [],
                "graph_executed": None,
                "fusion_graph_top10": [],
                "disambiguation_candidates": [],
                "ontology_filter": []
            }
            bond_confidence_data.append(error_info)
    
    # Close BOND matcher resources
    try:
        matcher.close()
        print("🔒 BOND matcher resources closed")
    except Exception as e:
        print(f"Warning: Error closing BOND matcher: {e}")
    
    # Calculate accuracy for evaluation
    df_eval = pd.DataFrame({
        'source_term': df_test['source_term'],
        'target_ontology_id': df_test['target_ontology_id'],
        'predictions': bond_predictions
    })
    calculate_accuracy(df_eval)
    
    # Add BOND prediction columns to original dataframe (same format as other methods)
    print("\n📊 Creating enhanced CSV output...")
    
    # Top-1 predictions
    df_test['bond_top1_predicted_id'] = [pred[0]['id'] if len(pred) >= 1 else None for pred in bond_predictions]
    df_test['bond_top1_predicted_label'] = [pred[0]['label'] if len(pred) >= 1 else None for pred in bond_predictions]
    
    # Top-3 predictions
    for i in range(3):
        df_test[f'bond_top3_rank{i+1}_id'] = [pred[i]['id'] if len(pred) > i else None for pred in bond_predictions]
        df_test[f'bond_top3_rank{i+1}_label'] = [pred[i]['label'] if len(pred) > i else None for pred in bond_predictions]
    
    # Top-5 predictions  
    for i in range(5):
        df_test[f'bond_top5_rank{i+1}_id'] = [pred[i]['id'] if len(pred) > i else None for pred in bond_predictions]
        df_test[f'bond_top5_rank{i+1}_label'] = [pred[i]['label'] if len(pred) > i else None for pred in bond_predictions]
    
    # Add confidence and reasoning columns
    df_test['bond_fusion_score'] = [conf['fusion_score'] for conf in bond_confidence_data]
    df_test['bond_retrieval_confidence'] = [conf['retrieval_confidence'] for conf in bond_confidence_data]
    df_test['bond_llm_confidence'] = [conf['llm_confidence'] for conf in bond_confidence_data]
    df_test['bond_llm_reason'] = [conf['llm_reason'] for conf in bond_confidence_data]
    
    # Add pipeline dissection columns
    df_test['bond_original_query'] = [conf['original_query'] for conf in bond_confidence_data]
    df_test['bond_expansion_terms'] = [str(conf['expansion_terms']) for conf in bond_confidence_data]
    df_test['bond_context_terms'] = [str(conf['context_terms']) for conf in bond_confidence_data]
    df_test['bond_candidates_exact'] = [conf['candidates_exact'] for conf in bond_confidence_data]
    df_test['bond_candidates_bm25'] = [conf['candidates_bm25'] for conf in bond_confidence_data]
    df_test['bond_candidates_dense'] = [conf['candidates_dense'] for conf in bond_confidence_data]
    df_test['bond_fusion_top10'] = [str(conf['fusion_top10']) for conf in bond_confidence_data]
    df_test['bond_graph_executed'] = [conf['graph_executed'] for conf in bond_confidence_data]
    df_test['bond_fusion_graph_top10'] = [str(conf['fusion_graph_top10']) for conf in bond_confidence_data]
    df_test['bond_disambiguation_candidates'] = [str(conf['disambiguation_candidates']) for conf in bond_confidence_data]
    df_test['bond_ontology_filter'] = [str(conf['ontology_filter']) for conf in bond_confidence_data]
    
    # Add retrieval confidence for top predictions
    df_test['bond_top1_retrieval_conf'] = [pred[0]['retrieval_confidence'] if len(pred) >= 1 else None for pred in bond_predictions]
    for i in range(3):
        df_test[f'bond_top3_rank{i+1}_retrieval_conf'] = [pred[i]['retrieval_confidence'] if len(pred) > i else None for pred in bond_predictions]
    for i in range(5):
        df_test[f'bond_top5_rank{i+1}_retrieval_conf'] = [pred[i]['retrieval_confidence'] if len(pred) > i else None for pred in bond_predictions]
    
    # Ensure output directory exists
    OUTPUT_CSV_PATH.parent.mkdir(exist_ok=True)
    
    # Save the enhanced CSV with original structure + BOND predictions
    df_test.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"✅ BOND baseline predictions appended and saved to {OUTPUT_CSV_PATH}")
    
    # Count new columns added
    bond_pred_cols = 2 + 6 + 10  # top1 + top3 + top5 predictions 
    bond_conf_cols = 4  # fusion_score, retrieval_confidence, llm_confidence, llm_reason
    bond_pipeline_cols = 11  # original_query, expansion_terms, context_terms, candidates_*, fusion_top10, graph_executed, fusion_graph_top10, disambiguation_candidates, ontology_filter
    bond_ret_conf_cols = 1 + 3 + 5  # retrieval confidence for top1, top3, top5
    total_bond_cols = bond_pred_cols + bond_conf_cols + bond_pipeline_cols + bond_ret_conf_cols
    
    print(f"📈 Added {total_bond_cols} BOND columns to original {len(df_test.columns) - total_bond_cols} columns")
    print(f"   • {bond_pred_cols} prediction columns (IDs + labels)")
    print(f"   • {bond_conf_cols} confidence/reasoning columns")
    print(f"   • {bond_pipeline_cols} pipeline dissection columns")
    print(f"   • {bond_ret_conf_cols} retrieval confidence columns")
    
    # Show some sample results with confidence
    print(f"\n📋 Sample predictions with confidence:")
    sample_cols = [
        'source_term', 'field', 'organism', 'target_ontology_id', 
        'bond_top1_predicted_id', 'bond_top1_predicted_label',
        'bond_fusion_score', 'bond_retrieval_confidence', 'bond_llm_confidence'
    ]
    existing_cols = [col for col in sample_cols if col in df_test.columns]
    if existing_cols:
        sample_df = df_test[existing_cols].head()
        # Truncate reason column if it exists for better display
        if 'bond_llm_reason' in df_test.columns:
            sample_df_display = sample_df.copy()
            for col in sample_df_display.columns:
                if 'reason' in col and sample_df_display[col].dtype == 'object':
                    sample_df_display[col] = sample_df_display[col].astype(str).str[:50] + '...'
        print(sample_df)

def main():
    """Main function to run BOND evaluation."""
    print("🎯 BOND Full Pipeline Evaluation")
    print("=" * 50)
    
    try:
        run_bond_evaluation()
        print(f"\n🎉 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Evaluation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
