#!/usr/bin/env python3
"""
Script to analyze pending batch results, deduplicate them, and create a CSV for remaining rows.

This script:
1. Reads all pending batch result files
2. Extracts the row indices that have been processed
3. Identifies which rows from the original CSV are still missing
4. Creates a new CSV file with only the unprocessed rows for batch processing
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
INPUT_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/bond_benchmark_test.csv")
PENDING_DIR = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/pending")
OUTPUT_CSV_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/remaining_rows.csv")
CONSOLIDATED_RESULTS_PATH = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/consolidated_results.jsonl")

def extract_row_index_from_custom_id(custom_id: str) -> int:
    """Extract row index from custom_id like 'row_123'"""
    return int(custom_id.split("_")[1])

def analyze_pending_results():
    """Analyze all pending batch result files and extract processed row indices."""
    
    print("🔍 Analyzing pending batch results...")
    
    # Track processed rows and their results
    processed_rows = set()
    duplicate_rows = defaultdict(list)  # row_index -> [file1, file2, ...]
    all_results = {}  # row_index -> result_data
    
    # Process each pending file
    pending_files = list(PENDING_DIR.glob("*.jsonl"))
    print(f"Found {len(pending_files)} pending result files")
    
    for file_path in pending_files:
        print(f"  📄 Processing {file_path.name}...")
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    custom_id = result.get("custom_id")
                    
                    if custom_id:
                        row_index = extract_row_index_from_custom_id(custom_id)
                        
                        if row_index in processed_rows:
                            # This is a duplicate
                            duplicate_rows[row_index].append(file_path.name)
                        else:
                            processed_rows.add(row_index)
                            all_results[row_index] = result
                            
                except json.JSONDecodeError as e:
                    print(f"    ⚠️  JSON decode error in {file_path.name} line {line_num}: {e}")
                except Exception as e:
                    print(f"    ⚠️  Error processing {file_path.name} line {line_num}: {e}")
    
    print(f"\n📊 Analysis Results:")
    print(f"  • Total unique rows processed: {len(processed_rows)}")
    print(f"  • Duplicate rows found: {len(duplicate_rows)}")
    
    if duplicate_rows:
        print(f"  📋 Duplicate details:")
        for row_index, files in list(duplicate_rows.items())[:5]:  # Show first 5
            print(f"    - Row {row_index}: found in {len(files) + 1} files")
        if len(duplicate_rows) > 5:
            print(f"    ... and {len(duplicate_rows) - 5} more duplicates")
    
    return processed_rows, all_results

def identify_missing_rows():
    """Identify which rows from the original CSV haven't been processed yet."""
    
    print("\n🔍 Loading original CSV and identifying missing rows...")
    
    # Load original CSV
    df = pd.read_csv(INPUT_CSV_PATH)
    total_rows = len(df)
    print(f"  • Total rows in original CSV: {total_rows}")
    
    # Get processed rows
    processed_rows, all_results = analyze_pending_results()
    
    # Find missing rows
    all_row_indices = set(range(total_rows))
    missing_rows = all_row_indices - processed_rows
    
    print(f"  • Rows already processed: {len(processed_rows)}")
    print(f"  • Rows still missing: {len(missing_rows)}")
    print(f"  • Progress: {len(processed_rows)/total_rows*100:.1f}%")
    
    return df, missing_rows, processed_rows, all_results

def create_remaining_csv():
    """Create a CSV file with only the unprocessed rows."""
    
    df, missing_rows, processed_rows, all_results = identify_missing_rows()
    
    if not missing_rows:
        print("\n✅ All rows have been processed! No remaining CSV needed.")
        return
    
    print(f"\n📝 Creating CSV with {len(missing_rows)} remaining rows...")
    
    # Create DataFrame with only missing rows
    missing_indices = sorted(list(missing_rows))
    remaining_df = df.iloc[missing_indices].copy()
    
    # Save to CSV
    remaining_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"  ✅ Saved remaining rows to: {OUTPUT_CSV_PATH}")
    
    # Show some statistics
    print(f"\n📊 Remaining rows breakdown:")
    if 'field' in remaining_df.columns:
        field_counts = remaining_df['field'].value_counts()
        for field, count in field_counts.head(5).items():
            print(f"  • {field}: {count} rows")
    
    if 'organism' in remaining_df.columns:
        organism_counts = remaining_df['organism'].value_counts()
        for organism, count in organism_counts.head(3).items():
            print(f"  • {organism}: {count} rows")
    
    return missing_indices

def consolidate_results():
    """Consolidate all processed results into a single JSONL file."""
    
    _, _, processed_rows, all_results = identify_missing_rows()
    
    if not all_results:
        print("\n⚠️  No results to consolidate.")
        return
    
    print(f"\n📦 Consolidating {len(all_results)} results into single file...")
    
    # Load original CSV to get the source data
    df = pd.read_csv(INPUT_CSV_PATH)
    
    # Write consolidated results
    with open(CONSOLIDATED_RESULTS_PATH, 'w') as f:
        for row_index in sorted(all_results.keys()):
            result = all_results[row_index]
            
            # Extract the original row data
            original_row = df.iloc[row_index].to_dict()
            
            # Extract prediction from the result
            prediction = None
            if result.get("result", {}).get("type") == "succeeded":
                message = result["result"]["message"]
                content = message.get("content", [])
                
                for block in content:
                    if (block.get("type") == "tool_use" and 
                        block.get("name") == "predict_ontology"):
                        prediction = block.get("input", {})
                        break
            
            # Create consolidated output
            output = {
                **original_row,
                "predicted_ontology_id": prediction.get("ontology_id") if prediction else None,
                "predicted_ontology_label": prediction.get("ontology_label") if prediction else None,
                "predicted_reason": prediction.get("reason", "No result returned") if prediction else "No result returned",
                "predicted_confidence": prediction.get("confidence", 0.0) if prediction else 0.0,
            }
            
            f.write(json.dumps(output) + "\n")
    
    print(f"  ✅ Consolidated results saved to: {CONSOLIDATED_RESULTS_PATH}")

def main():
    """Main function to run the analysis and create remaining CSV."""
    
    print("🚀 Starting analysis of pending batch results...\n")
    
    # Create remaining CSV for unprocessed rows
    missing_indices = create_remaining_csv()
    
    # Consolidate existing results
    consolidate_results()
    
    print(f"\n✅ Analysis complete!")
    print(f"  📄 Remaining rows CSV: {OUTPUT_CSV_PATH}")
    print(f"  📦 Consolidated results: {CONSOLIDATED_RESULTS_PATH}")
    
    if missing_indices:
        print(f"\n💡 Next steps:")
        print(f"  1. Use {OUTPUT_CSV_PATH} as input for your next batch processing")
        print(f"  2. Update the INPUT_CSV_PATH in anthropic_baseline.py to point to the remaining rows CSV")
        print(f"  3. Run the batch processing script again")

if __name__ == "__main__":
    main()
