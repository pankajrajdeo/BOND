#!/usr/bin/env python3
"""
Convert consolidated_results.jsonl to CSV format for easier analysis.
"""

import json
import pandas as pd
from pathlib import Path

# File paths
INPUT_FILE = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/consolidated_results.jsonl")
OUTPUT_FILE = Path("/Users/rajlq7/Downloads/Terms/BOND/evals/Anthropic-Only/consolidated_results.csv")

def convert_jsonl_to_csv():
    """Convert JSONL file to CSV format."""
    
    print(f"📄 Reading JSONL file: {INPUT_FILE}")
    
    # Read JSONL file
    data = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"📊 Found {len(data)} records")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns for better readability
    column_order = [
        # Original data
        'source_term', 'field', 'organism', 'tissue', 'target_ontology_id', 'target_term',
        # Predictions
        'predicted_ontology_id', 'predicted_ontology_label', 'predicted_reason', 'predicted_confidence',
        # Additional metadata
        'dataset_id', 'dataset_title', 'collection_name', 'tissue_ontology_id', 'citation',
        'mapping_direction', 'source_column', 'target_column', 'ontology_column',
        'frequency', 'confidence', 'mapping_type', 'alternatives_count', 'total_occurrences'
    ]
    
    # Only include columns that exist in the data
    existing_columns = [col for col in column_order if col in df.columns]
    df_reordered = df[existing_columns]
    
    # Save to CSV
    df_reordered.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ CSV saved to: {OUTPUT_FILE}")
    print(f"📋 Columns: {len(df_reordered.columns)}")
    print(f"📊 Rows: {len(df_reordered.columns)}")
    
    # Show sample of data
    print(f"\n📋 Sample data:")
    print(df_reordered[['source_term', 'field', 'organism', 'predicted_ontology_id', 'predicted_ontology_label']].head())
    
    # Show field distribution
    if 'field' in df_reordered.columns:
        print(f"\n📊 Field distribution:")
        field_counts = df_reordered['field'].value_counts()
        for field, count in field_counts.items():
            print(f"   • {field}: {count} rows")
    
    # Show organism distribution
    if 'organism' in df_reordered.columns:
        print(f"\n🐾 Organism distribution:")
        org_counts = df_reordered['organism'].value_counts()
        for org, count in org_counts.items():
            print(f"   • {org}: {count} rows")
    
    return df_reordered

if __name__ == "__main__":
    df = convert_jsonl_to_csv()