#!/usr/bin/env python3
import json
import os

def convert_arrays_to_strings(data):
    """Convert array fields to pipe-separated strings for better HF compatibility"""
    for item in data:
        array_fields = [
            'original_synonyms_exact', 'original_synonyms_narrow', 'original_synonyms_broad',
            'original_synonyms_related', 'original_replaced_by', 'original_consider',
            'resolved_synonyms_exact', 'resolved_synonyms_narrow', 'resolved_synonyms_broad',
            'resolved_synonyms_related', 'resolution_path'
        ]
        
        for field in array_fields:
            if field in item:
                value = item[field]
                if isinstance(value, list):
                    # Convert list to pipe-separated string, or empty string if empty
                    item[field] = '|'.join(value) if value else ""
                else:
                    item[field] = ""
    
    return data

def process_jsonl_file(input_file, output_file):
    """Process JSONL file to make it more HF-compatible"""
    print(f"Processing {input_file}...")
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} records")
    
    # Convert arrays to strings
    data = convert_arrays_to_strings(data)
    
    # Write back to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Wrote {len(data)} records to {output_file}")

def main():
    files = [
        ('bond_czi_benchmark_data_hydrated_train.jsonl', 'bond_czi_benchmark_data_hf_train.jsonl'),
        ('bond_czi_benchmark_data_hydrated_dev.jsonl', 'bond_czi_benchmark_data_hf_dev.jsonl'),
        ('bond_czi_benchmark_data_hydrated_test.jsonl', 'bond_czi_benchmark_data_hf_test.jsonl')
    ]
    
    for input_file, output_file in files:
        if os.path.exists(input_file):
            process_jsonl_file(input_file, output_file)
        else:
            print(f"⚠️  {input_file} not found, skipping...")
    
    print("\n🎉 All files processed!")
    print("\nYou can now upload these files to Hugging Face:")
    for _, output_file in files:
        print(f"  - {output_file}")

if __name__ == "__main__":
    main()
