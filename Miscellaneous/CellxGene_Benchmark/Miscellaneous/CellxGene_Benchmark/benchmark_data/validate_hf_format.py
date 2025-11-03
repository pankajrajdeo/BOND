#!/usr/bin/env python3
import json

def validate_hf_format(filename):
    """Validate the Hugging Face compatible format"""
    print(f"Validating {filename}...")
    
    issues = []
    line_count = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                line_count += 1
                
                # Check that array fields are now strings
                array_fields = [
                    'original_synonyms_exact', 'original_synonyms_narrow', 'original_synonyms_broad',
                    'original_synonyms_related', 'original_replaced_by', 'original_consider',
                    'resolved_synonyms_exact', 'resolved_synonyms_narrow', 'resolved_synonyms_broad',
                    'resolved_synonyms_related', 'resolution_path'
                ]
                
                for field in array_fields:
                    if field in data:
                        value = data[field]
                        if not isinstance(value, str):
                            issues.append(f"Line {line_num}: {field} is not a string (type: {type(value)})")
                
                # Check for None values in string fields
                string_fields = ['dataset_id', 'dataset_title', 'collection_name', 'field_type', 
                                'organism', 'tissue', 'author_term', 'original_ontology_id']
                for field in string_fields:
                    if field in data and data[field] is None:
                        issues.append(f"Line {line_num}: {field} is None")
                
            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: JSON decode error - {e}")
    
    print(f"Validated {filename}: {line_count} lines")
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
        return False
    else:
        print("✅ No issues found - file is ready for Hugging Face!")
        return True

if __name__ == "__main__":
    files = [
        "bond_czi_benchmark_data_hf_train.jsonl",
        "bond_czi_benchmark_data_hf_dev.jsonl", 
        "bond_czi_benchmark_data_hf_test.jsonl"
    ]
    
    all_valid = True
    for filename in files:
        print(f"\n{'='*50}")
        valid = validate_hf_format(filename)
        if not valid:
            all_valid = False
    
    print(f"\n{'='*50}")
    if all_valid:
        print("🎉 All files are ready for Hugging Face upload!")
        print("\nFiles to upload:")
        for filename in files:
            print(f"  - {filename}")
    else:
        print("❌ Some files have issues that need to be fixed.")
