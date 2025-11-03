import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def extract_unique_terms(metadata_files: List[str], output_dir: str = "./bond_input"):
    """
    Extract unique values for 7 CellxGene field types across all datasets.
    
    Field types:
    1. assay
    2. cell_type
    3. development_stage
    4. disease
    5. self_reported_ethnicity
    6. sex
    7. tissue
    """
    
    # Field mappings - handle different naming conventions across datasets
    field_mappings = {
        'assay': ['assay', 'Assay Index'],
        'cell_type': ['cell_type', 'celltype', 'Cell Type Index', 'ann_finest_level', 'Final_CT'],
        'development_stage': ['development_stage', 'Age Index', 'age'],
        'disease': ['disease', 'Covariate Index', 'condition', 'Disease_status'],
        'self_reported_ethnicity': ['self_reported_ethnicity', 'Ethnicity Index', 'race'],
        'sex': ['sex', 'Sex Index'],
        'tissue': ['tissue', 'Tissue Index']
    }
    
    # Store unique values for each field type
    unique_values = {
        'assay': set(),
        'cell_type': set(),
        'development_stage': set(),
        'disease': set(),
        'self_reported_ethnicity': set(),
        'sex': set(),
        'tissue': set()
    }
    
    # Track which dataset contributed which terms
    term_provenance = defaultdict(lambda: defaultdict(set))
    
    # Context information (same across all datasets)
    context = {
        'organism': 'Homo sapiens',
        'tissue': 'lung'
    }
    
    print("=" * 80)
    print("EXTRACTING UNIQUE TERMS FROM METADATA FILES")
    print("=" * 80)
    
    for metadata_file in metadata_files:
        print(f"\n📂 Processing: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        dataset_name = Path(metadata_file).stem
        
        # Navigate the JSON structure
        if 'dataset' in data:
            indexes = data['dataset'].get('**Indexes and Annotations**', {})
        elif 'annotations' in data:  # Spatial data
            indexes = data['annotations']
        else:
            indexes = data
        
        # Extract values for each field type
        for field_type, possible_keys in field_mappings.items():
            values = extract_values_from_nested_dict(indexes, possible_keys)
            
            if values:
                print(f"  ✓ {field_type}: Found {len(values)} unique terms")
                unique_values[field_type].update(values)
                for val in values:
                    term_provenance[field_type][val].add(dataset_name)
            else:
                print(f"  ✗ {field_type}: No terms found")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY OF UNIQUE TERMS")
    print("=" * 80)
    
    for field_type in unique_values:
        count = len(unique_values[field_type])
        print(f"\n{field_type.upper()}: {count} unique terms")
        
        # Convert set to sorted list for JSON serialization
        terms_list = sorted(list(unique_values[field_type]))
        
        # Save individual field file for BOND processing
        output = {
            'field_type': field_type,
            'context': context,
            'total_unique_terms': count,
            'terms': []
        }
        
        for term in terms_list:
            output['terms'].append({
                'original_term': term,
                'datasets': sorted(list(term_provenance[field_type][term]))
            })
        
        # Save to JSON
        output_file = Path(output_dir) / f"{field_type}_unique_terms.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"  💾 Saved to: {output_file}")
        
        # Print first 10 terms as preview
        print(f"  📋 Preview (first 10):")
        for term in terms_list[:10]:
            datasets = ', '.join(sorted(list(term_provenance[field_type][term])))
            print(f"     - {term} [{datasets}]")
        if len(terms_list) > 10:
            print(f"     ... and {len(terms_list) - 10} more")
    
    # Save combined summary
    summary = {
        'extraction_summary': {
            'total_datasets': len(metadata_files),
            'context': context,
            'field_types': {
                field_type: {
                    'unique_terms': len(unique_values[field_type]),
                    'output_file': f"{field_type}_unique_terms.json"
                }
                for field_type in unique_values
            }
        }
    }
    
    summary_file = Path(output_dir) / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to: {summary_file}")
    print("=" * 80)
    
    return unique_values, term_provenance


def extract_values_from_nested_dict(d: dict, possible_keys: List[str]) -> Set[str]:
    """
    Recursively extract values from nested dictionary given possible key names.
    """
    values = set()
    
    def recurse(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                # Check if key matches any possible key
                if any(pk.lower() in key.lower() for pk in possible_keys):
                    if isinstance(val, list):
                        values.update([str(v) for v in val if v])
                    elif isinstance(val, dict):
                        # Extract from nested dict
                        for nested_val in val.values():
                            if isinstance(nested_val, list):
                                values.update([str(v) for v in nested_val if v])
                            elif nested_val and not isinstance(nested_val, dict):
                                values.add(str(nested_val))
                    elif val and not isinstance(val, dict):
                        values.add(str(val))
                else:
                    recurse(val)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(d)
    
    # Clean up values
    cleaned_values = set()
    for v in values:
        v = v.strip()
        if v and v.lower() not in ['', 'null', 'none', 'unknown', 'na', 'n/a']:
            cleaned_values.add(v)
    
    return cleaned_values


# Example usage
if __name__ == "__main__":
    metadata_files = [
        "BPD_Sucre_normalized_log_deg.json",
        "BPD_Sun_normalized_log_deg.json",
        "HLCA_full_superadata_v3_norm_log_deg.json",
        "HCA_fetal_lung_normalized_log_deg.json",
        "LungMAP_pulmonary_fibrosis_spatial.json"
    ]
    
    unique_values, provenance = extract_unique_terms(
        metadata_files,
        output_dir="./bond_input"
    )
    
    print("\n✅ Extraction complete! Ready for BOND processing.")
    print("\nNext steps:")
    print("1. Run BOND on each field type file in ./bond_input/")
    print("2. Get advisor validation")
    print("3. Build unified mapping dictionary")