# BOND CZI Benchmark Dataset

A comprehensive benchmark dataset for ontology mapping in biomedical single-cell data, derived from CELLxGENE metadata.

## Dataset Description

This dataset contains 192,916 training examples of author terms mapped to standardized ontology identifiers across 7 field types: assay, cell_type, development_stage, disease, self_reported_ethnicity, sex, and tissue.

## Data Structure

Each example contains:
- **Dataset context**: dataset_id, dataset_title, collection_name
- **Anchor context**: field_type, organism, tissue, author_term
- **Original ontology mapping**: ontology_id, label, definition, synonyms, obsolete status
- **Resolved ontology mapping**: resolved obsolete terms with full metadata
- **Evidence metrics**: support counts and confidence scores

## Splits

- **Train**: 173,632 examples (90%)
- **Validation**: 9,639 examples (5%)
- **Test**: 9,645 examples (5%)

## Field Types

1. **assay**: Experimental methods and technologies
2. **cell_type**: Cell classifications and types
3. **development_stage**: Developmental stages and ages
4. **disease**: Disease conditions and phenotypes
5. **self_reported_ethnicity**: Ethnicity and ancestry information
6. **sex**: Biological sex classifications
7. **tissue**: Tissue and organ types

## Data Format

The dataset is provided in JSONL format with the following structure:

```json
{
  "dataset_id": "00ff600e-6e2e-4d76-846f-0eec4f0ae417",
  "dataset_title": "Human tonsil nonlymphoid cells scRNA",
  "collection_name": "Single-cell analysis of human B cell maturation...",
  "field_type": "cell_type",
  "organism": "Homo sapiens",
  "tissue": "tonsil",
  "author_term": "cDC1",
  "original_ontology_id": "CL:0000990",
  "original_is_obsolete": 0,
  "original_label": "conventional dendritic cell",
  "original_definition": "Conventional dendritic cell is a dendritic cell...",
  "original_synonyms_exact": "DC1|cDC|dendritic reticular cell|type 1 DC",
  "original_synonyms_narrow": "",
  "original_synonyms_broad": "interdigitating cell|veiled cell",
  "original_synonyms_related": "interdigitating cell|veiled cell|DC1|cDC|dendritic reticular cell|type 1 DC",
  "original_replaced_by": "",
  "original_consider": "",
  "resolved_ontology_id": "CL:0000990",
  "resolved_label": "conventional dendritic cell",
  "resolved_definition": "Conventional dendritic cell is a dendritic cell...",
  "resolved_synonyms_exact": "DC1|cDC|dendritic reticular cell|type 1 DC",
  "resolved_synonyms_narrow": "",
  "resolved_synonyms_broad": "interdigitating cell|veiled cell",
  "resolved_synonyms_related": "interdigitating cell|veiled cell|DC1|cDC|dendritic reticular cell|type 1 DC",
  "resolution_path": "CL:0000990",
  "match_status": "unchanged",
  "support_dataset_count": 2,
  "support_row_count": 56,
  "llm_predicted_author_column": "author_cell_type",
  "author_confidence": 0.98,
  "ontology_confidence": 1.0,
  "split": "train"
}
```

**Note**: Array fields (synonyms, replaced_by, consider, resolution_path) are stored as pipe-separated strings for better Hugging Face compatibility. Empty arrays are represented as empty strings.

## Usage

This dataset is designed for training models to map free-text author terms to standardized ontology identifiers in biomedical contexts. It can be used for:

- Ontology mapping and entity linking
- Biomedical named entity recognition
- Text classification in biomedical domains
- Training biomedical language models

## Files

- `bond_czi_benchmark_data_hf_train.jsonl`: Training split (173,632 examples)
- `bond_czi_benchmark_data_hf_dev.jsonl`: Validation split (9,639 examples)
- `bond_czi_benchmark_data_hf_test.jsonl`: Test split (9,645 examples)

## Citation

If you use this dataset, please cite the original CELLxGENE publications and the BOND project.
