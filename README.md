# BOND-CZI Benchmark Dataset

A comprehensive benchmark dataset for evaluating biomedical ontology normalization models, derived from the CELLxGENE Census single-cell transcriptomics data.

**Repository**: [pankajrajdeo/bond-czi-benchmark](https://huggingface.co/datasets/pankajrajdeo/bond-czi-benchmark)

## 📊 Dataset Overview

The BOND-CZI Benchmark is a large-scale dataset designed to evaluate the performance of language models and machine learning systems in mapping author-provided biological terms to standardized ontology identifiers. This benchmark addresses the critical challenge of harmonizing diverse biological terminology across different research studies and datasets.

### Key Statistics

- **Total Datasets**: 1,574 from CELLxGENE Census
- **Processed Datasets**: 1,561 with successful column mappings
- **Final Datasets**: 1,027 with extracted training examples
- **Total Training Examples**: 64,777 author-term to ontology-ID mappings
- **Metadata Files**: 1,573 downloaded and processed
- **Organisms**: Primarily Homo sapiens (57,044 examples) and Mus musculus (7,717 examples)
- **Field Types**: 7 standard biological categories (assay, cell_type, development_stage, disease, sex, self_reported_ethnicity, tissue)
- **Unique Tissues**: 247 different tissue types

## 🎯 Task Description

The benchmark evaluates the ability of models to:
1. **Map author terms to ontology IDs**: Convert researcher-provided biological terms to standardized ontology identifiers
2. **Handle diverse terminology**: Process variations in biological naming conventions across different studies
3. **Maintain biological accuracy**: Ensure mappings preserve the intended biological meaning
4. **Scale across multiple domains**: Work across different biological categories and organisms

### Supported Ontologies

- **Cell Ontology (CL)**: Cell types and cellular components
- **UBERON**: Anatomical structures and tissues
- **Experimental Factor Ontology (EFO)**: Experimental factors and conditions
- **Human Phenotype Ontology (HPO)**: Human phenotypes and diseases
- **Gene Ontology (GO)**: Biological processes and molecular functions

## 📁 Dataset Structure

### Files

- `bond_czi_benchmark_data_hydrated_train.jsonl` (58,390 examples) - Training set
- `bond_czi_benchmark_data_hydrated_dev.jsonl` (3,224 examples) - Development set  
- `bond_czi_benchmark_data_hydrated_test.jsonl` (3,163 examples) - Test set

### Data Format

Each example in the JSONL files contains:

```json
{
  "dataset_id": "unique_dataset_identifier",
  "dataset_title": "Study title",
  "collection_name": "Collection name",
  "field_type": "cell_type|assay|tissue|disease|sex|development_stage|self_reported_ethnicity",
  "organism": "Homo sapiens|Mus musculus",
  "tissue": "tissue_name",
  "author_term": "researcher_provided_term",
  "ontology_id": "CL:0000001",
  "support_dataset_count": 5,
  "support_row_count": 25,
  "llm_predicted_author_column": "column_name",
  "author_confidence": 0.95,
  "ontology_confidence": 0.99,
  "split": "train|dev|test",
  "mapping": {
    "original": {
      "ontology_id": "CL:0000001",
      "is_obsolete": 0,
      "label": "standard_ontology_label",
      "definition": "detailed_definition",
      "synonyms": {
        "exact": ["exact_synonym1", "exact_synonym2"],
        "narrow": ["narrow_synonym1"],
        "broad": ["broad_synonym1"],
        "related": ["related_synonym1"]
      },
      "replaced_by": [],
      "consider": []
    },
    "resolved": {
      "ontology_id": "CL:0000001",
      "label": "standard_ontology_label",
      "definition": "detailed_definition",
      "synonyms": {...},
      "resolution_path": ["CL:0000001"]
    }
  }
}
```

## 🔬 Data Generation Pipeline

The benchmark was created through a comprehensive 6-step pipeline:

### 1. Manifest Generation
- **Input**: CELLxGENE Census (2025-01-30 snapshot)
- **Output**: Dataset manifest with 1,574 datasets
- **Process**: Analyzed dataset coverage across 7 biological fields and multiple organisms

### 2. Manifest Analysis
- **Input**: Dataset manifest
- **Output**: Coverage statistics and publication-ready figures
- **Process**: Generated comprehensive coverage analysis and visualizations

### 3. Metadata Download
- **Input**: Dataset IDs from manifest
- **Output**: 1,573 metadata files
- **Process**: Downloaded H5AD files, extracted metadata, and cleaned up temporary files

### 4. LLM Column Mapping
- **Input**: Metadata files and representative sample
- **Output**: Column mappings for 1,561 datasets
- **Process**: Used GPT-5 to map author columns to standard ontology columns

### 5. Data Compilation
- **Input**: Column mappings and metadata files
- **Output**: 64,777 training examples
- **Process**: Extracted author-term to ontology-ID pairs with parallel processing

### 6. Ontology Hydration
- **Input**: Compiled data and ontology database
- **Output**: Final JSONL files with enriched ontology information
- **Process**: Added ontology metadata, resolved obsolete terms, and created train/dev/test splits

## 📈 Dataset Statistics

### Field Distribution
- **cell_type**: 53,140 examples (82.0%)
- **tissue**: 6,068 examples (9.4%)
- **development_stage**: 2,573 examples (4.0%)
- **assay**: 1,789 examples (2.8%)
- **disease**: 802 examples (1.2%)
- **self_reported_ethnicity**: 306 examples (0.5%)
- **sex**: 99 examples (0.2%)

### Organism Distribution
- **Homo sapiens**: 57,044 examples (88.1%)
- **Mus musculus**: 7,717 examples (11.9%)

### Split Distribution
- **Train**: 58,390 examples (90.2%)
- **Dev**: 3,224 examples (5.0%)
- **Test**: 3,163 examples (4.9%)

### Field Coverage in Mappings
- **cell_type**: 1,374 datasets
- **tissue**: 611 datasets
- **development_stage**: 657 datasets
- **assay**: 443 datasets
- **disease**: 496 datasets
- **self_reported_ethnicity**: 231 datasets
- **sex**: 175 datasets

## 🔍 Quality Assessment

### Human Expert Review
A comprehensive human expert review was conducted on 10% of the LLM-generated column mappings:

- **LLM Performance**: 98.9% accuracy (277/280 correct predictions)
- **Inter-Annotator Agreement**: 99.1% agreement between reviewers (κ=0.663 - substantial)
- **Human-LLM Agreement**: 98.9% - experts validated LLM predictions as correct

This high accuracy demonstrates the reliability of the automated mapping process and the quality of the benchmark dataset.

## 🚀 Usage

### Loading the Dataset

```python
import json

# Load training examples
train_examples = []
with open('bond_czi_benchmark_data_hydrated_train.jsonl', 'r') as f:
    for line in f:
        train_examples.append(json.loads(line.strip()))

# Example usage
example = train_examples[0]
print(f"Author term: {example['author_term']}")
print(f"Ontology ID: {example['ontology_id']}")
print(f"Field type: {example['field_type']}")
```

### Evaluation Metrics

Recommended evaluation metrics for this benchmark:

1. **Exact Match Accuracy**: Percentage of predictions that exactly match the target ontology ID
2. **Top-k Accuracy**: Percentage of correct predictions in top-k candidates
3. **Field-specific Performance**: Performance breakdown by biological field type
4. **Organism-specific Performance**: Performance breakdown by organism
5. **Confidence Calibration**: Correlation between model confidence and accuracy

## 🔧 Technical Details

### Data Quality Measures
- **Support Counts**: Number of datasets and rows supporting each mapping
- **Confidence Scores**: LLM-generated confidence for author and ontology mappings
- **Obsolete Term Resolution**: Automatic resolution of deprecated ontology terms
- **Data Type Consistency**: Ensured consistent data types across all fields

### Processing Features
- **Parallel Processing**: Used multiprocessing for efficient data compilation
- **Error Handling**: Robust error handling for malformed CSV files
- **Memory Optimization**: Efficient processing of large metadata files
- **Data Validation**: Comprehensive validation of ontology IDs and term mappings

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@dataset{bond_czi_benchmark_2024,
  title={BOND-CZI Benchmark: A Large-Scale Dataset for Biomedical Ontology Normalization},
  author={Rajdeo, Pankaj},
  year={2024},
  url={https://huggingface.co/datasets/pankajrajdeo/bond-czi-benchmark}
}
```

## 🤝 Contributing

We welcome contributions to improve the benchmark:

1. **Data Quality**: Report issues with ontology mappings or data inconsistencies
2. **New Fields**: Suggest additional biological fields for inclusion
3. **Evaluation Metrics**: Propose new evaluation approaches
4. **Documentation**: Help improve documentation and examples

## 📄 License

This dataset is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## 🙏 Acknowledgments

- **CELLxGENE Census**: For providing the foundational single-cell transcriptomics data
- **OpenAI**: For providing GPT-5 access for column mapping generation
- **Ontology Providers**: For maintaining comprehensive biological ontologies
- **Research Community**: For contributing the diverse biological datasets

---

*This benchmark represents a significant step toward standardized evaluation of biomedical ontology normalization systems, enabling more robust and comparable assessments of model performance in this critical domain.*
