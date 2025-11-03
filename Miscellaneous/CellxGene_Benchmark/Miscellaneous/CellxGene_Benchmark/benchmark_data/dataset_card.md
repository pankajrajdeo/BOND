---
annotations_creators:
- user-generated
language:
- en
language_creators:
- found
license:
- mit
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
source_datasets:
- original
task_categories:
- text-classification
- question-answering
task_ids:
- ontology-mapping
- entity-linking
- biomedical-named-entity-recognition
paperswithcode_id: null
config_name: bond_czi_benchmark
version:
  version_str: 1.0.0
  description: null
  major: 1
  minor: 0
  patch: 0
splits:
  train:
    name: train
    num_bytes: 411162791
    num_examples: 173632
    shard_lengths:
    - 173632
  validation:
    name: validation
    num_bytes: 23139805
    num_examples: 9639
    shard_lengths:
    - 9639
  test:
    name: test
    num_bytes: 23165775
    num_examples: 9645
    shard_lengths:
    - 9645
download_size: 457468371
dataset_size: 457468371
features:
  dataset_id:
    dtype: string
  dataset_title:
    dtype: string
  collection_name:
    dtype: string
  field_type:
    dtype: string
  organism:
    dtype: string
  tissue:
    dtype: string
  author_term:
    dtype: string
  original_ontology_id:
    dtype: string
  original_is_obsolete:
    dtype: int64
  original_label:
    dtype: string
  original_definition:
    dtype: string
  original_synonyms_exact:
    dtype: string
  original_synonyms_narrow:
    dtype: string
  original_synonyms_broad:
    dtype: string
  original_synonyms_related:
    dtype: string
  original_replaced_by:
    dtype: string
  original_consider:
    dtype: string
  resolved_ontology_id:
    dtype: string
  resolved_label:
    dtype: string
  resolved_definition:
    dtype: string
  resolved_synonyms_exact:
    dtype: string
  resolved_synonyms_narrow:
    dtype: string
  resolved_synonyms_broad:
    dtype: string
  resolved_synonyms_related:
    dtype: string
  resolution_path:
    dtype: string
  match_status:
    dtype: string
  support_dataset_count:
    dtype: int64
  support_row_count:
    dtype: int64
  llm_predicted_author_column:
    dtype: string
  author_confidence:
    dtype: float64
  ontology_confidence:
    dtype: float64
  split:
    dtype: string
