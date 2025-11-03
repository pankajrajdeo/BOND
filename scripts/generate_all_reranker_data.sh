#!/bin/bash
#
# Generate reranker training data for all splits (train, dev, test)
#

set -e  # Exit on error

# Activate virtual environment
source bond_venv/bin/activate

# Set environment variables
export BOND_ASSETS_PATH="/Users/rajlq7/Desktop/PhD_Projects_Oct_2025/BOND/assets"
export BOND_SQLITE_PATH="/Users/rajlq7/Desktop/PhD_Projects_Oct_2025/BOND/assets/ontologies.sqlite"
export BOND_RETRIEVAL_ONLY=1

# Base paths
BENCHMARK_BASE="Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data"
OUTPUT_BASE="reranker_training_data"

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "RERANKER TRAINING DATA GENERATION"
echo "=========================================="
echo ""

# Generate TRAIN split
echo ">>> Generating TRAIN split..."
python scripts/build_reranker_training_data.py \
  --benchmark_path "$BENCHMARK_BASE/bond_czi_benchmark_data_hydrated_train.jsonl" \
  --output_path "$OUTPUT_BASE/train.jsonl" \
  --num_hard_negatives 10 \
  --num_random_negatives 5 \
  --top_k_retrieval 50

echo ""
echo ">>> Generating DEV split..."
python scripts/build_reranker_training_data.py \
  --benchmark_path "$BENCHMARK_BASE/bond_czi_benchmark_data_hydrated_dev.jsonl" \
  --output_path "$OUTPUT_BASE/dev.jsonl" \
  --num_hard_negatives 10 \
  --num_random_negatives 5 \
  --top_k_retrieval 50

echo ""
echo ">>> Generating TEST split..."
python scripts/build_reranker_training_data.py \
  --benchmark_path "$BENCHMARK_BASE/bond_czi_benchmark_data_hydrated_test.jsonl" \
  --output_path "$OUTPUT_BASE/test.jsonl" \
  --num_hard_negatives 10 \
  --num_random_negatives 5 \
  --top_k_retrieval 50

echo ""
echo "=========================================="
echo "GENERATION COMPLETE!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_BASE"/*.jsonl

echo ""
echo "File statistics:"
for file in "$OUTPUT_BASE"/*.jsonl; do
    echo "  $file: $(wc -l < "$file") examples"
done

echo ""
echo "Label distribution (train):"
cat "$OUTPUT_BASE/train.jsonl" | jq -r '.example_type' | sort | uniq -c

echo ""
echo "Done! ✅"
