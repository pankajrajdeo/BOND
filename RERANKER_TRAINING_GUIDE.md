# BOND Reranker Training Guide

## Overview

This guide explains how to train a cross-encoder reranker for the BOND pipeline to improve ontology normalization accuracy from ~75-80% (retrieval only) to ~85-90% (with reranker).

## Why Reranker Instead of Encoder Finetuning?

**The Problem with Encoder Finetuning:**
- Your benchmark data has **one-to-many mappings** (e.g., "lymphocyte" → multiple different ontology IDs in different contexts)
- This is **biologically valid** - the same author term can refer to different specific cell types depending on tissue, organism, etc.
- Finetuning a bi-encoder on this data creates **contradictory gradients** that prevent convergence
- Result: **47% accuracy@10** (worse than base model's 65-70%)

**Why Reranker Works:**
- Operates on **(query, candidate)** pairs rather than embedding terms in isolation
- Learns **context-dependent relevance**: "lymphocyte" in tonsil vs. blood can map to different IDs
- The one-to-many problem becomes a **feature** - model learns disambiguation
- Expected improvement: **10-15% boost** in Hit@10 accuracy

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│ STAGE 1: RETRIEVAL (Existing BOND Pipeline)    │
├─────────────────────────────────────────────────┤
│ • Dense (FAISS) → Top-50                        │
│ • BM25 (SQLite) → Top-20                        │
│ • Exact matching → Top-5                        │
│ • RRF Fusion → Top-50 candidates                │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 2: RERANKER (NEW - Train on Benchmark)   │
├─────────────────────────────────────────────────┤
│ Model: Cross-Encoder (DeBERTa-v3-base)          │
│ Input: (query, candidate) pairs                 │
│ Output: Relevance scores → Top-10               │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ STAGE 3: LLM DISAMBIGUATION (Existing)         │
├─────────────────────────────────────────────────┤
│ Input: Top-10 reranked candidates               │
│ Output: Final chosen term + reasoning           │
└─────────────────────────────────────────────────┘
```

---

## Step 1: Generate Reranker Training Data

### What the Script Does

The script `build_reranker_training_data.py`:

1. Loads benchmark data (author_term → ontology_id pairs)
2. Runs BOND retrieval for each query (Dense + BM25 + Exact → RRF)
3. Labels candidates:
   - **Positive (label=1)**: candidate_id matches correct ontology_id
   - **Hard negative (label=0)**: retrieved but wrong ID
   - **Random negative (label=0)**: same field type, not retrieved
4. Formats data for cross-encoder training

### Running the Script

**Quick test (10 examples):**
```bash
cd /Users/rajlq7/Desktop/PhD_Projects_Oct_2025/BOND
source bond_venv/bin/activate

export BOND_ASSETS_PATH="/Users/rajlq7/Desktop/PhD_Projects_Oct_2025/BOND/assets"
export BOND_SQLITE_PATH="/Users/rajlq7/Desktop/PhD_Projects_Oct_2025/BOND/assets/ontologies.sqlite"
export BOND_RETRIEVAL_ONLY=1

python scripts/build_reranker_training_data.py \
  --benchmark_path "Miscellaneous/CellxGene_Benchmark/.../bond_czi_benchmark_data_hydrated_test.jsonl" \
  --output_path "reranker_training_data/test_sample.jsonl" \
  --num_hard_negatives 5 \
  --num_random_negatives 3 \
  --sample_size 10 \
  --top_k_retrieval 20
```

**Full data generation (all splits):**
```bash
bash scripts/generate_all_reranker_data.sh
```

This generates:
- `reranker_training_data/train.jsonl` (~1.5M examples from 173K queries)
- `reranker_training_data/dev.jsonl` (~90K examples)
- `reranker_training_data/test.jsonl` (~85K examples)

**Expected runtime:** 2-4 hours on CPU (depends on BOND retrieval speed)

### Data Format

Each training example is a JSON object:
```json
{
  "query": "cell_type: cDC1; tissue: tonsil; organism: Homo sapiens",
  "candidate": "label: conventional dendritic cell; synonyms: DC1 | cDC | ...; definition: ...",
  "candidate_id": "CL:0000990",
  "correct_id": "CL:0000990",
  "label": 1,
  "retrieval_score": 0.95,
  "retrieval_rank": 0,
  "example_type": "positive"
}
```

---

## Step 2: Train Cross-Encoder Reranker

### Script: `train_reranker.py`

```python
#!/usr/bin/env python3
"""
Train cross-encoder reranker for BOND.

Usage:
    python scripts/train_reranker.py \
        --train_path reranker_training_data/train.jsonl \
        --val_path reranker_training_data/dev.jsonl \
        --output_path reranker_checkpoints/bond-reranker-v1 \
        --model_name microsoft/deberta-v3-base \
        --epochs 3 \
        --batch_size 16 \
        --learning_rate 2e-5
"""

import json
import argparse
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader


def load_reranker_data(jsonl_path, max_examples=None):
    """Load (query, candidate, label) examples from JSONL."""
    examples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            row = json.loads(line)
            # CrossEncoder expects (text1, text2, label)
            examples.append(InputExample(
                texts=[row['query'], row['candidate']],
                label=float(row['label'])
            ))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--val_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--model_name', default='microsoft/deberta-v3-base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_train_examples', type=int, default=None)
    parser.add_argument('--max_val_examples', type=int, default=10000)

    args = parser.parse_args()

    # Load data
    print(f"Loading training data from {args.train_path}...")
    train_examples = load_reranker_data(args.train_path, args.max_train_examples)
    print(f"Loaded {len(train_examples)} training examples")

    print(f"Loading validation data from {args.val_path}...")
    val_examples = load_reranker_data(args.val_path, args.max_val_examples)
    print(f"Loaded {len(val_examples)} validation examples")

    # Initialize cross-encoder
    print(f"Initializing cross-encoder: {args.model_name}")
    model = CrossEncoder(
        args.model_name,
        num_labels=1,  # Binary relevance score
        max_length=512,
        default_activation_function=torch.nn.Sigmoid()
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size
    )

    # Evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        val_examples,
        name='bond_reranker_val'
    )

    # Train
    print(f"Training for {args.epochs} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        optimizer_params={'lr': args.learning_rate},
        warmup_steps=args.warmup_steps,
        output_path=args.output_path,
        save_best_model=True,
        show_progress_bar=True
    )

    print(f"Training complete! Model saved to {args.output_path}")


if __name__ == '__main__':
    import torch
    main()
```

### Training Command

```bash
python scripts/train_reranker.py \
  --train_path reranker_training_data/train.jsonl \
  --val_path reranker_training_data/dev.jsonl \
  --output_path reranker_checkpoints/bond-reranker-v1 \
  --model_name microsoft/deberta-v3-base \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --warmup_steps 1000
```

**Hardware requirements:**
- GPU: T4, V100, A100 (16GB+ VRAM recommended)
- Training time: 2-4 hours for 1.5M examples

**Alternative base models:**
- `microsoft/deberta-v3-base` (184M params, best quality)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (23M params, faster)
- `microsoft/MiniLM-L12-H384-uncased` (33M params, balanced)

---

## Step 3: Integrate Reranker into BOND Pipeline

### Modify `bond/pipeline.py`

Add reranker loading in `__init__`:

```python
class BondMatcher:
    def __init__(self, settings: Optional[BondSettings] = None):
        # ... existing code ...

        # Load reranker if available
        self.reranker = None
        reranker_path = os.getenv("BOND_RERANKER_PATH")
        if reranker_path and os.path.exists(reranker_path):
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_path)
            logger.info(f"Loaded reranker from {reranker_path}")
```

Add reranking step in `_query_internal` (after RRF fusion, before LLM):

```python
# After line ~790 (after metadata hydration, before soft boosts)
if self.reranker is not None and ranked:
    logger.info("Applying cross-encoder reranking...")
    ranked = self._rerank_with_cross_encoder(
        query, field_name, tissue, organism, ranked, top_k=10
    )
```

Add reranking method:

```python
def _rerank_with_cross_encoder(
    self,
    query: str,
    field_name: str,
    tissue: Optional[str],
    organism: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Rerank candidates using cross-encoder."""

    # Build query text (same format as training)
    query_text = f"{field_name}: {query}; tissue: {tissue or 'unknown'}; organism: {organism}"

    # Build (query, candidate) pairs
    pairs = []
    for cand in candidates:
        # Format candidate text (same as training)
        syns = []
        for syn_type in ['synonyms_exact', 'synonyms_related', 'synonyms_broad']:
            syns.extend(cand.get(syn_type, []) or [])
        syns = syns[:10]  # cap at 10

        definition = (cand.get('definition', '') or '')[:200]

        parts = [f"label: {cand.get('label', '')}"]
        if syns:
            parts.append(f"synonyms: {' | '.join(syns)}")
        if definition:
            parts.append(f"definition: {definition}")

        cand_text = "; ".join(parts)
        pairs.append([query_text, cand_text])

    # Score with cross-encoder
    scores = self.reranker.predict(pairs)

    # Add reranker scores to candidates
    for i, cand in enumerate(candidates):
        cand['reranker_score'] = float(scores[i])

    # Re-sort by reranker score
    reranked = sorted(candidates, key=lambda x: x['reranker_score'], reverse=True)

    logger.info(f"Reranker top-3 scores: {[c['reranker_score'] for c in reranked[:3]]}")

    return reranked[:top_k]
```

### Environment Variable

Add to `.env`:
```bash
BOND_RERANKER_PATH=/path/to/reranker_checkpoints/bond-reranker-v1
```

---

## Step 4: Evaluate Reranker Performance

### Evaluation Script

```bash
python scripts/evaluate_with_reranker.py \
  --benchmark_path benchmark_data/bond_czi_benchmark_data_hydrated_test.jsonl \
  --reranker_path reranker_checkpoints/bond-reranker-v1 \
  --output_path evals/reranker_results.jsonl
```

### Expected Results

| Pipeline Stage | Hit@1 | Hit@3 | Hit@10 | MRR |
|----------------|-------|-------|--------|-----|
| Retrieval only | 0.65  | 0.75  | 0.80   | 0.71 |
| + Reranker     | 0.75  | 0.85  | 0.90   | 0.81 |
| + LLM          | 0.88  | 0.94  | 0.96   | 0.91 |

**Expected improvements:**
- **10-15% boost** in Hit@10 before LLM stage
- **Higher quality top-10** for LLM → better final predictions
- **Reduced LLM costs**: Skip LLM if reranker confidence > 0.95

---

## Troubleshooting

### Issue: Out of memory during training

**Solution:** Reduce batch size or use gradient accumulation
```python
model.fit(
    ...,
    batch_size=8,  # Reduce from 16
    gradient_accumulation_steps=2  # Effective batch size = 16
)
```

### Issue: Training too slow

**Solutions:**
1. Use smaller base model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
2. Sample training data: `--max_train_examples 500000`
3. Use mixed precision training (FP16)

### Issue: Reranker not improving accuracy

**Check:**
1. Data quality: Are labels correct?
2. Text formatting: Does it match training format?
3. Positive/negative balance: Should be ~5-10% positives
4. Base retrieval quality: If retrieval is <50%, reranker can't help

---

## Next Steps

1. ✅ Generate reranker training data (DONE - running now)
2. ⏳ Train cross-encoder reranker (~3 hours)
3. ⏳ Integrate into BOND pipeline
4. ⏳ Evaluate on benchmark
5. ⏳ (Optional) Train field-specific rerankers for cell_type, disease, etc.
6. ⏳ (Optional) Ensemble multiple rerankers

---

## Summary

**DO:**
- ✅ Use reranker for context-dependent disambiguation
- ✅ Train on (query, candidate, label) pairs from retrieval
- ✅ Keep encoder frozen - don't finetune on contaminated data
- ✅ Use cross-encoder (DeBERTa) for best quality

**DON'T:**
- ❌ Finetune bi-encoder on author terms (creates contradictions)
- ❌ Try to "fix" one-to-many mappings (biologically valid)
- ❌ Expect perfect accuracy from retrieval alone
- ❌ Skip the reranker and go straight to LLM (expensive + lower quality)

**Expected outcome:** 85-90% Hit@10 with reranker (vs. 47% with encoder finetuning)
