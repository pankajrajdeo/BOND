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
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from datasets import Dataset
from typing import List


def load_reranker_data(jsonl_path: str, max_examples: int = None) -> List[dict]:
    """Load training data from JSONL file and convert to dataset format."""
    examples = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            if line.strip():
                row = json.loads(line)
                examples.append({
                    'sentence1': row['query'],
                    'sentence2': row['candidate'],
                    'label': float(row['label'])
                })
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
    train_data = load_reranker_data(args.train_path, args.max_train_examples)
    train_dataset = Dataset.from_list(train_data)
    print(f"Loaded {len(train_dataset)} training examples")

    print(f"Loading validation data from {args.val_path}...")
    val_data = load_reranker_data(args.val_path, args.max_val_examples)
    val_dataset = Dataset.from_list(val_data)
    print(f"Loaded {len(val_dataset)} validation examples")

    # Initialize cross-encoder
    print(f"Initializing cross-encoder: {args.model_name}")
    model = CrossEncoder(
        args.model_name,
        num_labels=1,  # Binary classification
        max_length=512
    )

    # Initialize loss function with class weighting for imbalanced data
    import torch
    pos_weight = torch.tensor([len(train_data) / sum(1 for r in train_data if r['label'] == 1)])
    loss = BinaryCrossEntropyLoss(model, pos_weight=pos_weight)

    # Initialize evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        [(row['sentence1'], row['sentence2'], row['label']) for row in val_data],
        name='bond_reranker_val'
    )

    # Initialize training arguments
    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        evaluation_strategy='steps',
        eval_steps=1000,
        save_strategy='steps',
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model='bond_reranker_val_ndcg@10',
        save_total_limit=3,
        logging_steps=100,
        log_level='info',
        report_to='none'  # Set to 'wandb' or 'tensorboard' if desired
    )

    # Initialize trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator
    )

    # Train
    print(f"Training for {args.epochs} epochs...")
    trainer.train()

    print(f"Training complete! Model saved to {args.output_path}")


if __name__ == '__main__':
    main()

