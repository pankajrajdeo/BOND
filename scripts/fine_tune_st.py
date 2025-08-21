#!/usr/bin/env python3
"""
Script for fine-tuning SentenceTransformer models for BOND.
This script provides a complete fine-tuning pipeline for biomedical ontology embeddings.
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data(data_path: str) -> List[InputExample]:
    """Load training data from various formats for MultipleNegativesRankingLoss"""
    data_path = Path(data_path)
    
    if data_path.suffix == '.csv':
        # CSV format: query, positive (negative is optional)
        df = pd.read_csv(data_path)
        examples = []
        for _, row in df.iterrows():
            # MultipleNegativesRankingLoss works with (anchor, positive) pairs
            examples.append(InputExample(
                texts=[row['query'], row['positive']], 
                label=1.0  # Always positive for ranking loss
            ))
        return examples
    
    elif data_path.suffix == '.json':
        # JSON format: list of training examples
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            if 'query' in item and 'positive' in item:
                examples.append(InputExample(
                    texts=[item['query'], item['positive']], 
                    label=1.0
                ))
            elif 'texts' in item and len(item['texts']) == 2:
                examples.append(InputExample(
                    texts=item['texts'], 
                    label=1.0
                ))
        return examples
    
    elif data_path.suffix == '.tsv':
        # TSV format: query, positive (negative is optional)
        df = pd.read_csv(data_path, sep='\t')
        examples = []
        for _, row in df.iterrows():
            examples.append(InputExample(
                texts=[row['query'], row['positive']], 
                label=1.0
            ))
        return examples
    
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}. Use .csv, .json, or .tsv")

def create_sample_training_data(output_path: str = "sample_training_data.csv"):
    """Create a sample training data file to help users get started"""
    sample_data = [
        {"query": "T-cell", "positive": "T lymphocyte"},
        {"query": "T-cell", "positive": "T cell"},
        {"query": "cancer", "positive": "malignant neoplasm"},
        {"query": "cancer", "positive": "carcinoma"},
        {"query": "protein", "positive": "polypeptide"},
        {"query": "protein", "positive": "amino acid sequence"},
        {"query": "heart", "positive": "cardiac"},
        {"query": "heart", "positive": "myocardium"},
        {"query": "brain", "positive": "cerebral"},
        {"query": "brain", "positive": "encephalon"},
        {"query": "blood", "positive": "hematologic"},
        {"query": "blood", "positive": "sanguineous"},
        {"query": "liver", "positive": "hepatic"},
        {"query": "liver", "positive": "hepatocyte"},
        {"query": "lung", "positive": "pulmonary"},
        {"query": "lung", "positive": "respiratory"},
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Sample training data created: {output_path}")
    logger.info(f"   Contains {len(sample_data)} (query, positive) pairs")
    logger.info(f"   Use this as a template for your own biomedical ontology data")

def create_evaluator(eval_data_path: str = None) -> InformationRetrievalEvaluator:
    """Create Information Retrieval evaluator for biomedical ontology evaluation"""
    if not eval_data_path or not os.path.exists(eval_data_path):
        return None
    
    try:
        eval_data = load_training_data(eval_data_path)
        
        # Convert to IR evaluation format
        queries = {}      # query_id -> query_text
        corpus = {}       # doc_id -> doc_text
        relevant_docs = {} # query_id -> set of relevant doc_ids
        
        # Create unique IDs for queries and documents
        query_counter = 0
        doc_counter = 0
        query_to_id = {}  # query_text -> query_id
        doc_to_id = {}    # doc_text -> doc_id
        
        for example in eval_data:
            if len(example.texts) == 2:
                query_text = example.texts[0]
                doc_text = example.texts[1]
                
                # Assign IDs
                if query_text not in query_to_id:
                    query_to_id[query_text] = f"q_{query_counter}"
                    queries[f"q_{query_counter}"] = query_text
                    query_counter += 1
                
                if doc_text not in doc_to_id:
                    doc_to_id[doc_text] = f"d_{doc_counter}"
                    corpus[f"d_{doc_counter}"] = doc_text
                    doc_counter += 1
                
                # Mark as relevant
                query_id = query_to_id[query_text]
                doc_id = doc_to_id[doc_text]
                
                if query_id not in relevant_docs:
                    relevant_docs[query_id] = set()
                relevant_docs[query_id].add(doc_id)
        
        logger.info(f"Created IR evaluator with {len(queries)} queries and {len(corpus)} documents")
        logger.info(f"Total relevant pairs: {sum(len(docs) for docs in relevant_docs.values())}")
        
        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="biomedical-ontology-eval",
            mrr_at_k=[5, 10],
            ndcg_at_k=[5, 10],
            accuracy_at_k=[1, 3, 5, 10],
            precision_recall_at_k=[1, 3, 5, 10],
            map_at_k=[10, 100],
            show_progress_bar=True,
            batch_size=32
        )
        
    except Exception as e:
        logger.warning(f"Could not create IR evaluator: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SentenceTransformer for BOND using MultipleNegativesRankingLoss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Data Format Examples:

CSV/TSV format:
  query,positive
  "T-cell","T lymphocyte"
  "cancer","malignant neoplasm"
  "protein","polypeptide"

JSON format:
  [
    {"query": "T-cell", "positive": "T lymphocyte"},
    {"query": "cancer", "positive": "malignant neoplasm"}
  ]

The script will automatically create in-batch negatives for training.
For evaluation, provide similar format data to measure retrieval performance.
        """
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Base model to fine-tune")
    parser.add_argument("--train_data", required=True, help="Path to training data (.csv, .json, or .tsv)")
    parser.add_argument("--eval_data", help="Path to evaluation data (optional, same format as training)")
    parser.add_argument("--output_dir", required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size (higher = more in-batch negatives)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--scale", type=float, default=20.0, help="Scale parameter for ranking loss (higher = harder negatives)")
    parser.add_argument("--create_sample", action="store_true", help="Create sample training data and exit")
    
    args = parser.parse_args()
    
    # Handle sample data creation
    if args.create_sample:
        create_sample_training_data()
        return 0
    
    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Fine-tuning {args.model} on {device}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Log recommendations for MultipleNegativesRankingLoss
    logger.info("🎯 MultipleNegativesRankingLoss Recommendations:")
    logger.info(f"  • Batch size: {args.batch_size} (higher = more in-batch negatives = better training)")
    logger.info(f"  • Scale: {args.scale} (higher = harder negatives, 20.0 is optimal)")
    logger.info("  • Training data should be (query, positive) pairs")
    logger.info("  • No need to provide negative examples - they're generated automatically")
    
    # Load training data
    try:
        train_examples = load_training_data(args.train_data)
        logger.info(f"Loaded {len(train_examples)} training examples")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return 1
    
    if len(train_examples) == 0:
        logger.error("No training examples found")
        return 1
    
    # Validate training data format for MultipleNegativesRankingLoss
    logger.info("📊 Training data validation:")
    logger.info(f"  • Total examples: {len(train_examples)}")
    logger.info(f"  • Format: (anchor, positive) pairs")
    
    # Check for potential issues
    unique_anchors = set()
    unique_positives = set()
    for example in train_examples:
        if len(example.texts) == 2:
            unique_anchors.add(example.texts[0])
            unique_positives.add(example.texts[1])
    
    logger.info(f"  • Unique anchors: {len(unique_anchors)}")
    logger.info(f"  • Unique positives: {len(unique_positives)}")
    
    if len(unique_anchors) < 10:
        logger.warning("⚠️  Very few unique anchors - consider adding more variety")
    if len(unique_positives) < 10:
        logger.warning("⚠️  Very few unique positives - consider adding more variety")
    
    # Recommend batch size based on data size
    recommended_batch_size = min(64, len(train_examples) // 2)
    if args.batch_size < recommended_batch_size:
        logger.info(f"💡 Recommended batch size: {recommended_batch_size} (current: {args.batch_size})")
        logger.info("   Higher batch size = more in-batch negatives = better training signal")
    
    # Load base model
    try:
        model = SentenceTransformer(args.model, device=device)
        logger.info(f"Loaded base model: {args.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create evaluator if evaluation data provided
    evaluator = create_evaluator(args.eval_data)
    if evaluator:
        logger.info("📊 Information Retrieval Evaluation Enabled:")
        logger.info("  • MRR@5, MRR@10: Mean Reciprocal Rank (higher is better)")
        logger.info("  • NDCG@5, NDCG@10: Normalized Discounted Cumulative Gain")
        logger.info("  • Accuracy@1,3,5,10: Top-k accuracy")
        logger.info("  • Precision@1,3,5,10: Precision at k")
        logger.info("  • Recall@1,3,5,10: Recall at k")
        logger.info("  • MAP@10, MAP@100: Mean Average Precision (primary metric)")
        logger.info("  • Primary metric: MAP@100 (higher is better)")
    else:
        logger.info("⚠️  No evaluation data provided - training without evaluation")
    
    # Training arguments - MultipleNegativesRankingLoss is perfect for biomedical ontology pairs
    train_loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=args.scale,  # Higher scale = harder negatives
        similarity_fct=losses.cos_sim  # Cosine similarity for better biomedical embeddings
    )
    
    logger.info(f"Using MultipleNegativesRankingLoss with scale={args.scale}")
    logger.info(f"Batch size: {args.batch_size} (higher batch size = more in-batch negatives = better training)")
    
    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_examples,
        loss=train_loss,
        evaluator=evaluator,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        show_progress_bar=True,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if evaluator else False,
        metric_for_best_model="biomedical-ontology-eval_cosine_map@100" if evaluator else None,
        greater_is_better=True if evaluator else False,  # MAP@100 is better when higher
    )
    
    # Start training
    logger.info("Starting fine-tuning...")
    try:
        trainer.train()
        logger.info("Fine-tuning completed successfully!")
        
        # Save the final model
        model.save(args.output_dir)
        logger.info(f"Model saved to: {args.output_dir}")
        
        # Save training configuration
        config = {
            "base_model": args.model,
            "training_data": args.train_data,
            "evaluation_data": args.eval_data if args.eval_data else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "max_seq_length": args.max_seq_length,
            "device": device,
            "training_examples": len(train_examples),
            "loss_function": "MultipleNegativesRankingLoss",
            "scale": args.scale,
            "similarity_function": "cosine_similarity",
            "evaluation_metrics": [
                "MRR@5", "MRR@10", "NDCG@5", "NDCG@10",
                "Accuracy@1,3,5,10", "Precision@1,3,5,10", "Recall@1,3,5,10",
                "MAP@10", "MAP@100"
            ],
            "primary_metric": "MAP@100"
        }
        
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info("Training configuration saved")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
