"""
Train cross-encoder reranker for BOND in Google Colab.

Usage:
1. Upload your reranker_training_data folder to Google Drive
2. Run this cell to install dependencies
3. Run the training cell

To use:
    - Upload training data to Drive or use gdown to download
    - Adjust paths in the CONFIG section
    - Run all cells
"""

import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

print("=" * 60)
print("BOND Reranker Training - Google Colab Setup")
print("=" * 60)

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================
print("\n>>> Installing dependencies...")
os.system("pip install -q sentence-transformers datasets transformers torch scikit-learn")

# ============================================================================
# CHECK GPU
# ============================================================================
print("\n>>> Checking GPU availability...")
if torch.cuda.is_available():
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No GPU available - training will be very slow!")

print("\n>>> Setting device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# MOUNT GOOGLE DRIVE (optional, uncomment if using Drive)
# ============================================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================================
# CONFIG - Adjust these paths for your setup
# ============================================================================
CONFIG = {
    # Path to training data (adjust based on your setup)
    'train_path': '/content/reranker_training_data/train.jsonl',  # Or: '/content/drive/MyDrive/BOND/reranker_training_data/train.jsonl'
    'val_path': '/content/reranker_training_data/dev.jsonl',
    
    # Model configuration
    'model_name': 'microsoft/deberta-v3-base',
    'output_path': '/content/reranker_checkpoints/bond-reranker-v1',
    
    # Training hyperparameters
    'epochs': 3,
    'batch_size': 8,  # Reduce if OOM
    'learning_rate': 2e-5,
    'warmup_steps': 1000,
    
    # Optional: limit data for quick testing
    'max_train_examples': None,  # Set to 1000 for quick test
    'max_val_examples': 10000,
}

print("\n" + "=" * 60)
print("CONFIGURATION")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# DOWNLOAD DATA (if using gdown or need to fetch from Drive)
# ============================================================================
print("\n>>> Checking data files...")
train_exists = os.path.exists(CONFIG['train_path'])
val_exists = os.path.exists(CONFIG['val_path'])

if not train_exists or not val_exists:
    print("⚠️  Data files not found at configured paths.")
    print("\nYou can:")
    print("1. Upload reranker_training_data folder to Colab")
    print("2. Use gdown to download from Google Drive")
    print("3. Mount Google Drive and point to your data")
    print("\nFor option 2, run:")
    print("  !gdown --folder <GOOGLE_DRIVE_FOLDER_ID>")
else:
    print("✓ Training data found")
    print("✓ Validation data found")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_reranker_data(jsonl_path: str, max_examples: int = None):
    """Load training data from JSONL file and convert to dataset format."""
    print(f"\nLoading data from {jsonl_path}...")
    examples = []
    
    # Check file size for progress estimation
    total_lines = 0
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                total_lines += 1
    
    with open(jsonl_path) as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="  Reading examples")):
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

# ============================================================================
# DATASET FORMATTING FOR CROSS-ENCODER
# ============================================================================
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

print("\n✓ All imports successful!")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_reranker(config):
    """Main training function."""
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    # Load training data
    train_data = load_reranker_data(config['train_path'], config['max_train_examples'])
    train_dataset = Dataset.from_list(train_data)
    print(f"\n✓ Loaded {len(train_dataset):,} training examples")
    
    # Print label distribution
    from collections import Counter
    label_counts = Counter(row['label'] for row in train_data)
    print(f"  Label distribution: {dict(label_counts)}")
    
    # Load validation data
    val_data = load_reranker_data(config['val_path'], config['max_val_examples'])
    val_dataset = Dataset.from_list(val_data)
    print(f"✓ Loaded {len(val_dataset):,} validation examples")
    
    # Initialize model
    print(f"\n>>> Initializing model: {config['model_name']}")
    model = CrossEncoder(
        config['model_name'],
        num_labels=1,  # Binary classification
        max_length=512,
        device=device
    )
    
    # Initialize loss function with class weighting to handle imbalanced data
    print(">>> Initializing loss function: BinaryCrossEntropyLoss with pos_weight")
    # pos_weight helps with imbalanced classes (93% negative, 7% positive)
    # Weight positive examples by ratio of negatives to positives
    pos_weight = torch.tensor([len(train_data) / sum(1 for r in train_data if r['label'] == 1)])
    loss = BinaryCrossEntropyLoss(model, pos_weight=pos_weight)
    
    # Initialize evaluator
    print(">>> Creating evaluator...")
    val_examples = [(row['sentence1'], row['sentence2'], row['label']) for row in val_data]
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        val_examples,
        name='bond_reranker_val'
    )
    
    # Setup training arguments
    print(">>> Setting up training arguments...")
    training_args = CrossEncoderTrainingArguments(
        output_dir=config['output_path'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'] if 'warmup_steps' in config else 1000,
        evaluation_strategy='steps',
        eval_steps=500,  # Evaluate every 500 steps
        save_strategy='steps',
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='bond_reranker_val_f1',
        save_total_limit=3,  # Keep only last 3 checkpoints
        logging_steps=100,
        log_level='info',
        report_to='none',  # Set to 'wandb' if you have wandb set up
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available()  # Use mixed precision on GPU
    )
    
    # Initialize trainer
    print(">>> Initializing trainer...")
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator
    )
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {config['output_path']}")
    
    # Save final model
    trainer.save_model()
    print("✓ Final model saved")
    
    return trainer

# ============================================================================
# RUN TRAINING
# ============================================================================
if __name__ == '__main__':
    # Check if data exists
    if not os.path.exists(CONFIG['train_path']):
        print(f"\n❌ ERROR: Training data not found at {CONFIG['train_path']}")
        print("\nPlease:")
        print("1. Upload your reranker_training_data folder to Colab")
        print("2. Or mount Google Drive and adjust CONFIG paths")
    else:
        print(f"\n✓ Training data found at {CONFIG['train_path']}")
        trainer = train_reranker(CONFIG)
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Download the trained model from:", CONFIG['output_path'])
        print("2. Use the model in your BOND pipeline")
        print("3. Or continue training with more epochs")
        print("=" * 60)

