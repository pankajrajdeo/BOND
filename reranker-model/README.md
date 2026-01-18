# Reranker Model

## Download Model

Download the reranker model from Hugging Face:

**Model URL:** https://huggingface.co/rgrupesh/reranker-model

## Installation

```bash
# Install required packages
pip install -U sentence-transformers

# Download the model files from the Hugging Face link above
# Place all model files in this directory (reranker-model/)
```

## Configuration

### Where to Update the Reranker Model Path

After downloading the model, you need to configure the path in BOND:

**Option 1: Environment Variable (Recommended)**
- Edit the `.env` file in the BOND root directory
- Set the `BOND_RERANKER_PATH` variable:
  ```bash
  BOND_RERANKER_PATH=/path/to/BOND/reranker-model
  ```
- Example:
  ```bash
  BOND_RERANKER_PATH=/Users/yourname/Desktop/conf/BOND/reranker-model
  ```

**Option 2: Direct Configuration**
- The reranker path is defined in `bond/config.py` (line 107)
- It reads from the environment variable: `BOND_RERANKER_PATH`
- You can also pass it as a parameter when initializing `BondSettings`

**Option 3: Script Override**
- In ablation scripts (e.g., `benchmark-metrics/20-configs/run_ablation_rerun.py`)
- Override the `reranker_path` in the config overrides dictionary
- Example:
  ```python
  overrides = {
      "reranker_path": "/path/to/BOND/reranker-model",
      # ... other settings
  }
  ```

## Model Details

- **Model Type:** Cross Encoder
- **Base Model:** bioformers/bioformer-16L
- **Maximum Sequence Length:** 512 tokens
- **Training Dataset Size:** 2,401,485 samples
- **Average Precision:** 0.8867

## Required Files

After downloading from Hugging Face, this directory should contain:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer_config.json`
- `vocab.txt`
- Other model-specific files

## Model Performance

- **Accuracy:** 0.975
- **F1 Score:** 0.824
- **Precision:** 0.796
- **Recall:** 0.854

