# BOND Installation Guide

This guide provides detailed instructions for installing and configuring BOND.

## System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Disk Space**: ~5GB for ontology database and FAISS indices
- **Optional**: GPU for faster embedding inference (CUDA-compatible)

## Step 1: Clone the Repository

```bash
git clone https://github.com/pankajrajdeo/BOND.git
cd BOND
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv bond_venv

# Activate (Linux/macOS)
source bond_venv/bin/activate

# Activate (Windows)
bond_venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install BOND package (editable mode)
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Step 4: Obtain Ontology Database

You need an SQLite database containing ontology terms. You have two options:

### Option A: Use Pre-built Database

If you have access to a pre-built `ontologies.sqlite` file:

```bash
mkdir -p assets
cp /path/to/ontologies.sqlite assets/ontologies.sqlite
```

### Option B: Generate Database from Ontology Files

```bash
# Generate SQLite database from OBO/OWL files
bond-generate-sqlite \
  --input_dir /path/to/ontology/files \
  --output_path assets/ontologies.sqlite
```

The script supports:
- OBO format files (`.obo`)
- OWL format files (`.owl`)
- JSON-LD format

Required ontologies for full functionality:
- Cell Ontology (CL)
- UBERON
- MONDO Disease Ontology
- Experimental Factor Ontology (EFO)
- PATO
- HANCESTRO
- NCBI Taxonomy
- Organism-specific development stage ontologies (HsapDv, MmusDv, etc.)

## Step 5: Build FAISS Index

Build the FAISS index for dense semantic search:

```bash
bond-build-faiss \
  --sqlite_path assets/ontologies.sqlite \
  --assets_path assets \
  --embed_model st:all-MiniLM-L6-v2
```

**Note**: This step requires:
- Embedding model access (see Step 6)
- Several hours for large ontology databases
- Sufficient disk space (~2-5GB)

## Step 6: Configure Environment

Create a `.env` file in the project root:

```bash
# Embedding Model Configuration
# Options:
# - st:all-MiniLM-L6-v2 (Sentence Transformers, default)
# - st:sentence-transformers/all-mpnet-base-v2
# - litellm/http://your-embedding-service

BOND_EMBED_MODEL=st:all-MiniLM-L6-v2

# LLM Providers for Expansion and Disambiguation
# You need at least one configured

# Option 1: Anthropic Claude
BOND_EXPANSION_LLM=anthropic/claude-3-5-sonnet-20241022
BOND_DISAMBIGUATION_LLM=anthropic/claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your-anthropic-api-key

# Option 2: OpenAI GPT
# BOND_EXPANSION_LLM=openai/gpt-4o
# BOND_DISAMBIGUATION_LLM=openai/gpt-4o
# OPENAI_API_KEY=your-openai-api-key

# Option 3: Other LiteLLM-compatible providers
# BOND_EXPANSION_LLM=cohere/command-r-plus
# BOND_DISAMBIGUATION_LLM=cohere/command-r-plus
# COHERE_API_KEY=your-cohere-api-key

# Paths (defaults shown)
BOND_ASSETS_PATH=assets
BOND_SQLITE_PATH=assets/ontologies.sqlite

# Optional: Retrieval-only mode (skip LLM stages)
# BOND_RETRIEVAL_ONLY=1

# Optional: API Authentication
# BOND_API_KEY=your-secret-api-key
# BOND_ALLOW_ANON=1  # Allow anonymous access (development only)
```

## Step 7: Verify Installation

Test the installation:

```bash
# Check CLI works
bond-query --help

# Test query (requires database and FAISS index)
bond-query \
  --query "T-cell" \
  --field cell_type \
  --organism "Homo sapiens" \
  --tissue "blood"

# Test server (if API key is set)
bond-serve
# In another terminal:
curl http://localhost:8000/health
```

## Docker Installation (Alternative)

A Dockerfile is provided for containerized deployment:

```bash
# Build image
docker build -t bond:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/assets:/app/assets \
  -e BOND_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  bond:latest
```

## Troubleshooting

### Issue: "Database not found: assets/ontologies.sqlite"

**Solution**: Ensure the ontology database exists at the specified path:
```bash
ls -lh assets/ontologies.sqlite
```

### Issue: "FAISS index not found"

**Solution**: Build the FAISS index:
```bash
bond-build-faiss --sqlite_path assets/ontologies.sqlite --assets_path assets
```

### Issue: LLM API errors

**Solutions**:
1. Verify API keys are set correctly
2. Check API key permissions (write access required)
3. Ensure sufficient API credits/quota
4. Try a different LLM provider

### Issue: Out of memory during FAISS build

**Solutions**:
1. Build index with smaller batch size
2. Use CPU-only FAISS (faiss-cpu) instead of GPU version
3. Process ontologies in chunks

### Issue: Import errors

**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source bond_venv/bin/activate
pip install -e .
```

## Next Steps

- Read the [README.md](README.md) for usage examples
- Check [API Documentation](docs/API.md) for REST API usage
- Explore [Hybrid Search Guide](README_hybrid_search.md) for advanced features
- Review [Reranker Training Guide](RERANKER_TRAINING_GUIDE.md) for custom model training

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pankajrajdeo/BOND/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pankajrajdeo/BOND/discussions)

## Additional Resources

- **Benchmark Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/pankajrajdeo/bond-czi-benchmark)
- **Paper**: See [paper.md](paper.md) for related multi-agent curation system

## Selecting Your Encoder (HF or Ollama)

You can use your published encoders with BOND.

### Option A: Ollama (local)

1) Pull the model:
```bash
ollama pull rajdeopankaj/bond-embed-v1-fp16
```
2) Set the env var (e.g., in `.env`):
```bash
BOND_EMBED_MODEL=ollama:rajdeopankaj/bond-embed-v1-fp16
# OLLAMA_API_BASE=http://localhost:11434  # if remote, set your host
```
3) Build FAISS:
```bash
bond-build-faiss --sqlite_path assets/ontologies.sqlite --assets_path assets
```

### Option B: Hugging Face TEI (hosted)

1) Deploy `pankajrajdeo/bond-embed-v1-fp16` behind a LiteLLM-compatible endpoint (e.g., TEI + gateway).
2) Set the env var to the routed model name, for example:
```bash
BOND_EMBED_MODEL=litellm:huggingface/teimodel
```
3) Build FAISS as usual.

References:
- HF model: https://huggingface.co/pankajrajdeo/bond-embed-v1-fp16
- Ollama model: https://ollama.com/rajdeopankaj/bond-embed-v1-fp16

