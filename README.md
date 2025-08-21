# BOND: Biomedical Ontology Normalization and Disambiguation

> **B**iomedical **O**ntology **N**ormalization and **D**isambiguation

BOND is a sophisticated, production-ready entity linking pipeline for biomedical data harmonization. It combines hybrid retrieval (Exact Match, BM25, and FAISS Vector Search) with modern LLM-powered query expansion and context-aware disambiguation to standardize messy metadata against authoritative biomedical ontologies.

## 🚀 Features

- **🔍 Hybrid Retrieval**: Combines lexical (BM25) and semantic (FAISS) search for robust matching
- **⚡ High-Performance Indices**: Prebuilt SQLite database and FAISS indices for 15+ major biomedical ontologies
- **🤖 AI-Powered**: LLM-driven query expansion and disambiguation via LiteLLM
- **🌐 Universal Providers**: Support for OpenAI, Anthropic, Google, Ollama, and custom HTTP endpoints
- **📊 Production Ready**: FastAPI server, CLI tool, and Python library
- **🔒 Secure**: API key authentication and configurable access controls
- **📈 Scalable**: True batch processing and intelligent caching
- **🔄 Cross-Platform**: Runs on Linux, macOS (Intel/Apple Silicon), and Windows

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query Input   │───▶│  Query Expansion│───▶│  Hybrid Search  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐            │
                       │   LLM Context   │◀───────────┘
                       │   Analysis      │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Final Output   │
                       │  (Structured)   │
                       └─────────────────┘
```

## 📦 Included Ontologies

The prebuilt assets include indices for these major biomedical ontologies:

- **Cell Ontology (CL)** - Cell types and cellular components
- **Experimental Factor Ontology (EFO)** - Experimental factors and phenotypes
- **Foundational Model of Anatomy (FMA)** - Human anatomy
- **Human Ancestry Ontology (HANCESTRO)** - Population and ancestry terms
- **Human Phenotype Ontology (HPO)** - Human phenotypic abnormalities
- **Human Developmental Stages (HsapDv)** - Human development stages
- **Medical Action Ontology (MAXO)** - Medical procedures and treatments
- **Mouse Developmental Stages (MmusDv)** - Mouse development stages
- **Mondo Disease Ontology (MONDO)** - Diseases and disorders
- **Mouse Pathology Ontology (MPATH)** - Mouse pathology
- **NCBI Taxonomy (NCBITaxon)** - Biological taxonomy
- **Ontology for Biomedical Investigations (OBI)** - Biomedical investigations
- **Ontology for General Medical Science (OGMS)** - General medical concepts
- **Phenotype and Trait Ontology (PATO)** - Qualities and attributes
- **Protein Ontology (PR)** - Proteins and protein families
- **RadLex** - Radiology lexicon
- **Sequence Ontology (SO)** - Sequence features and types
- **Uberon** - Multi-species anatomy ontology

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9+ (3.11+ recommended for best compatibility)
- **Memory**: 4GB+ RAM (8GB+ recommended for large datasets)
- **Storage**: 2GB+ free space for assets and working directory
- **OS**: Linux, macOS (Intel/Apple Silicon), or Windows

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd BOND

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the BOND package in development mode
pip install -e .
```

**macOS Users**: Install OpenMP support if needed:
```bash
brew install libomp
```

### 3. 🗂️ Extract Prebuilt Assets

**IMPORTANT**: The project includes a `assets.zip` file (~1.2GB) containing prebuilt SQLite database and FAISS indices for all supported ontologies. You must extract this before using BOND.

```bash
# Extract the prebuilt assets
unzip assets.zip

# This creates the assets/ directory with:
# - assets/ontology.sqlite          (SQLite database with all ontology terms)
# - assets/faiss_store/             (FAISS vector indices)
#   ├── embeddings.faiss            (Vector index)
#   ├── id_map.npy                  (ID mappings)
#   ├── rescore_vectors.npy         (Rescoring vectors)
#   └── embedding_signature.json    (Embedding metadata)
```

The extracted assets directory structure:
```
assets/
├── ontology.sqlite                 # SQLite database with all terms (~1.6GB)
├── faiss_store/
│   ├── embeddings.faiss           # FAISS vector index (~150MB)
│   ├── id_map.npy                 # Term ID mappings (~756MB)
│   ├── rescore_vectors.npy        # High-precision vectors (~1.3GB)
│   └── embedding_signature.json   # Embedding model metadata
└── last_update.txt                # Build timestamp
```

### 4. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit with your settings
nano .env  # or use your preferred editor
```

**Essential Environment Variables**:
```env
# Core Settings
BOND_ASSETS_PATH=assets

# LLM Configuration (required for query expansion and disambiguation)
BOND_EXPANSION_LLM=openai/gpt-4o-mini
BOND_DISAMBIGUATION_LLM=openai/gpt-4o-mini

# Embedding Model (prebuilt indices use bond-embed-v1-fp16)
BOND_EMBED_MODEL=ollama/rajdeopankaj/bond-embed-v1-fp16:latest

# API Keys (required for cloud providers)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Local Ollama (if using local models)
OLLAMA_API_BASE=http://localhost:11434

# Optional: Adjust retrieval parameters
BOND_TOPK_FINAL=10
BOND_TOPK_DENSE=50
BOND_TOPK_BM25=20
```

### 5. Test the System

```bash
# Basic query
bond-query --query "T-cell" --field_name "cell type"

# Verbose output (shows retrieval process)
bond-query --query "T-cell" --field_name "cell type" --verbose

# Query with context
bond-query --query "cancer" --field_name "disease" \
  --dataset_description "single-cell RNA-seq data from tumor samples"
```

### 6. Start the API Server

```bash
# Start server (default: http://localhost:8000)
bond-serve

# In another terminal, test the API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "T-cell", "field_name": "cell type"}'
```

## 🔧 Configuration

### Core Environment Variables

```env
# --- Core Paths ---
BOND_ASSETS_PATH=assets                    # Path to extracted assets directory

# --- LLM Configuration ---
BOND_EXPANSION_LLM=openai/gpt-4o-mini     # Model for query expansion
BOND_DISAMBIGUATION_LLM=openai/gpt-4o-mini # Model for final disambiguation

# --- Embeddings ---
BOND_EMBED_MODEL=ollama/rajdeopankaj/bond-embed-v1-fp16:latest
BOND_EMB_BATCH=16                          # Embedding batch size

# --- Retrieval Parameters ---
BOND_TOPK_EXACT=5                          # Top-k exact matches
BOND_TOPK_BM25=20                          # Top-k BM25 results
BOND_TOPK_DENSE=50                         # Top-k vector search results
BOND_TOPK_FINAL=10                         # Final top-k after fusion
BOND_RRF_K=60.0                           # Reciprocal Rank Fusion parameter

# --- Query Expansion ---
BOND_EXPANSION=1                           # Enable query expansion
BOND_N_EXPANSIONS=3                        # Number of expansions to generate

# --- API Security ---
BOND_API_KEY=                              # API key for authentication
BOND_ALLOW_ANON=1                         # Allow anonymous access (dev only)

# --- Provider API Keys ---
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_API_BASE=http://localhost:11434
```

### LLM Provider Configuration

#### OpenAI
```env
BOND_EXPANSION_LLM=openai/gpt-4o-mini
BOND_DISAMBIGUATION_LLM=openai/gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
```

#### Anthropic
```env
BOND_EXPANSION_LLM=anthropic/claude-3-5-sonnet-20241022
BOND_DISAMBIGUATION_LLM=anthropic/claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

#### Local Ollama
```env
BOND_EXPANSION_LLM=ollama/llama3.2
BOND_DISAMBIGUATION_LLM=ollama/llama3.2
OLLAMA_API_BASE=http://localhost:11434
```

## 📚 Usage

### Command Line Interface

```bash
# Basic query
bond-query --query "T-cell" --field_name "cell type"

# Advanced query with context
bond-query \
  --query "cancer" \
  --field_name "disease" \
  --dataset_description "single-cell RNA-seq data from tumor samples"

# Verbose output (shows all retrieved results and trace)
bond-query --query "T-cell" --field_name "cell type" --verbose

# Restrict to specific ontologies
bond-query --query "T-cell" --field_name "cell type" \
  --restrict_to_ontologies cl mondo

# Return detailed trace information
bond-query --query "T-cell" --field_name "cell type" --return_trace

# Override retrieval parameters
bond-query --query "T-cell" --field_name "cell type" \
  --topk_final 5 --topk_dense 25
```

### Python API

```python
from bond.config import BondSettings
from bond.pipeline import BondMatcher

# Initialize with default settings
matcher = BondMatcher()

# Single query
result = matcher.query(
    query="T-cell",
    field_name="cell type",
    dataset_description="single-cell RNA-seq data"
)

print(f"Chosen term: {result['chosen']['label']}")
print(f"Ontology ID: {result['chosen']['id']}")
print(f"Source: {result['chosen']['source']}")
print(f"Confidence: {result['chosen']['retrieval_confidence']}")

# Batch processing
queries = [
    {"query": "T-cell", "field_name": "cell type"},
    {"query": "cancer", "field_name": "disease"},
    {"query": "protein", "field_name": "molecule"}
]

results = matcher.batch_query(queries)

# Context manager (automatic cleanup)
with BondMatcher() as matcher:
    results = matcher.batch_query(queries)
    # Resources automatically cleaned up
```

### HTTP API

#### Start Server
```bash
# Start server with API key authentication
export BOND_API_KEY=$(openssl rand -hex 32)
bond-serve

# Or for development without authentication
export BOND_ALLOW_ANON=1
bond-serve
```

#### API Endpoints

**Single Query**:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": "T-cell",
    "field_name": "cell type",
    "dataset_description": "single-cell RNA-seq data"
  }'
```

**Batch Query**:
```bash
curl -X POST "http://localhost:8000/batch_query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "items": [
      {"query": "T-cell", "field_name": "cell type"},
      {"query": "cancer", "field_name": "disease"}
    ]
  }'
```

**Available Ontologies**:
```bash
curl "http://localhost:8000/ontologies"
```

**Health Check**:
```bash
curl "http://localhost:8000/health"
```

## 🔍 Understanding Output

### Query Result Structure

```json
{
  "id": "CL:0000623",
  "label": "natural killer cell",
  "definition": "A lymphocyte that can spontaneously kill a variety of target cells...",
  "source": "cl",
  "iri": "http://purl.obolibrary.org/obo/CL_0000623",
  "synonyms_exact": ["NK cell", "NK-cell"],
  "synonyms_related": ["natural killer lymphocyte"],
  "synonyms_broad": ["lymphocyte", "immune cell"],
  "synonyms_generic": ["cell"],
  "alt_ids": [],
  "xrefs": ["BTO:0000914", "FMA:84370"],
  "namespace": "cell",
  "subsets": ["cumulus_restricted"],
  "comments": [],
  "parents_is_a": ["CL:0000542"],
  "abstracts": [],
  "retrieval_confidence": 0.85,
  "llm_confidence": 0.92,
  "reason": "This is the most appropriate parent term for the query 'T-cell'..."
}
```

### Confidence Scores

- **`retrieval_confidence`**: Technical score from the hybrid retrieval pipeline (0-1)
- **`llm_confidence`**: LLM's confidence in the chosen term (0-1, if LLM enabled)

### Field Descriptions

- **`id`**: Ontology identifier (e.g., `CL:0000623`)
- **`label`**: Human-readable term name
- **`source`**: Ontology source (e.g., `cl`, `mondo`, `hpo`)
- **`iri`**: Full IRI for the term
- **`definition`**: Canonical definition from the ontology
- **`synonyms_*`**: Categorized synonyms (exact, related, broad, generic)
- **`xrefs`**: Cross-references to other ontologies
- **`parents_is_a`**: Direct parent terms

## 🏗️ Building Custom Indices (Advanced)

If you want to build indices from your own ontology files instead of using the prebuilt ones:

### 1. Prepare Ontology Files

```bash
# Create data directory and add your .owl files
mkdir -p data
# Copy your ontology files (OWL or OBO format) to data/
cp /path/to/your/ontologies/*.owl data/
```

### 2. Build Indices

```bash
# Remove existing assets (if you want to rebuild)
rm -rf assets/

# Build new indices
bond-build-index \
  --owl ./data/*.owl \
  --embed_spec "st:all-MiniLM-L6-v2" \
  --assets_path assets
```

**Note**: Building indices from scratch can take several hours depending on the size of your ontologies and your hardware.

## 🐛 Troubleshooting

### Common Issues

#### Missing Assets
```bash
# Error: Database not found: assets/ontology.sqlite
# Solution: Extract the assets.zip file
unzip assets.zip
```

#### OpenMP Runtime Conflicts (macOS)
```bash
# Error: OMP: Error #15: Initializing libomp.dylib
# Solution: Install OpenMP support
brew install libomp
```

#### FAISS Index Errors
```bash
# Error: Could not load FAISS index
# Solution: Ensure assets are properly extracted
unzip assets.zip
```

#### API Authentication Errors
```bash
# Error: BOND_API_KEY not set
# Solution: Set API key or allow anonymous access
export BOND_API_KEY=$(openssl rand -hex 32)
# or for development:
export BOND_ALLOW_ANON=1
```

#### LLM Provider Errors
```bash
# Error: Authentication failed
# Solution: Check your API keys
export OPENAI_API_KEY=sk-your-actual-key
```

### Performance Tuning

#### Memory Optimization
```bash
# Reduce batch sizes for memory-constrained environments
export BOND_EMB_BATCH=8
export BOND_TOPK_DENSE=25
```

#### Speed Optimization
```bash
# Reduce top-k values for faster responses
export BOND_TOPK_FINAL=5
export BOND_TOPK_BM25=10
```

## 🔄 Docker Deployment

```bash
# Build Docker image
docker build -t bond-pipeline .

# Run with assets mounted
docker run -d \
  --name bond-api \
  -p 8000:8000 \
  -v $(pwd)/assets:/app/assets \
  -e OPENAI_API_KEY=sk-your-key \
  -e BOND_ALLOW_ANON=1 \
  bond-pipeline
```

## 📖 Command Line Reference

### `bond-query`
```bash
bond-query [OPTIONS] --query TEXT --field_name TEXT

Options:
  --query TEXT                    Query text to harmonize [required]
  --field_name TEXT              Field name for context [required]
  --dataset_description TEXT     Dataset description for context
  --restrict_to_ontologies TEXT... Limit to specific ontologies (e.g., cl mondo)
  --topk_exact INT               Top-k exact matches [default: 5]
  --topk_bm25 INT                Top-k BM25 results [default: 20]
  --topk_dense INT               Top-k vector results [default: 50]
  --topk_final INT               Final top-k results [default: 10]
  --rrf_k FLOAT                  RRF fusion parameter [default: 60.0]
  --verbose                      Show detailed retrieval process
  --return_trace                 Include trace information
  --help                         Show help message
```

### `bond-serve`
```bash
bond-serve [OPTIONS]

Options:
  --host TEXT                   Host to bind to [default: 0.0.0.0]
  --port INTEGER               Port to bind to [default: 8000]
  --help                       Show help message
```

### `bond-build-index`
```bash
bond-build-index [OPTIONS] --owl PATH...

Options:
  --owl PATH...                OWL/OBO files to process [required]
  --embed_spec TEXT           Embedding model specification
  --assets_path PATH          Output directory [default: assets]
  --help                      Show help message
```

## 🤝 Contributing

We welcome contributions! To get started:

```bash
# Clone and setup development environment
git clone <repository-url>
cd BOND
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Extract assets for testing
unzip assets.zip

# Run tests
python -m pytest tests/  # if tests exist

# Code formatting
black bond/
ruff check bond/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FAISS**: Facebook AI Similarity Search for vector indexing
- **LiteLLM**: Universal LLM interface for provider abstraction
- **Pronto**: OWL/OBO ontology parsing
- **FastAPI**: Modern web framework for API development
- **SentenceTransformers**: Local embedding model support

## 📞 Support

- **Issues**: Report bugs and request features through GitHub issues
- **Documentation**: Check the inline help with `--help` flags
- **Community**: Join discussions about biomedical ontology harmonization

## 🔄 Version History

### v0.2.0 (Current)
- Prebuilt assets with 15+ major biomedical ontologies
- Improved hybrid retrieval pipeline
- Enhanced LLM integration
- Production-ready API server
- Comprehensive CLI tools

---

**Ready to harmonize your biomedical data? Extract the assets and start querying!** 🚀