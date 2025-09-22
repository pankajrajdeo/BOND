# BOND Hybrid Search - Usage Guide

This script provides a simplified interface to BOND's hybrid search capabilities, allowing you to perform exact, BM25, and vector search with ontology namespace filtering.

## Features

- **Exact Search**: Phrase-based exact matching against labels and synonyms
- **BM25 Search**: Keyword-based ranked search using BM25 algorithm  
- **Vector Search**: Semantic similarity search (simplified implementation)
- **Ontology Filtering**: Filter results by specific ontology namespaces using CURIE prefixes
- **Hybrid Fusion**: Combines all search methods using Reciprocal Rank Fusion (RRF)
- **Parallel Processing**: Runs searches concurrently for better performance

## Installation

Make sure you have the BOND environment set up and the virtual environment activated:

```bash
source bond_venv/bin/activate
```

## Usage Examples

### List Available Ontologies
```bash
python hybrid_search.py --list-ontologies
```

### Basic Search
```bash
# Search for "T cell" in CL ontology
python hybrid_search.py --query "T cell" --ontology cl

# Search for "lung" in UBERON ontology  
python hybrid_search.py --query "lung" --ontology uberon
```

### Multiple Ontologies
```bash
# Search across multiple ontologies
python hybrid_search.py --query "cancer" --ontologies cl,uberon,mondo
```

### Custom Top-K Parameters
```bash
# Adjust the number of results from each search method
python hybrid_search.py --query "T cell" --ontology cl --top-k-exact 5 --top-k-bm25 10 --top-k-vector 15
```

### Output Formats

**JSON Output (default):**
```bash
python hybrid_search.py --query "T cell" --ontology cl
```

**Table Output:**
```bash
python hybrid_search.py --query "T cell" --ontology cl --output-format table
```

## Output Structure

The script returns results in the following structure:

```json
{
  "query": "search query",
  "ontology_filter": ["cl"],
  "search_time_ms": 99.15,
  "results": {
    "exact": [...],      // Exact match results
    "bm25": [...],       // BM25 search results  
    "vector": [...],      // Vector search results
    "fused": [...]       // Combined results using RRF
  },
  "summary": {
    "exact_count": 1,
    "bm25_count": 3, 
    "vector_count": 0,
    "fused_count": 4
  }
}
```

Each result includes:
- `id`: CURIE identifier (e.g., "CL:0000084")
- `label`: Human-readable term name
- `definition`: Term definition (when available)
- `ontology_id`: Source ontology namespace
- `iri`: Full IRI for the term
- `search_type`: Which search method found this result
- `fusion_score`: RRF score for fused results

## Available Ontologies

- `cl`: Cell Ontology
- `uberon`: Uber Anatomy Ontology  
- `mondo`: MonDO Disease Ontology
- `pato`: Phenotype And Trait Ontology
- `efo`: Experimental Factor Ontology
- `ncbitaxon`: NCBI Taxonomy
- `fbbt`: Drosophila anatomy
- `zfa`: Zebrafish anatomy
- `wbbt`: C. elegans anatomy
- `hsapdv`: Human development stages
- `mmusdv`: Mouse development stages
- `fbdv`: Drosophila development stages
- `wbls`: C. elegans life stages
- `hancestro`: Human ancestry ontology

## Notes

- The script uses CURIE prefix filtering for accurate ontology filtering
- Vector search is currently simplified (random sampling) - in production you'd embed the query and search FAISS directly
- All searches run in parallel for optimal performance
- Results are combined using Reciprocal Rank Fusion with configurable weights
