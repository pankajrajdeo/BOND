"""
Pydantic models for BOND API requests, responses, and internal data structures.
These models provide automatic validation, strong typing, and self-documenting schemas.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

# =============================================================================
# REQUEST MODELS
# =============================================================================

class QueryItem(BaseModel):
    """Individual query request item"""
    query: str = Field(..., description="The text query to search for")
    field_name: str = Field(
        ...,
        description="Schema field being standardized (e.g., cell_type, tissue)",
    )
    organism: str = Field(..., description="Organism (canonical name)")
    tissue: str = Field(..., description="Tissue/organ context (e.g., lung, brain)")
    # Optional knobs used by /query endpoint
    n_expansions: Optional[int] = Field(None, description="Override default number of expansions")
    topk_final: Optional[int] = Field(None, description="Override default number of final results")
    num_choices: Optional[int] = Field(None, description="Number of top alternatives to include (in addition to chosen)")
    topk_exact: Optional[int] = Field(None, description="Override top-k for exact match retrieval")
    topk_bm25: Optional[int] = Field(None, description="Override top-k for BM25 retrieval")
    topk_dense: Optional[int] = Field(None, description="Override top-k for vector retrieval")
    rrf_k: Optional[float] = Field(None, description="Override RRF k parameter")
    exact_only: Optional[bool] = Field(None, description="If true, use exact channel only")
    graph_depth: Optional[int] = Field(None, description="Graph expansion depth (0 disables)")
    rerank_after_graph: Optional[bool] = Field(None, description="Run second RRF after graph expansion")
    return_trace: Optional[bool] = Field(None, description="Override default trace setting")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class QueryResultItem(BaseModel):
    """Individual result item from a query"""
    id: str = Field(..., description="Unique identifier for the ontology term")
    label: str = Field(..., description="Human-readable label for the term")
    source: str = Field(..., description="Source ontology (e.g., 'cl', 'mondo')")
    iri: Optional[str] = Field(None, description="Full IRI for the ontology term (e.g., PURL)")
    definition: Optional[str] = Field(None, description="Canonical definition text")
    synonyms_exact: Optional[List[str]] = Field(default=None, description="Exact synonyms")
    synonyms_related: Optional[List[str]] = Field(default=None, description="Related synonyms")
    synonyms_broad: Optional[List[str]] = Field(default=None, description="Broad synonyms")
    fusion_score: float = Field(..., description="Reciprocal Rank Fusion score")
    retrieval_confidence: Optional[float] = Field(None, description="Normalized retrieval confidence [0,1]")
    reason: Optional[str] = Field(None, description="LLM's reason for choosing this term")
    llm_confidence: Optional[float] = Field(None, description="LLM's confidence in its choice [0,1]")

class QueryResponse(BaseModel):
    """Deprecated in favor of custom JSON contract returned by server."""
    results: List[QueryResultItem] = Field(default_factory=list)
    chosen: Optional[QueryResultItem] = None
    alternatives: Optional[List[QueryResultItem]] = None
    trace: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BatchResponse(BaseModel):
    """Deprecated in favor of custom JSON contract returned by server."""
    results: List[QueryResponse] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class BondConfig(BaseModel):
    """Deprecated config model; server returns raw cfg now."""
    model_config = ConfigDict(extra="ignore")

class OntologyInfo(BaseModel):
    """Information about available ontologies"""
    sources: List[str] = Field(..., description="List of available ontology sources")
    total_terms: int = Field(..., description="Total number of terms across all ontologies")
    last_updated: Optional[datetime] = Field(None, description="When the index was last built")

# =============================================================================
# INTERNAL DATA MODELS
# =============================================================================

class EmbeddingSignature(BaseModel):
    """Model signature for embedding validation"""
    model_id: str = Field(..., description="Identifier for the embedding model")
    dimension: int = Field(..., description="Embedding vector dimension")
    anchor_text: str = Field(..., description="Reference text used for validation")
    anchor_vector: List[float] = Field(..., description="Reference embedding vector")

class IndexMetadata(BaseModel):
    """Metadata for FAISS index profiles"""
    profile: str = Field(..., description="Index profile name")
    method: str = Field(..., description="Indexing method used")
    embedding_model: str = Field(..., description="Source embedding model")
    normalize: bool = Field(..., description="Whether embeddings are normalized")
    dimension: int = Field(..., description="Embedding dimension")
    notes: str = Field(..., description="Additional implementation notes")
    created_at: Optional[str] = Field(None, description="When the index was created (ISO format)")

# =============================================================================
# LLM STRUCTURED RESPONSES
# =============================================================================

class ExpansionResponse(BaseModel):
    """Expected structure from expansion LLM."""
    expansions: List[str] = Field(default_factory=list)
    context_terms: List[str] = Field(default_factory=list)

class DisambiguationResponse(BaseModel):
    """Expected structure from disambiguation LLM."""
    chosen_id: str | None  # Allow None for abstain capability
    reason: Optional[str] = None
    llm_confidence: Optional[float] = None
    alternatives: Optional[List[str]] = None

class TraceInfo(BaseModel):
    """Detailed trace information for debugging"""
    query: str = Field(..., description="Original query text")
    profile: str = Field(..., description="Index profile used")
    expansions: List[str] = Field(default_factory=list, description="Generated query expansions")
    candidates: Dict[str, List[str]] = Field(..., description="Candidate IDs from each retrieval method")
    fusion: List[tuple] = Field(..., description="Fusion results with scores")
    disambiguation: Dict[str, Any] = Field(..., description="LLM disambiguation details")
    ontology_filter: Optional[List[str]] = Field(None, description="Applied ontology restrictions")

# =============================================================================
# HEALTH AND STATUS MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Current server timestamp")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")

# =============================================================================
# UTILITY MODELS
# =============================================================================

class SearchResult(BaseModel):
    """Generic search result structure"""
    id: str
    label: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = None

class FusionResult(BaseModel):
    """Result from reciprocal rank fusion"""
    id: str
    score: float
    rank: int
