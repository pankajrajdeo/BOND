import os
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass

def _parse_ontologies(env_var: str) -> Optional[List[str]]:
    """Parse comma-separated ontology list from environment variable"""
    val = os.getenv(env_var)
    if not val:
        return None
    return [item.strip() for item in val.split(',') if item.strip()]

class BondSettings(BaseModel):
    # Core paths (new schema only)
    assets_path: str = Field(default=os.getenv("BOND_ASSETS_PATH", "assets"))
    # Always derive sqlite path from assets path; new schema filename only
    sqlite_path: str = Field(default="assets/ontologies.sqlite")

    # Embeddings
    # Accept new env var BOND_EMBED_MODEL; fallback to legacy BOND_EMBED_SPEC for BC
    embed_model: str = Field(default=(os.getenv("BOND_EMBED_MODEL") or os.getenv("BOND_EMBED_SPEC") or "st:all-MiniLM-L6-v2"))
    litellm_api_base: str = Field(default=os.getenv("LITELLM_API_BASE", ""))

    # Separate LLMs for expansion and disambiguation (required)
    expansion_llm_model: str = Field(default=(os.getenv("BOND_EXPANSION_LLM") or os.getenv("BOND_EXPANSION_LLM_MODEL") or ""))
    disambiguation_llm_model: str = Field(default=(os.getenv("BOND_DISAMBIGUATION_LLM") or os.getenv("BOND_DISAMBIGUATION_LLM_MODEL") or ""))

    # Retrieval parameters
    topk_exact: int = Field(default=int(os.getenv("BOND_TOPK_EXACT", 5)))
    topk_bm25: int = Field(default=int(os.getenv("BOND_TOPK_BM25", 20)))
    topk_dense: int = Field(default=int(os.getenv("BOND_TOPK_DENSE", 50)))
    # topk_final: Drives both the fusion list shown in trace and the set of
    # candidates provided to the disambiguation LLM.
    topk_final: int = Field(default=int(os.getenv("BOND_TOPK_FINAL", 20)))
    rrf_k: float = Field(default=float(os.getenv("BOND_RRF_K", 60.0)))
    use_rrf: bool = Field(default=True, description="Use RRF fusion. If False, use simple concatenation with priority ordering.")

    # RRF source weighting (JSON string preferred; falls back to per-weight envs for BC)
    rrf_weights: Dict[str, float] = Field(
        default_factory=lambda: (
            (lambda s: (json.loads(s) if s else {
                "exact": float(os.getenv("BOND_RRF_EXACT_WEIGHT", 1.0)),
                "bm25": float(os.getenv("BOND_RRF_BM25_WEIGHT", 0.8)),
                "dense": float(os.getenv("BOND_RRF_DENSE_WEIGHT", 0.6)),
                # Context-augmented channels
                "bm25_ctx": float(os.getenv("BOND_RRF_BM25_CTX_WEIGHT", 0.72)),  # ~bm25*0.9 by default
                # New: intent-based dense channel (replaces dense_ctx)
                "dense_full": float(os.getenv("BOND_RRF_DENSE_FULL_WEIGHT", os.getenv("BOND_RRF_DENSE_CTX_WEIGHT", 0.6))),
            })) (os.getenv("BOND_RRF_WEIGHTS"))
        )
    )

    rescore_multiplier: int = Field(default=int(os.getenv("BOND_RESCORE_MULTIPLIER", 20)))
    enable_expansion: bool = Field(default=bool(int(os.getenv("BOND_EXPANSION", 1))))
    n_expansions: int = Field(default=int(os.getenv("BOND_N_EXPANSIONS", 3)))
    log_level: str = Field(default=os.getenv("BOND_LOG_LEVEL", "INFO"))
    use_gpu: bool = Field(default=bool(int(os.getenv("BOND_USE_GPU", 1))))
    restrict_to_ontologies: Optional[List[str]] = Field(
        default_factory=lambda: _parse_ontologies("BOND_RESTRICT_TO_ONTOLOGIES"),
        description="Default list of ontology sources to restrict search to."
    )
    # New schema tables only
    table_terms: str = "ontology_terms"
    table_terms_fts: str = "ontology_terms_fts"

    # Re-ranking boosts
    exact_match_boost: float = Field(default=float(os.getenv("BOND_EXACT_MATCH_BOOST", 10.0)))
    label_exact_match_boost: float = Field(default=float(os.getenv("BOND_LABEL_EXACT_MATCH_BOOST", 50.0)))
    ontology_prior_boost: float = Field(default=float(os.getenv("BOND_ONTOLOGY_PRIOR_BOOST", 0.5)))
    context_overlap_boost: float = Field(default=float(os.getenv("BOND_CONTEXT_OVERLAP_BOOST", 0.2)))

    # Graph expansion (auto by default)
    graph_mode: str = Field(default=os.getenv("BOND_GRAPH_MODE", "auto"))  # one of: off|auto|on
    graph_auto_fields: List[str] = Field(
        default_factory=lambda: [s.strip() for s in (os.getenv("BOND_GRAPH_AUTO_FIELDS", "tissue,development_stage,cell_type").split(",")) if s.strip()]
    )
    graph_auto_depth: int = Field(default=int(os.getenv("BOND_GRAPH_DEPTH_AUTO", 1)))
    graph_weight: float = Field(default=float(os.getenv("BOND_GRAPH_WEIGHT", 0.7)))
    graph_top1_min: float = Field(default=float(os.getenv("BOND_GRAPH_TOP1_MIN", 0.35)))
    graph_top3_mean_min: float = Field(default=float(os.getenv("BOND_GRAPH_TOP3_MEAN_MIN", 0.25)))

    # Post-fusion lexical boost over term_doc using query+expansions+context tokens
    term_doc_overlap_boost: float = Field(default=float(os.getenv("BOND_TERM_DOC_OVERLAP_BOOST", 0.15)))

    # Retrieval-only mode (skip LLM expansion/disambiguation)
    retrieval_only: bool = Field(default=bool(int(os.getenv("BOND_RETRIEVAL_ONLY", 0))))

    # LLM/Embedding provider resilience knobs
    llm_timeout_seconds: int = Field(default=int(os.getenv("BOND_LLM_TIMEOUT", 30)))
    llm_max_retries: int = Field(default=int(os.getenv("BOND_LLM_RETRIES", 3)))
    embed_timeout_seconds: int = Field(default=int(os.getenv("BOND_EMBED_TIMEOUT", 30)))
    embed_max_retries: int = Field(default=int(os.getenv("BOND_EMBED_RETRIES", 3)))

    # Semantic intent rerank using FAISS rescore vectors
    enable_semantic_intent_rerank: bool = Field(default=bool(int(os.getenv("BOND_SEMANTIC_INTENT_RERANK", 1))))
    semantic_intent_weight: float = Field(default=float(os.getenv("BOND_SEMANTIC_INTENT_WEIGHT", 0.5)))

    # Cross-encoder reranker
    reranker_path: Optional[str] = Field(default=os.getenv("BOND_RERANKER_PATH"))
    
    def model_post_init(self, __context: Dict) -> None:  # type: ignore[override]
        # Derive sqlite path from assets_path always (new schema filename)
        self.sqlite_path = os.path.join(self.assets_path, "ontologies.sqlite")
