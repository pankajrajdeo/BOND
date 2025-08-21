import os
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

def _parse_ontologies(env_var: str) -> Optional[List[str]]:
    """Parse comma-separated ontology list from environment variable"""
    val = os.getenv(env_var)
    if not val:
        return None
    return [item.strip() for item in val.split(',') if item.strip()]

class BondSettings(BaseModel):
    # Core paths
    assets_path: str = Field(default=os.getenv("BOND_ASSETS_PATH", "assets"))
    # Always derive sqlite path from assets path; no env override necessary
    sqlite_path: str = Field(default="assets/ontology.sqlite")

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
    topk_final: int = Field(default=int(os.getenv("BOND_TOPK_FINAL", 10)))
    rrf_k: float = Field(default=float(os.getenv("BOND_RRF_K", 60.0)))

    # RRF source weighting (JSON string preferred; falls back to per-weight envs for BC)
    rrf_weights: Dict[str, float] = Field(
        default_factory=lambda: (
            (lambda s: (json.loads(s) if s else {
                "exact": float(os.getenv("BOND_RRF_EXACT_WEIGHT", 1.0)),
                "bm25": float(os.getenv("BOND_RRF_BM25_WEIGHT", 0.8)),
                "dense": float(os.getenv("BOND_RRF_DENSE_WEIGHT", 0.6)),
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
    table_terms: str = "terms"
    table_terms_fts: str = "terms_fts"

    # Context-aware retrieval tuning
    context_keywords: Optional[List[str]] = Field(
        default_factory=lambda: _parse_ontologies("BOND_CONTEXT_KEYWORDS"),
        description="Preseeded context keywords (comma-separated). If empty, derived from dataset_description heuristics."
    )
    context_boost: float = Field(default=float(os.getenv("BOND_CONTEXT_BOOST", 0.15)))

    def model_post_init(self, __context: Dict) -> None:  # type: ignore[override]
        # Derive sqlite path from assets_path always
        self.sqlite_path = os.path.join(self.assets_path, "ontology.sqlite")
