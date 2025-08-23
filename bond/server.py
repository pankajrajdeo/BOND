from fastapi import FastAPI, HTTPException, Depends, Header
from typing import Optional, List, Any, Dict
import os
import threading
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass
from .runtime_env import configure_runtime
from .config import BondSettings
from .pipeline import BondMatcher
from .models import (
    QueryItem, BatchRequest, ErrorResponse
)
from .logger import logger

configure_runtime()
app = FastAPI(title="BOND API", version="0.1.0")
settings = BondSettings()
_matcher_lock = threading.Lock()
matcher = None  # Lazy initialization

def verify_api_key(authorization: str = Header(None)):
    """Basic API key authentication via static token.
    - Set BOND_API_KEY to any secret string to require clients to send 'Authorization: Bearer <BOND_API_KEY>'.
    - For local/dev without auth, set BOND_ALLOW_ANON=1.
    """
    api_key = os.getenv("BOND_API_KEY")
    if not api_key:
        if os.getenv("BOND_ALLOW_ANON", "0").lower() in ("1", "true", "yes"):
            return True
        raise HTTPException(
            status_code=401,
            detail="BOND_API_KEY not set. Set BOND_API_KEY or BOND_ALLOW_ANON=1 for local development."
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use 'Authorization: Bearer <BOND_API_KEY>'")
    if authorization[len("Bearer "):].strip() != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key. Check your BOND_API_KEY env variable")
    return True

def get_matcher():
    """Lazy initialization of the matcher to prevent crashes on missing assets"""
    global matcher
    if matcher is None:
        with _matcher_lock:
            if matcher is None:
                try:
                    matcher = BondMatcher(settings)
                except Exception as e:
                    raise HTTPException(
                        status_code=503, 
                        detail=f"BOND matcher initialization failed: {str(e)}. Please ensure assets are built with 'bond-build-index'."
                    )
    return matcher

# Models are now imported from bond.models

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "BOND API",
        "version": "0.2.0"
    }

@app.post("/query")
def query(item: QueryItem, auth: bool = Depends(verify_api_key), verbose: bool = False) -> Dict[str, Any]:
    try:
        matcher = get_matcher()
        result = matcher.query(
            item.query,
            field_name=item.field_name,
            dataset_description=item.dataset_description,
            n_expansions=item.n_expansions,
            topk_final=item.topk_final,
            num_choices=item.num_choices,
            topk_exact=item.topk_exact,
            topk_bm25=item.topk_bm25,
            topk_dense=item.topk_dense,
            rrf_k=item.rrf_k,
            return_trace=item.return_trace or verbose,
            restrict_to_ontologies=item.restrict_to_ontologies,
        )
        # Standardized JSON output
        def _chosen_clean(ch):
            if not ch:
                return None
            return {
                "id": ch.get("id"),
                "label": ch.get("label"),
                "definition": ch.get("definition"),
                "source": ch.get("source"),
                "iri": ch.get("iri"),
                "synonyms_exact": ch.get("synonyms_exact"),
                "synonyms_related": ch.get("synonyms_related"),
                "synonyms_broad": ch.get("synonyms_broad"),
                "synonyms_generic": ch.get("synonyms_generic"),
                "alt_ids": ch.get("alt_ids"),
                "xrefs": ch.get("xrefs"),
                "namespace": ch.get("namespace"),
                "subsets": ch.get("subsets"),
                "comments": ch.get("comments"),
                "parents_is_a": ch.get("parents_is_a"),
                "abstracts": ch.get("abstracts"),
            }
        chosen = _chosen_clean(result.get("chosen"))
        if item.return_trace or verbose:
            return {
                "expansions": result.get("trace", {}).get("expansions", []),
                "context_terms": result.get("trace", {}).get("context_terms", []),
                "fusion_top_k": result.get("trace", {}).get("fusion", []),
                "trace": result.get("trace", {}),
                "results": result.get("results", []),
                "disambiguation": {
                    **(chosen or {}),
                    "reason": (result.get("chosen") or {}).get("reason"),
                    "retrieval_confidence": (result.get("chosen") or {}).get("retrieval_confidence"),
                    "llm_confidence": (result.get("chosen") or {}).get("llm_confidence"),
                },
                "alternatives": result.get("alternatives", []),
            }
        else:
            payload = {
                **(chosen or {}),
                "reason": (result.get("chosen") or {}).get("reason"),
                "retrieval_confidence": (result.get("chosen") or {}).get("retrieval_confidence"),
                "llm_confidence": (result.get("chosen") or {}).get("llm_confidence"),
            }
            if item.num_choices and item.num_choices > 0:
                payload["alternatives"] = result.get("alternatives", [])
            return payload
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=ErrorResponse(
                error="Query processing failed",
                detail=str(e),
                timestamp=datetime.now()
            ).model_dump()
        )

@app.post("/batch_query")
def batch_query(req: BatchRequest, auth: bool = Depends(verify_api_key)) -> List[Dict[str, Any]]:
    try:
        matcher = get_matcher()
        results = matcher.batch_query([item.model_dump() for item in req.items])
        # Return list of standardized outputs
        payloads = []
        for res in results:
            def _chosen_clean(ch):
                if not ch:
                    return None
                return {
                    "id": ch.get("id"),
                    "label": ch.get("label"),
                    "definition": ch.get("definition"),
                    "source": ch.get("source"),
                    "iri": ch.get("iri"),
                    "synonyms_exact": ch.get("synonyms_exact"),
                    "synonyms_related": ch.get("synonyms_related"),
                    "synonyms_broad": ch.get("synonyms_broad"),
                    "synonyms_generic": ch.get("synonyms_generic"),
                    "alt_ids": ch.get("alt_ids"),
                    "xrefs": ch.get("xrefs"),
                    "namespace": ch.get("namespace"),
                    "subsets": ch.get("subsets"),
                    "comments": ch.get("comments"),
                    "parents_is_a": ch.get("parents_is_a"),
                    "abstracts": ch.get("abstracts"),
                }
            chosen = _chosen_clean(res.get("chosen"))
            payloads.append({
                "expansions": res.get("trace", {}).get("expansions", []),
                "context_terms": res.get("trace", {}).get("context_terms", []),
                "fusion_top_k": res.get("trace", {}).get("fusion", []),
                "trace": res.get("trace", {}),
                "results": res.get("results", []),
                "disambiguation": {
                    **(chosen or {}),
                    "reason": (res.get("chosen") or {}).get("reason"),
                    "retrieval_confidence": (res.get("chosen") or {}).get("retrieval_confidence"),
                    "llm_confidence": (res.get("chosen") or {}).get("llm_confidence"),
                },
                "alternatives": res.get("alternatives", []),
            })
        return payloads
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=ErrorResponse(
                error="Batch query processing failed",
                detail=str(e),
                timestamp=datetime.now()
            ).model_dump()
        )

@app.get("/config")
def get_config(auth: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    try:
        matcher = get_matcher()
        config_dict = matcher.cfg.model_dump()
        return config_dict
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/ontologies")
def get_ontologies(auth: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """Returns information about available ontologies in the index."""
    try:
        matcher = get_matcher()
        sources = matcher.get_available_ontologies()
        
        # Get total term count
        cur = matcher.get_connection().cursor()
        cur.execute(f"SELECT COUNT(*) FROM {matcher.cfg.table_terms}")
        total_terms = cur.fetchone()[0]
        
        return {"sources": sources, "total_terms": total_terms, "last_updated": None}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/cache/stats")
def get_cache_stats(auth: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """Cache removed; return zeros"""
    return {"cache_size": 0, "cache_hits": 0, "cache_misses": 0, "hit_rate": 0.0}

@app.post("/cache/clear")
def clear_cache(auth: bool = Depends(verify_api_key)) -> Dict[str, str]:
    """Cache removed"""
    return {"message": "Cache disabled"}

@app.get("/gpu-status")
def get_gpu_status(auth: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """GPU not applicable for binary FAISS"""
    try:
        matcher = get_matcher()
        return matcher.faiss.get_gpu_status() if hasattr(matcher, 'faiss') else {"error": "FAISS store not available"}
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        return {"error": str(e)}

@app.get("/metrics")
def get_metrics(auth: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """Get basic metrics for monitoring"""
    try:
        matcher = get_matcher()
        return {
            "cache": {"cache_size": 0, "cache_hits": 0, "cache_misses": 0, "hit_rate": 0.0},
            "gpu": matcher.faiss.get_gpu_status() if hasattr(matcher, 'faiss') else {},
            "uptime_seconds": (datetime.now() - matcher._start_time).total_seconds() if hasattr(matcher, '_start_time') else None,
            "request_count": getattr(matcher, '_request_count', 0),
            "start_time": matcher._start_time.isoformat() if hasattr(matcher, '_start_time') else None
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"error": str(e)}
    
def main():
    import uvicorn
    logger.info("Starting BOND API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
