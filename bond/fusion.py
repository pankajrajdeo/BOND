from typing import Dict, List, Tuple
from collections import defaultdict

def rrf_fuse(rankings: Dict[str, List[str]], k: float = 60.0, weights: Dict[str, float] | None = None) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion with optional source weighting.
    
    Args:
        rankings: Dict mapping source names to lists of ranked IDs
        k: RRF parameter (default 60.0)
        weights: Optional dict mapping source names to weights (default: all sources weighted equally)
    
    Returns:
        List of (id, score) tuples sorted by score descending
    """
    scores = defaultdict(float)
    
    # Default weights if none provided
    if weights is None:
        weights = {src: 1.0 for src in rankings.keys()}
    
    for src, ids in rankings.items():
        weight = weights.get(src, 1.0)  # Default weight 1.0 for unknown sources
        for rank, id_ in enumerate(ids, start=1):
            scores[id_] += weight / (k + rank)
    
    fused = list(scores.items())
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
