from typing import Dict, List, Tuple, Optional
from collections import defaultdict

def rrf_fuse(rankings: Dict[str, List[str]], k: float = 60.0, weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
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


def simple_fuse(rankings: Dict[str, List[str]], channel_priority: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """
    Simple fusion: concatenate results from channels in priority order, removing duplicates.
    Priority order: exact_label > exact > bm25 > dense > bm25_ctx > dense_full
    
    Args:
        rankings: Dict mapping source names to lists of ranked IDs
        channel_priority: Optional list of channel names in priority order.
                         If None, uses default priority: exact_label, exact, bm25, dense, bm25_ctx, dense_full
    
    Returns:
        List of (id, score) tuples. Score is 1.0 for all (no ranking score, just order)
    """
    if channel_priority is None:
        # Default priority order: exact matches first, then keyword, then semantic
        channel_priority = ["exact_label", "exact", "bm25", "dense", "bm25_ctx", "dense_full"]
    
    seen = set()
    fused = []
    
    # Process channels in priority order
    for channel in channel_priority:
        if channel in rankings:
            for id_ in rankings[channel]:
                if id_ not in seen:
                    seen.add(id_)
                    fused.append((id_, 1.0))  # Score of 1.0 for all (order matters, not score)
    
    # Process any remaining channels not in priority list
    for channel, ids in rankings.items():
        if channel not in channel_priority:
            for id_ in ids:
                if id_ not in seen:
                    seen.add(id_)
                    fused.append((id_, 1.0))
    
    return fused
