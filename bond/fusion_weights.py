from typing import Dict


def field_aware_weights(field_name: str) -> Dict[str, float]:
    """Return field-specific RRF weights.

    Adjusts source importance per domain to steer fusion sensibly.
    """
    field_lower = (field_name or "").lower()

    if field_lower in {"assay", "assay_type", "protocol"}:
        return {"exact": 1.5, "bm25": 0.9, "dense": 0.4}
    if field_lower in {"cell_type", "celltype", "cell"}:
        return {"exact": 1.0, "bm25": 0.7, "dense": 1.2}
    if field_lower in {"disease", "condition", "pathology"}:
        return {"exact": 1.2, "bm25": 0.8, "dense": 0.8}
    return {"exact": 1.0, "bm25": 0.8, "dense": 0.6}

