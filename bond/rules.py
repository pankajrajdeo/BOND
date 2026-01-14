"""Common heuristic rules for BOND pipeline.

Includes abstain detection and basic compatibility checks extracted
from the pipeline to keep orchestration code lean.
"""

from __future__ import annotations

import re
from typing import Tuple


# Precompiled, token-aware patterns for abstain triggers (case-insensitive)
_ABSTAIN_PATTERNS = [
    re.compile(r"\bunknown\b", re.IGNORECASE),
    re.compile(r"\bdoublets?\b", re.IGNORECASE),
    re.compile(r"\bdebris\b", re.IGNORECASE),
    re.compile(r"\bnull\b", re.IGNORECASE),
    # Match "na" and "n/a" as whole tokens
    re.compile(r"\bn/?a\b", re.IGNORECASE),
]


def should_abstain(query: str) -> Tuple[bool, str | None]:
    """Return (True, reason) if query is a no-value indicator.

    Triggers include: unknown, doublet(s), debris, null, na, n/a (case-insensitive).
    """
    text = (query or "").strip()
    for pat in _ABSTAIN_PATTERNS:
        if pat.search(text):
            return True, f"Query contains no-value indicator matching '{pat.pattern}'"
    return False, None


def context_violation(label_blob: str, tissue: str | None) -> bool:
    """Detect anatomical incompatibilities given tissue context.
    Conservative, token-based checks to avoid cross-system leakage.
    """
    t = (tissue or "").lower()
    x = (label_blob or "").lower()

    # Brain region conflicts (expanded cortical detection)
    cortical_tokens = [
        "cortex", "cortical", "neocortex", "gyrus", "sulcus",
        "temporal gyrus", "temporal cortex", "frontal lobe", "parietal lobe",
        "occipital lobe", "temporal lobe", "entorhinal", "hippocampus",
        "amygdala", "cingulate", "prefrontal", "neocortical",
    ]
    subcortical_striatal = ["striatal", "striatum", "basal ganglia", "putamen", "caudate", "globus pallidus", "substantia nigra"]
    cerebellar_tokens = ["cerebellum", "cerebellar", "purkinje"]

    is_cortical = any(tok in t for tok in cortical_tokens)
    is_striatal = any(tok in t for tok in subcortical_striatal)
    is_cerebellar = any(tok in t for tok in cerebellar_tokens)

    if is_cortical and (any(tok in x for tok in subcortical_striatal) or any(tok in x for tok in cerebellar_tokens)):
        return True
    if is_striatal and ("cortical" in x or "cortex" in x or "gyrus" in x or "neocortex" in x or "hippocampus" in x or "amygdala" in x):
        return True
    if is_cerebellar and ("cortex" in x or "cortical" in x or "striatal" in x or "striatum" in x or "basal ganglia" in x):
        return True

    # Organ-specific conflicts
    if "lung" not in t and any(tok in x for tok in ["alveolar", "bronchial", "airway", "pulmonary"]):
        return True
    if "kidney" not in t and "renal" in x:
        return True

    return False


def species_violation(label_blob: str, organism: str | None) -> bool:
    """Detect species incompatibilities based on label/definition markers.

    Heuristics:
      - Reject cross-markers like (Mmus) vs (Hsap) when organism disagrees.
      - Reject explicit species tokens in labels ("human", "mouse", etc.) when they disagree.
    """
    x = (label_blob or "").lower()
    org = (organism or "").lower()
    # Curated markers
    if org.startswith("homo sapiens") and "(mmus" in x:
        return True
    if org.startswith("mus musculus") and "(hsap" in x:
        return True
    # Token-based species words
    if org.startswith("homo sapiens"):
        if re.search(r"\b(mouse|mus\s+musculus|murine|rat|rattus|zebrafish|danio\s+rerio|drosophila|fruit\s+fly|fly|caenorhabditis|c\.?\s*elegans|nematode)\b", x):
            return True
    if org.startswith("mus musculus"):
        if re.search(r"\b(human|homo\s+sapiens)\b", x):
            return True
    return False


# -----------------
# Normalizers
# -----------------

_ORG_CANONICAL = {
    "homo sapiens": "Homo sapiens",
    "human": "Homo sapiens",
    "hs": "Homo sapiens",
    "hsa": "Homo sapiens",
    "hg": "Homo sapiens",
    
    "mus musculus": "Mus musculus",
    "mouse": "Mus musculus",
    "mice": "Mus musculus",
    "mm": "Mus musculus",
    "mmu": "Mus musculus",
    
    "danio rerio": "Danio rerio",
    "zebrafish": "Danio rerio",
    "dr": "Danio rerio",
    "zfish": "Danio rerio",
    
    "drosophila melanogaster": "Drosophila melanogaster",
    "fruit fly": "Drosophila melanogaster",
    "fly": "Drosophila melanogaster",
    "dm": "Drosophila melanogaster",
    "dmel": "Drosophila melanogaster",
    
    "caenorhabditis elegans": "Caenorhabditis elegans",
    "c. elegans": "Caenorhabditis elegans",
    "worm": "Caenorhabditis elegans",
    "cel": "Caenorhabditis elegans",
    "ce": "Caenorhabditis elegans",
}


def normalize_organism(name: str | None) -> str | None:
    if not name:
        return name
    key = name.strip().lower()
    return _ORG_CANONICAL.get(key, name)


# Require token to start with a letter to avoid matching bare numerics like "5+"
_MARKER_SUFFIX_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9-]*)([+-])\b")


def normalize_marker_suffixes(text: str | None) -> str | None:
    """Normalize marker suffixes like `CD25+` -> `CD25 positive`, `CD14-` -> `CD14 negative`.
    Leaves other text unchanged. Returns unchanged when input is falsy.
    """
    if not text:
        return text
    def _sub(m: re.Match[str]) -> str:
        token = m.group(1)
        sign = m.group(2)
        return f"{token} {'positive' if sign == '+' else 'negative'}"
    out = _MARKER_SUFFIX_RE.sub(_sub, text)
    # Ensure a space after the inserted positive/negative if next token is alnum (handles CD4+CD8+ -> CD4 positive CD8+)
    out = re.sub(r"\b(positive|negative)(?=[A-Za-z0-9])", r"\1 ", out)
    return out


def normalize_cell_type_hyphens(text: str | None) -> str | None:
    """Normalize hyphens in cell type names: remove hyphens in the middle of words, keep hyphens at the end.
    
    Rule: 
    - If hyphen is in the middle of words (e.g., "T-cell"), remove it (convert to "T cell").
    - If hyphen is at the end of a word (e.g., "CD4-"), leave it as is.
    - If hyphen is before "positive" or "negative" (marker expressions), leave it as is.
    
    Examples:
        'T-cell' -> 'T cell' (hyphen in middle, removed)
        'B-cell' -> 'B cell' (hyphen in middle, removed)
        'NK-cell' -> 'NK cell' (hyphen in middle, removed)
        'CD4-' -> 'CD4-' (hyphen at end, preserved)
        'CD4-negative' -> 'CD4-negative' (marker expression, preserved)
        'CD8-positive' -> 'CD8-positive' (marker expression, preserved)
    """
    if not text:
        return text
    
    # Pattern: hyphen between two word characters
    # But preserve if second word is "positive" or "negative" (marker expressions)
    pattern = r'\b([A-Za-z0-9]+)-([A-Za-z0-9]+)\b'
    
    def _replace_hyphen(m: re.Match[str]) -> str:
        # Check if the second word is "positive" or "negative" (case-insensitive)
        second_word = m.group(2).lower()
        if second_word in ('positive', 'negative'):
            # This is a marker expression, keep the hyphen
            return m.group(0)
        # Replace hyphen with space (hyphen is in middle of words)
        return f"{m.group(1)} {m.group(2)}"
    
    out = re.sub(pattern, _replace_hyphen, text)
    return out


# Field name normalization
_FIELD_CANONICAL = {
    "cell_type": "cell_type",
    "celltype": "cell_type",
    "cell": "cell_type",
    "tissue": "tissue",
    "organ": "tissue",
    "disease": "disease",
    "condition": "disease",
    "pathology": "disease",
    "development_stage": "development_stage",
    "dev_stage": "development_stage",
    "development-stage": "development_stage",
    "stage": "development_stage",
    "sex": "sex",
    "gender": "sex",
    "self_reported_ethnicity": "self_reported_ethnicity",
    "ethnicity": "self_reported_ethnicity",
    "race": "self_reported_ethnicity",
    "assay": "assay",
    "assay_type": "assay",
    "protocol": "assay",
    "organism": "organism",
    "species": "organism",
}


def normalize_field_name(name: str | None) -> str | None:
    if not name:
        return name
    key = name.strip().lower()
    return _FIELD_CANONICAL.get(key, name)
