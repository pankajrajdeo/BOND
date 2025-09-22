from typing import List, Optional, Dict

# Field -> default ontology_id set (broad)
FIELD_TO_ONTOLOGIES = {
    "cell_type": ["cl", "fbbt", "zfa", "wbbt"],
    "tissue": ["uberon", "fbbt", "zfa", "wbbt"],
    "disease": ["mondo", "pato"],  # PATO:0000461 used for healthy
    "development_stage": ["hsapdv", "mmusdv", "fbdv", "wbls", "zfa"],
    "sex": ["pato"],
    "self_reported_ethnicity": ["hancestro"],
    "assay": ["efo"],
    "organism": ["ncbitaxon"],
}

# Field -> Primary Ontology for boosting
FIELD_TO_PRIMARY_ONTOLOGY: Dict[str, str] = {
    "cell_type": "cl",
    "tissue": "uberon",
    "disease": "mondo",
    "sex": "pato",
    "self_reported_ethnicity": "hancestro",
    "assay": "efo",
    "organism": "ncbitaxon",
    # development_stage is organism-dependent via SPECIES_ROUTING
}

# Ontology ID -> expected CURIE prefix mapping (guardrail for malformed DB rows)
ONTOLOGY_CURIE_PREFIX: Dict[str, str] = {
    "cl": "CL:",
    "uberon": "UBERON:",
    "mondo": "MONDO:",
    "pato": "PATO:",
    "efo": "EFO:",
    "ncbitaxon": "NCBITaxon:",
    "fbbt": "FBbt:",
    "zfa": "ZFA:",
    "wbbt": "WBbt:",
    "hsapdv": "HsapDv:",
    "mmusdv": "MmusDv:",
    "fbdv": "FBdv:",
    "wbls": "WBls:",
    "hancestro": "HANCESTRO:",
}

# Species-aware overrides for field -> ontology_id set
SPECIES_ROUTING = {
    # human
    "NCBITaxon:9606": {
        "cell_type": ["cl"],
        "development_stage": ["hsapdv"],
        "tissue": ["uberon"],
    },
    # mouse
    "NCBITaxon:10090": {
        "cell_type": ["cl"],
        "development_stage": ["mmusdv"],
        "tissue": ["uberon"],
    },
    # zebrafish
    "NCBITaxon:7955": {
        "cell_type": ["cl", "zfa"],
        "development_stage": ["zfa"],
        "tissue": ["uberon", "zfa"],
    },
    # fruit fly
    "NCBITaxon:7227": {
        "cell_type": ["cl", "fbbt"],
        "development_stage": ["fbdv"],
        "tissue": ["uberon", "fbbt"],
    },
    # C. elegans
    "NCBITaxon:6239": {
        "cell_type": ["cl", "wbbt"],
        "development_stage": ["wbls"],
        "tissue": ["uberon", "wbbt"],
    },
}

# Supported organisms (canonical names only)
SUPPORTED_ORGANISMS = {
    "Homo sapiens": "NCBITaxon:9606",
    "Mus musculus": "NCBITaxon:10090",
    "Danio rerio": "NCBITaxon:7955",
    "Drosophila melanogaster": "NCBITaxon:7227",
    "Caenorhabditis elegans": "NCBITaxon:6239",
}

def supported_organisms() -> List[str]:
    return list(SUPPORTED_ORGANISMS.keys())

def _canonical_taxon(organism: Optional[str]) -> Optional[str]:
    if not organism:
        return None
    # Enforce exact match to supported names only
    try:
        return SUPPORTED_ORGANISMS[organism]
    except KeyError:
        raise ValueError(
            "Unsupported organism. Use one of: " + ", ".join(supported_organisms())
        )

def supported_fields() -> List[str]:
    return list(FIELD_TO_ONTOLOGIES.keys())

def allowed_ontologies_for(field_name: Optional[str], organism: Optional[str]) -> Optional[List[str]]:
    """Return ontology_id list for this field and organism, or None if no restriction.
    If field_name is provided and not supported, raises ValueError with a helpful message.
    """
    if not field_name:
        return None
    field = field_name.strip().lower()
    base = FIELD_TO_ONTOLOGIES.get(field)
    if not base:
        raise ValueError("Unsupported field. Use one of: " + ", ".join(supported_fields()))
    tax = _canonical_taxon(organism)
    if tax and tax in SPECIES_ROUTING and field in SPECIES_ROUTING[tax]:
        return SPECIES_ROUTING[tax][field]
    return base

def get_ontology_prior_for_field(field_name: str) -> Optional[str]:
    """Returns the primary ontology source for a given field."""
    if not field_name:
        return None
    return FIELD_TO_PRIMARY_ONTOLOGY.get(field_name.lower().strip())

def allowed_curie_prefixes(ontology_ids: Optional[List[str]]) -> Optional[List[str]]:
    """Return list of CURIE prefixes corresponding to ontology_ids.
    If unknown ontology id is provided, it is ignored.
    """
    if not ontology_ids:
        return None
    prefixes = [ONTOLOGY_CURIE_PREFIX[o] for o in ontology_ids if o in ONTOLOGY_CURIE_PREFIX]
    return prefixes or None
