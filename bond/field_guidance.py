# bond/field_guidance.py
from typing import Dict, Optional

FIELD_GUIDANCE: Dict[str, Dict[str, str]] = {
    "cell_type": {
        "semantic_constraints": "MUST be a cellular entity (from Cell Ontology or species-specific anatomy). PROHIBITED: genes, proteins, biological processes (GO terms), diseases, or anatomical structures.",
        "expansion_focus": "Lineage-preserving variants, activation states, common synonyms, and marker combinations (e.g., CD4-positive).",
        "context_priority": "Lineage, activation state, surface markers, anatomical subregion relevant to tissue context.",
        "avoid": "Gene symbols alone, GO processes, diseases, assay/platform names.",
        "disambiguation_rules": (
            "1. **SEMANTIC TYPE CHECK:** First, ensure the candidate is a cell type. Immediately REJECT any candidate that is a biological process (e.g., GO terms), a gene, or a protein. "
            "2. **CONTRADICTION CHECK:** Scrutinize for direct contradictions. If the query specifies 'M1', any 'M2 macrophage' candidate is INVALID and MUST be rejected. If the query specifies 'CD4-positive', a 'CD8-positive' candidate is INVALID. This rule overrides retrieval score. "
            "3. **POSITIVE SIGNAL MATCH:** After eliminating contradictions, give highest preference to candidates that positively match explicit signals (markers like 'CD25+', states like 'activated', provenance like 'monocyte-derived'). "
            "4. **LINEAGE CONSISTENCY:** The candidate must belong to the correct biological lineage implied by the query. "
            "5. **SPECIFICITY & TIE-BREAKING:** Match the query's level of detail; use retrieval score only to break ties between otherwise valid candidates."
        )
    },
    "tissue": {
        "semantic_constraints": "MUST be an anatomical entity (organ, tissue, or substructure from UBERON or species-specific anatomy). PROHIBITED: cell types, diseases, assays.",
        "expansion_focus": "Synonyms and immediate parent/child anatomical terms (e.g., 'lung' -> 'pulmonary tissue').",
        "context_priority": "Organ/system, subregion (e.g., lobe, cortex), laterality if applicable.",
        "avoid": "Cell types, biological processes, disease names, assay terms.",
        "disambiguation_rules": (
            "1. **SEMANTIC TYPE CHECK:** Candidate MUST be an anatomical structure. REJECT cell types or biological processes. "
            "2. **CONTEXTUAL ALIGNMENT:** Use provided context to resolve ambiguity (e.g., for 'cortex' with tissue 'adrenal gland', choose 'adrenal cortex'). REJECT anatomically incompatible locations. "
            "3. **HIERARCHICAL ACCURACY:** Choose the correct anatomical level. Prefer the direct organ/tissue match over a minute substructure unless the query is highly specific."
        )
    },
    "disease": {
        "semantic_constraints": "MUST be a pathological condition from MONDO. The term 'normal' or 'healthy' MUST map to PATO:0000461.",
        "expansion_focus": "Common clinical synonyms, disease subtypes, and related pathological processes.",
        "context_priority": "Anatomical site, etiology, subtype, onset/severity modifiers.",
        "avoid": "Physiological traits, experimental conditions, assays, non-pathological phenotypes unless explicitly 'healthy'.",
        "disambiguation_rules": (
            "1. **SEMANTIC TYPE CHECK:** Candidate MUST be a disease from MONDO or PATO:0000461 for 'healthy'. "
            "2. **KEYWORD ALIGNMENT:** Candidate label or synonyms should contain the core disease concept from the query. "
            "3. **SPECIFICITY MATCH:** If a subtype is specified (e.g., 'adenocarcinoma'), REJECT the generic parent ('carcinoma') when the specific subtype is available."
        )
    },
    "development_stage": {
        "semantic_constraints": "MUST be a developmental stage from the correct organism-specific ontology (e.g., HsapDv, MmusDv).",
        "expansion_focus": "Standard stage names, synonyms, and common temporal descriptions (e.g., 'embryonic day 14.5' -> 'E14.5').",
        "context_priority": "Species-appropriate stage terms, exact timepoint format (e.g., E14.5), prenatal/postnatal.",
        "avoid": "Anatomical structures, cell types, disease modifiers.",
        "disambiguation_rules": (
            "1. **ONTOLOGY SOURCE:** Candidate MUST come from the correct species-specific developmental ontology. "
            "2. **TEMPORAL ACCURACY:** Choose the term that most accurately reflects the age or stage mentioned in the query. Reject anatomical terms."
        )
    },
    "sex": {
        "semantic_constraints": "MUST be a biological sex from PATO.",
        "expansion_focus": "Canonical terms like 'male', 'female', 'hermaphrodite'.",
        "context_priority": "Exact sex terms only; no fuzzy synonyms.",
        "avoid": "Gender identity terms, karyotype-only phrases unless explicit, unrelated traits.",
        "disambiguation_rules": (
            "1. **KEYWORD FIRST:** Choose the candidate that is an exact or synonymous match to the query word. "
            "2. **REJECT SEMANTIC DRIFT:** Be highly skeptical of candidates that do not share exact keywords, regardless of vector similarity. The vocabulary is too small for fuzzy matching."
        )
    },
    "self_reported_ethnicity": {
        "semantic_constraints": "MUST be an ancestry or population category from HANCESTRO.",
        "expansion_focus": "Standard biomedical ancestry groups and geographical synonyms.",
        "context_priority": "Continental or major population groups unless a specific subpopulation is stated.",
        "avoid": "Nationality, language, religion; overly granular subpopulations without explicit mention.",
        "disambiguation_rules": (
            "1. **KEYWORD FIRST:** Prefer candidates that are an exact or synonymous match to the query. "
            "2. **HIERARCHY:** Choose the appropriate population resolution. Prefer broader continental groups unless a specific sub-population is explicitly mentioned."
        )
    },
    "assay": {
        "semantic_constraints": "MUST be an experimental method or assay from EFO.",
        "expansion_focus": "Platform names, technology synonyms, and full experimental names.",
        "context_priority": "Platform/chemistry, version, library prep specifics.",
        "avoid": "Biological entities, disease names, anatomical structures.",
        "disambiguation_rules": (
            "1. **KEYWORD FIRST:** The candidate label or synonym MUST contain the core technology or platform name from the query (e.g., '10x', 'SMART-seq'). "
            "2. **REJECT BIOLOGICAL TERMS:** Immediately reject any candidate that is a biological entity instead of a technical method."
        )
    },
    "organism": {
        "semantic_constraints": "MUST be a species or taxon from NCBITaxon.",
        "expansion_focus": "Canonical scientific names and common names (e.g., 'Homo sapiens' -> 'human').",
        "context_priority": "Species-level naming; strain only if explicitly mentioned.",
        "avoid": "Cell lines, tissues, diseases, processes.",
        "disambiguation_rules": (
            "1. **KEYWORD FIRST:** Choose the candidate that is an exact or synonymous match to the query name. "
            "2. **TAXONOMIC LEVEL:** Prefer the species level over subspecies or strains unless a specific strain is explicitly mentioned in the query."
        )
    },
}

def get_field_guidance(field_name: Optional[str]) -> Dict[str, str]:
    if not field_name:
        return {}
    return FIELD_GUIDANCE.get(field_name.lower(), {})
