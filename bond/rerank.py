from typing import Dict, List, Set


def _assay_tokens(text: str) -> set[str]:
    t = (text or "").lower()
    feats = set()
    if "10x" in t:
        feats.add("10x")
    if "smart-seq" in t or "smartseq" in t:
        feats.add("smart-seq")
    if "drop-seq" in t or "dropseq" in t or "drop seq" in t:
        feats.add("drop-seq")
    if "cel-seq" in t or "celseq" in t or "cel seq" in t:
        feats.add("cel-seq")
    if "atac" in t:
        feats.add("atac")
    if "rna-seq" in t or "rnaseq" in t or "rna seq" in t:
        feats.add("rna-seq")
    if "spatial" in t or "visium" in t:
        feats.add("spatial")
    if "multiome" in t:
        feats.add("multiome")
    if "snrna" in t:
        feats.add("snrna")
    if "scrna" in t:
        feats.add("scrna")
    if "tcr" in t:
        feats.add("tcr")
    return feats


def apply_soft_boosts(
    ranked: List[Dict],
    cfg,
    field_name: str,
    query: str,
    context_terms: List[str] | None,
    exact_ids: Set[str],
) -> None:
    """Apply domain-aware soft boosts to fusion_score in-place.

    Includes exact-match, ontology prior, context overlap, term_doc lexical overlap,
    assay version/protocol, language preference, and sex canonical preference.
    """
    # Build context tokens
    context_tokens = {tok for term in (context_terms or []) for tok in term.lower().split()}

    # Lexical token set from original query + expansions + context is assembled upstream;
    # here reuse context_tokens for overlap; use term_doc for longer matches separately.

    # Assay feature tokens from query
    q_assay_feats = _assay_tokens(query)

    # Normalize original query for label-level exact match check
    def _normalize_for_match(s: str) -> str:
        """Normalize string for exact match comparison (lowercase, whitespace normalized)"""
        return " ".join(s.lower().split())
    
    normalized_query = _normalize_for_match(query)
    label_exact_matches = []
    
    for r in ranked:
        # 1) Label-Level Exact Match Boost (highest priority)
        # Check if query exactly matches the label (not just synonyms)
        candidate_label = r.get("label") or ""
        normalized_label = _normalize_for_match(candidate_label)
        
        if normalized_query == normalized_label:
            # Perfect label match - give highest boost
            r["fusion_score"] += cfg.label_exact_match_boost
            r["_label_exact_match"] = True  # Flag for logging
            label_exact_matches.append(r.get("id"))
        # 2) Synonym/Other Exact Match Boost (lower priority)
        elif r.get("id") in exact_ids:
            r["fusion_score"] += cfg.exact_match_boost
    
    # Log label exact matches for debugging
    if label_exact_matches:
        from .logger import logger
        logger.info(f"Label-level exact matches found: {label_exact_matches} (boost: {cfg.label_exact_match_boost})")

        # 2) Ontology Prior Boost
        try:
            from .schema_policies import get_ontology_prior_for_field

            primary_ontology = get_ontology_prior_for_field(field_name)
            if primary_ontology and r.get("source") == primary_ontology:
                r["fusion_score"] += cfg.ontology_prior_boost
        except Exception:
            pass

        # 3) Context Overlap Boost (label/definition/synonyms_exact)
        if context_tokens:
            candidate_text = (
                (r.get("label") or "")
                + " "
                + (r.get("definition") or "")
                + " "
                + " ".join(r.get("synonyms_exact") or [])
            ).lower()
            candidate_tokens = set(candidate_text.split())
            overlap = len(context_tokens.intersection(candidate_tokens))
            if overlap > 0:
                r["fusion_score"] += (cfg.context_overlap_boost * overlap)

        # 3b) term_doc lexical overlap boost using combined tokens (simple contains)
        try:
            # Build a minimal token set from query and context for term_doc scan
            lex_tokens: set[str] = set()
            for token in (query or "").lower().replace("/", " ").replace("_", " ").split():
                if len(token) >= 3 and token.isalnum():
                    lex_tokens.add(token)
            for ct in context_terms or []:
                for token in ct.lower().split():
                    if len(token) >= 3 and token.isalnum():
                        lex_tokens.add(token)
            if lex_tokens and r.get("term_doc"):
                td = str(r["term_doc"]).lower()
                hit = sum(1 for t in lex_tokens if t in td)
                if hit > 0:
                    r["fusion_score"] += (cfg.term_doc_overlap_boost * hit)
        except Exception:
            pass

        # 4) Assay version/protocol boosts
        if field_name.lower() in {"assay", "assay_type", "protocol"}:
            candidate_text = (
                (r.get("label") or "")
                + " "
                + (r.get("definition") or "")
                + " "
                + " ".join(r.get("synonyms_exact") or [])
            ).lower()
            import re

            query_versions = re.findall(r"v(\d+)", (query or "").lower())
            candidate_versions = re.findall(r"v(\d+)", candidate_text)
            if query_versions and candidate_versions and any(qv in candidate_versions for qv in query_versions):
                r["fusion_score"] += 0.3

            cand_feats = _assay_tokens(candidate_text)
            if len(q_assay_feats) >= 2:
                matched = len(q_assay_feats.intersection(cand_feats))
                missing = len(q_assay_feats) - matched
                if matched == len(q_assay_feats):
                    r["fusion_score"] += 0.5
                else:
                    r["fusion_score"] += 0.1 * matched
                    r["fusion_score"] -= 0.2 * missing

        # 5) Language preference: prefer ASCII labels for ASCII queries
        try:
            q_non_ascii = sum(1 for ch in (query or "") if ord(ch) > 127)
            if q_non_ascii == 0:
                label_text = (r.get("label") or "")
                label_non_ascii = sum(1 for ch in label_text if ord(ch) > 127)
                if label_non_ascii > 0:
                    r["fusion_score"] -= 0.5
        except Exception:
            pass

        # 6) Sex canonical preference
        if field_name.lower() == "sex":
            lbl = (r.get("label") or "").strip().lower()
            if lbl in {"male", "female"}:
                r["fusion_score"] += 2.0
            elif " " in lbl:
                r["fusion_score"] -= 0.2

