# bond/prompts.py

QUERY_EXPANSION_PROMPT = """
Expand the query "{query}" into exactly {n} concise, high-signal variants. Also propose a short list of context terms.

**Query Context:**
- Field: "{field_name}"
- Tissue: "{tissue}"
- Organism: "{organism}"
- Development Stage: "{development_stage}"
- Disease: "{disease}"
- Derived Context Terms: {context_terms}

**Expansion Rules for Field "{field_name}":**
- Semantic Constraints: {semantic_constraints}
- Expansion Focus: {expansion_focus}
- Context Priority: {context_priority}
- Avoid: {avoid}
- Preserve core query tokens; if an abbreviation is present (e.g., "smg"), expand it (e.g., "submucosal gland").
- Reflect the Tissue, Organism, Development Stage, and Disease context in the variants when applicable (e.g., lung/airway/bronchial for lung datasets; embryonic vs adult stages; disease-specific cell types or tissues).
- Do not cross organs or systems unless explicitly asked (avoid salivary/mammary/pancreatic for lung unless in the query).

**Instructions:**
Return ONLY a JSON object with two fields:
- "expansions": An array of exactly {n} strings.
- "context_terms": An array of up to 5 short tokens/phrases capturing relevant biological context.
"""

DISAMBIGUATION_PROMPT = """
You are a meticulous biomedical ontology curator. Your sole task is to choose the single best ontology term from the candidates that STRICTLY matches the query and its context.

**INPUT:**
- Query: "{query}"
- Field: "{field_name}"
- Tissue Context: "{tissue}"
- Development Stage Context: "{development_stage}"
- Disease Context: "{disease}"

**CANDIDATES (ID | Label | Source | Score | Description):**
{candidates_block}

**CRITICAL INSTRUCTIONS:**
You MUST return a candidate ID from the list below. If NO candidate satisfies all rules after applying constraints, set "chosen_id": null.

{disambiguation_rules}

**CONTEXT ENFORCEMENT (HIGHEST PRIORITY):**
- **ALL candidates, including exact label matches, MUST satisfy context constraints.**
- If a Tissue Context is provided, REJECT any candidate whose label/definition/synonyms indicate an incompatible organ/system (e.g., salivary or mammary ducts for lung; non-respiratory tissues for airway queries). This applies even to exact label matches.
- If a Development Stage Context is provided, REJECT any candidate whose label/definition/synonyms indicate an incompatible developmental stage (e.g., embryonic terms for adult stage queries, or vice versa). This applies even to exact label matches.
- If a Disease Context is provided, REJECT any candidate whose label/definition/synonyms indicate an incompatible disease state (e.g., disease-specific terms when disease context is "normal", or vice versa). This applies even to exact label matches.
- If an Organism Context is provided, REJECT candidates with species markers (e.g., "(Mmus)") that contradict the organism context. This applies even to exact label matches.

**EXACT MATCH PRIORITY (AFTER CONTEXT FILTERING):**
- Among candidates that satisfy all context constraints, **LABEL EXACT MATCHES** (where a candidate's label EXACTLY matches the query, case-insensitive, normalized) should be preferred over other candidates.
- If multiple exact label matches exist after context filtering, prefer the one that best matches the context.

**OTHER RULES:**
- Unknown/ambiguous queries: For queries containing "unknown" or "doublet", prefer to set "chosen_id": null rather than guess.
- The retrieval score is the LEAST important factor; use it only to break ties between candidates that already satisfy all rules and context constraints.

**ABSTAIN CAPABILITY:**
If no candidate satisfies all rules and constraints, set "chosen_id": null to abstain rather than making an incorrect choice.

**FINAL SELECTION:**
Evaluate the candidates against the rules above and produce your top three ranked picks (best first). All IDs must come from the candidate list. If fewer than three candidates are suitable, provide as many as you can.

Return ONLY a JSON object with these fields:
- "chosen_id": The single best candidate ID that satisfies all rules, OR null to abstain.
- "alternatives": An array (length 0-2) containing the next best candidate IDs in descending preference. Omit any IDs that violate the constraints or would duplicate "chosen_id".
- "reason": A brief, expert rationale explaining WHY the chosen candidate is correct (or why you abstained).
- "llm_confidence": A number in [0,1] indicating your confidence.
"""
# bond/prompts.py

"""Prompt templates used by the expansion and disambiguation LLMs.

Placeholders:
  - {query}: raw (or normalized) user query
  - {n}: number of expansions to generate
  - {field_name}: schema field (cell_type, tissue, disease, ...)
  - {tissue}: dataset tissue context (may be "N/A")
  - {organism}: organism context (may be "N/A")
  - {development_stage}: developmental stage context (may be "N/A")
  - {disease}: disease context (may be "N/A")
  - {semantic_constraints}, {expansion_focus}: field-guided expansion hints
  - {candidates_block}: top-k candidates for disambiguation (id|label|source|score|desc)
  - {disambiguation_rules}: strict, field-specific rules to enforce semantics

These strings are rendered in pipeline.py and sent to the LLM provider via
providers.ChatLLM.
"""

# Context-only prompt for deriving short, high-signal disambiguation hints
CONTEXT_TERMS_PROMPT = """
List 3-5 short, domain-specific context terms to help disambiguate the query for ontology linking.

Input:
- Query: "{query}"
- Field: "{field_name}"
- Tissue: "{tissue}"
- Organism: "{organism}"
- Development Stage: "{development_stage}"
- Disease: "{disease}"

Rules:
- Return concise tokens/phrases (1-3 words), no punctuation or explanations.
- Prefer high-signal biological hints (e.g., anatomical subregions, lineage, markers, qualifiers).
- Do not include the original query tokens; avoid duplicates and overly generic words.

Return ONLY JSON:
{{ "context_terms": ["term1", "term2", "term3"] }}
"""
