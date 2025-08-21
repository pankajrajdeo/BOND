QUERY_EXPANSION_PROMPT = """\
Expand the query into exactly {n} concise, high-signal variants that strictly respect the dataset context, and also propose a small list of organ/tissue context terms.

Rules:
- Enforce the semantic type implied by Field. Examples:
  - If Field == "cell type": only cell-type terms; no genes/proteins/GO terms.
  - If Field == "gene" or "protein": gene/protein names only.
  - If Field == "disease": disease terms only.
- Use organ/tissue context from Dataset when present (e.g., lung, airway, pulmonary, brain, liver, kidney). Prefer variants that explicitly include the organ/tissue.
- Avoid variants that imply a different or unrelated organ/tissue than the Dataset context unless explicitly stated in the Query.
- Biomedical domain only; no speculative terms or subtypes beyond what the query implies.
- No species qualifiers unless present in the Query.

Return ONLY a JSON object with two fields:
- "expansions": an array of exactly {n} strings (one variant per array item)
- "context_terms": an array of up to 5 short tokens/phrases capturing organ/tissue context (ranked by relevance)

Query: "{query}"
Field: "{field_name}"
Dataset: "{dataset_description}"
"""

DISAMBIGUATION_PROMPT = """\
You are an expert biomedical ontology curator. Your task is to choose the single best standardized ontology term for a given query, based on the context of the dataset it came from and a list of retrieved candidate terms.

Query: "{query}"
Field Name: "{field_name}"
Dataset Description: "{dataset_description}"

Candidates (ID | Label | Source | Score | Description):
{candidates_block}

Analyze the candidates and select the one that is the most precise and contextually appropriate match for the query.

Guidance:
- Prioritize the canonical definition and label. Use the Description field, which includes: Definition, Exact Synonyms, Related Synonyms, Broad Synonyms.
- Respect dataset context (Field and Dataset). Prefer candidates whose Description clearly matches the organ/tissue and context implied by the Dataset (e.g., lung/airway for a lung dataset). Reject candidates that imply unrelated organs/tissues (e.g., salivary gland) unless the Query explicitly specifies them.
- When two candidates are comparable, prefer the one with higher Score. Score is a normalized retrieval confidence in [0,1].
- Prefer candidates whose Label directly contains the main query tokens when context is satisfied.
- Use the Description field (Definition, Exact/Related/Broad Synonyms) to verify semantic fit. The output should be biologically plausible and accurate.

Return ONLY a JSON object with these fields:
- "chosen_id": the single best candidate ID
- "reason": brief rationale for the choice
- "llm_confidence": a number in [0,1]
- "alternatives": OPTIONAL array of up to N additional IDs (if the top choice is low-confidence or ambiguous). If alternatives are included, order by preference.

Example Response:
{{
  "chosen_id": "CL:0000084",
  "reason": "The query 'T-cell' directly corresponds to the standard parent term in the Cell Ontology.",
  "llm_confidence": 0.92,
  "alternatives": ["CL:0000542", "CL:1001550", "CL:0000912"]
}}
"""