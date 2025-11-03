from bond import BondMatcher

# Initialize matcher
matcher = BondMatcher()

# Query per user's specified case
result = matcher.query(
    query="gcap",
    field_name="cell_type ",  # keep the space as provided
    dataset_description="HLCA LungMAP data",
    restrict_to_ontologies=["cl"],
)

chosen = result.get("chosen") or {}

print("=" * 80)
print("BOND CHOSEN TERM (Concise)")
print("=" * 80)
print(f"id: {chosen.get('id', '')}")
print(f"label: {chosen.get('label', '')}")
print(f"source: {chosen.get('source', '')}")
print(f"iri: {chosen.get('iri', '')}")
print(f"definition: {chosen.get('definition', '')}")
print(f"retrieval_confidence: {chosen.get('retrieval_confidence', 0):.3f}")
print(f"llm_confidence: {chosen.get('llm_confidence', 0):.3f}")
print(f"reason: {chosen.get('reason', '')}")
print("=" * 80)