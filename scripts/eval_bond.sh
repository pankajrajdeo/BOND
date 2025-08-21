#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/eval_bond.sh
# Requires: bond-query in PATH (your venv), jq (optional, for summaries)

OUTDIR="eval_outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

common_restrict="cl"
common_field="cell type"
lung_ctx="HLCA lungref full data for pediatric cases by LungMAP"

# TESTS: query | field_name | dataset_description | restrict_ontologies
cat > "$OUTDIR/tests.tsv" <<'TSV'
SMG|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
SMG duct|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
submucosal gland|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
goblet cell|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
airway basal cell|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
alveolar macrophage|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
ciliated cell|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
mucus secreting cell|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
club cell|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
AT2 cell|cell type|HLCA lungref full data for pediatric cases by LungMAP|cl
NK cells|cell type|PBMC immune single-cell dataset|cl
DC2|cell type|PBMC immune single-cell dataset|cl
TSV

slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_ -' | tr ' ' '_' | tr '-' '_' ; }

while IFS='|' read -r query field dataset restricts; do
  [ -z "${query:-}" ] && continue
  name="$(slug "${query}_${field}")"
  echo ">>> Running: $query | $field | $dataset | $restricts"

  # 1) Non-verbose, single choice (baseline)
  bond-query \
    --query "$query" \
    --field_name "$field" \
    --dataset_description "$dataset" \
    --restrict_to_ontologies $restricts \
    > "$OUTDIR/${name}.single.json"

  # 2) Non-verbose, with alternatives
  bond-query \
    --query "$query" \
    --field_name "$field" \
    --dataset_description "$dataset" \
    --restrict_to_ontologies $restricts \
    --num_choices 3 \
    > "$OUTDIR/${name}.alts.json"

  # 3) Verbose trace (for inspection)
  bond-query \
    --query "$query" \
    --field_name "$field" \
    --dataset_description "$dataset" \
    --restrict_to_ontologies $restricts \
    --num_choices 3 \
    --verbose \
    > "$OUTDIR/${name}.verbose.json"

  # Optional: One more run with “duct” nudged when relevant
  if echo "$query" | grep -qi 'smg'; then
    bond-query \
      --query "${query} duct" \
      --field_name "$field" \
      --dataset_description "$dataset" \
      --restrict_to_ontologies $restricts \
      --num_choices 3 \
      > "$OUTDIR/${name}.smg_duct_hint.json"
  fi

  # Summaries (jq optional)
  if command -v jq >/dev/null 2>&1; then
    echo "  [single]   chosen: $(jq -r '.label + " | " + (.id // "")' "$OUTDIR/${name}.single.json" 2>/dev/null || echo '-')"
    echo "  [alts]     chosen: $(jq -r '.chosen.label + " | " + (.chosen.id // "")' "$OUTDIR/${name}.alts.json" 2>/dev/null || echo '-')"
    echo "             alternatives:"
    jq -r '.alternatives[]? | "               - " + .label + " | " + (.id // "") + " | rc=" + ((.retrieval_confidence // 0) | tostring)' "$OUTDIR/${name}.alts.json" 2>/dev/null || true
    echo
  fi
done < "$OUTDIR/tests.tsv"

echo "All outputs saved under: $OUTDIR"