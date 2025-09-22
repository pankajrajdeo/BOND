import os
import re
import random
import unicodedata
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")


# ----------------------------
# 1) Helpers (format + normalization)
# ----------------------------
def _norm_text(s):
    """Unicode-normalize (NFKC) and strip whitespace; return None if empty."""
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s).strip()
    return s if s else None

def _nz(s, default="unknown"):
    """Normalize and provide default when missing/empty."""
    t = _norm_text(s)
    return t if t is not None else default

def _truncate(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    return text[:max_chars]

def make_anchor(field_type: str, author_term: str, tissue: str, organism: str) -> str:
    prefix = field_type or "cell_type"
    # Keep author term as-authored, but Unicode-normalized & stripped (no casing changes)
    at = _norm_text(author_term)
    return f"{prefix}: {at if at is not None else 'unknown'}; tissue: {_nz(tissue)}; organism: {_nz(organism)}"

def make_positive(label: str, synonyms: dict, definition: str) -> str:
    # Match FAISS text exactly: label; synonyms (exact→narrow→broad→related, cap 50); definition (cap 500)
    syn_exact = (synonyms or {}).get("exact", []) or []
    syn_narrow = (synonyms or {}).get("narrow", []) or []
    syn_broad = (synonyms or {}).get("broad", []) or []
    syn_related = (synonyms or {}).get("related", []) or []
    syn_all = []
    for lst in (syn_exact, syn_narrow, syn_broad, syn_related):
        for s in lst:
            syn_all.append(s.strip() if isinstance(s, str) else str(s))
    syn_all = [s for s in syn_all if s]
    syn_field = " | ".join(syn_all[:50]) if syn_all else ""
    def_short = _truncate(definition, max_chars=500)
    parts = [f"label: {_nz(label, default='')}"]
    if syn_field:
        parts.append(f"synonyms: {syn_field}")
    if def_short:
        parts.append(f"definition: {def_short}")
    return "; ".join(parts)


# ----------------------------
# 2) Load HF dataset and build anchor/positive pairs
# ----------------------------
HF_DATASET = "pankajrajdeo/BOND-Benchmark"

# ---------- Heuristics to drop ONLY useless anchors ----------
_NA_LIKE = {"na", "n/a", "null", "none", "nan", "[na]", "[none]"}

_num_re = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")         # pure int/float
_date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")               # YYYY-MM-DD

def _is_pure_number(s: str) -> bool:
    return bool(_num_re.fullmatch(s))

def _is_date_like(s: str) -> bool:
    return bool(_date_re.fullmatch(s))

def _is_na_like(s: str) -> bool:
    return s.lower() in _NA_LIKE

def should_keep_example(author_term: str, ontology_id: str, label: str) -> bool:
    """
    Keep anchors EXACTLY as authored (aside from strip/NFKC), but:
      - require a valid mapping (ontology_id AND label)
      - drop anchors that are clearly placeholders: empty/NA, pure numbers, dates
    Everything else (IDs like 'VLMC_6', 'Donor 2_Jej', etc.) is kept.
    """
    if not ontology_id or not label:
        return False
    s = _norm_text(author_term)
    if s is None:
        return False
    if _is_na_like(s) or _is_pure_number(s) or _is_date_like(s):
        return False
    return True

def build_pairs(split: str):
    ds = load_dataset(HF_DATASET, split=split)
    anchors, positives, curies = [], [], []
    for row in ds:
        field_type = row.get("field_type")
        author_term = row.get("author_term")
        tissue = row.get("tissue")
        organism = row.get("organism")

        mapping = row.get("mapping") or {}
        resolved = mapping.get("resolved") or {}
        original = mapping.get("original") or {}

        label = resolved.get("label") or original.get("label")
        definition = resolved.get("definition") or original.get("definition")
        synonyms = resolved.get("synonyms") or original.get("synonyms") or {}
        curie = resolved.get("ontology_id") or original.get("ontology_id")

        # Filter examples per heuristic
        if should_keep_example(author_term, curie, label):
            anchors.append(make_anchor(field_type, author_term, tissue, organism))
            positives.append(make_positive(label, synonyms, definition))
            curies.append(_nz(curie, default=''))
    return anchors, positives, curies

train_anchors, train_positives, train_ids = build_pairs("train")
val_anchors, val_positives, val_ids = build_pairs("validation")
test_anchors, test_positives, test_ids = build_pairs("test")

# Shuffle training pairs each epoch via DataLoader, but also pre-shuffle once
seed = 7
random.seed(seed)
idx = list(range(len(train_anchors)))
random.shuffle(idx)
train_examples = [
    InputExample(texts=[train_anchors[i], train_positives[i]]) for i in idx
]

train_loader = DataLoader(train_examples, batch_size=min(64, max(1, len(train_examples))), shuffle=True, drop_last=True)


# ----------------------------
# 3) Model + Loss (MNRL)
# ----------------------------
model = SentenceTransformer("pankajrajdeo/bond-embed-v1")
loss = losses.MultipleNegativesRankingLoss(model=model, scale=20.0)


# ----------------------------
# 4) Validation evaluator (IR) from validation split
# ----------------------------
def build_ir(split_anchors, split_positives, split_ids, name: str):
    queries = {f"q{i}": a for i, a in enumerate(split_anchors)}
    corpus = {}
    for pid, pos in zip(split_ids, split_positives):
        if pid and pid not in corpus:
            corpus[pid] = pos
    relevant = {f"q{i}": {pid} if pid else set() for i, pid in enumerate(split_ids)}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name=name,
        show_progress_bar=False,
    )

val_evaluator = build_ir(val_anchors, val_positives, val_ids, name="bond_benchmark_val")


# ----------------------------
# 5) Train with model.fit
# ----------------------------
warmup_steps = int(0.1 * (len(train_loader)))
model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=2,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": 2e-5},
    evaluator=val_evaluator,
    evaluation_steps=max(100, len(train_loader)//4),
    show_progress_bar=True,
    output_path="mnrl-bond-embed",
)

# ----------------------------
# 6) Final save
# ----------------------------
model.save("mnrl-bond-embed-final")
