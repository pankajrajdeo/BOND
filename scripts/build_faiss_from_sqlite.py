#!/usr/bin/env python3
"""
Build FAISS index from ontology_terms (label+synonyms+definition), skipping obsolete terms.
- Embedding text per term:
  "label: {LABEL}; synonyms: {EXACT | NARROW | BROAD | RELATED}; definition: {SHORT_DEF}"
- IDs are stored in id_map.npy; FAISS is binary sign + int8 rescoring over L2-normalized vectors.
"""
import argparse, os, sqlite3, json, numpy as np, logging, warnings, io
from tqdm import tqdm
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import faiss

# Optional env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Your embedding plumbing (unchanged)
from bond.runtime_env import configure_runtime
from bond.providers import resolve_embeddings
from bond.models import EmbeddingSignature, IndexMetadata

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def generate_signature(embed_model: str, embed_fn) -> dict:
    t = "the quick brown fox jumps over the lazy dog"
    v = embed_fn([t])[0]
    return EmbeddingSignature(model_id=embed_model, dimension=len(v), anchor_text=t, anchor_vector=v).model_dump()

def embed_resilient(batch_texts, batch_ids, embed_fn):
    try:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            out = embed_fn(batch_texts)
        if not isinstance(out, list) or len(out)!=len(batch_texts):
            raise RuntimeError("Embedding provider returned unexpected shape")
        return out
    except Exception as e:
        if len(batch_texts)==1:
            logger.error(f"Skipping malformed text for id={batch_ids[0]} due to error: {e}")
            return [None]
        mid = len(batch_texts)//2
        return embed_resilient(batch_texts[:mid], batch_ids[:mid], embed_fn) + \
               embed_resilient(batch_texts[mid:], batch_ids[mid:], embed_fn)

def positive_str(label, sx, sn, sb, sr, definition):
    def split(s): return [x for x in (s or "").split("|") if x]
    # Order: exact → narrow → broad → related
    syn_all = split(sx) + split(sn) + split(sb) + split(sr)
    syn_field = " | ".join(syn_all[:50]) if syn_all else ""
    def_short = (definition or "")[:500]
    parts = [f"label: {label}"]
    if syn_field: parts.append(f"synonyms: {syn_field}")
    if def_short: parts.append(f"definition: {def_short}")
    return "; ".join(parts)

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def build_faiss_from_sqlite(sqlite_path: str, embed_model: str, assets_path: str, faiss_rebuild: bool=False):
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite not found: {sqlite_path}")

    conn = sqlite3.connect(sqlite_path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ontology_terms'")
    if not cur.fetchone():
        raise ValueError("Missing table 'ontology_terms'")

    batch = int(os.getenv("BOND_EMB_BATCH","16"))
    embed_fn = resolve_embeddings(embed_model, batch_size=batch)

    store_path = os.path.join(assets_path,"faiss_store")
    os.makedirs(store_path, exist_ok=True)
    sig_path    = os.path.join(store_path,"embedding_signature.json")
    faiss_path  = os.path.join(store_path,"embeddings.faiss")       # binary
    id_map_path = os.path.join(store_path,"id_map.npy")             # curies
    rescore_path = os.path.join(store_path,"rescore_vectors.npy")   # int8
    meta_path   = os.path.join(store_path,"meta.json")

    signature = generate_signature(embed_model, embed_fn)

    need_full = faiss_rebuild or (not os.path.exists(faiss_path))
    if not need_full and os.path.exists(sig_path):
        try:
            with open(sig_path,"r") as f: old = json.load(f)
            if old.get("model_id")!=signature.get("model_id") or int(old.get("dimension",0))!=int(signature.get("dimension",0)):
                logging.warning("Embedding model/dimension changed; rebuilding FAISS.")
                need_full = True
        except Exception:
            need_full = True

    # Pull text (skip obsolete)
    logger.info("Fetching ontology texts (skipping obsolete) ...")
    cur.execute("""
        SELECT curie, label, definition, synonyms_exact, synonyms_narrow, synonyms_broad, synonyms_related, is_obsolete
        FROM ontology_terms
        WHERE label IS NOT NULL AND label != ''
        ORDER BY curie
    """)
    rows = cur.fetchall()

    all_ids, all_texts = [], []
    for curie, label, definition, sx, sn, sb, sr, is_obs in rows:
        if int(is_obs or 0) == 1:
            continue
        txt = positive_str(label, sx, sn, sb, sr, definition)
        if txt.strip():
            all_ids.append(curie)
            all_texts.append(txt)
    if not all_ids:
        logger.error("No indexable (non-obsolete) terms found.")
        return

    # Incremental?
    work_ids, work_texts = all_ids, all_texts
    if not need_full and os.path.exists(id_map_path):
        try:
            existing_ids = set(str(x) for x in np.load(id_map_path, allow_pickle=False).tolist())
        except Exception:
            existing_ids = set()
        mask = [i not in existing_ids for i in all_ids]
        work_ids   = [i for i,m in zip(all_ids,mask) if m]
        work_texts = [t for t,m in zip(all_texts,mask) if m]
        if not work_ids:
            with open(sig_path,"w") as f: json.dump(signature,f,indent=2)
            logger.info("No new terms to embed. Done.")
            return

    logger.info(f"Encoding {len(work_texts)} docs with {embed_model} ...")
    embs, failed = [], []
    for i in tqdm(range(0, len(work_texts), batch), desc="Embedding"):
        bt = work_texts[i:i+batch]; bi = work_ids[i:i+batch]
        if not bt: continue
        out = embed_resilient(bt, bi, embed_fn)
        if any(e is None for e in out):
            failed.extend([bi[j] for j,e in enumerate(out) if e is None])
        embs.extend(out)
    ok_embs = [e for e in embs if e is not None]
    ok_ids  = [id_ for j,id_ in enumerate(work_ids) if embs[j] is not None]
    if not ok_embs:
        logger.error("No embeddings produced."); return

    embs_fp32 = np.asarray(ok_embs, dtype=np.float32)
    embs_fp32 = l2_normalize(embs_fp32)
    d = embs_fp32.shape[1]

    def write_full(ids_list, fp32):
        bin_bytes = np.packbits((fp32 >= 0).astype(np.uint8), axis=-1)
        index_bin = faiss.IndexBinaryFlat(d)
        index_bin.add(bin_bytes)
        faiss.write_index_binary(index_bin, faiss_path)

        embs_int8 = np.clip(fp32*127, -127, 127).astype(np.int8)

        max_len = max((len(x) for x in ids_list), default=1)
        np.save(id_map_path, np.array(ids_list, dtype=f"<U{max_len}"))
        np.save(rescore_path, embs_int8)

        with open(sig_path,"w") as f: json.dump(signature,f,indent=2)
        meta = IndexMetadata(
            profile="faiss_store",
            method="binary+int8_rescore",
            embedding_model=embed_model,
            normalize=True,
            dimension=int(d),
            notes="Binary sign for candidates; int8 dot re-score on L2-normalized vectors.",
            created_at=datetime.now().isoformat()
        )
        with open(meta_path,"w") as f: json.dump(meta.model_dump(), f, indent=2)
        logger.info(f"✅ FAISS store written: {store_path}")

    if need_full:
        write_full(ok_ids, embs_fp32)
        conn.close(); return

    # append
    index = faiss.read_index_binary(faiss_path)
    try:
        rescore_existing = np.load(rescore_path, mmap_mode=None)
    except Exception:
        rescore_existing = None
    try:
        old_ids_list = [str(x) for x in np.load(id_map_path, allow_pickle=False).tolist()]
    except Exception:
        old_ids_list = []

    new_pairs = [(i,e) for i,e in zip(ok_ids, embs_fp32) if i not in set(old_ids_list)]
    if not new_pairs:
        with open(sig_path,"w") as f: json.dump(signature,f,indent=2)
        logger.info("No new terms to append. Done."); conn.close(); return

    new_ids = [p[0] for p in new_pairs]
    new_fp32 = np.asarray([p[1] for p in new_pairs], dtype=np.float32)
    new_bin = np.packbits((new_fp32 >= 0).astype(np.uint8), axis=-1)
    index.add(new_bin)
    faiss.write_index_binary(index, faiss_path)

    new_int8 = np.clip(new_fp32*127, -127, 127).astype(np.int8)
    if rescore_existing is not None and rescore_existing.size>0:
        rescore_concat = np.concatenate([rescore_existing, new_int8], axis=0)
    else:
        rescore_concat = new_int8
    np.save(rescore_path, rescore_concat)

    all_ids_list = old_ids_list + new_ids
    max_len = max((len(x) for x in all_ids_list), default=1)
    np.save(id_map_path, np.array(all_ids_list, dtype=f"<U{max_len}"))

    with open(sig_path,"w") as f: json.dump(signature,f,indent=2)
    logger.info(f"✅ Appended {len(new_ids)} new vectors to FAISS store")
    conn.close()

def main():
    configure_runtime()
    ap = argparse.ArgumentParser(description="Build FAISS from ontology_terms (label+synonyms+definition), skipping obsolete.")
    ap.add_argument("--sqlite_path", default=os.getenv("BOND_SQLITE_PATH") or "/Users/rajlq7/Downloads/Terms/BOND/assets/ontologies.sqlite")
    ap.add_argument("--assets_path", default=os.getenv("BOND_ASSETS_PATH") or "assets")
    ap.add_argument("--embed_model", default=os.getenv("BOND_EMBED_MODEL") or "ollama/rajdeopankaj/bond-embed-v1-fp16:latest")
    ap.add_argument("--faiss_rebuild", action="store_true", default=False)
    args = ap.parse_args()

    os.makedirs(args.assets_path, exist_ok=True)
    logger.info(f"🧮 SQLite: {args.sqlite_path}")
    logger.info(f"📁 Output: {args.assets_path}")
    logger.info(f"🤖 Model : {args.embed_model}")
    logger.info(f"📊 Batch : {os.getenv('BOND_EMB_BATCH','16')}")
    build_faiss_from_sqlite(args.sqlite_path, args.embed_model, args.assets_path, faiss_rebuild=args.faiss_rebuild)
    logger.info("✅ Done.")

if __name__ == "__main__":
    main()
