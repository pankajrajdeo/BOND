import os
import sqlite3
import threading
import numpy as np
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from .config import BondSettings
from .providers import resolve_embeddings, ChatLLM
from .retrieval.bm25_sqlite import search_exact, search_bm25
from .retrieval.faiss_store import FaissStore
from .fusion import rrf_fuse
from .prompts import QUERY_EXPANSION_PROMPT, DISAMBIGUATION_PROMPT
from .llm import extract_json_block
from .validate_signature import validate_embedding_signature
from .logger import logger

# use the shared configured logger from bond.logger

def _normalize(s: str) -> str:
    return " ".join(s.lower().split())

def _uniq(seq: List) -> List:
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def _tokenize_bio(text: Optional[str]) -> List[str]:
    if not text:
        return []
    tokens = []
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) >= 3:
            tokens.append(tok)
    return tokens

class BondMatcher:
    def __init__(self, settings: Optional[BondSettings] = None):
        self.cfg = settings or BondSettings()
        logger.info("Initializing BOND")
        
        # ✅ FIX: Don't create connection here - create per-thread with proper URI
        self.db_path = self.cfg.sqlite_path or os.path.join(self.cfg.assets_path, "ontology.sqlite")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        logger.info("✅ BOND matcher initialized (connections created per-thread)")
        
        # Thread-safe connection method
        self._connection_pool = {}
        self._connection_lock = threading.Lock()
        
        # ✅ FIX: Thread-safe connection with proper URI and pragmas
        def get_connection():
            thread_id = threading.get_ident()
            with self._connection_lock:
                if thread_id not in self._connection_pool:
                    conn = sqlite3.connect(
                        f"file:{self.db_path}?mode=ro&immutable=1", 
                        uri=True, 
                        check_same_thread=False
                    )
                    conn.row_factory = sqlite3.Row
                    # Set performance pragmas
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA journal_mode=OFF")
                    conn.execute("PRAGMA query_only=ON")
                    self._connection_pool[thread_id] = conn
                return self._connection_pool[thread_id]
        
        self.get_connection = get_connection
        
        # Basic metrics
        self._start_time = datetime.now()
        self._request_count = 0
        
        # Single FAISS store
        self.faiss = FaissStore(self.cfg.assets_path, self.cfg.rescore_multiplier)
        # Embedding model auto-resolved (supports litellm style strings)
        self.embed_fn = resolve_embeddings(self.cfg.embed_model)
        # Validate embedding signature
        validate_embedding_signature(self.faiss.signature_path, self.embed_fn)
        logger.info("✅ Embedding signature validated successfully.")
        
        # Initialize separate LLMs (required)
        if not self.cfg.expansion_llm_model or not self.cfg.disambiguation_llm_model:
            raise RuntimeError("BOND_EXPANSION_LLM and BOND_DISAMBIGUATION_LLM must be set")
        self.expansion_llm = ChatLLM(self.cfg.expansion_llm_model)
        self.disamb_llm = ChatLLM(self.cfg.disambiguation_llm_model)
        logger.info(f"✅ Expansion LLM: {self.cfg.expansion_llm_model}")
        logger.info(f"✅ Disambiguation LLM: {self.cfg.disambiguation_llm_model}")
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        logger.info("✅ ThreadPoolExecutor initialized for parallel retrieval")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.close()

    def close(self):
        """Explicit cleanup method for resources"""
        # ✅ FIX: Clean up all thread connections
        if hasattr(self, '_connection_pool'):
            try:
                for conn in self._connection_pool.values():
                    conn.close()
                self._connection_pool.clear()
                logger.info("✅ All SQLite connections closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing SQLite connections: {e}")
        
        if hasattr(self, 'executor'):
            try:
                self.executor.shutdown(wait=True)  # ✅ FIX: Wait for pending tasks to complete
                logger.info("✅ ThreadPoolExecutor shutdown (waited for completion)")
            except Exception as e:
                logger.warning(f"⚠️ Error shutting down executor: {e}")

    def __del__(self):
        """Fallback cleanup - not guaranteed to be called"""
        self.close()

    def _dense_search_batch(self, queries: List[str], k: Optional[int] = None) -> List[str]:
        """Batch version of dense search for better performance"""
        if not queries:
            return []
        
        # Batch embed all queries at once
        embeddings = self.embed_fn(queries)
        query_vectors = np.array(embeddings, dtype=np.float32)
        
        # Batch search
        search_k = k if k is not None else self.cfg.topk_dense
        batch_results = self.faiss.search(query_vectors, search_k)
        
        # Flatten results from all queries
        all_ids = []
        for query_results in batch_results:
            all_ids.extend(query_results)
        
        return all_ids

    def get_available_ontologies(self) -> List[str]:
        """Returns a list of all unique ontology sources available in the index."""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT source FROM {self.cfg.table_terms} ORDER BY source")
        sources = [row[0] for row in cur.fetchall()]
        return sources

    # Cache removed entirely

    # Cache removed entirely

    def get_expansion_cache_stats(self) -> Dict[str, Any]:
        return {"cache_size": 0, "cache_hits": 0, "cache_misses": 0, "hit_rate": 0.0}

    def clear_expansion_cache(self) -> None:
        logger.info("Cache disabled; nothing to clear")

    def query(
        self,
        query: str,
        field_name: Optional[str] = None,
        dataset_description: Optional[str] = None,
        n_expansions: Optional[int] = None,
        topk_final: Optional[int] = None,
        return_trace: Optional[bool] = None,
        restrict_to_ontologies: Optional[List[str]] = None,
        topk_exact: Optional[int] = None,
        topk_bm25: Optional[int] = None,
        topk_dense: Optional[int] = None,
        rrf_k: Optional[float] = None,
        num_choices: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Increment request count for metrics
        self._request_count += 1
        
        # Direct compute (no caching)
        return self._query_internal(
            query, field_name, dataset_description, n_expansions,
            topk_final, restrict_to_ontologies,
            topk_exact, topk_bm25, topk_dense, rrf_k, num_choices
        )

    def _query_internal(
        self,
        query: str,
        field_name: Optional[str] = None,
        dataset_description: Optional[str] = None,
        n_expansions: Optional[int] = None,
        topk_final: Optional[int] = None,
        restrict_to_ontologies: Optional[List[str]] = None,
        topk_exact: Optional[int] = None,
        topk_bm25: Optional[int] = None,
        topk_dense: Optional[int] = None,
        rrf_k: Optional[float] = None,
        num_choices: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        
        logger.info(f"Processing query: '{query}'")
        if restrict_to_ontologies:
            logger.info(f"Restricting search to ontologies: {restrict_to_ontologies}")
        
        # Override config with per-request parameters
        n_expansions = n_expansions if n_expansions is not None else cfg.n_expansions
        topk_final = topk_final if topk_final is not None else cfg.topk_final
        topk_exact_v = topk_exact if topk_exact is not None else cfg.topk_exact
        topk_bm25_v = topk_bm25 if topk_bm25 is not None else cfg.topk_bm25
        topk_dense_v = topk_dense if topk_dense is not None else cfg.topk_dense
        rrf_k_v = rrf_k if rrf_k is not None else cfg.rrf_k
        
        # Use provided filter or fall back to config default
        ontology_filter = restrict_to_ontologies if restrict_to_ontologies is not None else cfg.restrict_to_ontologies
        
        trace = {"query": query, "expansions": [], "candidates": {}, "fusion": [], "disambiguation": {}}

        queries = [query]
        context_terms: List[str] = []
        if cfg.enable_expansion and n_expansions > 0:
            logger.info(f"Generating {n_expansions} query expansions...")
            # ✅ FIX: Use the resolved n_expansions, not the global config
            n_expansions = n_expansions or cfg.n_expansions
            prompt = QUERY_EXPANSION_PROMPT.format(n=n_expansions, query=query, field_name=field_name or "N/A", dataset_description=dataset_description or "N/A")
            
            # Generate expansions via Expansion LLM (no non-LLM mode)
            exp_text = self.expansion_llm.text(prompt, temperature=0)
            data = extract_json_block(exp_text) or {}
            expansions = [ln.strip() for ln in (data.get("expansions") or []) if ln.strip()]
            context_terms = [t.strip() for t in (data.get("context_terms") or []) if t.strip()][:5]
            logger.info(f"Generated {len(expansions)} expansions and {len(context_terms)} context terms")
            
            queries.extend(expansions[:n_expansions])
            trace["expansions"] = expansions[:n_expansions]
            if context_terms:
                trace["context_terms"] = context_terms
            logger.info(f"Total expansions: {len(expansions[:n_expansions])}")
        
        q_norms = list({_normalize(q) for q in queries})
        logger.info(f"Processing {len(queries)} queries (including {len(q_norms)} unique normalized variants)")
        
        # --- Parallel Retrieval ---
        logger.info("Starting parallel retrieval across exact, BM25, and vector search...")
        # Base and context channels
        bm25_queries_base = list(queries)
        bm25_queries_ctx = []
        if context_terms:
            for tok in context_terms[:5]:
                bm25_queries_ctx.append(f"{query} {tok}")

        exact_future = self.executor.submit(
            lambda: [h["id"] for h in search_exact(self.get_connection(), cfg.table_terms, q_norms, topk_exact_v, sources=ontology_filter)]
        )
        bm25_base_future = self.executor.submit(
            lambda: [h["id"] for q in bm25_queries_base for h in search_bm25(self.get_connection(), cfg.table_terms, cfg.table_terms_fts, q, topk_bm25_v, sources=ontology_filter)]
        )
        bm25_ctx_future = self.executor.submit(
            lambda: [h["id"] for q in bm25_queries_ctx for h in search_bm25(self.get_connection(), cfg.table_terms, cfg.table_terms_fts, q, topk_bm25_v, sources=ontology_filter)]
        ) if bm25_queries_ctx else None

        # Dense base: batch queries
        dense_base_future = self.executor.submit(lambda: self._dense_search_batch(queries, k=topk_dense_v * 2))
        # Dense ctx: a single composed query
        if context_terms:
            ctx_phrase = " ".join(context_terms[:3])
            dense_ctx_future = self.executor.submit(
                lambda: self.faiss.search(np.array(self.embed_fn([f"{query} {ctx_phrase}"]), dtype=np.float32), topk_dense_v)[0]
            )
        else:
            dense_ctx_future = None
        
        exact_ids = _uniq(exact_future.result())
        bm25_base_ids = _uniq(bm25_base_future.result())
        bm25_ctx_ids = _uniq(bm25_ctx_future.result()) if bm25_ctx_future else []
        dense_base_ids = _uniq(dense_base_future.result())
        dense_ctx_ids = _uniq(dense_ctx_future.result()) if dense_ctx_future else []
        
        # Post-filter FAISS results if ontology filter is applied
        if ontology_filter and (dense_base_ids or dense_ctx_ids):
            # Validate dense_base_ids
            cur = self.get_connection().cursor()
            if dense_base_ids:
                qmarks = ",".join("?" for _ in dense_base_ids)
            cur.execute(f"SELECT id FROM {cfg.table_terms} WHERE id IN ({qmarks}) AND source IN ({','.join('?' for _ in ontology_filter)})", 
                           dense_base_ids + ontology_filter)
            valid_dense_ids = {row[0] for row in cur.fetchall()}
            dense_base_ids = [id_ for id_ in dense_base_ids if id_ in valid_dense_ids]
            # Validate dense_ctx_ids
            if dense_ctx_ids:
                qmarks2 = ",".join("?" for _ in dense_ctx_ids)
                cur.execute(f"SELECT id FROM {cfg.table_terms} WHERE id IN ({qmarks2}) AND source IN ({','.join('?' for _ in ontology_filter)})", 
                           dense_ctx_ids + ontology_filter)
                valid_dense_ids2 = {row[0] for row in cur.fetchall()}
                dense_ctx_ids = [id_ for id_ in dense_ctx_ids if id_ in valid_dense_ids2]
        
        trace["candidates"] = {
            "exact": exact_ids,
            "bm25_base": bm25_base_ids,
            "bm25_ctx": bm25_ctx_ids,
            "dense_base": dense_base_ids,
            "dense_ctx": dense_ctx_ids,
        }
        if ontology_filter:
            trace["ontology_filter"] = ontology_filter

        # --- Fusion ---
        logger.info("Fusing results from base/context channels...")
        # Dynamic context weights without exposing to users
        ctx_strength = min(1.0, len(context_terms) / 3.0) if context_terms else 0.0
        weights = dict(cfg.rrf_weights)
        # If exact matches exist, upweight exact slightly for this query
        if exact_ids:
            weights["exact"] = weights.get("exact", 1.0) * 1.2
        weights.update({
            "bm25_base": weights.get("bm25", 0.8),
            "bm25_ctx": weights.get("bm25", 0.8) * (1.2 + 0.3 * ctx_strength),
            "dense_base": weights.get("dense", 0.6),
            "dense_ctx": weights.get("dense", 0.6) * (1.2 + 0.3 * ctx_strength),
        })
        fused = rrf_fuse({
            "exact": exact_ids,
            "bm25_base": bm25_base_ids,
            "bm25_ctx": bm25_ctx_ids,
            "dense_base": dense_base_ids,
            "dense_ctx": dense_ctx_ids,
        }, k=rrf_k_v, weights=weights)[:topk_final]
        trace["fusion"] = fused
        logger.info(f"Fusion complete: {len(fused)} candidates ranked by RRF")
        
        ids = [fid for fid, _ in fused]
        if not ids:
            logger.warning("No candidates found after fusion")
            return {"results": [], "chosen": None}
        
        # --- Metadata Hydration ---
        logger.info("Hydrating metadata for ranked candidates...")
        cur = self.get_connection().cursor()
        qmarks = ",".join("?" for _ in ids)
        cur.execute(f"SELECT id, label, definition, source, iri, syn_exact, syn_related, syn_broad, syn_generic, alt_ids, xrefs, namespace, subsets, comments, parents_is_a, abstracts FROM {cfg.table_terms} WHERE id IN ({qmarks})", ids)
        meta = {}
        for row in cur.fetchall():
            (_id, _label, _definition, _source, _iri, _sx, _sr, _sb,
             _sg, _alt, _xr, _ns, _ss, _cm, _pa, _ab) = row
            try:
                syn_exact = json.loads(_sx) if _sx else None
            except Exception:
                syn_exact = None
            try:
                syn_related = json.loads(_sr) if _sr else None
            except Exception:
                syn_related = None
            try:
                syn_broad = json.loads(_sb) if _sb else None
            except Exception:
                syn_broad = None
            def _json_list(x):
                try:
                    return json.loads(x) if x else None
                except Exception:
                    return None
            syn_generic = _json_list(_sg)
            alt_ids = _json_list(_alt)
            xrefs = _json_list(_xr)
            namespace = _json_list(_ns)
            subsets = _json_list(_ss)
            comments = _json_list(_cm)
            parents_is_a = _json_list(_pa)
            abstracts = _json_list(_ab)
            meta[_id] = {
                "label": _label,
                "definition": _definition,
                "source": _source,
                "iri": _iri,
                "synonyms_exact": syn_exact,
                "synonyms_related": syn_related,
                "synonyms_broad": syn_broad,
                "synonyms_generic": syn_generic,
                "alt_ids": alt_ids,
                "xrefs": xrefs,
                "namespace": namespace,
                "subsets": subsets,
                "comments": comments,
                "parents_is_a": parents_is_a,
                "abstracts": abstracts,
            }
        
        ranked = []
        for fid, fscore in fused:
            if fid in meta:
                ranked.append({"id": fid, **meta[fid], "fusion_score": fscore})
        
        logger.info(f"Metadata hydration complete: {len(ranked)} candidates with full information")

        # Simplified re-ranking: Trust RRF fusion + exact matches + LLM disambiguation
        # Remove complex heuristic boosts that duplicate LLM capabilities

        # Exact-match bonus: if exact channel returned hits, reward them SIGNIFICANTLY
        if exact_ids:
            exact_set = set(exact_ids)
            for r in ranked:
                if r["id"] in exact_set:
                    # Large bonus to ensure exact matches always rank highest
                    r["fusion_score"] += 10.0  # Much larger than any other boost
            ranked.sort(key=lambda x: x["fusion_score"], reverse=True)

            # Context filter removed: LLM disambiguation already handles dataset context appropriately
            # Exact matches and high-ranking biological terms should never be filtered by keyword matching

        # Calculate confidence scores
        if ranked:
            # Normalize fusion scores to [0,1] range
            max_score = max(r["fusion_score"] for r in ranked)
            min_score = min(r["fusion_score"] for r in ranked)
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            for r in ranked:
                # Technical name: retrieval_confidence (normalized RRF fusion score in [0,1])
                r["retrieval_confidence"] = (r["fusion_score"] - min_score) / score_range
        
        # --- Rule-based pre-rerank to improve biological accuracy (ontology-agnostic) ---
        # Build head tokens from query and expansions
        head_tokens = set(_tokenize_bio(query))
        if isinstance(trace, dict):
            for exp in trace.get("expansions", [])[:5]:
                head_tokens.update(_tokenize_bio(exp))
        # Context terms (LLM-generated) are supportive
        ctx_tokens = set(_tokenize_bio(" ".join(context_terms or [])))

        def _score_evidence(r: Dict[str, Any]) -> float:
            hay = " ".join([
                r.get("label") or "",
                r.get("definition") or "",
                " ".join(r.get("synonyms_exact") or []),
                " ".join(r.get("synonyms_related") or []),
                " ".join(r.get("synonyms_broad") or []),
            ]).lower()
            ht_hits = sum(1 for t in head_tokens if t in hay)
            ct_hits = sum(1 for t in ctx_tokens if t in hay)
            # Give higher weight to head tokens; context terms are supportive
            return 0.5 * ht_hits + 0.2 * ct_hits

        if ranked:
            for r in ranked:
                r["fusion_score"] += _score_evidence(r)
            ranked.sort(key=lambda x: x["fusion_score"], reverse=True)
            # Recompute retrieval_confidence after evidence boosts
            max_score = max(r["fusion_score"] for r in ranked)
            min_score = min(r["fusion_score"] for r in ranked)
            span = max_score - min_score if max_score != min_score else 1.0
            for r in ranked:
                r["retrieval_confidence"] = (r["fusion_score"] - min_score) / span

        # --- LLM Disambiguation ---
        def _describe_candidate(r: Dict[str, Any]) -> str:
            definition = (r.get("definition") or "").replace("\n", " ").strip()
            ex = ", ".join((r.get("synonyms_exact") or [])[:8])
            rel = ", ".join((r.get("synonyms_related") or [])[:8])
            br = ", ".join((r.get("synonyms_broad") or [])[:8])
            gen = ", ".join((r.get("synonyms_generic") or [])[:6])
            parent = ", ".join((r.get("parents_is_a") or [])[:5])
            comments = ", ".join((r.get("comments") or [])[:3])
            parts = [
                f"Definition: {definition}" if definition else "Definition: N/A",
                f"Exact: {ex}" if ex else "Exact: N/A",
                f"Related: {rel}" if rel else "Related: N/A",
                f"Broad: {br}" if br else "Broad: N/A",
            ]
            if gen:
                parts.append(f"Synonyms: {gen}")
            if parent:
                parts.append(f"Parents: {parent}")
            if comments:
                parts.append(f"Comments: {comments}")
            return " | ".join(parts)

        # Trust fusion ranking - send top-K candidates to LLM without additional filtering
        # LLM is sophisticated enough to identify relevant vs irrelevant candidates
        llm_pool = ranked[:topk_final]

        candidates_block = "\n".join(
            f"{r['id']} | {r['label']} | {r['source']} | {r.get('retrieval_confidence', 0):.3f} | " + _describe_candidate(r)
            for r in llm_pool
        )
        prompt = DISAMBIGUATION_PROMPT.format(query=query, field_name=field_name or "N/A", dataset_description=dataset_description or "N/A", candidates_block=candidates_block)
        
        # Try LLM disambiguation with retry and fallback
        logger.info("Starting LLM disambiguation...")
        raw = self.disamb_llm.text(prompt, temperature=0.0, max_tokens=384)
        choice = extract_json_block(raw)
        if not choice or "chosen_id" not in choice:
            # one retry with explicit JSON instruction
            logger.warning("LLM response parsing failed, retrying with explicit JSON instruction...")
            raw = self.disamb_llm.text(prompt + "\n\nReturn ONLY valid JSON.", temperature=0.0, max_tokens=384)
            choice = extract_json_block(raw)
        chosen_id = choice.get("chosen_id") if isinstance(choice, dict) else None
        llm_conf = choice.get("llm_confidence") if isinstance(choice, dict) else None
        # Normalize llm_confidence to a float in [0,1], default to 0.0 when missing/invalid
        try:
            llm_conf = float(llm_conf)
            if llm_conf < 0.0:
                llm_conf = 0.0
            if llm_conf > 1.0:
                llm_conf = 1.0
        except Exception:
            llm_conf = 0.0
        if not chosen_id:
            # fallback to LLM pool top-1 (already re-ranked and head-token filtered)
            fallback_pool = llm_pool or ranked
            if fallback_pool:
                chosen_id = fallback_pool[0]["id"]
                choice = {"chosen_id": chosen_id, "reason": "LLM disambiguation failed, using pool top-1"}
                logger.warning("LLM disambiguation failed, using pool top-1 result")
        logger.info(f"LLM disambiguation completed: {choice.get('reason', 'No reason provided')}")
        
        # Attach LLM reason and confidence to chosen object for client visibility
        chosen_obj = next((r for r in ranked if r["id"] == chosen_id), None)
        if chosen_obj is not None:
            reason_text = choice.get("reason") if isinstance(choice, dict) else None
            payload = {
                "llm_confidence": llm_conf
            }
            if reason_text:
                payload["reason"] = reason_text
            chosen_obj = {**chosen_obj, **payload}

        # LLM is the final selector; no post-LLM algorithmic overrides
        request_n = max(0, int(num_choices or 0))

        # Build alternatives when requested (exclude chosen)
        alternatives = None
        if request_n > 0:
            alts = [r for r in ranked if r["id"] != (chosen_obj["id"] if chosen_obj else None)]
            alternatives = alts[:request_n]

        out = {"results": ranked, "chosen": chosen_obj, "alternatives": alternatives or []}
        trace["disambiguation"] = {"chosen_id": chosen_id, "reason": choice.get("reason")}
        out["trace"] = trace
        
        logger.info(f"Query processing complete. Chosen: {chosen_id}")
        return out
    
    # ✅ FIX: A more efficient batch processing method
    def batch_query(self, items: List[Dict[str, Optional[str]]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of queries efficiently by batching each pipeline stage.
        """
        # Increment request count for metrics
        self._request_count += 1
        
        logger.info(f"Processing batch of {len(items)} items with true batch processing...")
        if not items:
            return []

        cfg = self.cfg
        
        # --- Stage 1: Query Expansion (Batched where possible) ---
        # NOTE: True batching here depends on the LLM provider. This implementation
        # still loops, but a production system should use a batch-enabled client.
        queries_map = {} # Maps item_index -> list of its expanded queries
        all_queries = set()
        for i, item in enumerate(items):
            query = item.get("query", "")
            queries_map[i] = {query}
            if cfg.enable_expansion and cfg.n_expansions > 0:
                # ✅ FIX: Use the resolved n_expansions, not the global config
                n_expansions = item.get("n_expansions") or cfg.n_expansions
                prompt = QUERY_EXPANSION_PROMPT.format(
                    n=n_expansions,
                    query=query,
                    field_name=item.get("field_name") or "N/A",
                    dataset_description=item.get("dataset_description") or "N/A",
                )
                try:
                    # Generate new expansions via Expansion LLM
                    exp_text = self.expansion_llm.text(prompt, temperature=0.0)
                    data = extract_json_block(exp_text) or {}
                    expansions = {ln.strip() for ln in (data.get("expansions") or []) if ln.strip()}
                    
                    queries_map[i].update(list(expansions)[:cfg.n_expansions])
                except Exception as e:
                    logger.warning(f"Failed to expand query '{query}': {e}")
            all_queries.update(queries_map[i])

        # --- Stage 2: Embedding (Already batched and efficient) ---
        unique_queries = list(all_queries)
        logger.info(f"Batch embedding {len(unique_queries)} unique queries...")
        embeddings = self.embed_fn(unique_queries)
        embedding_map = {query: vec for query, vec in zip(unique_queries, embeddings)}

        # --- Stage 3: Retrieval (Batched) ---
        # Batch FAISS Search
        logger.info("Batch retrieving from FAISS...")
        query_vectors = np.array([embedding_map[q] for q in unique_queries if q in embedding_map], dtype=np.float32)
        if query_vectors.shape[0] > 0:
            # ✅ FIX: Handle new return type from FaissStore.search
            dense_results_batch = self.faiss.search(query_vectors, cfg.topk_dense)
            # dense_results_batch is now List[List[str]] where each inner list is results for one query
            dense_id_map = {q: results for q, results in zip(unique_queries, dense_results_batch)}
        else:
            dense_id_map = {}

        # NOTE: BM25 FTS5 MATCH does not support batching with an IN clause.
        # This part remains serial per query but is still efficient.
        
        # --- Stage 4 & 5: Per-Item Fusion, Hydration, and Disambiguation ---
        final_results = []
        
        for i, item in enumerate(items):
            try:
                # Collect all candidates for this specific item
                item_queries = list(queries_map[i])
                item_q_norms = {_normalize(q) for q in item_queries}
                
                # ✅ FIX: Apply ontology filtering consistently with single query
                sources = item.get("restrict_to_ontologies") or cfg.restrict_to_ontologies
                
                # Single batch call to search_exact for all normalized queries
                results_exact = search_exact(self.get_connection(), cfg.table_terms, list(item_q_norms), cfg.topk_exact, sources=sources)
                item_exact_ids = [r['id'] for r in results_exact]
                
                item_dense_ids = [id_ for q in item_queries for id_ in dense_id_map.get(q, [])]
                
                # BM25 must be run per query
                item_bm25_ids = [h["id"] for q in item_queries for h in search_bm25(self.get_connection(), cfg.table_terms, cfg.table_terms_fts, q, cfg.topk_bm25, sources=sources)]
                
                # ✅ FIX: Post-filter dense results by source if filter is set
                if sources and item_dense_ids:
                    qmarks = ",".join("?" for _ in item_dense_ids)
                    cur = self.get_connection().cursor()
                    cur.execute(
                        f"SELECT id FROM {cfg.table_terms} WHERE id IN ({qmarks}) AND source IN ({','.join('?' for _ in sources)})",
                        item_dense_ids + sources
                    )
                    valid = {row[0] for row in cur.fetchall()}
                    item_dense_ids = [x for x in item_dense_ids if x in valid]
                
                # Combine all candidates for hydration
                candidate_ids_for_item = set(item_exact_ids) | set(item_dense_ids) | set(item_bm25_ids)
                
                if not candidate_ids_for_item:
                    final_results.append({"results": [], "chosen": None})
                    continue
                
                # Hydrate metadata for this item's candidates
                qmarks = ",".join("?" for _ in candidate_ids_for_item)
                cur = self.get_connection().cursor()
                cur.execute(f"SELECT id, label, definition, source, iri, syn_exact, syn_related, syn_broad FROM {cfg.table_terms} WHERE id IN ({qmarks})", list(candidate_ids_for_item))
                meta = {}
                for row in cur.fetchall():
                    _id, _label, _definition, _source, _iri, _sx, _sr, _sb = row
                    try:
                        syn_exact = json.loads(_sx) if _sx else None
                    except Exception:
                        syn_exact = None
                    try:
                        syn_related = json.loads(_sr) if _sr else None
                    except Exception:
                        syn_related = None
                    try:
                        syn_broad = json.loads(_sb) if _sb else None
                    except Exception:
                        syn_broad = None
                    meta[_id] = {
                        "label": _label,
                        "definition": _definition,
                        "source": _source,
                        "iri": _iri,
                        "synonyms_exact": syn_exact,
                        "synonyms_related": syn_related,
                        "synonyms_broad": syn_broad,
                    }
                
                # Fuse and rank
                fused = rrf_fuse({
                    "exact": _uniq(item_exact_ids),
                    "bm25": _uniq(item_bm25_ids),
                    "dense": _uniq(item_dense_ids)
                }, k=cfg.rrf_k, weights=cfg.rrf_weights)[:cfg.topk_final]

                ranked = []
                for fid, fscore in fused:
                    if fid in meta:
                        ranked.append({"id": fid, **meta[fid], "fusion_score": fscore})

                # Calculate confidence scores
                if ranked:
                    max_score = max(r["fusion_score"] for r in ranked)
                    min_score = min(r["fusion_score"] for r in ranked)
                    score_range = max_score - min_score if max_score != min_score else 1.0
                    for r in ranked:
                        r["retrieval_confidence"] = (r["fusion_score"] - min_score) / score_range

                # LLM Disambiguation (can be batched in future)
                try:
                    def _describe_candidate_local(r: Dict[str, Any]) -> str:
                        definition = (r.get("definition") or "").replace("\n", " ").strip()
                        ex = ", ".join((r.get("synonyms_exact") or [])[:10])
                        rel = ", ".join((r.get("synonyms_related") or [])[:10])
                        br = ", ".join((r.get("synonyms_broad") or [])[:10])
                        gen = ", ".join((r.get("synonyms_generic") or [])[:6])
                        comments = ", ".join((r.get("comments") or [])[:3])
                        parts = [f"Definition: {definition}" if definition else "Definition: N/A",
                                 f"Exact Synonyms: {ex}" if ex else "Exact Synonyms: N/A",
                                 f"Related Synonyms: {rel}" if rel else "Related Synonyms: N/A",
                                 f"Broad Synonyms: {br}" if br else "Broad Synonyms: N/A"]
                        if gen:
                            parts.append(f"Synonyms: {gen}")
                        if comments:
                            parts.append(f"Comments: {comments}")
                        return " | ".join(parts)
                    candidates_block = "\n".join(
                        f"{r['id']} | {r['label']} | {r['source']} | {r.get('retrieval_confidence', 0):.3f} | " + _describe_candidate_local(r)
                        for r in ranked
                    )
                    prompt = DISAMBIGUATION_PROMPT.format(
                        query=item.get("query", ""), 
                        field_name=item.get("field_name") or "N/A", 
                        dataset_description=item.get("dataset_description") or "N/A", 
                        candidates_block=candidates_block
                    )
                    raw = self.disamb_llm.text(prompt, temperature=0.0, max_tokens=384)
                    choice = extract_json_block(raw)
                    chosen_id = choice.get("chosen_id") if isinstance(choice, dict) else None
                    if not chosen_id and ranked:
                        chosen_id = ranked[0]["id"]
                        choice = {"chosen_id": chosen_id, "reason": "LLM disambiguation failed, falling back to top-1"}
                except Exception as e:
                    logger.warning(f"LLM disambiguation failed for query '{item.get('query', '')}': {e}")
                    chosen_id = ranked[0]["id"] if ranked else None
                    choice = {"chosen_id": chosen_id, "reason": "LLM error, falling back to top-1"}

                # Attach reason to chosen record
                chosen = next((r for r in ranked if r["id"] == chosen_id), None)
                if chosen is not None:
                    reason_text = choice.get("reason") if isinstance(choice, dict) else None
                    payload = {}
                    if reason_text:
                        payload["reason"] = reason_text
                    llm_conf = choice.get("llm_confidence") if isinstance(choice, dict) else None
                    if isinstance(llm_conf, (int, float)):
                        try:
                            payload["llm_confidence"] = max(0.0, min(1.0, float(llm_conf)))
                        except Exception:
                            pass
                    if payload:
                        chosen = {**chosen, **payload}
                final_results.append({"results": ranked, "chosen": chosen})

            except Exception as e:
                logger.error(f"Failed to process batch item {i}: {e}")
                final_results.append({"error": str(e), "query": item.get("query"), "results": [], "chosen": None})
                
        logger.info(f"True batch processing complete: {len(final_results)} results generated")
        return final_results
