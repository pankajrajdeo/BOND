import os
import sqlite3
import threading
import numpy as np
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time
from .config import BondSettings
from .providers import resolve_embeddings, ChatLLM
from .retrieval.bm25_sqlite import search_exact, search_bm25
from .retrieval.faiss_store import FaissStore
from .fusion import rrf_fuse, simple_fuse
from .prompts import QUERY_EXPANSION_PROMPT, DISAMBIGUATION_PROMPT, CONTEXT_TERMS_PROMPT
from .models import ExpansionResponse, DisambiguationResponse
from .validate_signature import validate_embedding_signature
from .logger import logger
from .runtime_env import configure_runtime
from .schema_policies import allowed_ontologies_for
from .field_guidance import get_field_guidance
from .rules import should_abstain, context_violation, species_violation, normalize_organism, normalize_marker_suffixes, normalize_cell_type_hyphens, normalize_field_name
from .abbrev import AbbreviationExpander
from .fusion_weights import field_aware_weights
from .graph_utils import compute_graph_neighbors
configure_runtime()

# Helper to parse LLM JSON into a Pydantic model
import json, re
from typing import Type, TypeVar
from pydantic import BaseModel
_T = TypeVar("_T", bound=BaseModel)

def _extract_json_str(text: str):
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return text[s:e+1]
    return None

def parse_llm_json(text: str, model: Type[_T]) -> Optional[_T]:
    js = _extract_json_str(text)
    if not js:
        return None
    try:
        data = json.loads(js)
        return model.model_validate(data)
    except Exception:
        return None


# use the shared configured logger from bond.logger

def _normalize(s: str) -> str:
    return " ".join(s.lower().split())

def _uniq(seq: List) -> List:
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


class BondMatcher:
    def __init__(self, settings: Optional[BondSettings] = None):
        self.cfg = settings or BondSettings()
        logger.info("Initializing BOND")
        
        # Create per-thread SQLite connection using immutable, read-only URI
        self.db_path = self.cfg.sqlite_path or os.path.join(self.cfg.assets_path, "ontologies.sqlite")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        logger.info("BOND matcher initialized (connections created per-thread)")
        
        # Thread-safe connection method
        self._connection_pool = {}
        self._connection_lock = threading.Lock()
        
        # Thread-safe connection with proper URI and pragmas
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
        logger.info("Embedding signature validated successfully.")
        
        # Initialize separate LLMs unless in retrieval-only mode
        self.expansion_llm = None
        self.disamb_llm = None
        if not self.cfg.retrieval_only:
            if not self.cfg.expansion_llm_model or not self.cfg.disambiguation_llm_model:
                raise RuntimeError("BOND_EXPANSION_LLM and BOND_DISAMBIGUATION_LLM must be set (or set BOND_RETRIEVAL_ONLY=1)")
            self.expansion_llm = ChatLLM(self.cfg.expansion_llm_model)
            self.disamb_llm = ChatLLM(self.cfg.disambiguation_llm_model)
            logger.info(f"Expansion LLM: {self.cfg.expansion_llm_model}")
            logger.info(f"Disambiguation LLM: {self.cfg.disambiguation_llm_model}")
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        logger.info("ThreadPoolExecutor initialized for parallel retrieval")

        # Abbreviation expander (field-scoped)
        from .abbrev import AbbreviationExpander
        self.abbrev = AbbreviationExpander(self.cfg.assets_path)
        
        # Load cross-encoder reranker if available
        self.reranker = None
        if self.cfg.reranker_path and os.path.exists(self.cfg.reranker_path):
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(self.cfg.reranker_path)
                logger.info(f"Loaded reranker from {self.cfg.reranker_path}")
            except ImportError:
                logger.warning("sentence-transformers not installed, reranker disabled")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")

    def _rerank_with_cross_encoder(
        self,
        query: str,
        field_name: str,
        tissue: Optional[str],
        organism: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder reranker model.
        
        Args:
            query: Original query text
            field_name: Field type (cell_type, tissue, etc.)
            tissue: Tissue context (optional)
            organism: Organism name
            candidates: List of candidate dictionaries with metadata
            top_k: Number of top candidates to return
            
        Returns:
            Re-ranked list of candidates sorted by reranker score
        """
        if not self.reranker or not candidates:
            return candidates
        
        # Build query text (same format as training)
        query_text = f"{field_name}: {query}; tissue: {tissue or 'unknown'}; organism: {organism}"
        
        # Build (query, candidate) pairs
        pairs = []
        for cand in candidates:
            # Format candidate text (same as training)
            syns = []
            for syn_type in ['synonyms_exact', 'synonyms_related', 'synonyms_broad']:
                syns.extend(cand.get(syn_type, []) or [])
            syns = syns[:10]  # cap at 10 synonyms
            
            definition = (cand.get('definition', '') or '')[:200]
            
            parts = [f"label: {cand.get('label', '')}"]
            if syns:
                parts.append(f"synonyms: {' | '.join(syns)}")
            if definition:
                parts.append(f"definition: {definition}")
            
            cand_text = "; ".join(parts)
            pairs.append([query_text, cand_text])
        
        # Score with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Add reranker scores to candidates and re-sort
        for i, cand in enumerate(candidates):
            cand['reranker_score'] = float(scores[i])
            # Replace fusion_score with reranker_score for reranking
            # (soft boosts will add to this later)
            cand['fusion_score'] = float(scores[i])
        
        # Re-sort by reranker score
        reranked = sorted(candidates, key=lambda x: x['reranker_score'], reverse=True)
        
        logger.info(f"Reranker top-3 scores: {[c['reranker_score'] for c in reranked[:3]]}")
        
        return reranked[:top_k]
    
    def _build_intent_text(
        self,
        query: str,
        field_name: Optional[str],
        organism: Optional[str],
        tissue: Optional[str],
        queries_with_expansions: Optional[List[str]],
        context_terms: Optional[List[str]],
        development_stage: Optional[str] = None,
        disease: Optional[str] = None,
    ) -> str:
        """Construct a deterministic intent string for dense_full embedding.

        This captures base query, field context, organism/tissue/disease/development_stage hints, expansions,
        and derived context terms — without using any LLM.
        """
        parts: list[str] = []
        if field_name:
            parts.append(f"field={field_name}")
        if organism:
            parts.append(f"organism={organism}")
        if tissue:
            parts.append(f"tissue={tissue}")
        if development_stage:
            parts.append(f"development_stage={development_stage}")
        if disease:
            parts.append(f"disease={disease}")
        parts.append(f"query={query}")
        if queries_with_expansions and len(queries_with_expansions) > 1:
            alts = [q for q in queries_with_expansions[1:5] if q and q != query]
            if alts:
                parts.append("alts=" + " | ".join(alts))
        if context_terms:
            parts.append("ctx=" + ", ".join(context_terms[:5]))
        return " ; ".join(parts)

    def _expand_abbreviations(self, text: str, field_name: Optional[str]) -> str:
        try:
            return self.abbrev.expand(text, field_name)
        except Exception:
            return text

    def _derive_context_terms(
        self,
        base_query: str,
        field_name: Optional[str],
        tissue: Optional[str],
        organism: Optional[str],
        development_stage: Optional[str] = None,
        disease: Optional[str] = None,
    ) -> List[str]:
        """Derive up to 5 high-signal context terms using multiple methods.
        Methods:
          - Field-default domain vocabulary (lightweight priors)
          - BM25 over tissue ontologies to extract anatomical tokens (if tissue provided)
          - LLM-based suggestion via CONTEXT_TERMS_PROMPT
        """
        logger.info("Deriving context terms (multi-method)...")
        field_lower = (field_name or "").lower()
        out: List[str] = []

        # 1) Field-default priors
        FIELD_PRIORS: Dict[str, List[str]] = {
            "cell_type": ["lineage", "marker", "population"],
            "tissue": ["region", "layer", "subdivision"],
            "disease": ["pathology", "syndrome", "disorder"],
            "development_stage": ["embryonic", "fetal", "adult"],
            "sex": ["male", "female"],
            "self_reported_ethnicity": ["ancestry", "population", "continental"],
            "assay": ["protocol", "platform", "technology"],
            "organism": ["species", "strain"],
        }
        priors = FIELD_PRIORS.get(field_lower, [])

        # 2) BM25 anatomical tokens from tissue (restricted to tissue ontologies)
        bm25_tokens: List[str] = []
        if tissue:
            try:
                tissue_sources = allowed_ontologies_for("tissue", organism)
                if tissue_sources:
                    # Query BM25 for tissue string
                    rows: List[Dict[str, Any]] = search_bm25(
                        self.get_connection(), self.cfg.table_terms, self.cfg.table_terms_fts,
                        tissue, k=min(10, self.cfg.topk_bm25), sources=tissue_sources
                    )
                    labels = [(r.get("label") or "") for r in rows]
                    # Simple tokenization: alpha tokens >= 4 chars, lowercase
                    import re as _re
                    tokens = []
                    for lab in labels:
                        for tok in _re.split(r"[^A-Za-z0-9]+", lab.lower()):
                            if len(tok) >= 4 and not tok.isdigit():
                                tokens.append(tok)
                    # Frequency rank and take top a few
                    from collections import Counter as _Counter
                    common = [t for t, _ in _Counter(tokens).most_common(8)]
                    bm25_tokens = [t for t in common if t not in (base_query or "").lower()]
            except Exception as e:
                logger.debug(f"BM25 tissue token derivation skipped due to error: {e}")

        # 3) LLM-based context generation
        llm_terms: List[str] = []
        try:
            prompt = CONTEXT_TERMS_PROMPT.format(
                query=base_query or "",
                field_name=field_name or "N/A",
                tissue=tissue or "N/A",
                organism=organism or "N/A",
                development_stage=development_stage or "N/A",
                disease=disease or "N/A",
            )
            llm_out = self.expansion_llm.text(prompt, temperature=0)
            parsed = parse_llm_json(llm_out, ExpansionResponse)
            if parsed and parsed.context_terms:
                llm_terms = [t.strip() for t in parsed.context_terms if t and t.strip()]
        except Exception as e:
            logger.debug(f"LLM-based context term generation failed: {e}")
            # Re-raise rate limit errors so caller can handle retry logic
            error_str = str(e).lower()
            if "rate_limit" in error_str or "ratelimit" in error_str or "429" in str(e):
                raise  # Propagate rate limit errors to caller

        # Combine and truncate to top 5 unique
        combined = _uniq([*llm_terms, *bm25_tokens, *priors])
        if len(combined) > 5:
            combined = combined[:5]

        logger.info(
            f"Context terms derived: total={len(combined)} (llm={len(llm_terms)}, bm25={len(bm25_tokens)}, priors={len(priors)})"
        )
        return combined

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.close()

    def close(self):
        """Explicit cleanup method for resources"""
        # Clean up all thread connections
        if hasattr(self, '_connection_pool'):
            try:
                for conn in self._connection_pool.values():
                    conn.close()
                self._connection_pool.clear()
                logger.info("All SQLite connections closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing SQLite connections: {e}")
        
        if hasattr(self, 'executor'):
            try:
                self.executor.shutdown(wait=True)
                logger.info("ThreadPoolExecutor shutdown (waited for completion)")
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

    def _dense_search_dual(
        self,
        base_queries: List[str],
        ctx_queries: Optional[List[str]],
        k: Optional[int] = None,
    ) -> tuple[List[str], List[str]]:
        """Run a single FAISS search for base + context queries and split results.

        Returns a pair of flattened, deduped ID lists: (dense_ids, dense_ctx_ids).
        """
        bq = base_queries or []
        cq = ctx_queries or []
        if not bq and not cq:
            return [], []

        # Embed all queries at once
        combined = bq + cq
        embeddings = self.embed_fn(combined)
        query_vectors = np.array(embeddings, dtype=np.float32)

        search_k = k if k is not None else self.cfg.topk_dense
        batch_results = self.faiss.search(query_vectors, search_k)

        base_n = len(bq)
        base_results = batch_results[:base_n]
        ctx_results = batch_results[base_n:] if cq else []

        dense_ids: List[str] = []
        for r in base_results:
            dense_ids.extend(r)
        dense_ctx_ids: List[str] = []
        for r in ctx_results:
            dense_ctx_ids.extend(r)

        return _uniq(dense_ids), _uniq(dense_ctx_ids)

    def get_available_ontologies(self) -> List[str]:
        """List unique ontology IDs (new schema)."""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT ontology_id FROM {self.cfg.table_terms} ORDER BY ontology_id")
        sources = [row[0] for row in cur.fetchall()]
        return sources

    

    def query(
        self,
        query: str,
        field_name: str,
        organism: str,
        tissue: str,
        development_stage: str = "",
        disease: str = "",
        n_expansions: Optional[int] = None,
        topk_final: Optional[int] = None,
        return_trace: Optional[bool] = None,
        topk_exact: Optional[int] = None,
        topk_bm25: Optional[int] = None,
        topk_dense: Optional[int] = None,
        rrf_k: Optional[float] = None,
        num_choices: Optional[int] = None,
        exact_only: bool = False,
        graph_depth: Optional[int] = None,
        rerank_after_graph: bool = True,
        cached_expansions: Optional[List[str]] = None,
        cached_context_terms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Increment request count for metrics
        self._request_count += 1
        
        # Direct compute (no caching)
        return self._query_internal(
            query, field_name, organism, tissue, development_stage, disease, n_expansions,
            topk_final, return_trace,
            topk_exact, topk_bm25, topk_dense, rrf_k, num_choices,
            exact_only, graph_depth, rerank_after_graph,
            cached_expansions, cached_context_terms
        )

    def _query_internal(
        self,
        query: str,
        field_name: str,
        organism: str,
        tissue: str,
        development_stage: str = "",
        disease: str = "",
        n_expansions: Optional[int] = None,
        topk_final: Optional[int] = None,
        return_trace: Optional[bool] = None,
        topk_exact: Optional[int] = None,
        topk_bm25: Optional[int] = None,
        topk_dense: Optional[int] = None,
        rrf_k: Optional[float] = None,
        num_choices: Optional[int] = None,
        exact_only: bool = False,
        graph_depth: Optional[int] = None,
        rerank_after_graph: bool = True,
        cached_expansions: Optional[List[str]] = None,
        cached_context_terms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        
        logger.info(f"Processing query: '{query}'")
        timings: Dict[str, float] = {}
        t0 = time.monotonic()

        # Override config with per-request parameters
        n_expansions = n_expansions if n_expansions is not None else cfg.n_expansions
        topk_final = topk_final if topk_final is not None else cfg.topk_final
        topk_exact_v = topk_exact if topk_exact is not None else cfg.topk_exact
        topk_bm25_v = topk_bm25 if topk_bm25 is not None else cfg.topk_bm25
        topk_dense_v = topk_dense if topk_dense is not None else cfg.topk_dense
        rrf_k_v = rrf_k if rrf_k is not None else cfg.rrf_k
        
        # Normalize field and organism inputs to canonical names for routing
        field_name = normalize_field_name(field_name) or field_name
        organism = normalize_organism(organism) or organism

        # Field/organism-driven ontology scoping if not explicitly provided
        auto_scope = allowed_ontologies_for(field_name, organism)
        ontology_filter = auto_scope
        
        # Normalize obvious abbreviations before expansion/retrieval
        _orig_query = query
        logger.info(f"[FLOW] Input query: '{query}'")
        # Normalize cell type names FIRST (e.g., T-cell -> T cell)
        # This must run BEFORE marker suffix normalization to prevent "T-cell" from being treated as "T-"
        if field_name and field_name.lower() == "cell_type":
            query_before = query
            query = normalize_cell_type_hyphens(query) or query
            if query != query_before:
                logger.info(f"[FLOW] After hyphen normalization: '{query_before}' -> '{query}'")
            # Normalize marker suffixes for cell marker expressions (e.g., CD25+)
            query_before = query
            query = normalize_marker_suffixes(query) or query
            if query != query_before:
                logger.info(f"[FLOW] After marker suffix normalization: '{query_before}' -> '{query}'")
        # Apply user-provided abbreviations map
        query_before = query
        query = self._expand_abbreviations(query, field_name)
        if query != query_before:
            logger.info(f"[FLOW] After abbreviation expansion: '{query_before}' -> '{query}'")
        logger.info(f"[FLOW] Final normalized query for LLM: '{query}'")

        trace = {"query": _orig_query, "expansions": [], "candidates": {}, "fusion": [], "disambiguation": {}}
        if query != _orig_query:
            trace["normalized_query"] = query

        queries = [query]
        # Derive context terms first (used to guide expansion and retrieval)
        context_terms: List[str] = self._derive_context_terms(query, field_name, tissue, organism, development_stage, disease)

        # Early abstain short-circuit to avoid unnecessary LLM calls/costs
        _abstain_early, _reason_early = should_abstain(query)
        if _abstain_early:
            logger.info(f"Early abstain on query due to no-value indicator: '{query}'")
            trace = {"query": query, "expansions": [], "candidates": {}, "fusion": [], "disambiguation": {}}
            payload = {"results": [], "chosen": {"reason": _reason_early or f"Query '{query}' marked for abstain"}}
            if return_trace:
                payload["trace"] = trace
            return payload

        timings["normalize+context_ms"] = (time.monotonic() - t0) * 1000
        t_exp = time.monotonic()
        expansions: List[str] = []  # Initialize expansions list
        
        # CRITICAL FIX: Allow cached expansions even in retrieval_only mode
        # This enables: cached expansion + retrieval + reranking WITHOUT disambiguation
        if cached_expansions is not None and n_expansions > 0 and not exact_only:
            # Use cached expansions regardless of retrieval_only setting
            n_expansions = n_expansions or cfg.n_expansions
            logger.info(f"Using {len(cached_expansions)} cached expansions (retrieval_only={cfg.retrieval_only})")
            expansions = cached_expansions[:n_expansions]
            # Merge cached context terms if provided
            if cached_context_terms:
                context_terms = _uniq((context_terms or []) + cached_context_terms)[:5]
                logger.info(f"[FLOW] Using cached context terms: {cached_context_terms}")
            # Add expansions to queries list
            queries.extend(expansions[:n_expansions])
            trace["expansions"] = expansions[:n_expansions]
            if context_terms:
                trace["context_terms"] = context_terms
            logger.info(f"Total cached expansions: {len(expansions[:n_expansions])}")
        elif (not cfg.retrieval_only) and cfg.enable_expansion and n_expansions > 0 and not exact_only:
            # Generate expansions via LLM only if NOT in retrieval_only mode
            n_expansions = n_expansions or cfg.n_expansions
            logger.info(f"Generating {n_expansions} query expansions...")
            guidance = get_field_guidance(field_name)
            guidance_block = ""
            if guidance:
                guidance_block = (
                    "Field-Specific Guidance\n"
                    f"- Semantic Constraints: {guidance.get('semantic_constraints','')}\n"
                    f"- Expansion Focus: {guidance.get('expansion_focus','')}\n"
                    f"- Context Priority: {guidance.get('context_priority','')}\n"
                    f"- Avoid: {guidance.get('avoid','')}\n"
                )
            prompt = QUERY_EXPANSION_PROMPT.format(
                n=n_expansions,
                query=query,
                field_name=field_name or "N/A",
                tissue=tissue or "N/A",
                organism=organism or "N/A",
                    development_stage=development_stage or "N/A",
                    disease=disease or "N/A",
                context_terms=", ".join(context_terms) if context_terms else "[]",
                semantic_constraints=guidance.get('semantic_constraints', ''),
                expansion_focus=guidance.get('expansion_focus', ''),
                context_priority=guidance.get('context_priority', ''),
                avoid=guidance.get('avoid', ''),
            )
            
            # Generate expansions via LLM (resilient)
            try:
                logger.info(f"[FLOW] Calling expansion LLM with query: '{query}'")
                exp_text = self.expansion_llm.text(prompt, temperature=0, max_tokens=1024)
                logger.info(f"[FLOW] Expansion LLM returned response (length: {len(exp_text) if exp_text else 0})")
                parsed = parse_llm_json(exp_text, ExpansionResponse)
                if not parsed:
                    logger.warning(f"[FLOW] Failed to parse expansion LLM response as JSON: {exp_text[:500]}")
                expansions = [ln.strip() for ln in (parsed.expansions if parsed else []) if ln.strip()]
                logger.info(f"[FLOW] Generated {len(expansions)} expansions: {expansions}")
                # Merge any LLM-suggested context terms with derived ones
                llm_ctx = [t.strip() for t in (parsed.context_terms if parsed else []) if t.strip()]
                if llm_ctx:
                    context_terms = _uniq((context_terms or []) + llm_ctx)[:5]
                    logger.info(f"[FLOW] LLM suggested context terms: {llm_ctx}")
            except Exception as e:
                logger.error(f"[FLOW] Expansion LLM FAILED: {e}")
                import traceback
                logger.error(f"[FLOW] Expansion LLM traceback: {traceback.format_exc()}")
                # Re-raise rate limit errors so caller can handle retry logic
                error_str = str(e).lower()
                if "rate_limit" in error_str or "ratelimit" in error_str or "429" in str(e):
                    raise  # Propagate rate limit errors to caller
                # For other errors, continue without expansions
            logger.info(f"Generated {len(expansions)} expansions and {len(context_terms)} context terms")
            
            # Add LLM-generated expansions to queries list
            queries.extend(expansions[:n_expansions])
            trace["expansions"] = expansions[:n_expansions]
            if context_terms:
                trace["context_terms"] = context_terms
            logger.info(f"Total LLM expansions: {len(expansions[:n_expansions])}")
        
        timings["expansion_ms"] = (time.monotonic() - t_exp) * 1000
        
        q_norms = list({_normalize(q) for q in queries})
        logger.info(f"Processing {len(queries)} queries (including {len(q_norms)} unique normalized variants)")
        
        # --- Parallel Retrieval (base + context channels) ---
        logger.info("Starting retrieval across exact, BM25(base/ctx), dense(base), and dense_full(intent)...")
        t_ret = time.monotonic()
        exact_future = self.executor.submit(
            lambda: search_exact(self.get_connection(), cfg.table_terms, cfg.table_terms_fts, q_norms, topk_exact_v, sources=ontology_filter)
        )
        bm25_future = None if (exact_only or topk_bm25_v == 0) else self.executor.submit(
            lambda: [h["id"] for q in queries for h in search_bm25(self.get_connection(), cfg.table_terms, cfg.table_terms_fts, q, topk_bm25_v, sources=ontology_filter)]
        )
        # Context-enhanced BM25 channel: append derived context terms to each query
        ctx_queries: List[str] = []
        if (not exact_only) and context_terms:
            ctx_hint = " ".join(context_terms)
            ctx_queries = [f"{q} {ctx_hint}".strip() for q in queries]

        # Dense base channel (queries + expansions)
        dense_base_future = None if (exact_only or topk_dense_v == 0) else self.executor.submit(
            lambda: self._dense_search_batch(queries, k=topk_dense_v)
        )
        # Dense intent channel (single intent string)
        intent_text = self._build_intent_text(query, field_name, organism, tissue, queries, context_terms, development_stage, disease)
        intent_vec = None
        dense_full_future = None
        if not exact_only and topk_dense_v > 0:
            try:
                intent_vec_np = np.array(self.embed_fn([intent_text]), dtype=np.float32)
                intent_vec = intent_vec_np[0]
                dense_full_future = self.executor.submit(
                    lambda: self.faiss.search(intent_vec_np, topk_dense_v)[0]
                )
            except Exception as e:
                logger.debug(f"Intent channel embedding/search skipped: {e}")
        bm25_ctx_future = None if (exact_only or not ctx_queries or topk_bm25_v == 0) else self.executor.submit(
            lambda: [h["id"] for q in ctx_queries for h in search_bm25(self.get_connection(), cfg.table_terms, cfg.table_terms_fts, q, topk_bm25_v, sources=ontology_filter)]
        )

        # Get exact match results with labels (for label exact match detection)
        exact_results = exact_future.result()  # List of {id, label} dicts
        exact_ids = _uniq([h["id"] for h in exact_results])
        
        # Identify label-level exact matches (before RRF)
        # Normalize query for comparison
        def _normalize_for_match(s: str) -> str:
            return " ".join(s.lower().split())
        
        normalized_orig_query = _normalize_for_match(_orig_query or query)
        label_exact_match_ids = []
        
        logger.debug(f"Checking {len(exact_results)} exact match results for label exact match (normalized query: '{normalized_orig_query}')")
        for result in exact_results[:5]:  # Log first 5 for debugging
            candidate_label = result.get("label", "")
            normalized_label = _normalize_for_match(candidate_label)
            logger.debug(f"  Exact result: {result['id']} = '{candidate_label}' (normalized: '{normalized_label}')")
        
        for result in exact_results:
            candidate_label = result.get("label", "")
            normalized_label = _normalize_for_match(candidate_label)
            if normalized_orig_query == normalized_label:
                label_exact_match_ids.append(result["id"])
                logger.debug(f"Label exact match found: {result['id']} ('{candidate_label}')")
        
        if label_exact_match_ids:
            logger.info(f"Label-level exact matches found (before RRF): {label_exact_match_ids}")
        else:
            logger.debug(f"No label exact matches found. Checked {len(exact_results)} exact match results.")
        
        # Also query database directly for label exact matches (catches cases where they're not in exact match results)
        # This ensures we find label exact matches even if they come from other channels or aren't retrieved at all
        if not label_exact_match_ids and normalized_orig_query:
            cur = self.get_connection().cursor()
            cur.execute(
                f"SELECT curie, label FROM {cfg.table_terms} WHERE LOWER(TRIM(label)) = ? AND ontology_id IN ({','.join('?' for _ in ontology_filter)})",
                [normalized_orig_query] + ontology_filter
            )
            db_label_exact = cur.fetchall()
            for curie, label in db_label_exact:
                if curie not in label_exact_match_ids:
                    label_exact_match_ids.append(curie)
                    logger.info(f"Label exact match found via database query: {curie} ('{label}')")
        
        if label_exact_match_ids:
            logger.info(f"Total label-level exact matches found (before RRF): {label_exact_match_ids}")
        
        bm25_ids = _uniq(bm25_future.result()) if bm25_future else []
        dense_ids = _uniq(dense_base_future.result()) if dense_base_future else []
        dense_full_ids = _uniq(dense_full_future.result()) if dense_full_future else []
        bm25_ctx_ids = _uniq(bm25_ctx_future.result()) if bm25_ctx_future else []
        
        # Post-filter FAISS results by ontology_id if filter is applied
        if ontology_filter and (dense_ids or dense_full_ids):
            cur = self.get_connection().cursor()
            def _post_filter(ids: List[str]) -> List[str]:
                if not ids:
                    return ids
                qmarks = ",".join("?" for _ in ids)
                cur.execute(
                    f"SELECT curie FROM {cfg.table_terms} WHERE curie IN ({qmarks}) AND ontology_id IN ({','.join('?' for _ in ontology_filter)})",
                    ids + ontology_filter,
                )
                valid = {row[0] for row in cur.fetchall()}
                return [i for i in ids if i in valid]
            dense_ids = _post_filter(dense_ids)
            dense_full_ids = _post_filter(dense_full_ids)

        # Guardrail: Also restrict by CURIE prefix mapping to avoid malformed ontology_id entries
        if ontology_filter:
            try:
                from .schema_policies import allowed_curie_prefixes
                prefixes = allowed_curie_prefixes(ontology_filter)
            except Exception:
                prefixes = None
            if prefixes:
                def _by_prefix(ids: List[str]) -> List[str]:
                    return [i for i in ids if any(i.startswith(p) for p in prefixes)]
                exact_ids = _by_prefix(exact_ids)
                bm25_ids = _by_prefix(bm25_ids)
                dense_ids = _by_prefix(dense_ids)
                bm25_ctx_ids = _by_prefix(bm25_ctx_ids)
                dense_full_ids = _by_prefix(dense_full_ids)
        
        trace["candidates"] = {
            "exact": exact_ids,
            "bm25": bm25_ids,
            "dense": dense_ids,
            "bm25_ctx": bm25_ctx_ids,
            "dense_full": dense_full_ids,
        }
        timings["retrieval_ms"] = (time.monotonic() - t_ret) * 1000
        if ontology_filter:
            trace["ontology_filter"] = ontology_filter

        # --- Fusion ---
        # Combine channels with reciprocal rank fusion using configured weights.
        logger.info("Fusing results from base/context channels...")
        # --- Field-Aware RRF Weights (Phase 2 Enhancement) ---
        def get_field_aware_weights(field_name: str) -> Dict[str, float]:
            """Get field-specific RRF weights based on field characteristics."""
            field_lower = (field_name or "").lower()
            
            # Assay fields: boost exact matches (version numbers, specific protocols)
            if field_lower in {"assay", "assay_type", "protocol"}:
                return {
                    "exact": 1.5,  # Boost exact for version matching
                    "bm25": 0.9,   # Good for keyword matching
                    "dense": 0.4   # Less important for assays
                }
            
            # Cell type fields: boost dense search (semantic similarity important)
            elif field_lower in {"cell_type", "celltype", "cell"}:
                return {
                    "exact": 1.0,  # Standard weight
                    "bm25": 0.7,   # Reduced for cell types
                    "dense": 1.2   # Boost dense for semantic matching
                }
            
            # Disease fields: balanced approach
            elif field_lower in {"disease", "condition", "pathology"}:
                return {
                    "exact": 1.2,  # Slight boost for exact disease terms
                    "bm25": 0.8,   # Good for disease synonyms
                    "dense": 0.8   # Balanced semantic matching
                }
            
            # Default weights for other fields
            else:
                return {
                    "exact": 1.0,
                    "bm25": 0.8,
                    "dense": 0.6,
                }
        
        # Start with config weights, then apply field-aware overrides
        weights = dict(cfg.rrf_weights)
        
        # Get field-aware weights and apply them
        from .fusion_weights import field_aware_weights
        field_weights = field_aware_weights(field_name)
        weights.update(field_weights)
        
        # Add defaults for context channels based on their base types
        if "bm25_ctx" not in weights:
            weights["bm25_ctx"] = weights.get("bm25", 0.8) * 0.9
        if "dense_full" not in weights:
            weights["dense_full"] = weights.get("dense", 0.6) * 1.0

        # If exact matches exist, upweight exact slightly for this query
        if exact_ids:
            weights["exact"] = weights.get("exact", 1.0) * 1.2
            
        logger.info(f"Field-aware weights for '{field_name}': {weights}")
        channel_rankings = {"exact": exact_ids}
        
        # Add label exact match channel with high priority (if any found)
        if label_exact_match_ids:
            # Create separate channel for label exact matches at rank 1
            # This ensures they get maximum RRF score
            channel_rankings["exact_label"] = label_exact_match_ids
            # Give label exact matches very high weight (10x normal exact weight)
            weights["exact_label"] = weights.get("exact", 1.0) * 10.0
            logger.info(f"Label exact match channel added with weight {weights['exact_label']}: {label_exact_match_ids}")
        
        if bm25_ids:
            channel_rankings["bm25"] = bm25_ids
        if dense_ids:
            channel_rankings["dense"] = dense_ids
        if bm25_ctx_ids:
            channel_rankings["bm25_ctx"] = bm25_ctx_ids
        if 'dense_full_ids' in locals() and dense_full_ids:
            channel_rankings["dense_full"] = dense_full_ids
        t_fuse = time.monotonic()
        
        # Use RRF or simple fusion based on config
        if cfg.use_rrf:
            fused = rrf_fuse(channel_rankings, k=rrf_k_v, weights=weights)[:topk_final]
            logger.info(f"Fusion complete: {len(fused)} candidates ranked by RRF")
        else:
            # Simple concatenation fusion: priority order with deduplication
            fused = simple_fuse(channel_rankings)[:topk_final]
            logger.info(f"Fusion complete: {len(fused)} candidates using simple concatenation (no RRF)")
        
        trace["fusion"] = fused
        timings["fusion_ms"] = (time.monotonic() - t_fuse) * 1000

        # Log contribution by channel
        try:
            fused_ids_set = {fid for fid, _ in fused}
            contrib = {k: len(set(v).intersection(fused_ids_set)) for k, v in trace["candidates"].items() if v}
            logger.info(f"Channel contributions to fused top-k: {contrib}")
        except Exception:
            pass

        # Decide on GRAPH depth (auto mode or explicit override)
        eff_graph_depth = 0
        try:
            # Respect explicit override if provided (including 0)
            if graph_depth is not None:
                eff_graph_depth = int(graph_depth)
            else:
                mode = (cfg.graph_mode or "auto").lower()
                if mode == "on":
                    eff_graph_depth = max(1, cfg.graph_auto_depth)
                elif mode == "auto":
                    # Only consider for conservative fields
                    allow_fields = {f.strip().lower() for f in (cfg.graph_auto_fields or [])}
                    if field_name and field_name.lower() in allow_fields and not exact_ids:
                        # Estimate confidence from fused scores (pre-graph): normalize to [0,1]
                        if fused:
                            scores = [s for _, s in fused]
                            mx, mn = max(scores), min(scores)
                            span = (mx - mn) or 1.0
                            confs = [(s - mn) / span for s in scores]
                            top1 = confs[0] if confs else 0.0
                            top3_mean = sum(confs[:3]) / min(3, len(confs)) if confs else 0.0
                            if (top1 < cfg.graph_top1_min) or (top3_mean < cfg.graph_top3_mean_min):
                                eff_graph_depth = max(1, cfg.graph_auto_depth)
        except Exception:
            eff_graph_depth = 0

        # Graph expansion + second RRF (with distance decay)
        t_graph = time.monotonic()
        if eff_graph_depth > 0 and rerank_after_graph:
            seed_ids = [fid for fid, _ in fused]
            neighbors = compute_graph_neighbors(self.get_connection(), cfg.table_terms, seed_ids, eff_graph_depth)
            if neighbors:
                # Build distance-decayed ranked list for graph neighbors
                graph_ranked_list = []
                for nid, depth in neighbors.items():
                    score = 1.0 / (1.0 + depth)
                    graph_ranked_list.append((nid, score))
                graph_ranked_list.sort(key=lambda x: x[1], reverse=True)
                graph_ids_only = [nid for nid, _ in graph_ranked_list]
                
                if cfg.use_rrf:
                    fused2 = rrf_fuse(
                        {"fused": [fid for fid, _ in fused], "graph": graph_ids_only},
                        k=rrf_k_v,
                        weights={"fused": 1.0, "graph": float(getattr(cfg, 'graph_weight', 0.7))},
                    )[:topk_final]
                else:
                    # Simple fusion: fused results first, then graph neighbors (deduplicated)
                    fused2 = simple_fuse(
                        {"fused": [fid for fid, _ in fused], "graph": graph_ids_only},
                        channel_priority=["fused", "graph"]
                    )[:topk_final]
                # Log that AUTO/ON graph rerank was applied for transparency
                try:
                    mode = (cfg.graph_mode or "auto").lower()
                    logger.info(f"Graph rerank applied (mode={mode}, depth={eff_graph_depth}, graph_weight={getattr(cfg,'graph_weight',0.7)})")
                except Exception:
                    pass
                trace["fusion_graph"] = fused2
                fused = fused2
        timings["graph_ms"] = (time.monotonic() - t_graph) * 1000
        
        ids = [fid for fid, _ in fused]
        if not ids:
            logger.warning("No candidates found after fusion")
            payload = {"results": [], "chosen": None}
            if return_trace:
                payload["trace"] = trace
            return payload
        
        # --- Metadata Hydration ---
        t_hydrate = time.monotonic()
        logger.info("Hydrating metadata for ranked candidates...")
        cur = self.get_connection().cursor()
        qmarks = ",".join("?" for _ in ids)
        meta = {}
        cur.execute(
            f"SELECT curie, label, definition, ontology_id, iri, synonyms_exact, synonyms_related, synonyms_broad, synonyms_narrow, xrefs, comments, is_obsolete, term_doc "
            f"FROM {cfg.table_terms} WHERE curie IN ({qmarks})",
            ids,
        )
        for row in cur.fetchall():
            (
                _id,
                _label,
                _definition,
                _source,
                _iri,
                _sx,
                _sr,
                _sb,
                _sn,
                _xr,
                _cm,
                _obsolete,
                _term_doc,
            ) = row
            # Skip obsolete terms entirely
            if _obsolete:
                continue
            def _split_pipe(x):
                if not x:
                    return None
                vals = [t.strip() for t in str(x).split("|") if t.strip()]
                return vals or None
            syn_exact = _split_pipe(_sx)
            syn_related = _split_pipe(_sr)
            syn_broad = _split_pipe(_sb)
            xrefs = _split_pipe(_xr)
            comments = _cm
            meta[_id] = {
                "label": _label,
                "definition": _definition,
                "source": _source,
                "iri": _iri,
                "synonyms_exact": syn_exact,
                "synonyms_related": syn_related,
                "synonyms_broad": syn_broad,
                "synonyms_generic": None,
                "alt_ids": None,
                "xrefs": xrefs,
                "namespace": None,
                "subsets": None,
                "comments": comments,
                "parents_is_a": None,
                "abstracts": None,
                "term_doc": _term_doc,
            }
        
        ranked = []
        for fid, fscore in fused:
            if fid in meta:
                ranked.append({"id": fid, **meta[fid], "fusion_score": fscore})
        
        # IMPORTANT: If label exact matches exist but are not in fused list, add them with high score
        # This ensures label exact matches are always considered, even if RRF ranked them low
        if label_exact_match_ids:
            label_exact_match_set = set(label_exact_match_ids)
            ranked_ids = {r["id"] for r in ranked}
            missing_label_exact = label_exact_match_set - ranked_ids
            
            if missing_label_exact:
                logger.info(f"Label exact matches not in top-{len(fused)} after RRF, fetching metadata and adding them: {missing_label_exact}")
                # Fetch metadata for missing label exact matches
                cur = self.get_connection().cursor()
                for lid in missing_label_exact:
                    cur.execute(
                        f"SELECT curie, label, definition, ontology_id, iri, synonyms_exact, synonyms_related, synonyms_broad, xrefs, comments, term_doc, is_obsolete FROM {cfg.table_terms} WHERE curie = ?",
                        (lid,)
                    )
                    row = cur.fetchone()
                    if row:
                        _id, _label, _definition, _source, _iri, _sx, _sr, _sb, _xr, _cm, _term_doc, _obsolete = row
                        if _obsolete:
                            continue
                        def _split_pipe(x):
                            if not x:
                                return None
                            vals = [t.strip() for t in str(x).split("|") if t.strip()]
                            return vals or None
                        syn_exact = _split_pipe(_sx)
                        syn_related = _split_pipe(_sr)
                        syn_broad = _split_pipe(_sb)
                        xrefs = _split_pipe(_xr)
                        # Give them a high fusion score (higher than top candidate)
                        max_score = max([r.get("fusion_score", 0.0) for r in ranked], default=1.0)
                        ranked.append({
                            "id": _id,
                            "label": _label,
                            "definition": _definition,
                            "source": _source,
                            "iri": _iri,
                            "synonyms_exact": syn_exact,
                            "synonyms_related": syn_related,
                            "synonyms_broad": syn_broad,
                            "synonyms_generic": None,
                            "alt_ids": None,
                            "xrefs": xrefs,
                            "namespace": None,
                            "subsets": None,
                            "comments": _cm,
                            "parents_is_a": None,
                            "abstracts": None,
                            "term_doc": _term_doc,
                            "fusion_score": max_score + 1.0
                        })
                        logger.info(f"Added label exact match {lid} ({_label}) with boosted score {max_score + 1.0}")
        
        timings["hydrate_ms"] = (time.monotonic() - t_hydrate) * 1000
        logger.info(f"Metadata hydration complete: {len(ranked)} candidates with full information")

        # Re-check for label exact matches in ALL ranked candidates (not just exact match results)
        # This catches cases where label exact matches come from BM25/dense channels
        def _normalize_for_match(s: str) -> str:
            return " ".join(s.lower().split())
        
        normalized_orig_query = _normalize_for_match(_orig_query or query)
        label_exact_match_ids_all = []
        for r in ranked:
            candidate_label = r.get("label", "")
            normalized_label = _normalize_for_match(candidate_label)
            if normalized_orig_query == normalized_label:
                label_exact_match_ids_all.append(r["id"])
                if r["id"] not in label_exact_match_ids:
                    logger.info(f"Found label exact match in ranked candidates (not from exact channel): {r['id']} ('{candidate_label}')")
        
        # Merge with previously found label exact matches
        label_exact_match_ids.extend([x for x in label_exact_match_ids_all if x not in label_exact_match_ids])
        label_exact_match_set = set(label_exact_match_ids)
        if label_exact_match_set:
            logger.info(f"Total label exact matches (from all channels): {label_exact_match_set}")

        # --- Pre-LLM Hard Constraints (Phase 1 Critical Fix) ---
        logger.info("Applying pre-LLM hard constraints...")
        
        def _make_label_blob(r: dict) -> str:
            """Combine all text fields for conflict checking"""
            parts = [r.get("label") or "", r.get("definition") or ""]
            for synfield in ("synonyms_exact", "synonyms_related", "synonyms_broad"):
                parts.extend(r.get(synfield) or [])
            return " ".join(parts)
        
        # Check for immediate abstain conditions
        _abstain, _reason = should_abstain(query)
        if _abstain:
            logger.info(f"Abstaining on query due to no-value indicator: '{query}'")
            abstain_reason = _reason or f"Query '{query}' marked for abstain"
            payload = {"results": [], "chosen": {"reason": abstain_reason}}
            if return_trace:
                payload["trace"] = trace
            return payload
        
        # Apply hard filters for cell types
        # TESTING: Label exact matches now go through tissue context filtering
        if label_exact_match_set:
            logger.info(f"Label exact matches found (will be filtered by tissue context): {label_exact_match_set}")
        original_count = len(ranked)
        if field_name.lower() in {"cell_type"} and (tissue or organism):
            kept = []
            for r in ranked:
                # Label exact matches now go through normal filtering (testing tissue context awareness)
                blob = _make_label_blob(r)
                if tissue and context_violation(blob, tissue):
                    if r['id'] in label_exact_match_set:
                        logger.info(f"Filtered label exact match {r['id']} ({r.get('label', '')}) due to tissue conflict: {tissue}")
                    else:
                        logger.debug(f"Filtered {r['id']} due to tissue conflict: {r.get('label', '')}")
                    continue
                if organism and species_violation(blob, organism):
                    if r['id'] in label_exact_match_set:
                        logger.info(f"Filtered label exact match {r['id']} ({r.get('label', '')}) due to species conflict: {organism}")
                    else:
                        logger.debug(f"Filtered {r['id']} due to species conflict: {r.get('label', '')}")
                    continue
                kept.append(r)
            ranked = kept or ranked  # only drop if we still have something
        
        # Domain-specific whitelists
        field_lower = field_name.lower()
        
        # Sex field: only allow canonical terms
        if field_lower == "sex":
            allowed = {"male", "female"}
            def _is_allowed(r):
                label = (r.get("label") or "").lower()
                syns = [s.lower() for s in (r.get("synonyms_exact") or [])]
                return any(x in allowed for x in [label, *syns])
            new = [r for r in ranked if _is_allowed(r)]
            ranked = new or ranked
        
        # Disease field: prioritize PATO:0000461 for healthy/normal
        if field_lower == "disease":
            query_lower = (_orig_query or "").lower()
            if any(k in query_lower for k in ["healthy", "normal", "control", "no disease"]):
                for r in ranked:
                    if r["id"] == "PATO:0000461":
                        ranked = [r] + [x for x in ranked if x["id"] != "PATO:0000461"]
                        break
        
        # Ethnicity field: prefer continental HANCESTRO
        if field_lower == "self_reported_ethnicity":
            def _continental(r):
                blob = _make_label_blob(r).lower()
                return any(tok in blob for tok in ["european", "african", "east asian", "south asian", "admixed", "latino", "american"])
            ranked.sort(key=lambda r: (not _continental(r), -r.get("retrieval_confidence", 0.0)))
        
        filtered_count = len(ranked)
        if filtered_count < original_count:
            logger.info(f"Hard constraints filtered {original_count - filtered_count} candidates, {filtered_count} remaining")
        
        # Check if no candidates remain after filtering
        if not ranked:
            logger.info("No candidates remain after hard constraint filtering - returning abstain")
            payload = {"results": [], "chosen": {"reason": "No compatible candidates after filtering"}}
            if return_trace:
                payload["trace"] = trace
            return payload

        # --- Re-ranking ---
        t_rerank = time.monotonic()
        exact_set = set(exact_ids)

        # Cross-encoder reranker (if available)
        if self.reranker is not None and ranked:
            logger.info("Applying cross-encoder reranking...")
            ranked = self._rerank_with_cross_encoder(
                _orig_query or query, field_name, tissue, organism, ranked, top_k=topk_final or len(ranked)
            )
            logger.info(f"Reranker reordered candidates, top-3: {[r['id'] for r in ranked[:3]]}")
            # Reranker replaces all rule-based reranking - skip semantic intent and soft boosts
            ranked.sort(key=lambda x: x["fusion_score"], reverse=True)
        else:
            # When reranker_path=None, skip reranking entirely (for ablation baselines)
            # Use fusion scores as-is from RRF
            logger.info("Skipping reranking (reranker_path=None) - using fusion scores as-is")
            ranked.sort(key=lambda x: x["fusion_score"], reverse=True)
        
        timings["rerank_ms"] = (time.monotonic() - t_rerank) * 1000

        # Calculate confidence scores
        if ranked:
            max_score = max(r["fusion_score"] for r in ranked)
            min_score = min(r["fusion_score"] for r in ranked)
            span = (max_score - min_score) or 1.0
            for r in ranked:
                r["retrieval_confidence"] = float((r["fusion_score"] - min_score) / span)

        candidate_ids = {r["id"] for r in ranked}

        # Disambiguation via LLM
        chosen = None
        reason = None
        llm_conf = None
        llm_abstained = False
        llm_was_called = False  # Track if LLM was actually invoked
        llm_ranked_ids: List[str] = []
        t_llm = time.monotonic()
        if ranked and not cfg.retrieval_only:
            logger.info(f"[FLOW] Starting disambiguation with {len(ranked)} candidates")
            logger.info(f"[FLOW] Disambiguation query: '{query}'")
            logger.info(f"[FLOW] Top 3 candidates for disambiguation: {[r.get('id') + ':' + r.get('label', '')[:50] for r in ranked[:3]]}")
            # Prepare candidates block
            # Mark label exact matches for LLM visibility (label_exact_match_ids is defined earlier in the function)
            label_exact_match_set = set(label_exact_match_ids) if label_exact_match_ids else set()
            cand_lines = []
            for r in ranked[: topk_final or len(ranked)]:
                is_exact = r['id'] in label_exact_match_set
                exact_marker = " [EXACT LABEL MATCH]" if is_exact else ""
                cand_lines.append(f"{r['id']} | {r.get('label') or ''}{exact_marker} | {r.get('source') or ''} | {r.get('retrieval_confidence', 0.0):.3f} | { (r.get('definition') or '')[:200] }")
            candidates_block = "\n".join(cand_lines)
            guidance = get_field_guidance(field_name)
            guidance_block = ""
            if guidance:
                guidance_block = (
                    "Field-Specific Guidance\n"
                    f"- Semantic Constraints: {guidance.get('semantic_constraints','')}\n"
                    f"- Context Priority: {guidance.get('context_priority','')}\n"
                    f"- Avoid: {guidance.get('avoid','')}\n"
                    f"- Disambiguation Rules: {guidance.get('disambiguation_rules','')}\n"
                )
            prompt = DISAMBIGUATION_PROMPT.format(
                query=query,
                field_name=field_name or "N/A",
                tissue=tissue or "N/A",
                development_stage=development_stage or "N/A",
                disease=disease or "N/A",
                candidates_block=candidates_block,
                disambiguation_rules=guidance.get('disambiguation_rules', ''),
            )
            try:
                logger.info(f"[FLOW] Calling disambiguation LLM with query: '{query}'")
                llm_was_called = True  # Mark that we're calling the LLM
                out = self.disamb_llm.text(prompt, temperature=0, max_tokens=1024)
                logger.info(f"[FLOW] Disambiguation LLM returned response (length: {len(out) if out else 0})")
                parsed = parse_llm_json(out, DisambiguationResponse)
                if parsed:
                    logger.info(f"[FLOW] Disambiguation LLM chose: {parsed.chosen_id}, confidence: {parsed.llm_confidence}, reason: {parsed.reason[:100] if parsed.reason else 'N/A'}")
                else:
                    logger.warning(f"[FLOW] Failed to parse disambiguation LLM response: {out[:500]}")
                
                # Validate LLM response (Phase 1 Critical Fix)
                if parsed and parsed.chosen_id:
                    # Check if chosen_id is in candidate list
                    valid_choice = any(r["id"] == parsed.chosen_id for r in ranked)
                    if valid_choice:
                        llm_ranked_ids.append(parsed.chosen_id)
                        for r in ranked:
                            if r["id"] == parsed.chosen_id:
                                chosen = dict(r)
                                reason = parsed.reason
                                llm_conf = parsed.llm_confidence
                                break
                        if parsed.alternatives:
                            for alt_id in parsed.alternatives:
                                if (
                                    alt_id
                                    and alt_id in candidate_ids
                                    and alt_id not in llm_ranked_ids
                                ):
                                    llm_ranked_ids.append(alt_id)
                                    if len(llm_ranked_ids) >= 3:
                                        break
                    else:
                        # Treat out-of-candidate selection as an abstain to avoid misleading fallbacks
                        llm_abstained = True
                        reason = parsed.reason or f"LLM selected ID not in candidates: {parsed.chosen_id}"
                        logger.warning(f"LLM chose out-of-candidate ID '{parsed.chosen_id}', abstaining")
                elif parsed and parsed.chosen_id is None:
                    # LLM explicitly abstained; record reason but keep candidates and trace
                    llm_abstained = True
                    reason = parsed.reason or "LLM chose to abstain"
            except Exception as e:
                logger.error(f"[FLOW] Disambiguation LLM FAILED: {e}")
                import traceback
                logger.error(f"[FLOW] Disambiguation LLM traceback: {traceback.format_exc()}")
                # Re-raise rate limit errors so caller can handle retry logic
                error_str = str(e).lower()
                if "rate_limit" in error_str or "ratelimit" in error_str or "429" in str(e):
                    raise  # Propagate rate limit errors to caller
                # For other errors, continue with fallback
                pass

        timings["llm_ms"] = (time.monotonic() - t_llm) * 1000 if not cfg.retrieval_only else 0.0
        # Fallback to top-1 if LLM did not choose (including when it abstained)
        # Even if LLM abstained, use top retrieval result as educated guess based on context
        if not chosen and ranked and (cfg.retrieval_only or not llm_abstained):
            logger.info(f"[FLOW] Disambiguation LLM did not choose, falling back to top retrieval result: {ranked[0].get('id')} - {ranked[0].get('label', '')[:50]}")
            chosen = dict(ranked[0])
            # If LLM was called but didn't choose/abstain, mark it as called
            if llm_was_called:
                chosen["_llm_called"] = True
        elif not chosen and ranked and llm_abstained:
            # LLM abstained but we should still use top retrieval as educated guess
            logger.info(f"[FLOW] Disambiguation LLM abstained, using top retrieval result as educated guess: {ranked[0].get('id')} - {ranked[0].get('label', '')[:50]}")
            chosen = dict(ranked[0])
            # Mark that LLM was called and add abstain reason as context
            if llm_was_called:
                chosen["_llm_called"] = True
                if reason:
                    # Append LLM's abstain reason to the chosen result for context
                    chosen["llm_abstain_reason"] = reason
                    chosen["reason"] = f"LLM abstained but using top retrieval result based on context. LLM reason: {reason}"
        elif chosen:
            logger.info(f"[FLOW] Disambiguation LLM chose: {chosen.get('id')} - {chosen.get('label', '')[:50]}")
            if llm_conf is not None:
                logger.info(f"[FLOW] LLM completed successfully with confidence: {llm_conf}")
            else:
                logger.warning(f"[FLOW] LLM chose but confidence is None (may be abstain or parsing issue)")
        elif llm_abstained:
            logger.info(f"[FLOW] Disambiguation LLM abstained, reason: {reason}")
        if chosen is not None:
            if reason is not None and "reason" not in chosen:
                chosen["reason"] = reason
            if llm_conf is not None:
                chosen["llm_confidence"] = llm_conf
            # Mark that LLM was called (even if confidence is None, it means LLM ran)
            if llm_was_called:
                chosen["_llm_called"] = True

        # Build top-3 stack for downstream consumers (LLM preference first, fall back to retrieval order)
        if chosen and chosen.get("id"):
            primary_id = chosen["id"]
        else:
            primary_id = None

        # Ensure chosen id leads the list (LLM order) and pad with retrieval-based results
        dedup_ids: List[str] = []
        if primary_id:
            dedup_ids.append(primary_id)
        for cand_id in llm_ranked_ids:
            if cand_id and cand_id not in dedup_ids:
                dedup_ids.append(cand_id)
        for r in ranked:
            if len(dedup_ids) >= 3:
                break
            if r["id"] not in dedup_ids:
                dedup_ids.append(r["id"])
        llm_ranked_ids = dedup_ids[:3]

        # Alternatives if requested
        alternatives: List[Dict[str, Any]] = []
        if chosen and num_choices and num_choices > 0:
            alternatives = [
                {k: v for k, v in r.items() if k != "fusion_score"}
                for r in ranked[1 : 1 + num_choices]
            ]

        # Assemble output
        logger.info(f"[FLOW] Generating final output - chosen: {chosen.get('id') if chosen else 'None'} - {chosen.get('label', '')[:50] if chosen else 'None'}")
        result_payload = {
            "results": [
                {k: v for k, v in r.items() if k != "fusion_score"}
                for r in ranked
            ]
        }
        if chosen:
            result_payload["chosen"] = chosen
        if alternatives:
            result_payload["alternatives"] = alternatives
        if llm_ranked_ids:
            ranked_map = {r["id"]: r for r in ranked}
            llm_ranked_payload = []
            for idx, cand_id in enumerate(llm_ranked_ids, start=1):
                cand = ranked_map.get(cand_id)
                if not cand:
                    continue
                entry = {k: v for k, v in cand.items() if k != "fusion_score"}
                entry["llm_rank"] = idx
                llm_ranked_payload.append(entry)
            result_payload["llm_ranked"] = llm_ranked_payload
        else:
            result_payload["llm_ranked"] = []
        if return_trace:
            trace["timings_ms"] = {k: round(v, 2) for k, v in timings.items()}
            result_payload["trace"] = trace
        return result_payload
"""BOND pipeline orchestrator.

This module implements the end-to-end normalization flow used by the CLI and
API:
  1) Field-specific abbreviation normalization (assets/abbreviations.json)
  2) LLM query expansion + context term generation (prompts.py + field_guidance)
  3) Parallel multi-channel retrieval (Exact FTS, BM25 FTS, FAISS dense)
  4) Reciprocal Rank Fusion (optional second RRF after graph expansion)
  5) Metadata hydration from SQLite (filters obsolete terms)
  6) Contextual re-ranking (exact/ontology-prior/context-overlap boosts)
  7) LLM disambiguation with strict, field-specific guidance

Graph expansion runs in one of three modes: off | auto | on. In auto mode
(the default), a shallow depth-1 expansion is applied only for conservative
fields (tissue, development_stage, cell_type) when there are no exact matches
and the fused confidence is low. All graph candidates remain scoped to the
field's ontology.
"""
