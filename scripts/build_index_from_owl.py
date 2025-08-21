import argparse
from bond.runtime_env import configure_runtime
import os
import sqlite3
import json
import numpy as np
from tqdm import tqdm
from pronto import Ontology
from rdflib import Graph, URIRef, RDFS
from sentence_transformers import SentenceTransformer

from bond.providers import resolve_embeddings
import faiss
from datetime import datetime
import logging
import warnings
from glob import glob
from contextlib import redirect_stdout, redirect_stderr
import io
import re
import tempfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Suppress noisy warnings broadly where safe
warnings.simplefilter("ignore", SyntaxWarning)
# Use Pronto's NotImplementedWarning if available; else fall back to generic Warning
try:
	from pronto.utils import NotImplementedWarning as ProntoNotImplementedWarning
except Exception:
	try:
		from pronto import NotImplementedWarning as ProntoNotImplementedWarning
	except Exception:
		ProntoNotImplementedWarning = Warning
warnings.filterwarnings("ignore", category=ProntoNotImplementedWarning, module="pronto")

# Suppress date parsing errors from malformed OWL files
warnings.filterwarnings("ignore", category=UserWarning, module="dateutil.parser")
warnings.filterwarnings("ignore", message=".*Unknown string format.*", category=UserWarning)

# Suppress noisy Unicode encoding warnings during ontology parsing (safe to ignore)
warnings.filterwarnings("ignore", category=UnicodeWarning)

# Reduce verbosity of external libraries
logging.getLogger('bond').setLevel(logging.WARNING)
logging.getLogger('litellm').setLevel(logging.ERROR)
logging.getLogger('LiteLLM').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

def create_sqlite(sqlite_path: str) -> sqlite3.Connection:
	os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
	conn = sqlite3.connect(sqlite_path)
	cur = conn.cursor()
	# Incremental-friendly: do not drop; create if missing
	cur.execute("""
		CREATE TABLE IF NOT EXISTS terms (
			id TEXT PRIMARY KEY,
			label TEXT,
			def TEXT,
			norm_label TEXT,
			source TEXT,
			iri TEXT,
			definition TEXT,
			syn_exact TEXT,
			syn_related TEXT,
			syn_broad TEXT,
			syn_generic TEXT,
			alt_ids TEXT,
			xrefs TEXT,
			namespace TEXT,
			subsets TEXT,
			comments TEXT,
			parents_is_a TEXT,
			abstracts TEXT,
			ingested_via TEXT,
			provenance_rank INTEGER DEFAULT 0,
			updated_at TEXT
		)""")
	cur.execute("""
		CREATE TABLE IF NOT EXISTS source_status (
			source TEXT PRIMARY KEY,
			last_updated_authoritative TEXT,
			last_seen_via_import TEXT,
			data_version TEXT
		)
	""")
	cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS terms_fts USING fts5(label, def, content='terms', content_rowid='rowid', tokenize='porter')")
	conn.commit()
	return conn

def normalize(s: str) -> str:
	return " ".join((s or "").lower().split())

# --- rdflib-based annotation harvesting (labels/defs/synonyms) ---
def _build_rdflib_ann(path: str, source_name: str) -> dict:
	try:
		from rdflib import RDFS, URIRef
		g = Graph()
		g.parse(path)
		ann: dict = {}
		source_l = (source_name or "").lower()
		RADLEX = source_l == "radlex"

		def ensure(subj: str) -> dict:
			return ann.setdefault(subj, {
				"labels": set(),
				"pref": None,
				"syn_exact": set(),
				"syn_related": set(),
				"syn_broad": set(),
				"defs": set(),
				"parents_is_a": set(),
			})

		for s, p, o in g.triples((None, None, None)):
			try:
				ps = str(p)
				ps_l = ps.lower()
				ent = ensure(str(s))
				
				# Parents (is_a relationships)
				if p == RDFS.subClassOf and isinstance(o, URIRef):
					ent["parents_is_a"].add(str(o))
					continue
				
				# Labels
				if ps_l.endswith("preflabel") or ps_l.endswith("label"):
					ent["labels"].add(str(o))
				# Definitions (IAO_0000115 or ontology-specific 'definition')
				elif ps_l.endswith("iao_0000115") or ps_l.endswith("/definition") or ps_l.endswith("definition"):
					ent["defs"].add(str(o))
				# Synonyms (oboInOwl variants)
				elif "hasexactsynonym" in ps_l:
					ent["syn_exact"].add(str(o))
				elif "hasrelatedsynonym" in ps_l:
					ent["syn_related"].add(str(o))
				elif "hasbroadsynonym" in ps_l:
					ent["syn_broad"].add(str(o))
				# Generic synonyms: any other predicate containing 'synonym'
				elif "synonym" in ps_l:
					ent.setdefault("syn_generic", set()).add(str(o))
				# RadLex specific
				elif RADLEX and (ps_l.endswith("preferred_name") or ps_l.endswith("preferred name") or ps_l.endswith("/preferred_name") or "preferred_name" in ps_l):
					ent["pref"] = str(o)
				elif RADLEX and (ps_l.endswith("synonym_english") or ps_l.endswith("synonym_german") or ps_l.endswith("related_modality") or ps_l.endswith("/synonym") or ps_l.endswith("synonym")):
					ent["syn_related"].add(str(o))
				
				# DBpedia abstracts
				if "dbpedia.org/ontology/abstract" in ps_l:
					ent.setdefault("abstracts", []).append(str(o))
				# Xrefs
				if ps_l.endswith("hasdbxref") or ps_l.endswith("database_cross_reference"):
					ent.setdefault("xrefs", set()).add(str(o))
				# Alternate IDs
				if ps_l.endswith("hasalternativeid") or ps_l.endswith("alternative_id"):
					ent.setdefault("alt_ids", set()).add(str(o))
				# Namespace / ontology name (e.g., GO namespace)
				if ps_l.endswith("hasobonamespace"):
					ent.setdefault("namespace", set()).add(str(o))
				# Subsets
				if ps_l.endswith("insubset") or ps_l.endswith("in_subset"):
					ent.setdefault("subsets", set()).add(str(o))
				# Comments
				if ps_l.endswith("comment"):
					ent.setdefault("comments", set()).add(str(o))
			except Exception:
				continue

		# Build auxiliary key map for subjects by tail for resilient matching (especially RadLex)
		aux = {}
		for subj, info in list(ann.items()):
			tail = str(subj).rstrip('/').split('/')[-1].lower()
			aux[tail] = info
			# RadLex variants: RID####, RID_####
			if RADLEX:
				m = re.match(r"rid[_-]?(\d+)$", tail, flags=re.IGNORECASE)
				if m:
					num = m.group(1)
					aux[f"rid{num}"] = info
					aux[f"rid_{num}"] = info
		ann["__aux__"] = aux
		return ann
	except Exception:
		return {}

# --- Robust HTTP IRI normalization to CURIE + source ---
def _normalize_http_to_curie_and_source(http_id: str, default_record_id: str, default_source: str) -> tuple[str, str]:
	"""
	Given a full HTTP IRI, try to convert it to a CURIE and a canonical source prefix.
	Returns (record_id, source_prefix). Falls back to defaults when conversion not possible.
	"""
	try:
		iri = http_id
		lower_iri = iri.lower()
		# OBO PURL -> CURIE (PREFIX:LOCAL)
		if lower_iri.startswith("http://purl.obolibrary.org/obo/") or lower_iri.startswith("https://purl.obolibrary.org/obo/"):
			tail = iri.rstrip("/").split("/")[-1]
			if "_" in tail:
				prefix, local = tail.split("_", 1)
				return f"{prefix}:{local}", prefix.lower()
			return default_record_id, default_source
		# FMA PURLs at purl.org/sig/ont/fma/fma######## -> FMA:########
		if "purl.org/sig/ont/fma/" in lower_iri:
			last = iri.rstrip("/").split("/")[-1]
			up = last.upper()
			if up.startswith("FMA") and up[3:].isdigit():
				return f"FMA:{up[3:]}", "fma"
			return default_record_id, "fma"
		# RadLex -> RID:####, source=radlex
		if "//www.radlex.org/rid/" in lower_iri:
			last = iri.rstrip("/").split("/")[-1]
			up = last.upper()
			if up.startswith("RID") and up[3:].isdigit():
				return f"RID:{up[3:]}", "radlex"
			return default_record_id, "radlex"
		# identifiers.org -> extract CURIE after last '/'
		if lower_iri.startswith("http://identifiers.org/") or lower_iri.startswith("https://identifiers.org/"):
			# Common patterns:
			#  - https://identifiers.org/hgnc/10001 -> HGNC:10001
			#  - https://identifiers.org/ncbitaxon/9606 -> NCBITaxon:9606
			#  - https://identifiers.org/ensembl/ENSG00000141510 -> ENSG...
			parts = iri.strip("/").split("/")
			# parts[-2] is namespace key, parts[-1] is accession
			if len(parts) >= 2:
				key = parts[-2].lower()
				acc = parts[-1]
				mapping = {
					"hgnc": "HGNC",
					"ncbitaxon": "NCBITaxon",
					"omim": "OMIM",
					"orphanet": "ORPHA",
					"uniprot": "UniProt",
					"ensembl": "ensembl",  # handle Ensembl gene IDs
					"ncbigene": "ncbigene",  # handle NCBI gene IDs
				}
				if key in mapping:
					if key == "ensembl" and acc.upper().startswith("ENS"):
						# Extract clean gene ID (e.g., ENSG00000141510)
						return acc, "ensembl"
					elif key == "ncbigene" and acc.isdigit():
						# Extract clean gene ID (e.g., 123456)
						return acc, "ncbigene"
					elif mapping[key] is not None:
						return f"{mapping[key]}:{acc}", mapping[key].lower()
				# Fallback: if accession already has a prefix (e.g., HGNC:10001)
				if ":" in acc:
					pref = acc.split(":", 1)[0]
					return acc, pref.lower()
			return default_record_id, default_source
		# EFO release IRIs (www.ebi.ac.uk)
		if "www.ebi.ac.uk" in lower_iri:
			last = iri.rstrip("/").split("/")[-1]
			if last.startswith("EFO_") and last[4:].isdigit():
				return f"EFO:{last[4:]}", "efo"
			return default_record_id, default_source
		# Orphanet
		if "www.orpha.net" in lower_iri:
			last = iri.rstrip("/").split("/")[-1]
			if last.startswith("Orphanet_") and last.split("_")[-1].isdigit():
				return f"ORPHA:{last.split('_')[-1]}", "orphanet"
			return default_record_id, "orphanet"
		# BAO
		if "www.bioassayontology.org" in lower_iri:
			last = iri.rstrip("/").split("/")[-1]
			if last.startswith("BAO_") and last[4:].isdigit():
				return f"BAO:{last[4:]}", "bao"
			return default_record_id, "bao"
		# Gene Ontology site direct
		if "www.geneontology.org" in lower_iri:
			last = iri.rstrip("/").split("/")[-1]
			if last.startswith("GO_") and last[3:].isdigit():
				return f"GO:{last[3:]}", "go"
			return default_record_id, default_source
		# DBpedia
		if "dbpedia.org/resource/" in lower_iri:
			last = iri.rstrip("/").split("/")[-1]
			return f"DBPEDIA:{last}", "dbpedia"
		return default_record_id, default_source
	except Exception:
		return default_record_id, default_source

def _to_iso_date(day: str, mon: str, year: str) -> str:
	mon_map = {
		'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
		'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
	}
	mm = mon_map.get(mon.lower(), None)
	if not mm:
		return f"{year}-{mon}-{day}"
	d2 = day.zfill(2)
	return f"{year}-{mm}-{d2}"

def sanitize_owl_file_if_needed(owl_path: str) -> str:
	"""Sanitize known malformed metadata patterns (e.g., 'creation: 16MAY2017') to ISO dates.

	Returns a temporary sanitized file path if changes were made, otherwise the original path.
	"""
	try:
		with open(owl_path, 'r', encoding='utf-8', errors='ignore') as f:
			text = f.read()
	except Exception:
		return owl_path

	changed = False

	# Replace patterns like 'creation: 16MAY2017' or 'creation_date: 16MAY2017' with ISO '2017-05-16'
	def repl_creation(m):
		nonlocal changed
		changed = True
		day, mon, year = m.group(1), m.group(2), m.group(3)
		return _to_iso_date(day, mon, year)

	pattern = r"(?i)creation(?:[_\-\s]?date)?\s*[:=]\s*(\d{1,2})([A-Za-z]{3})(\d{4})"
	text2 = re.sub(pattern, repl_creation, text)

	if not changed:
		return owl_path

	tmp = tempfile.NamedTemporaryFile(prefix="san_owl_", suffix=".owl", delete=False)
	tmp.write(text2.encode('utf-8', errors='ignore'))
	tmp.flush()
	tmp.close()
	return tmp.name

def remove_creation_metadata(owl_path: str) -> str:
	"""As a last resort, remove lines containing creation metadata that break parsers."""
	try:
		with open(owl_path, 'r', encoding='utf-8', errors='ignore') as f:
			lines = f.readlines()
	except Exception:
		return owl_path

	filtered = []
	removed = False
	for line in lines:
		if re.search(r"(?i)creation[_\- ]?date|(?i)creation\s*[:=]", line):
			removed = True
			continue
		filtered.append(line)

	if not removed:
		return owl_path

	tmp = tempfile.NamedTemporaryFile(prefix="nocre_owl_", suffix=".owl", delete=False)
	tmp.write("".join(filtered).encode('utf-8', errors='ignore'))
	tmp.flush()
	tmp.close()
	return tmp.name

def ingest_owl(conn: sqlite3.Connection, owl_paths: list):
	cur = conn.cursor()
	total_terms = 0
	for owl_path in owl_paths:
		logger.info(f"Ingesting {os.path.basename(owl_path)}...")
		# Try raw parse, then sanitized, then metadata-stripped
		try:
			onto = Ontology(owl_path)
		except Exception as e1:
			logger.warning(f"Initial parse failed for {os.path.basename(owl_path)}: {e1}")
			san_path = sanitize_owl_file_if_needed(owl_path)
			try:
				onto = Ontology(san_path)
				logger.info(f"Parsed {os.path.basename(owl_path)} after sanitizing malformed dates")
			except Exception as e2:
				logger.warning(f"Sanitized parse failed for {os.path.basename(owl_path)}: {e2}")
				strip_path = remove_creation_metadata(owl_path)
				try:
					onto = Ontology(strip_path)
					logger.info(f"Parsed {os.path.basename(owl_path)} after removing creation metadata")
				except Exception as e3:
					logger.warning(f"Failed to parse {os.path.basename(owl_path)} after all attempts: {e3}")
					logger.warning("Skipping this ontology file due to parsing errors")
					continue
		source_name = os.path.splitext(os.path.basename(owl_path))[0]

		# Build generic rdflib annotations for augmentation (labels/defs/synonyms)
		rdf_ann = _build_rdflib_ann(owl_path, source_name)
		now_iso = datetime.now().isoformat()
		authoritative_sources_updated = set()
		imported_sources_seen = set()
		for term in tqdm(onto.terms(), desc=f"Processing {source_name}"):
			if term.obsolete: continue
			label = term.name or ""
			if not label: continue
			
			# Determine source and IDs robustly (handle HTTP IRIs like Ensembl first)
			tid = term.id or ""
			record_id = tid
			curie_prefix = source_name.lower()
			
			# Collect synonyms by scope for output; also aggregate for FTS/embeddings
			syns_all = []
			syn_exact, syn_related, syn_broad = [], [], []
			# Extended fields
			syn_generic = []
			alt_ids: list[str] = []
			xrefs: list[str] = []
			namespace: list[str] = []
			subsets: list[str] = []
			comments: list[str] = []
			parents_is_a: list[str] = []
			abstracts: list[str] = []
			# Initialize definition for this term
			text_def = (term.definition or "").strip()
			if hasattr(term, 'synonyms'):
				for syn in term.synonyms:
					text = None
					if hasattr(syn, 'description') and syn.description:
						text = syn.description
					elif hasattr(syn, 'name') and syn.name:
						text = syn.name
					if not text:
						continue
					syns_all.append(text)
					scope_val = None
					try:
						scope_val = str(getattr(syn, 'scope', None) or '').upper()
					except Exception:
						scope_val = None
					if scope_val == 'EXACT':
						syn_exact.append(text)
					elif scope_val == 'RELATED':
						syn_related.append(text)
					elif scope_val == 'BROAD':
						syn_broad.append(text)
			
			# Generic rdflib augmentation across ontologies (labels/defs/synonyms/xrefs/etc.)
			try:
				iri_self = None
				if tid.startswith("http://") or tid.startswith("https://"):
					iri_self = tid
				elif ":" in tid:
					pref, local = tid.split(":", 1)
					if source_name.lower() == "radlex" and pref.upper() == "RID":
						iri_self = f"http://www.radlex.org/RID/{pref}{local}"
					else:
						iri_self = f"http://purl.obolibrary.org/obo/{pref}_{local}"
				info = None
				if iri_self and iri_self in rdf_ann:
					info = rdf_ann[iri_self]
				elif iri_self and "__aux__" in rdf_ann:
					tail = iri_self.rstrip('/').split('/')[-1].lower()
					info = rdf_ann["__aux__"].get(tail)
				if info:
					# Prefer preferred label when current looks like a code
					if (not label) or (source_name.lower()=="radlex" and label.upper().startswith("RID")):
						if info.get("pref"):
							label = info["pref"]
						else:
							for lb in info.get("labels", set()):
								if source_name.lower()=="radlex" and lb.upper().startswith("RID"):
									continue
								label = lb
								break
					# Definition fallback if pronto missing - use prototype's cleaner logic
					rdefs = list(info.get("defs", set()))
					if (not text_def) and rdefs:
						text_def = rdefs[0]
					# Merge synonyms
					for s in info.get("syn_exact", set()):
						if s not in syn_exact:
							syn_exact.append(s)
							if s not in syns_all: syns_all.append(s)
					for s in info.get("syn_related", set()):
						if s not in syn_related:
							syn_related.append(s)
							if s not in syns_all: syns_all.append(s)
					for s in info.get("syn_broad", set()):
						if s not in syn_broad:
							syn_broad.append(s)
							if s not in syns_all: syns_all.append(s)
					for s in info.get("syn_generic", set()):
						if s not in syn_generic:
							syn_generic.append(s)
					# Extras - match prototype's handling
					for s in info.get("alt_ids", set()):
						if s not in alt_ids:
							alt_ids.append(s)
					for s in info.get("xrefs", set()):
						if s not in xrefs:
							xrefs.append(s)
					for s in info.get("namespace", set()):
						if s not in namespace:
							namespace.append(s)
					for s in info.get("subsets", set()):
						if s not in subsets:
							subsets.append(s)
					for s in info.get("comments", set()):
						if s not in comments:
							comments.append(s)
					for s in info.get("parents_is_a", set()):
						if s not in parents_is_a:
							parents_is_a.append(s)
					for s in info.get("abstracts", []):
						if s not in abstracts:
							abstracts.append(s)
			except Exception:
				pass

			# Combine label, definition, and synonyms for full-text search
			# Note: 'def' column contains more than just definition - it's full searchable text
			full_text = " ".join([label, text_def] + syns_all + syn_generic)
			
			# Compute IRI and normalize special HTTP identifiers (e.g., Ensembl)
			try:
				iri = None
				if tid.startswith("http://") or tid.startswith("https://"):
					# Full IRI provided by the ontology
					iri = tid
					lower_iri = tid.lower()
					# Try robust HTTP normalization first
					record_id, curie_prefix = _normalize_http_to_curie_and_source(tid, record_id, curie_prefix)
					# If still generic http and not mapped, handle specific known patterns (Ensembl)
					if curie_prefix == "http" and "/ensembl/" in lower_iri:
						try:
							extracted = tid.rstrip("/").split("/")[-1]
							if extracted and extracted.upper().startswith("ENS"):
								record_id = extracted
								curie_prefix = "ensembl"
							else:
								curie_prefix = "http"
						except Exception:
							curie_prefix = "http"
				elif ":" in tid:
					# OBO-style CURIE -> stable OBO PURL and source from prefix
					prefix, local = tid.split(":", 1)
					iri = f"http://purl.obolibrary.org/obo/{prefix}_{local}"
					curie_prefix = prefix.lower()
				else:
					# Fallback
					iri = None
					curie_prefix = source_name.lower()
			except Exception:
				iri = None

			# provenance rank: 2 if ingested from its authoritative source; 1 if via import
			prov_rank = 2 if curie_prefix == source_name.lower() else 1
			if prov_rank == 2:
				authoritative_sources_updated.add(curie_prefix)
			else:
				imported_sources_seen.add(curie_prefix)

			cur.execute(
				"""
				INSERT INTO terms(
					id,label,def,norm_label,source,iri,definition,
					syn_exact,syn_related,syn_broad,syn_generic,
					alt_ids,xrefs,namespace,subsets,comments,parents_is_a,abstracts,
					ingested_via,provenance_rank,updated_at
				) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
				ON CONFLICT(id) DO UPDATE SET
					label=excluded.label,
					def=excluded.def,
					norm_label=excluded.norm_label,
					source=excluded.source,
					iri=excluded.iri,
					definition=excluded.definition,
					syn_exact=excluded.syn_exact,
					syn_related=excluded.syn_related,
					syn_broad=excluded.syn_broad,
					syn_generic=excluded.syn_generic,
					alt_ids=excluded.alt_ids,
					xrefs=excluded.xrefs,
					namespace=excluded.namespace,
					subsets=excluded.subsets,
					comments=excluded.comments,
					parents_is_a=excluded.parents_is_a,
					abstracts=excluded.abstracts,
					ingested_via=excluded.ingested_via,
					provenance_rank=excluded.provenance_rank,
					updated_at=excluded.updated_at
				WHERE excluded.provenance_rank >= terms.provenance_rank
				""",
				(
					record_id,
					label,
					full_text,
					normalize(label),
					curie_prefix,
					iri,
					text_def,
					json.dumps(syn_exact, ensure_ascii=False),
					json.dumps(syn_related, ensure_ascii=False),
					json.dumps(syn_broad, ensure_ascii=False),
					json.dumps(syn_generic, ensure_ascii=False),
					json.dumps(alt_ids, ensure_ascii=False),
					json.dumps(xrefs, ensure_ascii=False),
					json.dumps(namespace, ensure_ascii=False),
					json.dumps(subsets, ensure_ascii=False),
					json.dumps(comments, ensure_ascii=False),
					json.dumps(parents_is_a, ensure_ascii=False),
					json.dumps(abstracts, ensure_ascii=False),
					source_name.lower(),
					prov_rank,
					now_iso,
				)
			)
			total_terms += 1
		# Update source_status for this ontology file
		data_version = None
		try:
			meta = getattr(onto, 'metadata', None)
			if meta is not None:
				data_version = getattr(meta, 'data_version', None) or getattr(meta, 'version', None)
		except Exception:
			data_version = None
		cur = conn.cursor()
		# Authoritative updates
		for s in authoritative_sources_updated:
			cur.execute(
				"""
				INSERT INTO source_status(source,last_updated_authoritative,last_seen_via_import,data_version)
				VALUES(?,?,COALESCE((SELECT last_seen_via_import FROM source_status WHERE source=?), NULL),?)
				ON CONFLICT(source) DO UPDATE SET
					last_updated_authoritative=excluded.last_updated_authoritative,
					data_version=COALESCE(excluded.data_version, source_status.data_version)
				""",
				(s, now_iso, s, data_version)
			)
		# Imported sightings
		for s in imported_sources_seen - authoritative_sources_updated:
			cur.execute(
				"""
				INSERT INTO source_status(source,last_seen_via_import,last_updated_authoritative,data_version)
				VALUES(?,?,COALESCE((SELECT last_updated_authoritative FROM source_status WHERE source=?), NULL),COALESCE((SELECT data_version FROM source_status WHERE source=?), NULL))
				ON CONFLICT(source) DO UPDATE SET
					last_seen_via_import=excluded.last_seen_via_import
				""",
				(s, now_iso, s, s)
			)
	logger.info("Rebuilding FTS index (FTS5 content='terms')...")
	# Efficient rebuild for content-backed FTS5
	try:
		cur.execute("INSERT INTO terms_fts(terms_fts) VALUES('rebuild')")
	except Exception:
		# Fallback: repopulate explicitly
		cur.execute("DELETE FROM terms_fts")
		cur.execute("INSERT INTO terms_fts(rowid, label, def) SELECT rowid, label, def FROM terms")
	conn.commit()
	logger.info(f"Total terms ingested: {total_terms}")

def generate_signature(embed_model: str, embed_fn) -> dict:
	from bond.models import EmbeddingSignature
	anchor_text = "the quick brown fox jumps over the lazy dog"
	v = embed_fn([anchor_text])[0]
	signature = EmbeddingSignature(
		model_id=embed_model,
		dimension=len(v),
		anchor_text=anchor_text,
		anchor_vector=v
	)
	return signature.model_dump()

def build_faiss_profiles(conn: sqlite3.Connection, embed_model: str, assets_path: str, faiss_rebuild: bool = False):
	embed_fn = resolve_embeddings(embed_model)
	cur = conn.cursor()

	# Prepare store paths and current signature (also yields embedding dimension via anchor)
	signature = generate_signature(embed_model, embed_fn)
	store_path = os.path.join(assets_path, "faiss_store")
	os.makedirs(store_path, exist_ok=True)

	sig_path = os.path.join(store_path, "embedding_signature.json")
	faiss_path = os.path.join(store_path, "embeddings.faiss")
	id_map_path = os.path.join(store_path, "id_map.npy")
	rescore_path = os.path.join(store_path, "rescore_vectors.npy")

	# Decide on full rebuild vs incremental before any expensive embedding
	need_full_rebuild = faiss_rebuild or (not os.path.exists(faiss_path))
	if not need_full_rebuild and os.path.exists(sig_path):
		try:
			with open(sig_path, "r") as f:
				old_sig = json.load(f)
			if old_sig.get("model_id") != signature.get("model_id"):
				logger.warning("Embedding model changed; performing full FAISS rebuild.")
				need_full_rebuild = True
			elif int(old_sig.get("dimension", signature.get("dimension", 0))) != int(signature.get("dimension", 0)):
				logger.warning("Embedding dimension changed; performing full FAISS rebuild.")
				need_full_rebuild = True
		except Exception:
			need_full_rebuild = True

	# Fetch all ids/texts once (cheap) and decide embedding workload
	logger.info("Fetching terms from database for embedding...")
	cur.execute("SELECT id, label, def FROM terms")
	rows = cur.fetchall()
	all_ids = [r[0] for r in rows]
	all_texts = [(r[1] or "") + " " + (r[2] or "") for r in rows]

	work_ids = all_ids
	work_texts = all_texts

	if not need_full_rebuild:
		# Load existing id map to compute only-new workload
		try:
			existing_id_map = np.load(id_map_path, allow_pickle=False)
			existing_ids = set(str(x) for x in existing_id_map.tolist())
		except Exception:
			existing_ids = set()
		new_mask = [i not in existing_ids for i in all_ids]
		work_ids = [i for i, keep in zip(all_ids, new_mask) if keep]
		work_texts = [t for t, keep in zip(all_texts, new_mask) if keep]
		if not work_ids:
			logger.info("No new terms to embed. FAISS store remains unchanged.")
			# Still update signature file to reflect current model
			with open(sig_path, "w") as f:
				json.dump(signature, f, indent=2)
			return

	logger.info(f"Encoding {len(work_texts)} corpus documents with {embed_model} (this may take a while)...")

	# --- START OF NEW BATCHING LOGIC ---
	
	batch_size = int(os.getenv("BOND_EMB_BATCH", "16"))
	embs = []
	failed_batches = []
	
	def embed_resilient(batch_texts: list, batch_ids: list) -> list:
		"""Embed a batch, recursively bisecting on failure to isolate malformed items.
		Returns a list of embeddings aligned to batch_texts length with None for failed items.
		"""
		try:
			# Silence noisy provider prints during embedding
			sink = io.StringIO()
			with redirect_stdout(sink), redirect_stderr(sink):
				out = embed_fn(batch_texts)
			# Ensure correct length
			if not isinstance(out, list) or len(out) != len(batch_texts):
				raise RuntimeError("Embedding provider returned unexpected shape")
			return out
		except Exception as e:
			# If single item fails, mark as None and continue
			if len(batch_texts) == 1:
				logger.error(f"Skipping malformed text for id={batch_ids[0]} due to error: {e}")
				return [None]
			# Bisect and embed halves independently
			mid = len(batch_texts) // 2
			left = embed_resilient(batch_texts[:mid], batch_ids[:mid])
			right = embed_resilient(batch_texts[mid:], batch_ids[mid:])
			return left + right

	pbar = tqdm(range(0, len(work_texts), batch_size), desc="Generating embeddings in batches")
	for i in pbar:
		batch_texts = work_texts[i:i + batch_size]
		batch_ids = work_ids[i:i + batch_size]
		
		if not batch_texts:
			continue

		# Resilient embedding: isolate bad items instead of dropping the whole batch
		batch_embs = embed_resilient(batch_texts, batch_ids)
		# Track failed items (None) for logging
		if any(e is None for e in batch_embs):
			batch_failed = [batch_ids[j] for j, e in enumerate(batch_embs) if e is None]
			failed_batches.append({"start_index": i, "ids": batch_failed})
		embs.extend(batch_embs)
		pbar.set_postfix({'Processed': len([e for e in embs if e is not None])})
	
	# --- END OF NEW BATCHING LOGIC ---
	
	# Filter out failed embeddings and their corresponding IDs
	successful_embs = [e for e in embs if e is not None]
	successful_ids = [id_ for i, id_ in enumerate(work_ids) if embs[i] is not None]
	
	if not successful_embs:
		logger.error("No embeddings were generated successfully. Aborting.")
		return
	
	logger.info(f"Successfully generated {len(successful_embs)} embeddings.")
	if failed_batches:
		skipped_items = sum(len(b.get("ids", [])) for b in failed_batches)
		logger.warning(f"Skipped {skipped_items} items across {len(failed_batches)} batches (isolated via recursive bisection). See per-id errors above.")
	
	embs_fp32 = np.asarray(successful_embs, dtype=np.float32)
	d = embs_fp32.shape[1]

	def _write_full_store(ids_list, fp32_embs):
		embs_binary_full = np.packbits(np.where(fp32_embs >= 0, 1, 0), axis=-1)
		index_binary_full = faiss.IndexBinaryFlat(d)
		index_binary_full.add(embs_binary_full)
		faiss.write_index_binary(index_binary_full, faiss_path)
		embs_int8_full = np.clip(fp32_embs * 127, -127, 127).astype(np.int8)
		# Save id_map with correct fixed-width dtype
		max_len = max((len(x) for x in ids_list), default=1)
		id_map_array_full = np.array(ids_list, dtype=f"<U{max_len}")
		np.save(id_map_path, id_map_array_full)
		np.save(rescore_path, embs_int8_full)
		with open(sig_path, "w") as f:
			json.dump(signature, f, indent=2)

		from bond.models import IndexMetadata
		meta = IndexMetadata(
			profile="faiss_store", method="binary+int8_rescore", embedding_model=embed_model, normalize=True, dimension=int(d),
			notes="Fast binary index for candidates, followed by precise int8 rescoring.", created_at=datetime.now().isoformat()
		)
		with open(os.path.join(store_path, "meta.json"), "w") as f:
			json.dump(meta.model_dump(), f, indent=2)
		logger.info(f"✅ FAISS store written: {store_path}")

	# Fresh build case: when we decided upfront
	if need_full_rebuild:
		_write_full_store(successful_ids, embs_fp32)
		return

	# Incremental append for new IDs (we already restricted work_ids to only new)
	try:
		existing_id_map = np.load(id_map_path, allow_pickle=False)
		existing_ids = set(str(x) for x in existing_id_map.tolist())
	except Exception:
		existing_ids = set()

	new_pairs = [(i, t, e) for i, t, e in zip(work_ids, work_texts, successful_embs) if e is not None and i not in existing_ids]
	logger.info(f"Found {len(new_pairs)} new terms to append to FAISS store (existing={len(existing_ids)} total)" )
	if not new_pairs:
		logger.info("No new terms to append. FAISS store remains unchanged.")
		# Still update signature file to reflect current model
		with open(sig_path, "w") as f:
			json.dump(signature, f, indent=2)
		return

	# Load existing index and vectors
	index = faiss.read_index_binary(faiss_path)
	try:
		rescore_existing = np.load(rescore_path, mmap_mode=None)
	except Exception:
		rescore_existing = None

	# Prepare new embeddings
	new_ids = [p[0] for p in new_pairs]
	new_fp32 = np.asarray([p[2] for p in new_pairs], dtype=np.float32)
	new_bin = np.packbits(np.where(new_fp32 >= 0, 1, 0), axis=-1)
	index.add(new_bin)
	faiss.write_index_binary(index, faiss_path)

	# Append rescoring vectors and id_map
	new_int8 = np.clip(new_fp32 * 127, -127, 127).astype(np.int8)
	if rescore_existing is not None and rescore_existing.size > 0:
		rescore_concat = np.concatenate([rescore_existing, new_int8], axis=0)
	else:
		rescore_concat = new_int8
	np.save(rescore_path, rescore_concat)

	try:
		old_ids_list = [str(x) for x in existing_id_map.tolist()] if 'existing_id_map' in locals() else []
	except Exception:
		old_ids_list = []
	all_ids_list = old_ids_list + new_ids
	max_len = max((len(x) for x in all_ids_list), default=1)
	id_map_array_new = np.array(all_ids_list, dtype=f"<U{max_len}")
	np.save(id_map_path, id_map_array_new)

	# Update signature (no change in model/dim)
	with open(sig_path, "w") as f:
		json.dump(signature, f, indent=2)
	logger.info(f"✅ Appended {len(new_ids)} new vectors to FAISS store")

def main():
	# Ensure safe OpenMP behavior for build runs
	configure_runtime()
	ap = argparse.ArgumentParser(description="Build BOND SQLite and Faiss indices from OWL/OBO files.")
	ap.add_argument("--owl", nargs="+", required=False, help="Path(s) to OWL/OBO ontology files. If omitted, auto-detects ./data/*.owl|*.obo")
	ap.add_argument("--sqlite_path", default=None, help="Output path for the SQLite database. Defaults to <assets_path>/ontology.sqlite")
	ap.add_argument("--assets_path", default=None, help="Root directory for asset output. Defaults to ./assets")
	ap.add_argument("--embed_model",
					default=(os.getenv("BOND_EMBED_MODEL") or "st:pankajrajdeo/bond-embed-v1-fp16"),
					help="Embedding model: 'st:<hf-model>', 'litellm:<provider/model>' (env BOND_EMBED_MODEL)")
	ap.add_argument("--faiss_rebuild", action="store_true", help="Force full FAISS rebuild instead of incremental append")
	ap.add_argument("--faiss_only", action="store_true", help="Build FAISS embeddings only (skip SQLite ingestion)")
	args = ap.parse_args()

	# Resolve assets path and sqlite path defaults and ensure directories exist
	assets_path = args.assets_path or os.getenv("BOND_ASSETS_PATH") or "assets"
	os.makedirs(assets_path, exist_ok=True)
	sqlite_path = args.sqlite_path or os.path.join(assets_path, "ontology.sqlite")

	owl_paths = args.owl
	if not owl_paths:
		detected = []
		detected.extend(glob(os.path.join("data", "*.owl")))
		detected.extend(glob(os.path.join("data", "*.obo")))
		# Also support assets/data
		detected.extend(glob(os.path.join(assets_path, "data", "*.owl")))
		detected.extend(glob(os.path.join(assets_path, "data", "*.obo")))
		owl_paths = sorted(set(detected))
		if not owl_paths:
			raise SystemExit("No OWL/OBO files provided and none found under ./data or assets/data")

	conn = create_sqlite(sqlite_path)
	
	if args.faiss_only:
		logger.info("🧮 Building FAISS embeddings only (skipping SQLite ingestion)")
		if not os.path.exists(sqlite_path):
			raise FileNotFoundError(f"SQLite database not found: {sqlite_path}. Run without --faiss_only first.")
	else:
		ingest_owl(conn, owl_paths)
	
	build_faiss_profiles(conn, args.embed_model, assets_path, faiss_rebuild=args.faiss_rebuild)
	conn.close()
	logger.info("\nAll indices built successfully.")

if __name__ == "__main__":
	main()
