#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust ontology -> SQLite loader for your filtered_ontologies.json

✔ Keeps existing hierarchy behavior:
   - ontology_edges: relation='directParent' and 'hierarchicalAncestor' exactly as before.

✔ Adds richer, normalized tables:
   - term_synonym (exact/narrow/broad/related)
   - term_xref
   - term_equivalent / term_disjoint
   - term_subset
   - term_replacement (replaced_by / consider)
   - ontology / ontology_imports (minimal metadata)

✔ Future-proof relations:
   - ANY property IRI whose values look like class IRIs are recorded as edges
     in ontology_edges with (relation, relation_iri). This includes RO relations
     like 'part_of', if/when they appear in your JSON.

✔ BM25-ready:
   - ontology_terms_fts mirrors ontology_terms.term_doc (label + synonyms + definition + comments + xrefs)
"""

import json
import sqlite3
import re
from tqdm import tqdm

DB_FILE   = "/Users/rajlq7/Downloads/Terms/BOND/assets/ontologies.sqlite"
JSON_FILE = "/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/filtered_ontologies.json"

# --------- CURIE prefix mapping (unchanged) ----------
CURIE_PREFIX_TO_ONTOLOGY = {
    "CL": "cl", "UBERON": "uberon", "MONDO": "mondo", "PATO": "pato", "EFO": "efo",
    "NCBITaxon": "ncbitaxon", "FBbt": "fbbt", "ZFA": "zfa", "WBbt": "wbbt",
    "HsapDv": "hsapdv", "MmusDv": "mmusdv", "FBdv": "fbdv", "WBls": "wbls", "HANCESTRO": "hancestro",
}

ALLOWED_ONTOLOGIES = {
    "cl","fbbt","fbdv","hancestro","hsapdv","mmusdv","mondo",
    "ncbitaxon","pato","uberon","wbbt","wbls","zfa","efo"
}

# --------- helpers ---------
def extract_value(obj):
    if isinstance(obj, dict):
        v = obj.get("value")
        if isinstance(v, dict) and "value" in v:
            return v.get("value")
        return v
    return obj

def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

def extract_text_values(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        v = obj.get("value")
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict) and "value" in v and isinstance(v["value"], str):
            return [v["value"]]
        return []
    if isinstance(obj, list):
        out = []
        for it in obj:
            out.extend(extract_text_values(it))
        return out
    return []

def iri_to_curie(iri: str|None) -> str|None:
    if not iri or not isinstance(iri, str):
        return None
    m = re.search(r"/obo/([A-Za-z]+)_([0-9]+)$", iri)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    m2 = re.search(r"/obo/(NCBITaxon)_([0-9]+)$", iri)
    if m2:
        return f"{m2.group(1)}:{m2.group(2)}"
    return None

def is_iri(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def looks_like_class_iri(s: str) -> bool:
    # conservative: only treat PURLs that we can CURIE-ify as class IRIs
    if not is_iri(s):
        return False
    return bool(re.search(r"/obo/[A-Za-z]+_[0-9]+$", s)) or bool(re.search(r"/obo/(NCBITaxon)_[0-9]+$", s))

def short_rel(rel_iri: str) -> str:
    # last fragment after # or /, fallback to full IRI
    if not isinstance(rel_iri, str):
        return str(rel_iri)
    if "#" in rel_iri:
        return rel_iri.rsplit("#", 1)[-1]
    return rel_iri.rstrip("/").rsplit("/", 1)[-1] if "/" in rel_iri else rel_iri

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def dedup_preserve(seq, key=lambda x: x.lower()):
    seen, out = set(), []
    for s in seq:
        k = key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out

# --------- content extraction ---------
def extract_synonyms(cls):
    syns = {"exact": [], "broad": [], "narrow": [], "related": []}
    for k, v in cls.items():
        key = str(k)
        vals = as_list(v)
        if "hasExactSynonym" in key:
            syns["exact"].extend([str(extract_value(x)) for x in vals if extract_value(x)])
        elif "hasBroadSynonym" in key:
            syns["broad"].extend([str(extract_value(x)) for x in vals if extract_value(x)])
        elif "hasNarrowSynonym" in key:
            syns["narrow"].extend([str(extract_value(x)) for x in vals if extract_value(x)])
        elif "hasRelatedSynonym" in key or key.lower() == "synonym" or "relatedSynonym" in key:
            syns["related"].extend([str(extract_value(x)) for x in vals if extract_value(x)])
    for t in syns:
        syns[t] = dedup_preserve([s.strip() for s in syns[t] if s and s.strip()])
    return syns

def extract_xrefs(cls):
    KEY = "http://www.geneontology.org/formats/oboInOwl#hasDbXref"
    out = []
    if KEY in cls:
        for v in as_list(cls[KEY]):
            val = extract_value(v)
            if val:
                out.append(str(val))
    return dedup_preserve(out)

def extract_subsets(cls):
    KEY = "http://www.geneontology.org/formats/oboInOwl#inSubset"
    vals = []
    if KEY in cls:
        for v in as_list(cls[KEY]):
            vv = extract_value(v) if isinstance(v, dict) else v
            if vv:
                vals.append(str(vv))
    return dedup_preserve(vals, key=lambda x: x)

def extract_equiv_disjoint(cls):
    eq_key = "http://www.w3.org/2002/07/owl#equivalentClass"
    dj_key = "http://www.w3.org/2002/07/owl#disjointWith"
    equiv, disj = [], []
    if eq_key in cls:
        for v in as_list(cls[eq_key]):
            vv = extract_value(v) if isinstance(v, dict) else v
            c = iri_to_curie(vv) if isinstance(vv, str) else None
            if c:
                equiv.append(c)
    if dj_key in cls:
        for v in as_list(cls[dj_key]):
            vv = extract_value(v) if isinstance(v, dict) else v
            c = iri_to_curie(vv) if isinstance(vv, str) else None
            if c:
                disj.append(c)
    return sorted(set(equiv)), sorted(set(disj))

def extract_replacements(cls):
    # Common replacement/consider slots in OBO/OWL exports
    CAND_KEYS_REPLACED_BY = [
        "http://purl.obolibrary.org/obo/IAO_0100001",               # IAO:0100001 'term replaced by'
        "http://www.geneontology.org/formats/oboInOwl#replacedBy",
    ]
    CAND_KEYS_CONSIDER = [
        "http://www.geneontology.org/formats/oboInOwl#consider",
    ]
    repl, consider = [], []
    for k in CAND_KEYS_REPLACED_BY:
        if k in cls:
            for v in as_list(cls[k]):
                vv = extract_value(v) if isinstance(v, dict) else v
                c = iri_to_curie(vv) if isinstance(vv, str) else None
                if c:
                    repl.append(c)
    for k in CAND_KEYS_CONSIDER:
        if k in cls:
            for v in as_list(cls[k]):
                vv = extract_value(v) if isinstance(v, dict) else v
                c = iri_to_curie(vv) if isinstance(vv, str) else None
                if c:
                    consider.append(c)
    return sorted(set(repl)), sorted(set(consider))

def build_term_doc(label, definition, syns, comments, xrefs):
    parts = []
    if label:
        parts.append(str(label))
    if definition:
        parts.append(str(definition))
    for bucket in ("exact", "narrow", "broad", "related"):
        parts.extend(syns.get(bucket, []))
    if comments:
        parts.append(str(comments))
    if xrefs:
        parts.extend(xrefs)
    # lowercase, normalize & dedup by full span
    seen, cleaned = set(), []
    for p in parts:
        t = normalize_spaces(p).lower()
        if t and t not in seen:
            seen.add(t)
            cleaned.append(t)
    return normalize_spaces(" ".join(cleaned))

# --------- schema ---------
def create_schema(conn):
    c = conn.cursor()
    c.executescript("""
    DROP TABLE IF EXISTS ontology;
    DROP TABLE IF EXISTS ontology_imports;
    DROP TABLE IF EXISTS ontology_terms;
    DROP TABLE IF EXISTS ontology_edges;
    DROP TABLE IF EXISTS term_synonym;
    DROP TABLE IF EXISTS term_xref;
    DROP TABLE IF EXISTS term_equivalent;
    DROP TABLE IF EXISTS term_disjoint;
    DROP TABLE IF EXISTS term_subset;
    DROP TABLE IF EXISTS term_replacement;
    DROP TABLE IF EXISTS ontology_terms_fts;
    """)
    # ontology metadata
    c.execute("""
    CREATE TABLE ontology (
      ontology_id TEXT PRIMARY KEY,
      title TEXT, description TEXT,
      version_iri TEXT, version_info TEXT,
      homepage TEXT, tracker TEXT, repository TEXT,
      license_url TEXT, license_label TEXT,
      contact_name TEXT, contact_email TEXT,
      source_timestamp TEXT
    );
    """)
    c.execute("""
    CREATE TABLE ontology_imports (
      ontology_id TEXT,
      imported_iri TEXT,
      PRIMARY KEY (ontology_id, imported_iri)
    );
    """)
    # terms (keeps previous columns + a few useful extras)
    c.execute("""
    CREATE TABLE ontology_terms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        curie TEXT UNIQUE,
        iri TEXT,
        ontology_id TEXT,
        label TEXT,
        definition TEXT,
        synonyms_exact TEXT,
        synonyms_broad TEXT,
        synonyms_narrow TEXT,
        synonyms_related TEXT,
        xrefs TEXT,
        comments TEXT,
        is_obsolete BOOLEAN,
        short_form TEXT,
        obo_namespace TEXT,
        num_descendants INTEGER,
        num_hier_descendants INTEGER,
        term_doc TEXT
    );
    """)
    # edges; keep your relation field and add relation_iri for future-proofing
    c.execute("""
    CREATE TABLE ontology_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_curie TEXT,
        target_curie TEXT,
        relation TEXT,
        relation_iri TEXT
    );
    """)
    # normalized
    c.execute("""
    CREATE TABLE term_synonym (
      curie TEXT,
      syn TEXT,
      syn_type TEXT CHECK (syn_type IN ('exact','narrow','broad','related')),
      src_property TEXT,
      UNIQUE (curie, syn, syn_type)
    );
    """)
    c.execute("""
    CREATE TABLE term_xref (
      curie TEXT,
      xref TEXT,
      src_property TEXT,
      UNIQUE (curie, xref)
    );
    """)
    c.execute("""
    CREATE TABLE term_equivalent (
      a_curie TEXT,
      b_curie TEXT,
      PRIMARY KEY (a_curie, b_curie)
    );
    """)
    c.execute("""
    CREATE TABLE term_disjoint (
      a_curie TEXT,
      b_curie TEXT,
      PRIMARY KEY (a_curie, b_curie)
    );
    """)
    c.execute("""
    CREATE TABLE term_subset (
      curie TEXT,
      subset_iri TEXT,
      PRIMARY KEY (curie, subset_iri)
    );
    """)
    c.execute("""
    CREATE TABLE term_replacement (
      curie TEXT,
      replacement_curie TEXT,
      relation TEXT CHECK (relation IN ('replaced_by','consider')),
      PRIMARY KEY (curie, replacement_curie, relation)
    );
    """)
    # FTS mirror (BM25)
    c.execute("""
    CREATE VIRTUAL TABLE ontology_terms_fts USING fts5(
        curie,
        label,
        synonyms_exact,
        synonyms_broad,
        synonyms_narrow,
        synonyms_related,
        xrefs,
        definition,
        comments,
        term_doc,
        content='ontology_terms',
        content_rowid='id'
    );
    """)
    # indexes
    c.execute("CREATE INDEX idx_terms_curie ON ontology_terms(curie);")
    c.execute("CREATE INDEX idx_terms_label ON ontology_terms(label);")
    c.execute("CREATE INDEX idx_terms_xrefs ON ontology_terms(xrefs);")
    c.execute("CREATE INDEX idx_edges_source ON ontology_edges(source_curie);")
    c.execute("CREATE INDEX idx_edges_target ON ontology_edges(target_curie);")
    conn.commit()

# --------- edge writer with safety ---------
def add_edge_safe(cur, source_curie, target_iri, relation_label, relation_iri=None):
    try:
        t_curie = iri_to_curie(target_iri)
        if not t_curie:
            return
        rel = relation_label or (short_rel(relation_iri) if relation_iri else None) or "relatedTo"
        cur.execute(
            "INSERT INTO ontology_edges (source_curie, target_curie, relation, relation_iri) VALUES (?,?,?,?)",
            (source_curie, t_curie, rel, relation_iri or rel),
        )
    except Exception:
        # swallow and continue; never let a bad edge stop the pipeline
        pass

# --------- populate ---------
def populate(conn, data):
    cur = conn.cursor()

    # --- ontology metadata ---
    for o in data.get("ontologies", []):
        try:
            oid = o.get("ontologyId")
            if ALLOWED_ONTOLOGIES and oid not in ALLOWED_ONTOLOGIES:
                continue
            
            # More robust title extraction
            title = None
            if o.get("title"):
                title = extract_value(o.get("title"))
            elif o.get("http://purl.org/dc/elements/1.1/title"):
                title = extract_value(o.get("http://purl.org/dc/elements/1.1/title"))
            
            # More robust description extraction
            desc = None
            if o.get("description"):
                desc = o.get("description")
            elif o.get("http://purl.org/dc/elements/1.1/description"):
                desc = extract_value(o.get("http://purl.org/dc/elements/1.1/description"))
            
            # Other fields with better error handling
            viri = o.get("http://www.w3.org/2002/07/owl#versionIRI")
            vinfo = extract_value(o.get("http://www.w3.org/2002/07/owl#versionInfo"))
            home = o.get("homepage") or o.get("uri")  # Some ontologies use "uri" instead of "homepage"
            track = o.get("tracker")
            repo = o.get("repository")
            
            # License handling
            lic_u = None
            lic_l = None
            if o.get("http://purl.org/dc/terms/license"):
                lic_u = o.get("http://purl.org/dc/terms/license")
            elif isinstance(o.get("license"), dict):
                lic_u = o.get("license", {}).get("url")
                lic_l = o.get("license", {}).get("label")
            elif isinstance(o.get("license"), str):
                lic_u = o.get("license")
            
            # Contact handling
            contact_name = None
            contact_email = None
            if isinstance(o.get("contact"), dict):
                contact_name = o.get("contact", {}).get("label")
                contact_email = o.get("contact", {}).get("email")
            
            # Source timestamp
            src_ts = None
            if o.get("sourceFileTimestamp"):
                src_ts = extract_value(o.get("sourceFileTimestamp"))
            elif o.get("loaded"):
                src_ts = extract_value(o.get("loaded"))

            # Insert with NULL handling and type safety
            # Ensure all values are strings or None, not dictionaries
            def safe_value(val):
                if isinstance(val, dict):
                    return str(val) if val else None
                return val
            
            cur.execute("""
            INSERT OR REPLACE INTO ontology
            (ontology_id,title,description,version_iri,version_info,homepage,tracker,repository,
             license_url,license_label,contact_name,contact_email,source_timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,(oid,safe_value(title),safe_value(desc),safe_value(viri),safe_value(vinfo),
                  safe_value(home),safe_value(track),safe_value(repo),safe_value(lic_u),
                  safe_value(lic_l),safe_value(contact_name),safe_value(contact_email),safe_value(src_ts)))

            # Handle dependencies
            for dep in as_list(o.get("dependencies")):
                try:
                    di = extract_value(dep) if isinstance(dep, dict) else dep
                    if di:
                        cur.execute("INSERT OR IGNORE INTO ontology_imports (ontology_id, imported_iri) VALUES (?,?)",
                                    (oid, str(di)))
                except Exception:
                    # Skip problematic dependencies
                    continue
                    
        except Exception as e:
            # Log the error but continue processing
            print(f"Warning: Failed to process ontology metadata for {o.get('ontologyId', 'unknown')}: {str(e)}")
            # Still try to insert basic record with just the ontology_id
            try:
                oid = o.get("ontologyId")
                if oid and (not ALLOWED_ONTOLOGIES or oid in ALLOWED_ONTOLOGIES):
                    cur.execute("""
                    INSERT OR IGNORE INTO ontology (ontology_id) VALUES (?)
                    """, (oid,))
            except Exception:
                pass

    # --- terms ---
    for ontology in tqdm(data.get("ontologies", []), desc="Ontologies"):
        oid = ontology.get("ontologyId")
        if ALLOWED_ONTOLOGIES and oid not in ALLOWED_ONTOLOGIES:
            continue

        for cls in tqdm(ontology.get("classes", []), desc=f"{oid}", leave=False):
            try:
                curie = extract_value(cls.get("curie"))
                iri   = extract_value(cls.get("iri"))
                if not curie:
                    # if there's a class without CURIE, try derive from IRI
                    curie = iri_to_curie(iri)

                # label
                ldata = cls.get("label")
                label = extract_value(ldata[0]) if isinstance(ldata, list) and ldata else extract_value(ldata)
                if not label and "http://www.w3.org/2000/01/rdf-schema#label" in cls:
                    rdl = extract_value(cls["http://www.w3.org/2000/01/rdf-schema#label"])
                    label = rdl

                # definition - prefer 'definition' then IAO:0000115
                ddata = cls.get("definition")
                if ddata:
                    def_texts = extract_text_values(ddata)
                    definition = " ".join(def_texts) if def_texts else None
                else:
                    iao_def = cls.get("http://purl.obolibrary.org/obo/IAO_0000115")
                    dt = extract_text_values(iao_def) if iao_def else []
                    definition = " ".join(dt) if dt else None

                comments_data = cls.get("http://www.w3.org/2000/01/rdf-schema#comment")
                comments = " ".join(extract_text_values(comments_data)) if comments_data else None

                is_obsolete = bool(cls.get("isObsolete"))
                short_form  = extract_value(cls.get("shortForm"))
                obo_ns      = extract_value(cls.get("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"))
                num_desc    = cls.get("numDescendants")
                num_hdesc   = cls.get("numHierarchicalDescendants")

                syns  = extract_synonyms(cls)
                xrefs = extract_xrefs(cls)
                term_doc = build_term_doc(label, definition, syns, comments, xrefs)

                # normalize ontology_id from CURIE prefix when possible
                oid_for_row = oid
                if isinstance(curie, str) and ":" in curie:
                    pref = curie.split(":", 1)[0]
                    mapped = CURIE_PREFIX_TO_ONTOLOGY.get(pref)
                    if mapped:
                        oid_for_row = mapped
                if ALLOWED_ONTOLOGIES and oid_for_row not in ALLOWED_ONTOLOGIES:
                    continue

                # insert term row
                cur.execute("""
                INSERT OR IGNORE INTO ontology_terms
                (curie, iri, ontology_id, label, definition,
                 synonyms_exact, synonyms_broad, synonyms_narrow, synonyms_related,
                 xrefs, comments, is_obsolete, short_form, obo_namespace,
                 num_descendants, num_hier_descendants, term_doc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,(
                    curie, iri, oid_for_row, label, definition,
                    "|".join(syns["exact"]),
                    "|".join(syns["broad"]),
                    "|".join(syns["narrow"]),
                    "|".join(syns["related"]),
                    "|".join(xrefs),
                    comments, 1 if is_obsolete else 0, short_form, obo_ns,
                    int(num_desc) if isinstance(num_desc,int) else None,
                    int(num_hdesc) if isinstance(num_hdesc,int) else None,
                    term_doc
                ))
                rowid = cur.lastrowid

                # FTS mirror
                cur.execute("""
                INSERT INTO ontology_terms_fts
                (rowid, curie, label, synonyms_exact, synonyms_broad, synonyms_narrow, synonyms_related,
                 xrefs, definition, comments, term_doc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,(rowid, curie, label, "|".join(syns["exact"]), "|".join(syns["broad"]),
                     "|".join(syns["narrow"]), "|".join(syns["related"]),
                     "|".join(xrefs), definition, comments, term_doc))

                # normalized synonyms/xrefs
                for s in syns["exact"]:
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_synonym (curie,syn,syn_type,src_property) VALUES (?,?,?,?)",
                                    (curie, s, "exact", "oboInOwl#hasExactSynonym"))
                    except Exception:
                        pass
                for s in syns["narrow"]:
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_synonym (curie,syn,syn_type,src_property) VALUES (?,?,?,?)",
                                    (curie, s, "narrow", "oboInOwl#hasNarrowSynonym"))
                    except Exception:
                        pass
                for s in syns["broad"]:
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_synonym (curie,syn,syn_type,src_property) VALUES (?,?,?,?)",
                                    (curie, s, "broad", "oboInOwl#hasBroadSynonym"))
                    except Exception:
                        pass
                for s in syns["related"]:
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_synonym (curie,syn,syn_type,src_property) VALUES (?,?,?,?)",
                                    (curie, s, "related", "oboInOwl#hasRelatedSynonym"))
                    except Exception:
                        pass
                for x in xrefs:
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_xref (curie,xref,src_property) VALUES (?,?,?)",
                                    (curie, x, "oboInOwl#hasDbXref"))
                    except Exception:
                        pass

                # subsets
                for sub_iri in extract_subsets(cls):
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_subset (curie, subset_iri) VALUES (?,?)",
                                    (curie, sub_iri))
                    except Exception:
                        pass

                # --- hierarchy edges (unchanged semantics) ---
                for parent in cls.get("directParent", []):
                    p_iri = extract_value(parent) if isinstance(parent, dict) else parent
                    add_edge_safe(cur, curie, p_iri, relation_label="directParent", relation_iri="directParent")
                for ancestor in cls.get("hierarchicalAncestor", []):
                    a_iri = extract_value(ancestor) if isinstance(ancestor, dict) else ancestor
                    add_edge_safe(cur, curie, a_iri, relation_label="hierarchicalAncestor", relation_iri="hierarchicalAncestor")

                # capture explicit rdfs:subClassOf if present
                try:
                    sc = cls.get("http://www.w3.org/2000/01/rdf-schema#subClassOf")
                    for v in as_list(sc):
                        vv = extract_value(v) if isinstance(v, dict) else v
                        if vv and looks_like_class_iri(vv):
                            add_edge_safe(cur, curie, vv, relation_label="subClassOf",
                                          relation_iri="http://www.w3.org/2000/01/rdf-schema#subClassOf")
                except Exception:
                    pass

                # --- logical links (and also materialize as edges) ---
                equiv, disj = extract_equiv_disjoint(cls)
                for b in equiv:
                    try:
                        a1, b1 = sorted([curie, b])
                        cur.execute("INSERT OR IGNORE INTO term_equivalent (a_curie,b_curie) VALUES (?,?)", (a1,b1))
                        # also add a generic edge row for uniform graph queries
                        cur.execute(
                            "INSERT INTO ontology_edges (source_curie,target_curie,relation,relation_iri) VALUES (?,?,?,?)",
                            (curie, b, "equivalentClass", "http://www.w3.org/2002/07/owl#equivalentClass")
                        )
                    except Exception:
                        pass
                for b in disj:
                    try:
                        a1, b1 = sorted([curie, b])
                        cur.execute("INSERT OR IGNORE INTO term_disjoint (a_curie,b_curie) VALUES (?,?)", (a1,b1))
                        cur.execute(
                            "INSERT INTO ontology_edges (source_curie,target_curie,relation,relation_iri) VALUES (?,?,?,?)",
                            (curie, b, "disjointWith", "http://www.w3.org/2002/07/owl#disjointWith")
                        )
                    except Exception:
                        pass

                # obsolete → replacement / consider (if present)
                repl, consider = extract_replacements(cls)
                for r in repl:
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_replacement (curie,replacement_curie,relation) VALUES (?,?,?)",
                                    (curie, r, "replaced_by"))
                    except Exception:
                        pass
                for r in consider:
                    try:
                        cur.execute("INSERT OR IGNORE INTO term_replacement (curie,replacement_curie,relation) VALUES (?,?,?)",
                                    (curie, r, "consider"))
                    except Exception:
                        pass

                # --- catch-all: record ANY object-property edges we can safely CURIE-ify ---
                # This makes the loader future-proof: if your JSON starts including RO relations (e.g., part_of),
                # they will be captured without schema/code changes.
                for key, val in cls.items():
                    try:
                        if not (isinstance(key, str) and is_iri(key)):
                            continue
                        # Skip common annotation IRIs we already handled or that are not object properties
                        if key in {
                            "http://www.w3.org/2000/01/rdf-schema#label",
                            "http://www.w3.org/2000/01/rdf-schema#comment",
                            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                            "http://purl.obolibrary.org/obo/IAO_0000115",
                            "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
                            "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
                            "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
                            "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
                            "http://www.geneontology.org/formats/oboInOwl#inSubset",
                            "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
                            "http://www.w3.org/2002/07/owl#equivalentClass",
                            "http://www.w3.org/2002/07/owl#disjointWith",
                        }:
                            continue
                        # Treat as relation only if values look like class IRIs we can CURIE-ify
                        for v in as_list(val):
                            vv = extract_value(v) if isinstance(v, dict) else v
                            if isinstance(vv, str) and looks_like_class_iri(vv):
                                add_edge_safe(cur, curie, vv, relation_label=short_rel(key), relation_iri=key)
                    except Exception:
                        # never stop on odd shapes
                        pass

            except Exception:
                # swallow any unexpected class-level issue and keep going
                pass

    conn.commit()

# --------- main ---------
if __name__ == "__main__":
    print("Loading JSON...")
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    conn = sqlite3.connect(DB_FILE)
    create_schema(conn)
    populate(conn, data)
    conn.close()
    print(f"✅ SQLite DB created: {DB_FILE}")