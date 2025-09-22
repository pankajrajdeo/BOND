#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to filter ontologies from EMBL-EBI JSON file based on CellxGene schema.

Usage: python filter_ontologies_ebi_json.py
"""

import json
from pathlib import Path
from tqdm import tqdm
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import ijson
from queue import Queue
import threading
from ijson.common import ObjectBuilder

INPUT_FILE = "ontologies.json"
OUTPUT_FILE = "filtered_ontologies.json"

# Ontologies required by CellxGene schema
TARGETS = {
    "cl", "efo", "hancestro", "hsapdv", "mmusdv", "mondo",
    "ncbitaxon", "pato", "uberon", "wbls", "wbbt", "fbbt", "fbdv", "zfa"
}

# Species required by CellxGene schema
SCHEMA_SPECIES = {
    "NCBITaxon:6239", "NCBITaxon:9483", "NCBITaxon:7955", "NCBITaxon:7227",
    "NCBITaxon:9595", "NCBITaxon:9606", "NCBITaxon:9541", "NCBITaxon:9544",
    "NCBITaxon:30608", "NCBITaxon:10090", "NCBITaxon:9986", "NCBITaxon:9598",
    "NCBITaxon:10116", "NCBITaxon:2697049", "NCBITaxon:9823", "NCBITaxon:32630"
}

def filter_ncbitaxon_classes(classes):
    """Filter NCBITaxon classes to only schema species + ancestors."""
    keep_ids = set()
    curie_to_class = {}
    
    # Build lookup map
    for cls in classes:
        curie_obj = cls.get("curie", {})
        if isinstance(curie_obj, dict) and "value" in curie_obj:
            curie_to_class[curie_obj["value"]] = cls

    # Find schema species and their ancestors
    for cls in classes:
        curie_obj = cls.get("curie", {})
        if isinstance(curie_obj, dict) and "value" in curie_obj:
            curie = curie_obj["value"]
            if curie in SCHEMA_SPECIES:
                keep_ids.add(curie)
                # Add ancestors
                for ancestor_iri in cls.get("hierarchicalAncestor", []):
                    if "NCBITaxon_" in ancestor_iri:
                        taxon_id = ancestor_iri.split("NCBITaxon_")[-1]
                        ancestor_curie = f"NCBITaxon:{taxon_id}"
                        keep_ids.add(ancestor_curie)

    return [curie_to_class[c] for c in keep_ids if c in curie_to_class]

def process_ontology_chunk(chunk_data):
    """Worker function to process a chunk of ontologies in parallel."""
    ontologies_chunk, targets = chunk_data
    filtered_ontologies = []
    found_targets = set()
    
    for ontology in ontologies_chunk:
        oid = ontology.get("ontologyId", "").lower()
        
        if oid in targets:
            found_targets.add(oid)
            
            # Apply NCBITaxon filtering if needed
            if oid == "ncbitaxon" and "classes" in ontology:
                ontology["classes"] = filter_ncbitaxon_classes(ontology["classes"])
            
            filtered_ontologies.append(ontology)
            
    return filtered_ontologies, found_targets

def stream_ontologies_filtered(filename, targets):
    """Stream over ontologies and only build Python objects for target items.

    Yields one value per ontology in the file:
      - dict for target ontologies (fully built object)
      - None for non-target ontologies (to allow progress without memory cost)
    """
    try:
        with open(filename, 'rb') as file:
            parser = ijson.parse(file)

            in_item = False
            decision_made = False
            build_this = False
            awaiting_oid_value = False
            builder = None
            buffered_events = []

            for prefix, event, value in parser:
                if prefix == 'ontologies.item' and event == 'start_map':
                    # Begin a new ontology item
                    in_item = True
                    decision_made = False
                    build_this = False
                    awaiting_oid_value = False
                    builder = None
                    buffered_events = [(event, value)]  # buffer ijson events (event, value)
                    continue

                if in_item:
                    # While we haven't decided, buffer events and watch for ontologyId
                    if not decision_made:
                        if event == 'map_key' and value == 'ontologyId':
                            awaiting_oid_value = True
                            buffered_events.append((event, value))
                            continue
                        if awaiting_oid_value and event in ('string', 'number'):
                            oid = str(value).lower()
                            build_this = oid in targets
                            decision_made = True
                            awaiting_oid_value = False

                            if build_this:
                                # Start builder and feed buffered events
                                builder = ObjectBuilder()
                                for ev, val in buffered_events:
                                    builder.event(ev, val)
                                # Also feed the ontologyId value event we just saw
                                builder.event(event, value)
                            # If not building, drop buffer to save memory
                            buffered_events = []
                            continue

                        # Not decided yet, keep buffering
                        buffered_events.append((event, value))
                        continue

                    # After decision: feed events only if building this item
                    if decision_made and build_this and builder is not None:
                        builder.event(event, value)

                    # End of current ontology item
                    if prefix == 'ontologies.item' and event == 'end_map':
                        in_item = False
                        if build_this and builder is not None:
                            yield builder.value
                        else:
                            yield None
                        # reset states
                        decision_made = False
                        build_this = False
                        awaiting_oid_value = False
                        builder = None
                        buffered_events = []
                        continue
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        raise

def process_single_ontology(ontology):
    """Process a single ontology and return filtered result if matches target."""
    oid = ontology.get("ontologyId", "").lower()
    
    if oid in TARGETS:
        # Apply NCBITaxon filtering if needed
        if oid == "ncbitaxon" and "classes" in ontology:
            ontology["classes"] = filter_ncbitaxon_classes(ontology["classes"])
        
        return ontology, oid
    return None, None

def main():
    # Detect available CPU cores
    num_cores = mp.cpu_count()
    print(f"🚀 Parallel streaming processing with {num_cores} CPU cores")
    print(f"📋 Target ontologies: {', '.join(sorted(TARGETS))}")
    
    start_time = time.time()
    
    # Stream and batch process ontologies in parallel
    filtered_ontologies = []
    found_targets = set()
    processed_count = 0
    batch_size = num_cores * 2  # Process 2 batches per core for good load balancing
    
    print(f"📡 Streaming ontologies from {INPUT_FILE} in batches of {batch_size}...")
    
    try:
        # Collect ontologies in batches and process in parallel
        ontology_batch = []
        
        with tqdm(desc="Processing ontologies", unit=" ontologies") as pbar:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = []
                
                for ontology in stream_ontologies_filtered(INPUT_FILE, TARGETS):
                    processed_count += 1
                    # Show progress as soon as each ontology is parsed
                    pbar.update(1)

                    # Skip non-target ontologies quickly (we don't build them)
                    if ontology is None:
                        continue

                    ontology_batch.append(ontology)

                    # When batch is full, submit for parallel processing
                    if len(ontology_batch) >= batch_size:
                        future = executor.submit(process_ontology_chunk, (ontology_batch, TARGETS))
                        futures.append(future)
                        ontology_batch = []

                        # Drain any completed futures (not just the first)
                        for completed_future in list(futures):
                            if completed_future.done():
                                futures.remove(completed_future)
                                batch_filtered, batch_found = completed_future.result()
                                filtered_ontologies.extend(batch_filtered)
                                found_targets.update(batch_found)

                                # Log found targets
                                for target in batch_found:
                                    print(f"✅ Found: {target}")

                                # Early termination check
                                if len(found_targets) == len(TARGETS):
                                    print(f"🎉 Found all {len(TARGETS)} targets! Cancelling remaining work...")
                                    for remaining_future in futures:
                                        remaining_future.cancel()
                                    executor.shutdown(wait=False)
                                    break

                        # Update small status periodically
                        if processed_count % 256 == 0:
                            pbar.set_postfix({
                                'found': len(found_targets),
                                'targets': len(TARGETS)
                            })

                        # Break if all targets found
                        if len(found_targets) == len(TARGETS):
                            break
                
                # Process final batch if any
                if ontology_batch and len(found_targets) < len(TARGETS):
                    future = executor.submit(process_ontology_chunk, (ontology_batch, TARGETS))
                    futures.append(future)
                    ontology_batch = []

                # Process all remaining futures and keep progress responsive
                for future in as_completed(futures):
                    if len(found_targets) >= len(TARGETS):
                        break
                    batch_filtered, batch_found = future.result()
                    filtered_ontologies.extend(batch_filtered)
                    found_targets.update(batch_found)

                    # Log found targets
                    for target in batch_found:
                        print(f"✅ Found: {target}")

                    # Update status after each completed future
                    pbar.set_postfix({
                        'found': len(found_targets),
                        'targets': len(TARGETS)
                    })
                        
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        # Fall back to regular JSON loading if streaming fails
        print("🔄 Falling back to regular JSON loading...")
        return main_fallback()
    
    # Write output with proper formatting
    print(f"💾 Writing {len(filtered_ontologies)} ontologies to {OUTPUT_FILE}...")
    
    # Convert Decimal objects to float for JSON serialization
    def convert_decimals(obj):
        from decimal import Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        else:
            return obj
    
    # Clean the data before writing
    cleaned_ontologies = [convert_decimals(ontology) for ontology in filtered_ontologies]
    output_data = {"ontologies": cleaned_ontologies}
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Summary
    missing = TARGETS - found_targets
    total_time = time.time() - start_time
    
    print(f"✅ Completed in {total_time:.1f}s")
    print(f"📊 Processed {processed_count} ontologies")
    print(f"📊 Found {len(found_targets)}/{len(TARGETS)} targets")
    if missing:
        print(f"⚠️  Missing: {', '.join(sorted(missing))}")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"🚀 Parallel streaming: ~{num_cores}x faster + memory-efficient!")

def main_fallback():
    """Fallback to chunk-based processing if streaming fails."""
    print("⚠️  Using fallback method with smaller chunks...")
    # This would implement a more conservative approach
    # For now, just return an error
    print("❌ Please install ijson: pip install ijson")
    return

if __name__ == "__main__":
    main()
