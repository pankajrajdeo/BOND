#!/usr/bin/env python3
"""
Generate cross-encoder training data from BOND SQLite database.

This script creates training pairs for cross-encoder models by:
1. Sampling queries from existing ontology terms
2. Creating positive pairs (query + correct term)
3. Creating negative pairs (query + incorrect terms)
4. Exporting in a format suitable for cross-encoder training
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_db(assets_path: str) -> sqlite3.Connection:
    """Connect to the BOND SQLite database."""
    db_path = Path(assets_path) / "ontology.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def get_all_terms(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Retrieve all terms from the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, label, definition, source, iri,
               syn_exact AS synonyms_exact,
               syn_related AS synonyms_related,
               syn_broad AS synonyms_broad
        FROM terms
        WHERE label IS NOT NULL AND label != ''
        ORDER BY source, label
    """)
    
    terms = []
    for row in cursor.fetchall():
        term = dict(row)
        # Convert JSON strings back to lists
        for key in ['synonyms_exact', 'synonyms_related', 'synonyms_broad']:
            if term[key]:
                try:
                    term[key] = json.loads(term[key])
                except json.JSONDecodeError:
                    term[key] = []
            else:
                term[key] = []
        terms.append(term)
    
    return terms

def create_query_variants(term: Dict[str, Any]) -> List[str]:
    """Create query variants from a term's label and synonyms."""
    variants = []
    
    # Add the main label
    if term['label']:
        variants.append(term['label'])
    
    # Add exact synonyms
    for syn in term.get('synonyms_exact', [])[:3]:  # Limit to 3
        if syn and syn != term['label']:
            variants.append(syn)
    
    # Add related synonyms (shorter ones are better for queries)
    for syn in term.get('synonyms_related', [])[:2]:
        if syn and len(syn.split()) <= 3:  # Prefer shorter phrases
            variants.append(syn)
    
    # Add broad synonyms (only if they're not too generic)
    for syn in term.get('synonyms_broad', [])[:1]:
        if syn and len(syn.split()) <= 2:
            variants.append(syn)
    
    return list(set(variants))  # Remove duplicates

def create_training_pairs(terms: List[Dict[str, Any]], 
                         num_pairs: int = 10000,
                         positive_ratio: float = 0.5) -> List[Dict[str, Any]]:
    """Create training pairs for cross-encoder."""
    training_data = []
    
    # Create positive pairs (query + correct term)
    num_positive = int(num_pairs * positive_ratio)
    positive_pairs = []
    
    for term in terms:
        variants = create_query_variants(term)
        for variant in variants:
            positive_pairs.append({
                'query': variant,
                'candidate': term,
                'label': 1,  # Positive pair
                'source': term['source']
            })
    
    # Sample positive pairs
    if len(positive_pairs) > num_positive:
        positive_pairs = random.sample(positive_pairs, num_positive)
    
    training_data.extend(positive_pairs)
    
    # Create negative pairs (query + incorrect terms)
    num_negative = num_pairs - len(positive_pairs)
    negative_pairs = []
    
    for i, term in enumerate(terms):
        variants = create_query_variants(term)
        for variant in variants:
            # Find negative candidates from different sources or different semantic areas
            negative_candidates = []
            
            # Strategy 1: Different source
            for other_term in terms:
                if other_term['source'] != term['source']:
                    negative_candidates.append(other_term)
                    if len(negative_candidates) >= 3:
                        break
            
            # Strategy 2: Same source but different semantic area
            if len(negative_candidates) < 3:
                for other_term in terms:
                    if (other_term['source'] == term['source'] and 
                        other_term['id'] != term['id']):
                        negative_candidates.append(other_term)
                        if len(negative_candidates) >= 5:
                            break
            
            # Create negative pairs
            for neg_candidate in negative_candidates[:2]:  # Max 2 negative per positive
                negative_pairs.append({
                    'query': variant,
                    'candidate': neg_candidate,
                    'label': 0,  # Negative pair
                    'source': f"{term['source']}_vs_{neg_candidate['source']}"
                })
    
    # Sample negative pairs
    if len(negative_pairs) > num_negative:
        negative_pairs = random.sample(negative_pairs, num_negative)
    
    training_data.extend(negative_pairs)
    
    # Shuffle the data
    random.shuffle(training_data)
    
    return training_data

def export_training_data(training_data: List[Dict[str, Any]], 
                        output_path: str,
                        format_type: str = 'json') -> None:
    """Export training data in the specified format."""
    output_file = Path(output_path)
    
    if format_type == 'json':
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    elif format_type == 'tsv':
        with open(output_file, 'w') as f:
            # Write header
            f.write("query\tcandidate_id\tcandidate_label\tcandidate_definition\tcandidate_source\tlabel\tsource\n")
            
            for pair in training_data:
                candidate = pair['candidate']
                f.write(f"{pair['query']}\t{candidate['id']}\t{candidate['label']}\t{candidate.get('definition', '')}\t{candidate['source']}\t{pair['label']}\t{pair['source']}\n")
    
    elif format_type == 'jsonl':
        with open(output_file, 'w') as f:
            for pair in training_data:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    logger.info(f"Exported {len(training_data)} training pairs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate cross-encoder training data from BOND database")
    parser.add_argument("--assets", default="assets", help="Path to assets directory")
    parser.add_argument("--output", default="cross_encoder_training_data.json", help="Output file path")
    parser.add_argument("--format", choices=['json', 'tsv', 'jsonl'], default='json', 
                       help="Output format")
    parser.add_argument("--num-pairs", type=int, default=10000, 
                       help="Number of training pairs to generate")
    parser.add_argument("--positive-ratio", type=float, default=0.5,
                       help="Ratio of positive pairs (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    try:
        # Connect to database
        logger.info("Connecting to BOND database...")
        conn = connect_db(args.assets)
        
        # Get all terms
        logger.info("Retrieving terms from database...")
        terms = get_all_terms(conn)
        logger.info(f"Found {len(terms)} terms in database")
        
        # Create training pairs
        logger.info("Creating training pairs...")
        training_data = create_training_pairs(
            terms, 
            num_pairs=args.num_pairs,
            positive_ratio=args.positive_ratio
        )
        
        # Export data
        logger.info("Exporting training data...")
        export_training_data(training_data, args.output, args.format)
        
        # Print statistics
        positive_count = sum(1 for pair in training_data if pair['label'] == 1)
        negative_count = len(training_data) - positive_count
        
        logger.info(f"Training data generation complete!")
        logger.info(f"Total pairs: {len(training_data)}")
        logger.info(f"Positive pairs: {positive_count}")
        logger.info(f"Negative pairs: {negative_count}")
        logger.info(f"Output file: {args.output}")
        
        # Show sample data
        logger.info("\nSample positive pair:")
        positive_sample = next(pair for pair in training_data if pair['label'] == 1)
        logger.info(f"Query: '{positive_sample['query']}'")
        logger.info(f"Candidate: '{positive_sample['candidate']['label']}' ({positive_sample['candidate']['source']})")
        
        logger.info("\nSample negative pair:")
        negative_sample = next(pair for pair in training_data if pair['label'] == 0)
        logger.info(f"Query: '{negative_sample['query']}'")
        logger.info(f"Candidate: '{negative_sample['candidate']['label']}' ({negative_sample['candidate']['source']})")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
