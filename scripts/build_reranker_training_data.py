#!/usr/bin/env python3
"""
Generate training data for cross-encoder reranker from BOND benchmark.

This script:
1. Loads benchmark data (author_term → ontology_id pairs)
2. Runs BOND retrieval (Dense + BM25 + Exact → RRF fusion)
3. Labels candidates as positive (matches correct ID) or negative
4. Creates (query, candidate, label) training examples
5. Outputs JSONL format for cross-encoder training

Usage:
    python build_reranker_training_data.py \
        --benchmark_path path/to/bond_czi_benchmark_data_hydrated_train.jsonl \
        --output_path reranker_training_data/train.jsonl \
        --num_hard_negatives 10 \
        --num_random_negatives 5 \
        --sample_size 100  # optional, for testing
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import defaultdict

# Add BOND to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bond.pipeline import BondMatcher
from bond.config import BondSettings
from bond.logger import logger


def format_query_text(
    field_type: str,
    author_term: str,
    tissue: Optional[str],
    organism: Optional[str]
) -> str:
    """Format query text in the same format as anchor text during training."""
    tissue_str = tissue if tissue else "unknown"
    organism_str = organism if organism else "unknown"
    return f"{field_type}: {author_term}; tissue: {tissue_str}; organism: {organism_str}"


def format_candidate_text(candidate: Dict[str, Any], max_def_chars: int = 200) -> str:
    """
    Format candidate text to match the positive text format used in embedding training.

    Args:
        candidate: Dictionary with keys: label, synonyms_exact, synonyms_related,
                   synonyms_broad, definition
        max_def_chars: Maximum characters for definition

    Returns:
        Formatted candidate text
    """
    label = candidate.get('label', '')

    # Gather synonyms (exact -> related -> broad, capped at 10 total)
    synonyms = []
    for syn_type in ['synonyms_exact', 'synonyms_related', 'synonyms_broad']:
        syn_list = candidate.get(syn_type, []) or []
        synonyms.extend(syn_list)

    # Deduplicate and cap
    seen = set()
    unique_syns = []
    for syn in synonyms:
        if syn and syn not in seen:
            seen.add(syn)
            unique_syns.append(syn)
        if len(unique_syns) >= 10:
            break

    # Truncate definition
    definition = (candidate.get('definition', '') or '')[:max_def_chars]

    # Build formatted text
    parts = [f"label: {label}"]
    if unique_syns:
        parts.append(f"synonyms: {' | '.join(unique_syns)}")
    if definition:
        parts.append(f"definition: {definition}")

    return "; ".join(parts)


def generate_reranker_training_data(
    benchmark_jsonl: str,
    output_jsonl: str,
    num_hard_negatives: int = 10,
    num_random_negatives: int = 5,
    sample_size: Optional[int] = None,
    retrieval_only: bool = True,
    top_k_retrieval: int = 50
):
    """
    Generate reranker training data from benchmark.

    Args:
        benchmark_jsonl: Path to benchmark JSONL file
        output_jsonl: Path to output JSONL file
        num_hard_negatives: Number of hard negatives (from retrieval but wrong)
        num_random_negatives: Number of random negatives (same field, not retrieved)
        sample_size: If set, only process this many examples (for testing)
        retrieval_only: Skip LLM stage (faster)
        top_k_retrieval: Number of candidates to retrieve
    """

    logger.info(f"Loading benchmark data from: {benchmark_jsonl}")

    # Load benchmark data
    benchmark_data = []
    with open(benchmark_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                benchmark_data.append(json.loads(line))

    logger.info(f"Loaded {len(benchmark_data)} benchmark examples")

    # Sample if requested
    if sample_size and sample_size < len(benchmark_data):
        logger.info(f"Sampling {sample_size} examples for testing")
        benchmark_data = random.sample(benchmark_data, sample_size)

    # Initialize BOND pipeline
    logger.info("Initializing BOND pipeline...")
    settings = BondSettings()
    settings.retrieval_only = retrieval_only  # Skip LLM for faster processing
    bond = BondMatcher(settings=settings)

    # Collect all ontology IDs per field for random negatives
    field_to_ontology_ids = defaultdict(set)
    for row in benchmark_data:
        field_type = row.get('field_type', '')
        ontology_id = row.get('original_ontology_id', '')
        if field_type and ontology_id:
            field_to_ontology_ids[field_type].add(ontology_id)

    logger.info(f"Field type distribution: {dict((k, len(v)) for k, v in field_to_ontology_ids.items())}")

    # Generate training examples
    training_examples = []
    stats = {
        'total_queries': 0,
        'positives': 0,
        'hard_negatives': 0,
        'random_negatives': 0,
        'missing_correct_id': 0,
        'retrieval_errors': 0
    }

    logger.info("Generating reranker training examples...")

    for row in tqdm(benchmark_data, desc="Processing benchmark"):
        try:
            # Extract query info
            query = row.get('author_term', '')
            field_type = row.get('field_type', '')
            tissue = row.get('tissue')
            organism = row.get('organism', '')
            correct_id = row.get('original_ontology_id', '')

            if not query or not field_type or not correct_id:
                continue

            stats['total_queries'] += 1

            # Run BOND retrieval (no LLM)
            try:
                result = bond.query(
                    query=query,
                    field_name=field_type,
                    organism=organism,
                    tissue=tissue,
                    topk_final=top_k_retrieval,
                    return_trace=False
                )
            except Exception as e:
                logger.warning(f"Retrieval error for '{query}': {e}")
                stats['retrieval_errors'] += 1
                continue

            # Get candidates
            candidates = result.get('results', [])

            if not candidates:
                logger.debug(f"No candidates retrieved for '{query}'")
                continue

            # Build query text
            query_text = format_query_text(field_type, query, tissue, organism)

            # Track retrieved IDs
            retrieved_ids = {c['id'] for c in candidates}

            # Process each candidate as positive or hard negative
            for rank, candidate in enumerate(candidates):
                candidate_id = candidate.get('id', '')
                candidate_text = format_candidate_text(candidate)
                retrieval_score = candidate.get('retrieval_confidence', 0.0)

                # Label: 1 if matches correct_id, 0 otherwise
                label = 1 if candidate_id == correct_id else 0

                if label == 1:
                    stats['positives'] += 1
                else:
                    stats['hard_negatives'] += 1
                    # Limit hard negatives
                    if rank >= num_hard_negatives:
                        continue

                training_examples.append({
                    'query': query_text,
                    'candidate': candidate_text,
                    'candidate_id': candidate_id,
                    'correct_id': correct_id,
                    'label': label,
                    'retrieval_score': retrieval_score,
                    'retrieval_rank': rank,
                    'example_type': 'positive' if label == 1 else 'hard_negative'
                })

            # If correct ID not in retrieved candidates, add it as a positive
            if correct_id not in retrieved_ids:
                stats['missing_correct_id'] += 1

                # Fetch correct ontology term from database
                try:
                    conn = bond.get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        f"SELECT curie, label, definition, synonyms_exact, synonyms_related, "
                        f"synonyms_broad FROM {bond.cfg.table_terms} WHERE curie = ?",
                        (correct_id,)
                    )
                    row_data = cur.fetchone()

                    if row_data:
                        def _split_pipe(x):
                            if not x:
                                return []
                            return [t.strip() for t in str(x).split("|") if t.strip()]

                        correct_term = {
                            'id': row_data[0],
                            'label': row_data[1],
                            'definition': row_data[2],
                            'synonyms_exact': _split_pipe(row_data[3]),
                            'synonyms_related': _split_pipe(row_data[4]),
                            'synonyms_broad': _split_pipe(row_data[5])
                        }

                        candidate_text = format_candidate_text(correct_term)

                        training_examples.append({
                            'query': query_text,
                            'candidate': candidate_text,
                            'candidate_id': correct_id,
                            'correct_id': correct_id,
                            'label': 1,
                            'retrieval_score': 0.0,
                            'retrieval_rank': 999,
                            'example_type': 'positive_missed'
                        })
                        stats['positives'] += 1
                except Exception as e:
                    logger.debug(f"Could not fetch correct term {correct_id}: {e}")

            # Add random negatives (same field, different ontology ID)
            if num_random_negatives > 0:
                field_ids = field_to_ontology_ids.get(field_type, set())
                # Exclude correct ID and already retrieved IDs
                available_ids = field_ids - retrieved_ids - {correct_id}

                if available_ids:
                    # Sample random negatives
                    num_to_sample = min(num_random_negatives, len(available_ids))
                    random_ids = random.sample(list(available_ids), num_to_sample)

                    try:
                        conn = bond.get_connection()
                        cur = conn.cursor()
                        qmarks = ",".join("?" for _ in random_ids)
                        cur.execute(
                            f"SELECT curie, label, definition, synonyms_exact, synonyms_related, "
                            f"synonyms_broad FROM {bond.cfg.table_terms} WHERE curie IN ({qmarks})",
                            random_ids
                        )

                        for row_data in cur.fetchall():
                            def _split_pipe(x):
                                if not x:
                                    return []
                                return [t.strip() for t in str(x).split("|") if t.strip()]

                            random_term = {
                                'id': row_data[0],
                                'label': row_data[1],
                                'definition': row_data[2],
                                'synonyms_exact': _split_pipe(row_data[3]),
                                'synonyms_related': _split_pipe(row_data[4]),
                                'synonyms_broad': _split_pipe(row_data[5])
                            }

                            candidate_text = format_candidate_text(random_term)

                            training_examples.append({
                                'query': query_text,
                                'candidate': candidate_text,
                                'candidate_id': row_data[0],
                                'correct_id': correct_id,
                                'label': 0,
                                'retrieval_score': 0.0,
                                'retrieval_rank': 999,
                                'example_type': 'random_negative'
                            })
                            stats['random_negatives'] += 1
                    except Exception as e:
                        logger.debug(f"Could not fetch random negatives: {e}")

        except Exception as e:
            logger.error(f"Error processing row: {e}")
            continue

    # Write to JSONL
    logger.info(f"Writing {len(training_examples)} training examples to {output_jsonl}")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("RERANKER TRAINING DATA GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total queries processed: {stats['total_queries']}")
    logger.info(f"Total training examples: {len(training_examples)}")
    logger.info(f"  - Positives: {stats['positives']} ({stats['positives']/len(training_examples)*100:.1f}%)")
    logger.info(f"    - Retrieved: {stats['positives'] - stats['missing_correct_id']}")
    logger.info(f"    - Missed (added manually): {stats['missing_correct_id']}")
    logger.info(f"  - Hard negatives: {stats['hard_negatives']} ({stats['hard_negatives']/len(training_examples)*100:.1f}%)")
    logger.info(f"  - Random negatives: {stats['random_negatives']} ({stats['random_negatives']/len(training_examples)*100:.1f}%)")
    logger.info(f"Retrieval errors: {stats['retrieval_errors']}")
    logger.info(f"\nOutput written to: {output_jsonl}")
    logger.info("="*60)

    # Close BOND
    bond.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate reranker training data from BOND benchmark"
    )
    parser.add_argument(
        '--benchmark_path',
        type=str,
        required=True,
        help='Path to benchmark JSONL file (e.g., bond_czi_benchmark_data_hydrated_train.jsonl)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output JSONL file for reranker training data'
    )
    parser.add_argument(
        '--num_hard_negatives',
        type=int,
        default=10,
        help='Number of hard negatives per query (retrieved but wrong) (default: 10)'
    )
    parser.add_argument(
        '--num_random_negatives',
        type=int,
        default=5,
        help='Number of random negatives per query (same field, not retrieved) (default: 5)'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Sample size for testing (default: None = use all data)'
    )
    parser.add_argument(
        '--top_k_retrieval',
        type=int,
        default=50,
        help='Number of candidates to retrieve from BOND (default: 50)'
    )

    args = parser.parse_args()

    generate_reranker_training_data(
        benchmark_jsonl=args.benchmark_path,
        output_jsonl=args.output_path,
        num_hard_negatives=args.num_hard_negatives,
        num_random_negatives=args.num_random_negatives,
        sample_size=args.sample_size,
        top_k_retrieval=args.top_k_retrieval
    )


if __name__ == '__main__':
    main()
