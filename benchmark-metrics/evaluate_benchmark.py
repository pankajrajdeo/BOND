#!/usr/bin/env python3
"""
BOND Benchmark Evaluation Script

Evaluates BOND on the CZI benchmark dataset with proper rate limiting for OpenAI API.

Rate Limits (OpenAI Tier 4):
- Requests: 10,000/min, unlimited/day
- Tokens: 10M/min, unlimited/day

Usage:
    python evaluate_benchmark.py --test_set test_set.jsonl --output results.json
    python evaluate_benchmark.py --test_set test_set.jsonl --output results.json --resume results.json --start_idx 100
"""

import argparse
import json
import time
import sys
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import statistics

# Add parent directory to path to import bond
sys.path.insert(0, str(Path(__file__).parent.parent))

from bond.pipeline import BondMatcher
from bond.config import BondSettings


class RateLimiter:
    """Rate limiter for OpenAI API with per-minute and per-day limits for both requests and tokens."""
    
    def __init__(self, requests_per_min: int = 10000, requests_per_day: int = 1000000000, 
                 tokens_per_min: int = 10000000, tokens_per_day: int = 1000000000):
        self.requests_per_min = requests_per_min
        self.requests_per_day = requests_per_day
        self.tokens_per_min = tokens_per_min
        self.tokens_per_day = tokens_per_day
        self.min_interval = 60.0 / requests_per_min  # seconds between requests
        self.request_times: List[float] = []
        self.token_usage: List[tuple[float, int]] = []  # (timestamp, tokens)
        self.day_start = time.time()
        self.daily_count = 0
        self.daily_tokens = 0
        
    def wait_if_needed(self, estimated_tokens: int = 2000):
        """
        Wait if we're hitting rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for this request (default 2000 for safety)
        """
        now = time.time()
        
        # Reset daily counters if new day
        if now - self.day_start > 86400:  # 24 hours
            self.daily_count = 0
            self.daily_tokens = 0
            self.day_start = now
        
        # Check daily token limit
        if self.daily_tokens + estimated_tokens > self.tokens_per_day:
            wait_time = 86400 - (now - self.day_start)
            if wait_time > 0:
                print(f"[RATE LIMIT] Daily token limit would be exceeded. Waiting {wait_time/3600:.1f} hours...")
                time.sleep(wait_time)
                self.daily_tokens = 0
                self.daily_count = 0
                self.day_start = time.time()
        
        # Check daily request limit
        if self.daily_count >= self.requests_per_day:
            wait_time = 86400 - (now - self.day_start)
            if wait_time > 0:
                print(f"[RATE LIMIT] Daily request limit reached. Waiting {wait_time/3600:.1f} hours...")
                time.sleep(wait_time)
                self.daily_count = 0
                self.daily_tokens = 0
                self.day_start = time.time()
        
        # Remove requests/tokens older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if now - t < 60]
        
        # Calculate current minute's token usage
        current_minute_tokens = sum(tokens for _, tokens in self.token_usage)
        
        # Check per-minute token limit
        if current_minute_tokens + estimated_tokens > self.tokens_per_min:
            # Find when we can make the request
            if self.token_usage:
                oldest_token_time = min(t for t, _ in self.token_usage)
                wait_time = 60 - (now - oldest_token_time) + 1.0  # Add 1s buffer
                if wait_time > 0:
                    print(f"[RATE LIMIT] Token limit ({current_minute_tokens}/{self.tokens_per_min}). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # Clean up after waiting
                    now = time.time()
                    self.token_usage = [(t, tokens) for t, tokens in self.token_usage if now - t < 60]
                    current_minute_tokens = sum(tokens for _, tokens in self.token_usage)
        
        # Check per-minute request limit
        if len(self.request_times) >= self.requests_per_min:
            oldest = min(self.request_times)
            wait_time = 60 - (now - oldest) + 0.1  # Add small buffer
            if wait_time > 0:
                print(f"[RATE LIMIT] Per-minute request limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                # Clean up again after waiting
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Ensure minimum interval between requests (to spread out token usage)
        # For Tier 4: 10,000 RPM and 10M TPM
        # With ~2200 tokens/request, token limit allows ~4545 requests/min
        # So ideal interval ≈ 60/4545 ≈ 0.013s (13ms), but use 20ms for safety margin
        if self.request_times:
            last_request = max(self.request_times)
            elapsed = now - last_request
            # Calculate token-based minimum interval
            # Token-limited throughput = tokens_per_min / estimated_tokens_per_request
            max_requests_by_tokens = self.tokens_per_min / max(estimated_tokens, 1)
            # Use 85% of theoretical max to stay safe
            safe_requests_per_min = max_requests_by_tokens * 0.85
            token_based_interval = 60.0 / max(safe_requests_per_min, 1)
            # Use the larger of: request-based interval or token-based interval
            min_interval = max(self.min_interval, token_based_interval)
            # Add small safety margin (5ms) to avoid hitting limits
            min_interval = min_interval + 0.005
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                time.sleep(wait_time)
        
        # Record this request
        now = time.time()
        self.request_times.append(now)
        self.token_usage.append((now, estimated_tokens))
        self.daily_count += 1
        self.daily_tokens += estimated_tokens


def load_test_set(jsonl_path: str, random_sample: Optional[int] = None, seed: Optional[int] = None, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load test set from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file (default: test.jsonl)
        random_sample: If provided, randomly sample this many examples
        seed: Random seed for reproducibility
        dataset_id: If provided, filter to only examples with this dataset_id
    """
    if seed is not None:
        random.seed(seed)
    
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                # Filter by dataset_id if provided
                if dataset_id is None or example.get("dataset_id") == dataset_id:
                    examples.append(example)
    
    if dataset_id:
        print(f"[INFO] Filtered to {len(examples)} examples with dataset_id: {dataset_id}")
    
    # Random sampling if requested
    if random_sample is not None and random_sample < len(examples):
        examples = random.sample(examples, random_sample)
        print(f"[INFO] Randomly sampled {random_sample} examples from {len(examples)} total")
    
    return examples


def evaluate_example(
    matcher: BondMatcher,
    example: Dict[str, Any],
    rate_limiter: Optional[RateLimiter] = None,
    n_expansions: Optional[int] = None,
    exact_only: bool = False,
    skip_disambiguation: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single example.
    
    Returns:
        {
            "example_idx": int,
            "author_term": str,
            "field_type": str,
            "organism": Optional[str],
            "tissue": Optional[str],
            "ground_truth_id": str,
            "ground_truth_label": Optional[str],
            "predicted_id": Optional[str],
            "top_3_ids_llm": List[str],  # From LLM ranking (llm_ranked), falls back to retrieval if LLM not available
            "top_5_ids_retrieval": List[str],  # From retrieval ranking (results)
            "rank": Optional[int],  # Rank of ground truth (1-indexed, None if not found)
            "correct": bool,
            "top_3_correct": bool,
            "top_5_correct": bool,
            "mrr": float,  # 1/rank if found, 0 otherwise
            "error": Optional[str],
            "retrieval_confidence": Optional[float],
            "llm_confidence": Optional[float],
            "llm_reason": Optional[str],
            "predicted_label": Optional[str],
            "top_3_labels_llm": List[str],  # From LLM ranking
            "top_5_labels_retrieval": List[str],  # From retrieval ranking
        }
    """
    result = {
        "author_term": example["author_term"],
        "field_type": example["field_type"],
        "organism": example.get("organism"),
        "tissue": example.get("tissue"),
        "ground_truth_id": example["resolved_ontology_id"],
        "ground_truth_label": example.get("resolved_label"),
        "predicted_id": None,
        "top_3_ids_llm": [],
        "top_5_ids_retrieval": [],
        "rank": None,
        "correct": False,
        "top_3_correct": False,
        "top_5_correct": False,
        "mrr": 0.0,
        "error": None,
        "retrieval_confidence": None,
        "llm_confidence": None,
        "llm_reason": None,
        "predicted_label": None,
        "top_3_labels_llm": [],
        "top_5_labels_retrieval": [],
    }
    
    try:
        # Apply rate limiting before LLM calls
        # Estimate tokens based on what LLM calls will be made:
        # - Expansion: ~500 tokens (small prompt + response)
        # - Disambiguation: ~1700 tokens (large prompt with candidates + response)
        estimated_tokens = 0
        if not skip_disambiguation and (n_expansions is None or n_expansions > 0):
            estimated_tokens += 500  # Expansion LLM
        if not skip_disambiguation:
            estimated_tokens += 1700  # Disambiguation LLM (larger, more tokens)
        elif n_expansions == 0:
            estimated_tokens = 1700  # Only disambiguation
        
        if rate_limiter:
            rate_limiter.wait_if_needed(estimated_tokens=estimated_tokens)
        
        # Call BOND
        query_kwargs = {
            "query": example["author_term"],
            "field_name": example["field_type"],
            "organism": example["organism"],
            "tissue": example.get("tissue") or "",
            "return_trace": True,
        }
        
        # Add optional parameters
        if n_expansions is not None:
            query_kwargs["n_expansions"] = n_expansions
        if exact_only:
            query_kwargs["exact_only"] = True
        
        bond_result = matcher.query(**query_kwargs)
        
        # If skipping disambiguation, use top-1 from retrieval instead
        if skip_disambiguation and bond_result.get("chosen"):
            # Replace chosen with top retrieval result
            results_list = bond_result.get("results", [])
            if results_list:
                bond_result["chosen"] = results_list[0]
                bond_result["chosen"]["reason"] = "Top retrieval result (disambiguation skipped)"
                bond_result["chosen"]["llm_confidence"] = None
        
        # Extract predictions
        chosen = bond_result.get("chosen")
        if chosen:
            result["predicted_id"] = chosen.get("id")
            result["predicted_label"] = chosen.get("label")
            result["retrieval_confidence"] = chosen.get("retrieval_confidence")
            result["llm_confidence"] = chosen.get("llm_confidence")
            result["llm_reason"] = chosen.get("reason")
            # Check if LLM was actually called (even if confidence is None)
            llm_was_called = chosen.get("_llm_called", False)
            result["_llm_called"] = llm_was_called  # Store for retry logic
            
            # Log LLM completion status
            if result["llm_confidence"] is not None:
                print(f"[SUCCESS] LLM completed for '{example['author_term']}' - confidence: {result['llm_confidence']}")
            elif llm_was_called:
                print(f"[INFO] LLM completed for '{example['author_term']}' but confidence is None (may be abstain)")
            elif not skip_disambiguation:
                print(f"[WARNING] LLM should have run but was not called for: {example['author_term']}")
        else:
            # No chosen result - LLM definitely didn't run
            result["_llm_called"] = False
        
        # Get top-5 from retrieval results (retrieval ranking)
        results_list = bond_result.get("results", [])
        result["top_5_ids_retrieval"] = [r.get("id") for r in results_list[:5] if r.get("id")]
        result["top_5_labels_retrieval"] = [r.get("label", "") for r in results_list[:5] if r.get("id")]
        
        # Get top-3 from LLM ranked (LLM ranking) - falls back to top-3 from retrieval if LLM not available
        llm_ranked = bond_result.get("llm_ranked", [])
        if llm_ranked:
            result["top_3_ids_llm"] = [r.get("id") for r in llm_ranked[:3] if r.get("id")]
            result["top_3_labels_llm"] = [r.get("label", "") for r in llm_ranked[:3] if r.get("id")]
        else:
            result["top_3_ids_llm"] = result["top_5_ids_retrieval"][:3]
            result["top_3_labels_llm"] = result["top_5_labels_retrieval"][:3]
        
        # Check if ground truth is in predictions
        ground_truth = example["resolved_ontology_id"]
        
        # Get all result IDs (for checking rank beyond top-5)
        all_ids = [r.get("id") for r in results_list if r.get("id")]
        
        # Check top-1
        if result["predicted_id"] == ground_truth:
            result["correct"] = True
            result["rank"] = 1
            result["mrr"] = 1.0
        else:
            # Check if ground truth appears anywhere in results
            if ground_truth in all_ids:
                result["rank"] = all_ids.index(ground_truth) + 1
                result["mrr"] = 1.0 / result["rank"]
                
                # Check if in top-3 or top-5
                if ground_truth in result["top_3_ids_llm"]:
                    result["top_3_correct"] = True
                if ground_truth in result["top_5_ids_retrieval"]:
                    result["top_5_correct"] = True
            else:
                # Ground truth not found in any results
                result["rank"] = None
                result["mrr"] = 0.0
        
    except Exception as e:
        error_str = str(e)
        result["error"] = error_str
        # Check if this is a rate limit error (should be retried)
        if "rate_limit" in error_str.lower() or "ratelimit" in error_str.lower() or "429" in error_str:
            result["is_rate_limit_error"] = True
        # Check if this is a validation error (shouldn't be retried)
        error_lower = error_str.lower()
        if "unsupported organism" in error_lower:
            result["is_validation_error"] = True
            print(f"[ERROR] Example failed (validation error - will not retry): {e}", file=sys.stderr)
        else:
            print(f"[ERROR] Example failed: {e}", file=sys.stderr)
    
    return result


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics from evaluation results."""
    if not results:
        return {}
    
    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    successful = total - errors
    
    if successful == 0:
        return {"error": "No successful evaluations"}
    
    # Accuracy metrics
    top1_correct = sum(1 for r in results if r.get("correct"))
    top3_correct = sum(1 for r in results if r.get("top_3_correct") or r.get("correct"))
    top5_correct = sum(1 for r in results if r.get("top_5_correct") or r.get("top_3_correct") or r.get("correct"))
    
    top1_accuracy = top1_correct / successful
    top3_accuracy = top3_correct / successful
    top5_accuracy = top5_correct / successful
    
    # MRR
    mrrs = [r.get("mrr", 0.0) for r in results if not r.get("error")]
    mean_mrr = statistics.mean(mrrs) if mrrs else 0.0
    
    # Rank statistics
    ranks = [r.get("rank") for r in results if r.get("rank") is not None]
    median_rank = statistics.median(ranks) if ranks else None
    mean_rank = statistics.mean(ranks) if ranks else None
    
    # Field-specific metrics
    field_metrics = defaultdict(lambda: {"total": 0, "correct": 0, "top3_correct": 0, "top5_correct": 0, "mrrs": []})
    for r in results:
        if r.get("error"):
            continue
        field = r.get("field_type", "unknown")
        field_metrics[field]["total"] += 1
        if r.get("correct"):
            field_metrics[field]["correct"] += 1
        if r.get("top_3_correct") or r.get("correct"):
            field_metrics[field]["top3_correct"] += 1
        if r.get("top_5_correct") or r.get("top_3_correct") or r.get("correct"):
            field_metrics[field]["top5_correct"] += 1
        if r.get("mrr", 0.0) > 0:
            field_metrics[field]["mrrs"].append(r.get("mrr"))
    
    field_breakdown = {}
    for field, metrics in field_metrics.items():
        if metrics["total"] > 0:
            field_breakdown[field] = {
                "total": metrics["total"],
                "top1_accuracy": metrics["correct"] / metrics["total"],
                "top3_accuracy": metrics["top3_correct"] / metrics["total"],
                "top5_accuracy": metrics["top5_correct"] / metrics["total"],
                "mean_mrr": statistics.mean(metrics["mrrs"]) if metrics["mrrs"] else 0.0,
            }
    
    return {
        "total_examples": total,
        "successful": successful,
        "errors": errors,
        "overall": {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "top5_accuracy": top5_accuracy,
            "mean_mrr": mean_mrr,
            "median_rank": median_rank,
            "mean_rank": mean_rank,
        },
        "by_field": field_breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BOND on benchmark dataset")
    parser.add_argument("--test_set", default="test.jsonl", help="Path to test set JSONL file (default: test.jsonl)")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--resume", help="Resume from existing results file")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--max_examples", type=int, help="Maximum number of examples to evaluate sequentially (first N examples, use without --random_sample)")
    parser.add_argument("--random_sample", type=int, help="Randomly sample N examples from test set (use this for random sampling, or omit for sequential)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility (only used with --random_sample)")
    parser.add_argument("--dataset_id", type=str, help="Filter to only examples with this dataset_id")
    parser.add_argument("--retrieval_only", action="store_true", help="Use retrieval-only mode (no LLM expansion or disambiguation)")
    parser.add_argument("--no_expansion", action="store_true", help="Skip query expansion (set n_expansions=0)")
    parser.add_argument("--no_disambiguation", action="store_true", help="Skip disambiguation LLM (use top retrieval result)")
    parser.add_argument("--exact_only", action="store_true", help="Use exact match retrieval only (skip BM25 and dense)")
    parser.add_argument("--rate_limit_requests_per_min", type=int, default=10000, help="Requests per minute limit")
    parser.add_argument("--rate_limit_requests_per_day", type=int, default=1000000000, help="Requests per day limit")
    parser.add_argument("--rate_limit_tokens_per_min", type=int, default=10000000, help="Tokens per minute limit")
    parser.add_argument("--rate_limit_tokens_per_day", type=int, default=1000000000, help="Tokens per day limit")
    
    args = parser.parse_args()
    
    # Load test set - default to test.jsonl in benchmark-metrics directory
    test_set_path = args.test_set
    if test_set_path == "test.jsonl":
        # If just "test.jsonl", look in benchmark-metrics directory (where script is)
        script_dir = Path(__file__).parent
        test_set_path = str(script_dir / "test.jsonl")
    
    # If test_set is a directory, look for test.jsonl inside it
    if os.path.isdir(test_set_path):
        test_set_path = os.path.join(test_set_path, "test.jsonl")
    
    # Check if file exists, if not try current directory
    if not os.path.exists(test_set_path):
        if os.path.exists("test.jsonl"):
            test_set_path = "test.jsonl"
            print(f"[INFO] Using test.jsonl from current directory")
        else:
            print(f"[ERROR] Test set file not found: {args.test_set}")
            print(f"[ERROR] Tried: {test_set_path}")
            sys.exit(1)
    
    print(f"[INFO] Loading test set from {test_set_path}...")
    examples = load_test_set(test_set_path, random_sample=args.random_sample, seed=args.seed, dataset_id=args.dataset_id)
    print(f"[INFO] Loaded {len(examples)} examples")
    
    # Load existing results if resuming
    results = []
    start_idx = args.start_idx
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Resuming from {args.resume}...")
        with open(args.resume, 'r') as f:
            existing = json.load(f)
            results = existing.get("results", [])
            start_idx = len(results)
            print(f"[INFO] Resuming from index {start_idx}")
    
    # Apply start_idx first, then limit examples if specified
    # This ensures max_examples works correctly with start_idx
    if start_idx > 0:
        examples = examples[start_idx:]
        print(f"[INFO] Starting from index {start_idx}, {len(examples)} examples remaining")
    
    # Limit examples if specified (after applying start_idx)
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"[INFO] Limited to {len(examples)} examples")
    
    # Reset start_idx to 0 since we've already sliced the examples list
    start_idx = 0
    
    # Initialize BOND
    print("[INFO] Initializing BOND matcher...")
    settings = BondSettings()
    
    # Determine pipeline configuration
    pipeline_mode = []
    if args.retrieval_only:
        os.environ["BOND_RETRIEVAL_ONLY"] = "1"
        settings.retrieval_only = True
        pipeline_mode.append("RETRIEVAL_ONLY")
    if args.no_expansion:
        pipeline_mode.append("NO_EXPANSION")
    if args.no_disambiguation:
        pipeline_mode.append("NO_DISAMBIGUATION")
    if args.exact_only:
        pipeline_mode.append("EXACT_ONLY")
    
    if not pipeline_mode:
        pipeline_mode.append("FULL_PIPELINE")
        print("[INFO] Using full pipeline (with LLM calls)")
        print(f"[INFO] Expansion LLM: {settings.expansion_llm_model}")
        print(f"[INFO] Disambiguation LLM: {settings.disambiguation_llm_model}")
    else:
        print(f"[INFO] Pipeline mode: {' + '.join(pipeline_mode)}")
        if not args.retrieval_only:
            if not args.no_expansion:
                print(f"[INFO] Expansion LLM: {settings.expansion_llm_model}")
            if not args.no_disambiguation:
                print(f"[INFO] Disambiguation LLM: {settings.disambiguation_llm_model}")
    
    matcher = BondMatcher(settings)
    
    # Initialize rate limiter (only if using LLM)
    rate_limiter = None
    # Use LLM if: not retrieval_only AND (expansion enabled OR disambiguation enabled)
    uses_expansion_llm = not args.retrieval_only and not args.no_expansion
    uses_disamb_llm = not args.retrieval_only and not args.no_disambiguation
    uses_llm = uses_expansion_llm or uses_disamb_llm
    if uses_llm:
        rate_limiter = RateLimiter(
            requests_per_min=args.rate_limit_requests_per_min,
            requests_per_day=args.rate_limit_requests_per_day,
            tokens_per_min=args.rate_limit_tokens_per_min,
            tokens_per_day=args.rate_limit_tokens_per_day
        )
        print(f"[INFO] Rate limiter:")
        print(f"  Requests: {args.rate_limit_requests_per_min}/min, {args.rate_limit_requests_per_day}/day")
        print(f"  Tokens: {args.rate_limit_tokens_per_min}/min, {args.rate_limit_tokens_per_day}/day")
    
    # Evaluate examples
    print(f"[INFO] Starting evaluation from index {start_idx}...")
    max_retries_per_example = 5  # Maximum retries for rate limit errors
    
    for idx, example in enumerate(examples[start_idx:], start=start_idx):
        print(f"[INFO] Evaluating example {idx + 1}/{len(examples)}: {example['author_term']} ({example['field_type']})")
        
        # Determine n_expansions
        n_expansions = 0 if args.no_expansion else None
        
        # Retry loop for rate limit errors and LLM failures
        retry_count = 0
        should_use_llm = not args.retrieval_only and not args.no_disambiguation
        bond_result = None  # Store bond_result to check _llm_called flag
        
        while True:
            # Call evaluate_example and capture bond_result if possible
            result = evaluate_example(
                matcher, 
                example, 
                rate_limiter,
                n_expansions=n_expansions,
                exact_only=args.exact_only,
                skip_disambiguation=args.no_disambiguation,
            )
            
            # Check if we need to retry:
            # 1. Rate limit error
            # 2. Any error when LLM should have run
            # 3. LLM should have run but didn't (llm_confidence is null when it shouldn't be)
            needs_retry = False
            retry_reason = None
            
            if result.get("is_rate_limit_error") and retry_count < max_retries_per_example:
                needs_retry = True
                retry_reason = "rate limit"
            elif should_use_llm and result.get("error") and retry_count < max_retries_per_example:
                # Check if this is a validation error that shouldn't be retried
                is_validation_error = result.get("is_validation_error", False)
                if not is_validation_error:
                    # Also check error message for validation keywords
                    error_msg = result.get("error", "").lower()
                    non_retryable_errors = [
                        "unsupported organism",
                        "unsupported field",
                        "invalid organism",
                        "validation error",
                    ]
                    is_validation_error = any(err in error_msg for err in non_retryable_errors)
                
                if is_validation_error:
                    # Validation errors won't be fixed by retrying - skip retry
                    needs_retry = False
                    print(f"[INFO] Skipping retry for validation error (non-retryable): {result.get('error')[:150]}")
                else:
                    # Other errors might be transient - retry
                    needs_retry = True
                    retry_reason = f"error: {result.get('error')[:100]}"
            elif should_use_llm and result.get("llm_confidence") is None and not result.get("error") and retry_count < max_retries_per_example:
                # Check if LLM was actually called using the _llm_called flag
                llm_was_called = result.get("_llm_called", False)
                
                if not llm_was_called:
                    # LLM was not called when it should have been - definitely retry
                    needs_retry = True
                    retry_reason = "LLM was not called (no _llm_called flag)"
                elif llm_was_called and result.get("llm_reason"):
                    # LLM was called and provided a reason (even if confidence is None)
                    # This is a valid completion (may be abstain) - don't retry
                    # The log already shows "[INFO] LLM completed... but confidence is None (may be abstain)"
                    pass
                elif llm_was_called and result.get("predicted_id") is None and not result.get("llm_reason"):
                    # LLM was called but no prediction and no reason - might have failed silently
                    # Only retry if we don't have a reason (abstain should have a reason)
                    needs_retry = True
                    retry_reason = "LLM was called but no prediction and no reason (possible silent failure)"
                else:
                    # LLM was called, we have a prediction, but no llm_confidence
                    # This might be an abstain or parsing issue - don't retry if we have a prediction
                    # (LLM completed but abstained, which is valid)
                    pass
            
            if needs_retry:
                retry_count += 1
                # Calculate wait time with exponential backoff
                wait_time = min(60 * (2 ** (retry_count - 1)), 300)  # Max 5 minutes
                print(f"[RETRY] {retry_reason} detected, waiting {wait_time}s before retry {retry_count}/{max_retries_per_example}...")
                time.sleep(wait_time)
                # Reset rate limiter's recent tracking after waiting
                if rate_limiter:
                    now = time.time()
                    rate_limiter.request_times = [t for t in rate_limiter.request_times if now - t < 60]
                    rate_limiter.token_usage = [(t, tokens) for t, tokens in rate_limiter.token_usage if now - t < 60]
                continue
            else:
                # Success or max retries reached
                if result.get("is_rate_limit_error") or (should_use_llm and result.get("llm_confidence") is None):
                    if retry_count >= max_retries_per_example:
                        print(f"[ERROR] Max retries ({max_retries_per_example}) reached for example {idx + 1}, moving on")
                    else:
                        print(f"[WARNING] Example {idx + 1} completed but LLM may not have run (llm_confidence={result.get('llm_confidence')})")
                # Remove the internal flags before storing result
                result.pop("is_rate_limit_error", None)
                result.pop("is_validation_error", None)  # Internal flag, don't save to results
                result.pop("_llm_called", None)  # Internal flag, don't save to results
                break
        
        result["example_idx"] = idx
        if retry_count > 0:
            result["retry_count"] = retry_count
        results.append(result)
        
        # Save progress every 10 examples
        if (idx + 1) % 10 == 0:
            metrics = calculate_metrics(results)
            output_data = {
                "results": results,
                "metrics": metrics,
                "config": {
                    "retrieval_only": args.retrieval_only,
                    "no_expansion": args.no_expansion,
                    "no_disambiguation": args.no_disambiguation,
                    "exact_only": args.exact_only,
                    "pipeline_mode": pipeline_mode,
                    "total_examples": len(examples),
                    "evaluated": len(results),
                }
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"[INFO] Progress saved. Current top-1 accuracy: {metrics.get('overall', {}).get('top1_accuracy', 0):.4f}")
    
    # Calculate final metrics
    print("[INFO] Calculating final metrics...")
    metrics = calculate_metrics(results)
    
    # Save final results
    output_data = {
        "results": results,
        "metrics": metrics,
        "config": {
            "retrieval_only": args.retrieval_only,
            "no_expansion": args.no_expansion,
            "no_disambiguation": args.no_disambiguation,
            "exact_only": args.exact_only,
            "pipeline_mode": pipeline_mode,
            "total_examples": len(examples),
            "evaluated": len(results),
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Check if metrics calculation was successful
    if not metrics or "error" in metrics:
        print(f"Total examples in dataset: {len(examples)}")
        print(f"Examples evaluated: {len(results)}")
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
        else:
            print("Error: No metrics calculated (no results or all failed)")
        print(f"\nResults saved to: {args.output}")
        print("="*60)
        return
    
    # Print metrics if available
    print(f"Total examples: {metrics.get('total_examples', len(results))}")
    print(f"Successful: {metrics.get('successful', 0)}")
    print(f"Errors: {metrics.get('errors', 0)}")
    
    overall = metrics.get('overall', {})
    if overall:
        print(f"\nOverall Metrics:")
        print(f"  Top-1 Accuracy: {overall.get('top1_accuracy', 0.0):.4f}")
        print(f"  Top-3 Accuracy: {overall.get('top3_accuracy', 0.0):.4f}")
        print(f"  Top-5 Accuracy: {overall.get('top5_accuracy', 0.0):.4f}")
        print(f"  Mean MRR: {overall.get('mean_mrr', 0.0):.4f}")
        print(f"  Median Rank: {overall.get('median_rank', 'N/A')}")
        print(f"  Mean Rank: {overall.get('mean_rank', 0.0):.2f}")
    
    if metrics.get('by_field'):
        print(f"\nBy Field:")
        for field, field_metrics in metrics['by_field'].items():
            print(f"  {field}:")
            print(f"    Top-1: {field_metrics['top1_accuracy']:.4f} ({field_metrics['total']} examples)")
            print(f"    Top-3: {field_metrics['top3_accuracy']:.4f}")
            print(f"    Mean MRR: {field_metrics['mean_mrr']:.4f}")
    
    print(f"\nResults saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()


