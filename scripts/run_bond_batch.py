#!/usr/bin/env python
"""
Batch runner for BOND.

Reads each *_unique_terms.json file in bond_input/, runs the matcher, and writes
CSV summaries with the top three LLM-ranked predictions per term.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bond.config import BondSettings  # noqa: E402
from bond.pipeline import BondMatcher  # noqa: E402


CSV_FIELDS = [
    "original term",
    "Observed in",
    "Prediction-1 Term",
    "Prediction-1 ontology ID",
    "Prediction-2 Term",
    "Prediction-2 ontology ID",
    "Prediction-3 Term",
    "Prediction-3 ontology ID",
]


def format_datasets(datasets: Iterable[str]) -> str:
    return "; ".join(sorted(set(datasets)))


def extract_predictions(resp: Dict) -> List[Tuple[str, str]]:
    ranked = resp.get("llm_ranked") or []
    out: List[Tuple[str, str]] = []
    for entry in ranked[:3]:
        out.append((entry.get("label") or "", entry.get("id") or ""))
    while len(out) < 3:
        out.append(("", ""))
    return out


def process_file(
    matcher: BondMatcher,
    json_path: Path,
    output_dir: Path,
    cache: Dict[Tuple[str, str, str | None, str | None], Dict],
    max_terms: int | None = None,
    delay: float = 0.0,
) -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    field = data["field_type"]
    context = data.get("context", {})
    organism = context.get("organism")
    tissue = context.get("tissue")
    terms = data.get("terms", [])

    output_path = output_dir / f"{field.lower().replace(' ', '_')}_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for idx, item in enumerate(terms):
            if max_terms is not None and idx >= max_terms:
                break

            original = item.get("original_term", "").strip()
            datasets = item.get("datasets", [])
            cache_key = (field.lower(), original, tissue, organism)

            if cache_key not in cache:
                try:
                    cache[cache_key] = matcher.query(
                        original,
                        field_name=field,
                        organism=organism,
                        tissue=tissue,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[warn] Failed to harmonize '{original}' for field '{field}': {exc}",
                        file=sys.stderr,
                    )
                    cache[cache_key] = {}
                    time.sleep(delay)
                    continue
                time.sleep(delay)

            resp = cache[cache_key] or {}
            predictions = extract_predictions(resp)
            row = {
                "original term": original,
                "Observed in": format_datasets(datasets),
                "Prediction-1 Term": predictions[0][0],
                "Prediction-1 ontology ID": predictions[0][1],
                "Prediction-2 Term": predictions[1][0],
                "Prediction-2 ontology ID": predictions[1][1],
                "Prediction-3 Term": predictions[2][0],
                "Prediction-3 ontology ID": predictions[2][1],
            }
            writer.writerow(row)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BOND on batch JSON inputs.")
    parser.add_argument(
        "--input-dir",
        default="bond_input",
        type=Path,
        help="Directory containing *_unique_terms.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="bond_output",
        type=Path,
        help="Directory to write CSV outputs.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=None,
        help="Optional cap of terms per file (for dry runs).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between API calls.",
    )
    args = parser.parse_args()

    json_files = sorted(args.input_dir.glob("*_unique_terms.json"))
    if not json_files:
        raise SystemExit(f"No *_unique_terms.json files found in {args.input_dir}")

    matcher = BondMatcher(BondSettings())
    cache: Dict[Tuple[str, str, str | None, str | None], Dict] = {}

    generated: List[Path] = []
    for path in json_files:
        print(f"[info] Processing {path.name}...")
        generated.append(
            process_file(
                matcher,
                path,
                args.output_dir,
                cache,
                max_terms=args.max_terms,
                delay=args.delay,
            )
        )

    print("\n[info] Generated CSV files:")
    for out_path in generated:
        print(f" - {out_path}")


if __name__ == "__main__":
    main()
