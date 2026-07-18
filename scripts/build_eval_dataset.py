"""Build a reproducible Kubernetes evaluation JSONL dataset from source chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.dataset import dataset_fingerprint, write_dataset
from evaluation.dataset_builder import (
    REQUIRED_CATEGORIES,
    DatasetBuildError,
    build_dataset,
    load_database_chunks,
    load_source_chunks,
)

DEFAULT_INPUT = Path("evaluation/datasets/kubernetes_source_catalog.jsonl")
DEFAULT_OUTPUT = Path("evaluation/datasets/kubernetes_eval.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Portable source-chunk JSONL (default: %(default)s)",
    )
    source.add_argument(
        "--database-url",
        help="Read active chunks from PostgreSQL instead of --input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated evaluation JSONL (default: %(default)s)",
    )
    parser.add_argument("--count", type=int, default=52, help="Number of cases to write")
    parser.add_argument(
        "--seed",
        type=int,
        default=20250717,
        help="Fixed random seed used for sampling and ordering",
    )
    parser.add_argument(
        "--database-chunk-limit",
        type=int,
        default=200,
        help="Maximum active database chunks loaded into the candidate pool",
    )
    parser.add_argument(
        "--id-prefix",
        default="k8s",
        help="Stable prefix for generated evaluation case identifiers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.database_url:
            chunks = load_database_chunks(
                args.database_url,
                seed=args.seed,
                limit=args.database_chunk_limit,
            )
        else:
            chunks = load_source_chunks(args.input)
        cases = build_dataset(
            chunks,
            count=args.count,
            seed=args.seed,
            id_prefix=args.id_prefix,
        )
        written = write_dataset(args.output, cases)
    except DatasetBuildError as exc:
        raise SystemExit(f"Dataset generation failed: {exc}") from exc

    categories = sorted({case.category for case in cases})
    answerable = sum(case.answerable for case in cases)
    print(f"Wrote {written} unreviewed cases to {args.output}")
    print(f"Fingerprint: {dataset_fingerprint(cases)}")
    print(f"Answerable/unanswerable: {answerable}/{written - answerable}")
    print(f"Categories ({len(categories)}/{len(REQUIRED_CATEGORIES)}): {', '.join(categories)}")


if __name__ == "__main__":
    main()
