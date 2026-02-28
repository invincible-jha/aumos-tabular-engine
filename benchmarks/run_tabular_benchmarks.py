"""AumOS Tabular Engine benchmark runner — SDGym compatible.

Runs all generators against standard datasets and outputs a JSON results file.

Usage:
    python benchmarks/run_tabular_benchmarks.py --output results.json
    python benchmarks/run_tabular_benchmarks.py --generators ctgan gaussian_copula
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BENCHMARK_DATASETS: list[str] = [
    "adult",
    "alarm",
    "asia",
    "census",
    "child",
    "covtype",
    "credit",
    "insurance",
    "intrusion",
    "news",
]

SUPPORTED_GENERATORS = ["ctgan", "gaussian_copula", "tvae", "smartnoise"]


async def run_benchmark(dataset_name: str, generator: str) -> dict[str, Any]:
    """Run a single benchmark and return metric scores.

    Args:
        dataset_name: SDGym dataset name to benchmark against.
        generator: Generator type to evaluate.

    Returns:
        Dict with KSComplement, TVComplement, BNLogLikelihood metrics and timing.
    """
    start_ts = time.monotonic()

    try:
        from sdgym import benchmark_single_table
        from sdgym.synthesizers import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

        synthesizer_map = {
            "ctgan": CTGANSynthesizer,
            "gaussian_copula": GaussianCopulaSynthesizer,
            "tvae": TVAESynthesizer,
        }

        if generator not in synthesizer_map:
            return {
                "dataset": dataset_name,
                "generator": generator,
                "error": f"Generator '{generator}' not supported in SDGym mode",
                "duration_s": 0.0,
            }

        synthesizer_class = synthesizer_map[generator]
        scores = await asyncio.to_thread(
            benchmark_single_table,
            synthesizer=synthesizer_class(),
            datasets=[dataset_name],
        )

        duration = time.monotonic() - start_ts
        return {
            "dataset": dataset_name,
            "generator": generator,
            "ks_complement": float(scores.get("KSComplement", 0.0)),
            "tv_complement": float(scores.get("TVComplement", 0.0)),
            "bn_log_likelihood": float(scores.get("BNLogLikelihood", 0.0)),
            "duration_s": duration,
        }

    except ImportError:
        return {
            "dataset": dataset_name,
            "generator": generator,
            "error": "sdgym not installed. pip install sdgym>=0.6",
            "duration_s": 0.0,
        }
    except Exception as exc:
        return {
            "dataset": dataset_name,
            "generator": generator,
            "error": str(exc),
            "duration_s": time.monotonic() - start_ts,
        }


async def main() -> None:
    """Run benchmarks across all dataset/generator combinations."""
    parser = argparse.ArgumentParser(
        description="AumOS Tabular Engine benchmark runner"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/benchmark-results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["ctgan", "gaussian_copula", "tvae"],
        choices=SUPPORTED_GENERATORS,
        help="Generators to benchmark",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=BENCHMARK_DATASETS,
        help="Dataset names to benchmark against",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {
        "run_date": datetime.utcnow().isoformat(),
        "generators": args.generators,
        "datasets": args.datasets,
        "scores": {},
    }

    for dataset in args.datasets:
        for generator in args.generators:
            print(f"Benchmarking {generator} on {dataset}...")
            key = f"{dataset}_{generator}"
            results["scores"][key] = await run_benchmark(dataset, generator)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
