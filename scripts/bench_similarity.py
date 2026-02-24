"""Benchmark: Python loop vs NumPy vectorized similarity scoring.

Usage:
    python scripts/bench_similarity.py --config configs/m1.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.similarity import (
    cosine_similarity,
    cosine_similarity_loop,
    euclidean_distance,
    euclidean_distance_loop,
)

TOLERANCE = 1e-6


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_benchmark(N: int, D: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((N, D))
    b = rng.standard_normal((N, D))

    results = {}

    # --- Cosine similarity ---------------------------------------------------
    t0 = time.perf_counter()
    cos_loop = cosine_similarity_loop(a, b)
    t_cos_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    cos_vec = cosine_similarity(a, b)
    t_cos_vec = time.perf_counter() - t0

    cos_max_diff = float(np.max(np.abs(cos_loop - cos_vec)))
    cos_correct = cos_max_diff < TOLERANCE

    results["cosine"] = {
        "loop_time_s": round(t_cos_loop, 6),
        "vectorized_time_s": round(t_cos_vec, 6),
        "speedup": round(t_cos_loop / t_cos_vec, 2) if t_cos_vec > 0 else float("inf"),
        "max_abs_diff": cos_max_diff,
        "correct": cos_correct,
    }

    # --- Euclidean distance ---------------------------------------------------
    t0 = time.perf_counter()
    euc_loop = euclidean_distance_loop(a, b)
    t_euc_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    euc_vec = euclidean_distance(a, b)
    t_euc_vec = time.perf_counter() - t0

    euc_max_diff = float(np.max(np.abs(euc_loop - euc_vec)))
    euc_correct = euc_max_diff < TOLERANCE

    results["euclidean"] = {
        "loop_time_s": round(t_euc_loop, 6),
        "vectorized_time_s": round(t_euc_vec, 6),
        "speedup": round(t_euc_loop / t_euc_vec, 2) if t_euc_vec > 0 else float("inf"),
        "max_abs_diff": euc_max_diff,
        "correct": euc_correct,
    }

    results["params"] = {"N": N, "D": D, "seed": seed, "tolerance": TOLERANCE}
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark similarity scoring")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench_cfg = cfg["benchmark"]
    seed = cfg["seed"]
    N = bench_cfg["N"]
    D = bench_cfg["D"]

    print(f"Running benchmark: N={N}, D={D}, seed={seed}")
    print("-" * 60)

    results = run_benchmark(N, D, seed)

    # Print summary
    for metric in ("cosine", "euclidean"):
        r = results[metric]
        status = "PASS" if r["correct"] else "FAIL"
        print(f"\n{metric.upper()} SIMILARITY:")
        print(f"  Loop time:       {r['loop_time_s']:.6f} s")
        print(f"  Vectorized time: {r['vectorized_time_s']:.6f} s")
        print(f"  Speedup:         {r['speedup']}x")
        print(f"  Max abs diff:    {r['max_abs_diff']:.2e}")
        print(f"  Correctness:     {status}")

    # Save results
    output_dir = os.path.join(cfg.get("output_dir", "outputs"), "bench")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Exit with error if correctness check failed
    all_correct = results["cosine"]["correct"] and results["euclidean"]["correct"]
    if not all_correct:
        print("\nERROR: Correctness check FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
