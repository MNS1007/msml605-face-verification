#!/usr/bin/env python3
"""Concurrency / load test for the face-verification inference pipeline.

Runs a configurable number of requests across multiple threads, using
a deterministic set of input pairs, and reports throughput and latency
distribution (including p95).

Usage:
    python scripts/load_test.py
    python scripts/load_test.py --workers 8 --requests 200
    python scripts/load_test.py --config configs/m3.yaml
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import infer_pair

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "m3.yaml")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_once(args):
    idx, row, threshold = args
    left, right = row["left_path"], row["right_path"]
    try:
        result = infer_pair(left, right, threshold)
        return {"latency": result["latency"], "success": True}
    except Exception as e:
        return {"latency": 0.0, "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Load test for face verification CLI")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    lt_cfg = cfg.get("load_test", {})

    workers = args.workers or lt_cfg.get("num_workers", 4)
    num_requests = args.requests or lt_cfg.get("num_requests", 50)
    pairs_csv = args.pairs or lt_cfg.get("pairs_csv", "outputs/pairs/test.csv")

    # Resolve threshold
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = cfg.get("inference", {}).get("threshold")
        if threshold is None:
            thresh_path = os.path.join(cfg.get("output_dir", "outputs"), "threshold", "selected_threshold.json")
            with open(thresh_path) as f:
                threshold = json.load(f)["threshold"]

    if not os.path.exists(pairs_csv):
        sys.exit(f"Pairs CSV not found: {pairs_csv}")

    pairs_df = pd.read_csv(pairs_csv)
    print(f"Load test: {num_requests} requests, {workers} workers, threshold={threshold:.4f}")
    print(f"Pairs source: {pairs_csv} ({len(pairs_df)} pairs)")

    # Build work items (cycle through pairs deterministically)
    work = [
        (i, pairs_df.iloc[i % len(pairs_df)], threshold)
        for i in range(num_requests)
    ]

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_once, w) for w in work]
        for f in as_completed(futures):
            results.append(f.result())
    total_time = time.time() - start

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = np.array([r["latency"] for r in successes])

    print(f"\n--- Results ---")
    print(f"Total requests:  {num_requests}")
    print(f"Successes:       {len(successes)}")
    print(f"Failures:        {len(failures)}")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Throughput:      {len(successes) / total_time:.2f} req/s")
    if len(latencies) > 0:
        print(f"Avg latency:     {latencies.mean():.4f}s")
        print(f"P50 latency:     {np.percentile(latencies, 50):.4f}s")
        print(f"P95 latency:     {np.percentile(latencies, 95):.4f}s")
        print(f"Min latency:     {latencies.min():.4f}s")
        print(f"Max latency:     {latencies.max():.4f}s")

    # Save runtime summary
    output_dir = cfg.get("output_dir", "outputs")
    summary_dir = os.path.join(output_dir, "load_test")
    os.makedirs(summary_dir, exist_ok=True)

    summary = {
        "num_requests": num_requests,
        "num_workers": workers,
        "threshold": threshold,
        "pairs_csv": pairs_csv,
        "successes": len(successes),
        "failures": len(failures),
        "total_wall_time_s": round(total_time, 2),
        "throughput_req_per_s": round(len(successes) / total_time, 2),
    }
    if len(latencies) > 0:
        summary["latency_avg_s"] = round(float(latencies.mean()), 4)
        summary["latency_p50_s"] = round(float(np.percentile(latencies, 50)), 4)
        summary["latency_p95_s"] = round(float(np.percentile(latencies, 95)), 4)
        summary["latency_min_s"] = round(float(latencies.min()), 4)
        summary["latency_max_s"] = round(float(latencies.max()), 4)

    summary_path = os.path.join(summary_dir, "runtime_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved runtime summary: {summary_path}")


if __name__ == "__main__":
    main()
