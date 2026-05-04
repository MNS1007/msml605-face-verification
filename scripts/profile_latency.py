#!/usr/bin/env python3
"""Hardware-aware latency profiler for the face-verification system (Milestone 4).

Measures three stages of the embedding-based inference pipeline separately, plus
end-to-end latency, across a sweep of batch sizes:

  1. preprocessing  : disk read + RGB resize to 160x160 + [-1, 1] normalize
  2. embedding      : FaceNet (InceptionResnetV1, VGGFace2) forward pass
  3. scoring        : vectorized cosine similarity (src.similarity)

Outputs (under outputs/profiling/):
  - per_stage_latency.json         : single-pair stage breakdown (mean/p50/p95)
  - batch_size_sensitivity.csv     : per-stage and end-to-end latency vs batch size
  - profiling_environment.json     : OS / CPU / library / device info
  - profiling_summary.md           : human-readable summary of the run

Usage:
  python scripts/profile_latency.py                        # CPU baseline (default)
  python scripts/profile_latency.py --batch-sizes 1 4 16   # custom batch sizes
  python scripts/profile_latency.py --device cuda          # optional GPU run
  python scripts/profile_latency.py --pairs-csv outputs/pairs/test.csv \
      --num-single-pairs 30 --warmup 3 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.similarity import cosine_similarity  # noqa: E402


# ---------------------------------------------------------------------------
# Stage helpers - these mirror the production inference path exactly.
# ---------------------------------------------------------------------------

def preprocess_one(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((160, 160))
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))
    return torch.tensor(x, dtype=torch.float32)


def preprocess_batch(paths: list[str]) -> torch.Tensor:
    return torch.stack([preprocess_one(p) for p in paths])


@torch.no_grad()
def embed_batch(model: torch.nn.Module, batch: torch.Tensor, device: str) -> np.ndarray:
    return model(batch.to(device)).cpu().numpy()


def score_batch(left_emb: np.ndarray, right_emb: np.ndarray) -> np.ndarray:
    return cosine_similarity(left_emb, right_emb)


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

def time_block(fn: Callable, repeat: int) -> list[float]:
    """Run fn repeat times and return the per-call wall-time list (seconds)."""
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def summarize(latencies_s: list[float]) -> dict:
    arr_ms = [t * 1000.0 for t in latencies_s]
    return {
        "n": len(arr_ms),
        "mean_ms": round(statistics.mean(arr_ms), 3),
        "median_ms": round(statistics.median(arr_ms), 3),
        "p95_ms": round(float(np.percentile(arr_ms, 95)), 3) if len(arr_ms) > 1 else round(arr_ms[0], 3),
        "min_ms": round(min(arr_ms), 3),
        "max_ms": round(max(arr_ms), 3),
    }


# ---------------------------------------------------------------------------
# Single-pair stage breakdown
# ---------------------------------------------------------------------------

def profile_per_stage(model, device, pair_paths, warmup, repeat):
    """For each of `repeat` distinct pairs, time the three stages separately."""
    pre_lat, emb_lat, scr_lat, e2e_lat = [], [], [], []

    # warm up the model + filesystem caches
    for left, right in pair_paths[:warmup]:
        _ = preprocess_one(left); _ = preprocess_one(right)
        x = torch.stack([preprocess_one(left), preprocess_one(right)])
        _ = embed_batch(model, x, device)

    for left, right in pair_paths[:repeat]:
        # End-to-end (single pair)
        t0 = time.perf_counter()

        t = time.perf_counter()
        x1 = preprocess_one(left); x2 = preprocess_one(right)
        pre_lat.append(time.perf_counter() - t)

        t = time.perf_counter()
        e1 = embed_batch(model, x1.unsqueeze(0), device)
        e2 = embed_batch(model, x2.unsqueeze(0), device)
        emb_lat.append(time.perf_counter() - t)

        t = time.perf_counter()
        _ = score_batch(e1, e2)
        scr_lat.append(time.perf_counter() - t)

        e2e_lat.append(time.perf_counter() - t0)

    return {
        "preprocessing_per_pair": summarize(pre_lat),
        "embedding_per_pair": summarize(emb_lat),
        "scoring_per_pair": summarize(scr_lat),
        "end_to_end_per_pair": summarize(e2e_lat),
    }


# ---------------------------------------------------------------------------
# Batch-size sensitivity sweep
# ---------------------------------------------------------------------------

def profile_batch_sizes(model, device, all_pairs, batch_sizes, warmup, repeat):
    rows = []

    for bsz in batch_sizes:
        if len(all_pairs) < bsz:
            print(f"[WARN] not enough pairs for batch={bsz}; skipping")
            continue
        pairs = all_pairs[:bsz]
        left_paths = [p[0] for p in pairs]
        right_paths = [p[1] for p in pairs]

        # Per-batch warmup so model graph + caches are hot for *this* size.
        for _ in range(warmup):
            wx = preprocess_batch(left_paths + right_paths)
            _ = embed_batch(model, wx, device)

        pre_t, emb_t, scr_t, e2e_t = [], [], [], []
        for _ in range(repeat):
            t0 = time.perf_counter()

            t = time.perf_counter()
            left_x = preprocess_batch(left_paths)
            right_x = preprocess_batch(right_paths)
            pre_t.append(time.perf_counter() - t)

            t = time.perf_counter()
            le = embed_batch(model, left_x, device)
            re = embed_batch(model, right_x, device)
            emb_t.append(time.perf_counter() - t)

            t = time.perf_counter()
            _ = score_batch(le, re)
            scr_t.append(time.perf_counter() - t)

            e2e_t.append(time.perf_counter() - t0)

        # aggregate
        pre_ms = statistics.mean(pre_t) * 1000
        emb_ms = statistics.mean(emb_t) * 1000
        scr_ms = statistics.mean(scr_t) * 1000
        e2e_ms = statistics.mean(e2e_t) * 1000

        # per-pair view
        rows.append({
            "batch_size": bsz,
            "preprocessing_total_ms": round(pre_ms, 3),
            "embedding_total_ms": round(emb_ms, 3),
            "scoring_total_ms": round(scr_ms, 3),
            "end_to_end_total_ms": round(e2e_ms, 3),
            "preprocessing_per_pair_ms": round(pre_ms / bsz, 3),
            "embedding_per_pair_ms": round(emb_ms / bsz, 3),
            "scoring_per_pair_ms": round(scr_ms / bsz, 3),
            "end_to_end_per_pair_ms": round(e2e_ms / bsz, 3),
            "throughput_pairs_per_s": round(bsz / (e2e_ms / 1000.0), 2),
        })
    return rows


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------

def collect_environment(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device_used": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available(),
    }
    if device == "cuda" and torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_markdown_summary(env, per_stage, batch_rows, out_path):
    lines = []
    lines.append("# Profiling Summary\n")
    lines.append("## Environment\n")
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Per-pair stage latency (single-pair inference path)\n")
    lines.append("| Stage | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for stage in ["preprocessing_per_pair", "embedding_per_pair", "scoring_per_pair", "end_to_end_per_pair"]:
        s = per_stage[stage]
        lines.append(f"| {stage} | {s['mean_ms']} | {s['median_ms']} | {s['p95_ms']} | {s['min_ms']} | {s['max_ms']} |")
    lines.append("")

    lines.append("## Batch-size sensitivity (totals + per-pair amortized)\n")
    if batch_rows:
        cols = list(batch_rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in batch_rows:
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", default="outputs/pairs/test.csv",
                    help="CSV with left_path,right_path,label,split (default: test pairs).")
    ap.add_argument("--out-dir", default="outputs/profiling")
    ap.add_argument("--batch-sizes", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32],
                    help="Batch sizes to sweep for the sensitivity study.")
    ap.add_argument("--num-single-pairs", type=int, default=20,
                    help="Number of distinct single pairs to time per stage.")
    ap.add_argument("--warmup", type=int, default=2,
                    help="Warm-up iterations before timing (per stage and per batch).")
    ap.add_argument("--repeat", type=int, default=20,
                    help="Repetitions for the per-pair stage measurement.")
    ap.add_argument("--device", default=None,
                    help="cpu (default) or cuda. If omitted, picks CPU explicitly so "
                         "the baseline is reproducible across machines.")
    args = ap.parse_args()

    device = args.device or "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    print(f"Loading pairs CSV: {args.pairs_csv}")
    df = pd.read_csv(args.pairs_csv)
    df = df[df["left_path"].apply(os.path.exists) & df["right_path"].apply(os.path.exists)]
    pair_paths = list(zip(df["left_path"].tolist(), df["right_path"].tolist()))
    if not pair_paths:
        sys.exit("No valid pairs found. Did the dataset move?")

    needed = max(args.num_single_pairs + args.warmup, max(args.batch_sizes))
    if len(pair_paths) < needed:
        print(f"[WARN] Only {len(pair_paths)} valid pairs; some batch sizes may be skipped.")

    print(f"Loading FaceNet (VGGFace2) on device={device}")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    print("Profiling per-stage single-pair latency...")
    per_stage = profile_per_stage(
        model=model,
        device=device,
        pair_paths=pair_paths[: args.num_single_pairs + args.warmup],
        warmup=args.warmup,
        repeat=args.num_single_pairs,
    )

    print(f"Profiling batch-size sensitivity over {args.batch_sizes}...")
    batch_rows = profile_batch_sizes(
        model=model,
        device=device,
        all_pairs=pair_paths,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    env = collect_environment(device)
    env["num_pairs_loaded"] = len(pair_paths)
    env["pairs_csv"] = args.pairs_csv
    env["num_single_pairs"] = args.num_single_pairs
    env["warmup"] = args.warmup
    env["repeat"] = args.repeat
    env["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z") or time.strftime("%Y-%m-%dT%H:%M:%S")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "per_stage_latency.json"), "w") as f:
        json.dump(per_stage, f, indent=2)
    with open(os.path.join(args.out_dir, "profiling_environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    pd.DataFrame(batch_rows).to_csv(
        os.path.join(args.out_dir, "batch_size_sensitivity.csv"), index=False
    )
    write_markdown_summary(
        env, per_stage, batch_rows,
        os.path.join(args.out_dir, "profiling_summary.md"),
    )

    print(f"\nWrote artifacts to {args.out_dir}/")
    print(json.dumps(per_stage, indent=2))


if __name__ == "__main__":
    main()
