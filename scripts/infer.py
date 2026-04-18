#!/usr/bin/env python3
"""CLI inference for the face-verification system.

Usage examples
--------------
Single pair:
    python scripts/infer.py --left img_a.jpg --right img_b.jpg

Batch file (CSV with columns left_path, right_path):
    python scripts/infer.py --batch pairs.csv

Override threshold or config:
    python scripts/infer.py --left a.jpg --right b.jpg --threshold 0.40
    python scripts/infer.py --batch pairs.csv --config configs/m3.yaml
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference import infer_pair

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "m3.yaml")
DEFAULT_THRESHOLD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "threshold", "selected_threshold.json"
)


# ------------------------------------------------------------------
# Threshold loading
# ------------------------------------------------------------------

def load_threshold(config_path: str | None, explicit: float | None) -> float:
    """Resolve the operating threshold (explicit flag > config > saved JSON)."""
    if explicit is not None:
        return explicit

    # Try config file
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        t = cfg.get("inference", {}).get("threshold")
        if t is not None:
            return float(t)

    # Fall back to saved threshold from milestone 2
    if os.path.exists(DEFAULT_THRESHOLD_PATH):
        with open(DEFAULT_THRESHOLD_PATH) as f:
            return float(json.load(f)["threshold"])

    raise RuntimeError(
        "No threshold found. Pass --threshold, set inference.threshold in config, "
        f"or ensure {DEFAULT_THRESHOLD_PATH} exists."
    )


# ------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------

def print_result(result: dict) -> None:
    """Pretty-print one inference result."""
    decision_str = "same" if result["decision"] else "different"
    print(f"  Pair:       {result['left']}  |  {result['right']}")
    print(f"  Score:      {result['score']:.6f}")
    print(f"  Threshold:  {result['threshold']:.6f}")
    print(f"  Decision:   {decision_str}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Latency:    {result['latency_ms']:.2f} ms")
    print()


def format_result(left, right, raw, threshold):
    """Wrap raw infer_pair output with extra fields for display."""
    decision_str = "same" if raw["decision"] else "different"
    return {
        "left": left,
        "right": right,
        "score": round(raw["score"], 6),
        "threshold": round(threshold, 6),
        "decision": decision_str,
        "confidence": round(float(raw["confidence"]), 4),
        "latency_ms": round(raw["latency"] * 1000.0, 2),
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Face-verification CLI: score, decision, confidence, latency.",
    )
    p.add_argument("--left", type=str, help="Path to the first (left) image.")
    p.add_argument("--right", type=str, help="Path to the second (right) image.")
    p.add_argument(
        "--batch", type=str,
        help="CSV file with columns left_path, right_path (one pair per row).",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Override operating threshold (default: loaded from config or saved JSON).",
    )
    p.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to YAML config file (default: configs/m3.yaml).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit results as JSON (one object per line).",
    )
    return p


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_parser().parse_args(argv)

    if args.batch and (args.left or args.right):
        sys.exit("Error: use --batch OR --left/--right, not both.")
    if not args.batch and not (args.left and args.right):
        sys.exit("Error: provide --left and --right, or --batch <csv>.")

    threshold = load_threshold(args.config, args.threshold)

    print(f"Threshold: {threshold:.6f}", file=sys.stderr)

    # Build pair list
    if args.batch:
        df = pd.read_csv(args.batch)
        pairs = list(zip(df["left_path"], df["right_path"]))
    else:
        pairs = [(args.left, args.right)]

    # Run inference
    results = []
    for left, right in pairs:
        raw = infer_pair(left, right, threshold)
        result = format_result(left, right, raw, threshold)
        results.append(result)
        if args.json:
            print(json.dumps(result))
        else:
            print_result(result)

    return results


if __name__ == "__main__":
    main()
