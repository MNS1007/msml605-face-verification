"""Smoke test for the CLI inference path (scripts/infer.py).

Creates two tiny synthetic images, runs the CLI in single-pair mode,
and verifies the output contains the expected fields.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFER_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "infer.py")


def _make_dummy_image(path: str, seed: int) -> None:
    """Write a tiny 160x160 RGB image with a fixed random pattern."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture()
def image_pair(tmp_path):
    """Create a temporary directory with two dummy face images."""
    left = str(tmp_path / "left.jpg")
    right = str(tmp_path / "right.jpg")
    _make_dummy_image(left, seed=1)
    _make_dummy_image(right, seed=2)
    return left, right


class TestCLISmoke:
    """Minimal smoke tests — model loads, inference runs, output is sane."""

    def test_single_pair_runs(self, image_pair):
        """CLI completes without error on a single pair."""
        left, right = image_pair
        result = subprocess.run(
            [
                sys.executable, INFER_SCRIPT,
                "--left", left,
                "--right", right,
                "--threshold", "0.35",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Output should mention score, decision, confidence, latency
        out = result.stdout
        assert "Score:" in out
        assert "Decision:" in out
        assert "Confidence:" in out
        assert "Latency:" in out

    def test_json_output(self, image_pair):
        """CLI --json flag produces valid JSON with required keys."""
        left, right = image_pair
        result = subprocess.run(
            [
                sys.executable, INFER_SCRIPT,
                "--left", left,
                "--right", right,
                "--threshold", "0.35",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        obj = json.loads(result.stdout.strip())
        for key in ("score", "threshold", "decision", "confidence", "latency_ms"):
            assert key in obj, f"Missing key: {key}"
        assert obj["decision"] in ("same", "different")
        assert 0.5 <= obj["confidence"] <= 1.0

    def test_batch_mode(self, image_pair, tmp_path):
        """CLI --batch flag works with a small CSV."""
        left, right = image_pair
        csv_path = str(tmp_path / "pairs.csv")
        with open(csv_path, "w") as f:
            f.write("left_path,right_path\n")
            f.write(f"{left},{right}\n")
            f.write(f"{right},{left}\n")

        result = subprocess.run(
            [
                sys.executable, INFER_SCRIPT,
                "--batch", csv_path,
                "--threshold", "0.35",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().split("\n") if l]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "decision" in obj
