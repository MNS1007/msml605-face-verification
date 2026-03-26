"""Small integration test for the Milestone 2 evaluation pipeline.

Uses a tiny synthetic fixture — no dataset download required.
Runs the full path: pairs → scores → threshold sweep → select threshold →
confusion matrix → error analysis → validate outputs.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_centric import rebalance_labels
from src.evaluation import (
    apply_threshold,
    compute_metrics,
    roc_curve_data,
    select_threshold,
    threshold_sweep,
)
from src.error_analysis import run_error_analysis
from src.similarity import cosine_similarity
from src.validation import (
    validate_metrics_complete,
    validate_pair_schema,
    validate_score_count,
)


@pytest.fixture
def synthetic_pipeline():
    """Create a tiny deterministic fixture that mimics the real pipeline.

    - 20 pairs (10 same-identity, 10 different-identity)
    - Synthetic 64-dim embeddings where same-identity pairs share a base
      vector with small noise, and different-identity pairs use independent
      random vectors.
    """
    rng = np.random.default_rng(42)
    n_pos = 10
    n_neg = 10
    dim = 64

    rows = []
    left_vecs = []
    right_vecs = []

    # Same-identity pairs: high cosine similarity
    for i in range(n_pos):
        base = rng.standard_normal(dim)
        base = base / np.linalg.norm(base)
        noise = rng.standard_normal(dim) * 0.1
        left_vecs.append(base)
        right_vecs.append(base + noise)
        rows.append({
            "left_path": f"/synthetic/Person{i}/Person{i}_0001.jpg",
            "right_path": f"/synthetic/Person{i}/Person{i}_0002.jpg",
            "label": 1,
            "split": "val",
        })

    # Different-identity pairs: low cosine similarity
    for i in range(n_neg):
        v1 = rng.standard_normal(dim)
        v2 = rng.standard_normal(dim)
        left_vecs.append(v1 / np.linalg.norm(v1))
        right_vecs.append(v2 / np.linalg.norm(v2))
        rows.append({
            "left_path": f"/synthetic/PersonA{i}/PersonA{i}_0001.jpg",
            "right_path": f"/synthetic/PersonB{i}/PersonB{i}_0001.jpg",
            "label": 0,
            "split": "val",
        })

    df = pd.DataFrame(rows)
    left_arr = np.array(left_vecs)
    right_arr = np.array(right_vecs)
    scores = cosine_similarity(left_arr, right_arr)

    return df, scores


class TestIntegrationPipeline:
    """End-to-end integration test on synthetic data."""

    def test_full_pipeline(self, synthetic_pipeline):
        df, scores = synthetic_pipeline
        labels = df["label"].values

        # Step 1: Validate inputs
        validate_pair_schema(df)
        validate_score_count(scores, expected_n=len(df))

        # Step 2: Threshold sweep
        thresholds = np.linspace(-1.0, 1.0, 51)
        sweep_results = threshold_sweep(labels, scores, thresholds)
        assert len(sweep_results) == 51
        assert all("f1" in r for r in sweep_results)

        # Step 3: Select threshold
        best = select_threshold(sweep_results, rule="max_f1")
        assert "threshold" in best
        threshold = best["threshold"]

        # Step 4: Apply threshold and compute metrics
        predictions = apply_threshold(scores, threshold)
        metrics = compute_metrics(labels, predictions)
        validate_metrics_complete(metrics)

        # The synthetic data is designed to be separable, so accuracy should be high
        assert metrics["accuracy"] >= 0.7, f"Accuracy too low: {metrics['accuracy']}"

        # Step 5: ROC curve data
        roc = roc_curve_data(labels, scores, thresholds)
        assert len(roc["fpr"]) == len(thresholds)
        assert len(roc["tpr"]) == len(thresholds)

        # Step 6: Error analysis
        analysis = run_error_analysis(
            df, scores, predictions, threshold=threshold,
            max_images=4, boundary_width=0.1,
        )
        assert "slices" in analysis
        assert len(analysis["slices"]) == 2
        for s in analysis["slices"]:
            assert "slice_name" in s
            assert "total_in_slice" in s
            assert "errors_in_slice" in s
            assert "hypothesis" in s

        # Step 7: Write outputs to temp dir and verify structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save metrics
            metrics_path = os.path.join(tmpdir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            assert os.path.isfile(metrics_path)

            # Save sweep
            sweep_path = os.path.join(tmpdir, "sweep.json")
            with open(sweep_path, "w") as f:
                json.dump(sweep_results, f, indent=2)
            assert os.path.isfile(sweep_path)

            # Save error analysis
            analysis_path = os.path.join(tmpdir, "error_analysis.json")
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
            assert os.path.isfile(analysis_path)

            # Verify written files are loadable
            loaded_metrics = json.load(open(metrics_path))
            assert loaded_metrics["accuracy"] == metrics["accuracy"]

    def test_data_centric_then_evaluate(self, synthetic_pipeline):
        """Test that data-centric rebalancing integrates with evaluation."""
        df, scores = synthetic_pipeline

        # Make an unbalanced version
        extra_pos = df[df["label"] == 1].iloc[:3]
        unbalanced = pd.concat([df, extra_pos], ignore_index=True)
        assert (unbalanced["label"] == 1).sum() != (unbalanced["label"] == 0).sum()

        # Rebalance
        balanced = rebalance_labels(unbalanced, seed=42)
        assert (balanced["label"] == 1).sum() == (balanced["label"] == 0).sum()
        validate_pair_schema(balanced)
