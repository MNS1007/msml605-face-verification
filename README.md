# MSML 605 - Face Verification System

End-to-end face verification system built on the LFW (Labeled Faces in the Wild) dataset.
Given two face images, the system outputs whether they match (same identity or different),
along with a similarity score, calibrated confidence, and latency measurement.

**Milestone 1** builds the reproducible, deterministic foundation: dataset ingestion,
verification pair generation, and fast vectorized similarity scoring.

**Milestone 2** adds a disciplined evaluation loop: experiment tracking, threshold
calibration on a validation split, error analysis with two defined slices, a data-centric
improvement (identity capping + label rebalancing), pipeline validation checks, and tests.

## Repo Layout

| Directory | Contents |
|-----------|----------|
| `src/` | Importable Python package — similarity, evaluation, validation, data-centric, error analysis |
| `scripts/` | CLI entrypoints: `ingest_lfw.py`, `make_pairs.py`, `bench_similarity.py`, `apply_data_centric.py` |
| `configs/` | YAML config files — `m1.yaml` (Milestone 1), `m2.yaml` (Milestone 2) |
| `tests/` | Unit tests and integration test |
| `reports/` | Milestone 2 evaluation report (PDF) |
| `data/` | Downloaded dataset cache (**gitignored**) |
| `outputs/` | Generated artifacts — pairs, runs, plots, metrics (**gitignored**) |

## Milestone 2 Summary

### Baseline
The baseline system uses cosine similarity on face embeddings with deterministic
LFW pairs (seed=42, 70/15/15 train/val/test split). Score direction: **higher
cosine similarity = more likely same person**.

### Threshold Selection Rule
The operating threshold is chosen by **maximising F1 score on the validation split**.
This rule is applied consistently for both the baseline and post-improvement runs.

### Data-Centric Improvement
**Problem:** Some identities (e.g., George_W_Bush) dominate the pair set, skewing
aggregate metrics.

**Change:** (1) Cap each identity to at most 30 pair appearances, (2) filter pairs
with missing images, (3) downsample the majority label class to restore 1:1 balance.
All operations are deterministic (seed=42).

**Effect:** Evaluation becomes more representative of the long tail of identities.

### Selected Threshold
See `outputs/runs/` for the tracked threshold sweep and selected value.

## How to Run

Reproduce all milestone outputs from a clean clone:

```bash
# 1. Clone and enter project
git clone <repo-url>
cd msml605-face-verification

# 2. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Ingest LFW dataset (downloads data, creates deterministic splits)
python scripts/ingest_lfw.py --config configs/m1.yaml

# 4. Generate verification pairs
python scripts/make_pairs.py --config configs/m1.yaml

# 5. Apply data-centric improvements (identity capping + rebalancing)
python scripts/apply_data_centric.py --config configs/m2.yaml

# 6. Run similarity benchmark (loop vs vectorized)
python scripts/bench_similarity.py --config configs/m1.yaml

# 7. Run evaluation, threshold sweep, and tracked runs
# (See scripts/ for evaluation entry points)

# 8. Run tests
python -m pytest tests/ -v
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/manifest.json` | Dataset manifest (counts, seed, split policy, data source) |
| `outputs/pairs/train.csv` | Training verification pairs (left_path, right_path, label, split) |
| `outputs/pairs/val.csv` | Validation verification pairs |
| `outputs/pairs/test.csv` | Test verification pairs |
| `outputs/pairs_improved/` | Pairs after data-centric improvement |
| `outputs/pairs_improved/data_centric_summary.json` | Summary of what changed |
| `outputs/runs/` | Tracked experiment runs (JSON logs) |
| `outputs/bench/benchmark_results.json` | Loop vs vectorized timing and correctness |
| `reports/` | Milestone 2 evaluation report (PDF) |

## Determinism

- All random operations use seed `42` (configured in YAML config files).
- Dataset splits are deterministic via `sklearn.model_selection.train_test_split` with fixed `random_state`.
- Pair generation sorts candidates before output for consistent ordering.
- Data-centric operations use `np.random.default_rng(42)` for reproducible sampling.
- Rerunning any script with the same config produces identical outputs.

## Report Location

The Milestone 2 evaluation report is at `reports/milestone2_report.pdf`.
