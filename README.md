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
| `src/` | Importable Python package — similarity, evaluation, validation, data-centric, error analysis, run tracking |
| `scripts/` | CLI entrypoints for ingestion, pair generation, scoring, threshold sweep, evaluation, data-centric improvement |
| `configs/` | YAML config files — `m1.yaml` (Milestone 1), `m2.yaml` (Milestone 2) |
| `tests/` | Unit tests and integration test |
| `reports/` | Milestone 2 evaluation report (PDF) |
| `data/` | Downloaded dataset cache (**gitignored**) |
| `outputs/` | Generated artifacts — pairs, scores, runs, plots, metrics (**gitignored**) |

## Milestone 2 Summary

### Baseline
The baseline system uses **FaceNet (InceptionResnetV1, VGGFace2)** to extract 512-dim
face embeddings, then computes **cosine similarity** between pairs. Deterministic LFW
pairs are generated with seed=42 and a 70/15/15 train/val/test split. Score direction:
**higher cosine similarity = more likely same person**.

### Threshold Selection Rule
The operating threshold is chosen by **maximising F1 score on the validation split**.
This rule is applied consistently for both the baseline and post-improvement runs.
The threshold is selected before evaluating on the held-out test split.

### Data-Centric Improvement
**Problem:** Some identities (e.g., George_W_Bush) dominate the pair set, skewing
aggregate metrics. Additionally, some image paths reference files that do not exist.

**Change:** (1) Filter pairs with missing/invalid image paths (applied in `make_pairs.py`),
(2) cap each identity to at most 30 pair appearances, (3) downsample the majority label
class to restore 1:1 balance. All operations are deterministic (seed=42).

**Effect:** The evaluation set becomes more representative of the long tail of identities,
giving a fairer picture of verifier robustness.

### Selected Threshold
See `outputs/threshold/selected_threshold.json` and `outputs/runs/` for tracked runs.

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

# 3. Ingest dataset (downloads LFW, creates deterministic splits)
python scripts/ingest_lfw.py --config configs/m1.yaml

# 4. Generate pairs (filters invalid image paths)
python scripts/make_pairs.py --config configs/m1.yaml

# 5. Score pairs (FaceNet embeddings + cosine similarity)
python scripts/score_pairs.py

# 6. Run validation checks
python scripts/validate_data.py

# 7. Baseline threshold sweep + selection + test evaluation
python scripts/run_threshold_sweep.py
python scripts/select_threshold.py
python scripts/evaluate_test.py

# 8. Apply data-centric improvement (identity capping + rebalancing)
python scripts/apply_data_centric.py --config configs/m2.yaml

# 9. Re-score improved pairs, re-sweep, re-select, re-evaluate
python scripts/score_pairs.py --pairs-dir outputs/pairs_improved/
python scripts/run_threshold_sweep.py
python scripts/select_threshold.py
python scripts/evaluate_test.py

# 10. Run error analysis
python scripts/run_error_analysis.py

# 11. Run tests
python -m pytest tests/ -v

# 12. Run load/concurrency test
python scripts/load_test.py
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/manifest.json` | Dataset manifest (counts, seed, split policy, data source) |
| `outputs/pairs/{train,val,test}.csv` | Verification pairs (left_path, right_path, label, split) |
| `outputs/pairs_improved/` | Pairs after data-centric improvement |
| `outputs/scores/` | Scored pairs with cosine similarity column |
| `outputs/sweeps/val_sweep.csv` | Threshold sweep results |
| `outputs/sweeps/roc_val.png` | ROC-style plot |
| `outputs/threshold/selected_threshold.json` | Selected threshold and metrics |
| `outputs/threshold/val_confusion_matrix.json` | Confusion matrix at selected threshold |
| `outputs/eval/` | Test set evaluation metrics and confusion matrix |
| `outputs/runs/` | Tracked experiment runs (JSON logs + runs.csv) |
| `outputs/error_analysis/` | Error slice analysis results |
| `reports/milestone2_report.pdf` | Milestone 2 evaluation report |

## Determinism

- All random operations use seed `42` (configured in YAML config files).
- Dataset splits are deterministic via `sklearn.model_selection.train_test_split` with fixed `random_state`.
- Pair generation sorts candidates before output for consistent ordering.
- Data-centric operations use `np.random.default_rng(42)` for reproducible sampling.
- Rerunning any script with the same config produces identical outputs.

## Report Location

The Milestone 2 evaluation report is at `reports/milestone2_report.pdf`.

## Milestone 3 Summary

### Model Choice

We use the Facenet model (InceptionResnetV1 pretrained on VGGFace2) for this milestone. Due to its pretraining, there was no extra training required for our task. The model was lightweight enough to run on our local machines, but was more complex than the simple pixel-based representations used earlier, which provided us with more accurate results for this milestone. FaceNet is also already a well-established baseline. This helped us in keeping our pipeline as reproducible as possible with minimal additions or trainings. This model produces embeddings where images of the same person are close together and different identities are far apart, which aligns directly with our task: similarity-based verification.

### Load Testing Results

We evaluated the system under concurrent inference using a thread-based load test.

Configuration:

* Requests: 100
* Workers: 4

Results:

* Throughput: ~14.5 requests/second
* Average latency: ~0.274 seconds
* P95 latency: ~0.277 seconds
* Min latency: ~0.200 seconds
* Max latency: ~0.827 seconds

The close alignment between average and P95 latency indicates stable performance under concurrency, with minimal tail-latency degradation.

