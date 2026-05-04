# MSML 605 — Face Verification System

End-to-end face-verification system built on the LFW (Labeled Faces in the Wild)
dataset. Given two face images the system outputs whether they match (same
identity or different), along with a similarity score, calibrated confidence,
and per-pair latency.

**Final release: `v1.0-final` (Milestone 4).**
The deployed system is a five-stage pipeline:
`preprocess → FaceNet (InceptionResnetV1, VGGFace2) → cosine similarity →
threshold → calibrated confidence`. The operating threshold is **0.351759**
(max F1 on validation), test F1 is **0.9744** (accuracy 0.9734). See the System
Card for the full audit.

## Final-release artifacts

| Deliverable | Location |
|-------------|----------|
| **System Card** (intended use, limitations, fairness risks, threshold, metrics, operational constraints) | [`reports/system_card.pdf`](reports/system_card.pdf) |
| **Profiling report** (per-stage latency, batch-size sensitivity, CPU baseline) | [`reports/profiling_report.pdf`](reports/profiling_report.pdf) |
| **Reproducibility checklist** | [`reports/reproducibility_checklist.md`](reports/reproducibility_checklist.md) |
| Selected threshold + metrics | `outputs/threshold/selected_threshold.json`, `outputs/eval/test_metrics.json` |
| Profiling raw artifacts | `outputs/profiling/` |
| Final config | `configs/m3.yaml` |
| Dockerfile / CLI entry point | `Dockerfile`, `scripts/infer.py` |
| Final Git tag | `v1.0-final` |

## Repo Layout

| Directory | Contents |
|-----------|----------|
| `src/` | Importable Python package — embeddings, similarity, inference, confidence, evaluation, validation, data-centric, error analysis, run tracking |
| `scripts/` | CLI entry points: ingestion, pair generation, scoring, threshold sweep + selection, evaluation, data-centric improvement, error analysis, **CLI inference (`infer.py`)**, **profiling (`profile_latency.py`)**, load test |
| `configs/` | YAML configs — `m1.yaml`, `m2.yaml`, **`m3.yaml` (final)** |
| `tests/` | 74 unit + smoke + integration tests |
| `reports/` | **System Card, profiling report, reproducibility checklist, Milestone 2 report** |
| `data/` | Downloaded dataset cache (**gitignored**) |
| `outputs/` | Generated artifacts — pairs, scores, runs, threshold, eval, **profiling**, load test (**gitignored except for small summaries**) |

## How to Run (final release)

The reproducibility checklist (`reports/reproducibility_checklist.md`) is the
single source of truth. The minimal copy-pastable path is:

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Reproduce data + scores
python scripts/ingest_lfw.py    --config configs/m1.yaml
python scripts/make_pairs.py    --config configs/m1.yaml
python scripts/score_pairs.py

# 3. Reproduce threshold + test metrics (System Card numbers)
python scripts/run_threshold_sweep.py
python scripts/select_threshold.py
python scripts/evaluate_test.py

# 4. Reproduce the profiling report (CPU baseline)
python scripts/profile_latency.py \
    --batch-sizes 1 2 4 8 16 32 \
    --num-single-pairs 25 --warmup 3 --repeat 5

# 5. Run the released CLI on a single pair
python scripts/infer.py --left <image_a.jpg> --right <image_b.jpg>

# 6. Run the test suite (74 tests)
python -m pytest tests/

# 7. (Optional) concurrency / load test
python scripts/load_test.py
```

### Dockerized CLI (final-release path)

```bash
# Build (downloads + bakes FaceNet weights into the image)
docker build -t face-verifier:v1.0-final .

# Run on a pair
docker run --rm \
    -v "$HOME/.cache/kagglehub:/root/.cache/kagglehub:ro" \
    face-verifier:v1.0-final \
    --left  /root/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0001.jpg \
    --right /root/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0002.jpg

# Batch from CSV
docker run --rm \
    -v "$HOME/.cache/kagglehub:/root/.cache/kagglehub:ro" \
    -v "$PWD/outputs:/app/outputs:ro" \
    face-verifier:v1.0-final \
    --batch /app/outputs/pairs/test.csv
```

## What changed in Milestone 4

Milestone 4 finalizes and audits the Milestone 3 system; it does not introduce
new representation work. Concretely:

- **Aligned the inference preprocessing with the scoring path.** `src/embeddings.py`
  now normalizes RGB inputs to `[-1, 1]` (the standard FaceNet/VGGFace2 input
  range) so the threshold calibrated against scored CSVs applies to the live
  CLI / Docker inference path.
- **Re-selected and re-evaluated the operating threshold** against the current
  scored splits. `outputs/threshold/selected_threshold.json` and
  `outputs/eval/test_metrics.json` are now mutually consistent at `θ = 0.351759`.
- **Synced `configs/m3.yaml::inference.threshold` to `0.351759`** so the
  System Card, README, config, CLI default, and saved JSON all agree.
- **Added `scripts/profile_latency.py`** — reproduces the production stages
  exactly, sweeps batch sizes, and writes machine-readable artifacts to
  `outputs/profiling/`.
- **Authored `reports/system_card.pdf`** — intended use, limitations, failure
  modes, fairness-risk discussion, operational constraints, reproducibility
  pointer.
- **Authored `reports/profiling_report.pdf`** — environment, per-stage latency,
  batch-size sensitivity, CPU baseline, interpretation. No GPU run; the host
  has no CUDA.
- **Authored `reports/reproducibility_checklist.md`** — exact commands for
  every artifact above.
- **All 74 tests pass** after the preprocessing fix.

## Headline metrics (final system, test split)

| Metric | Value |
|--------|-------|
| Threshold (cosine, max F1 on val) | **0.351759** |
| Accuracy | 0.9734 |
| Precision | 0.9723 |
| Recall | 0.9765 |
| F1 | 0.9744 |
| Confusion matrix (TP, FP, TN, FN) | 457, 13, 420, 11 |

CPU baseline (single pair, mean): preprocessing 7.0 ms, embedding 118.7 ms,
scoring 0.18 ms, end-to-end 125.9 ms. Throughput peaks at batch 8
(≈ 12.2 pairs/s) and degrades past batch 16 on the 8-core profiling host
(see `reports/profiling_report.pdf`).

## Determinism

- Every random operation uses seed `42` (configured in YAML).
- Splits are deterministic via `train_test_split(random_state=42)`.
- Pair generation sorts candidates before output for stable ordering.
- Data-centric operations use `np.random.default_rng(42)`.
- Re-running any script with the same config produces identical outputs.

## Earlier milestones (kept for context)

- **Milestone 1** built the deterministic dataset/pairs/scoring backbone.
- **Milestone 2** added threshold calibration, tracked runs, error analysis,
  data-centric improvement, validation checks, and tests. Report:
  `reports/milestone2_report.pdf`.
- **Milestone 3** introduced FaceNet embeddings, the `infer_pair` API,
  `scripts/infer.py`, the Dockerfile, and the load test.
- **Milestone 4** (this release) is the audit / profiling / release milestone.
