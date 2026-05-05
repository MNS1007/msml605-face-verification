# Reproducibility Checklist — Milestone 4 (`v1.0-final`)

This checklist is the minimum exact path to recreate the released system from a
clean clone. Every command is copy-pastable from a Unix shell at the repository
root. Numbers below come from the run that produced the System Card and the
profiling report.

> **Note on what is reproduced.** The generated `outputs/pairs/*.csv` and
> `outputs/scores/*_scored.csv` files contain **absolute filesystem paths** to
> the local kagglehub dataset cache and are therefore machine-specific. They are
> intentionally **gitignored**. The repository ships the deterministic
> *generators* (`ingest_lfw.py`, `make_pairs.py`, `score_pairs.py`),
> `configs/m1.yaml`, and seed `42` — running steps 2 and 3 below regenerates
> them on your machine. Because the seed and sort order are fixed, the same
> identity pairs emerge in the same order; only the path prefix differs across
> machines. The committed lightweight evidence under
> `outputs/threshold/`, `outputs/eval/`, `outputs/profiling/`, and
> `outputs/load_test/` is path-free and directly comparable across machines.

## 0. Pinned references

| Item | Value |
|------|-------|
| Final tag | `v1.0-final` |
| Final config | `configs/m3.yaml` |
| Operating threshold (cosine) | `0.351759` |
| Threshold selection rule | max F1 on validation split |
| Embedding | FaceNet `InceptionResnetV1`, pretrained `vggface2`, 512-dim |
| Score metric | cosine similarity (higher = more likely same) |
| Preprocessing | RGB → resize 160×160 → normalize to `[-1, 1]` |
| Random seed | `42` (everywhere) |

## 1. Environment

```bash
# Python 3.11 or 3.12 (we test with 3.12).
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

LFW dataset is downloaded on first use via `kagglehub`. If you have the dataset
cached elsewhere, edit `configs/m1.yaml::data_path`.

## 2. Reproduce the dataset, pairs, and scoring artifacts

```bash
# 2a. Ingest LFW and create deterministic 70/15/15 splits
python scripts/ingest_lfw.py --config configs/m1.yaml

# 2b. Generate verification pairs (filters unreadable image paths)
python scripts/make_pairs.py --config configs/m1.yaml

# 2c. Score all splits with FaceNet + cosine similarity
python scripts/score_pairs.py
```

Artifacts produced:
- `outputs/manifest.json`
- `outputs/pairs/{train,val,test}.csv` (4 199 / 900 / 901 pairs)
- `outputs/scores/{train,val,test}_scored.csv`

## 3. Reproduce the operating threshold and test metrics

```bash
# 3a. Sweep thresholds and select max-F1 on validation
python scripts/run_threshold_sweep.py
python scripts/select_threshold.py

# 3b. Lock the selected threshold and evaluate on the held-out test split
python scripts/evaluate_test.py
```

Artifacts produced:
- `outputs/sweeps/val_sweep.csv`, `outputs/sweeps/roc_val.png`
- `outputs/threshold/selected_threshold.json` (threshold = **0.351759**)
- `outputs/threshold/val_confusion_matrix.json`
- `outputs/eval/test_metrics.json` (test F1 = **0.9744**, accuracy = **0.9734**)
- `outputs/eval/test_confusion_matrix.json` (TP 457, FP 13, TN 420, FN 11)

## 4. Reproduce the profiling report (CPU baseline)

```bash
python scripts/profile_latency.py \
    --batch-sizes 1 2 4 8 16 32 \
    --num-single-pairs 25 \
    --warmup 3 \
    --repeat 5
```

Artifacts produced (under `outputs/profiling/`):
- `per_stage_latency.json` — preprocessing / embedding / scoring / end-to-end
- `batch_size_sensitivity.csv` — totals + per-pair amortized + throughput
- `profiling_environment.json` — OS, CPU, library versions, device used
- `profiling_summary.md` — human-readable summary table

## 5. Run the released CLI on a single pair

```bash
python scripts/infer.py \
    --left  outputs/pairs/test.csv \
    --right outputs/pairs/test.csv  # placeholder; replace with real image paths
```

A real example using two LFW images of the same identity:

```bash
LEFT=$(awk -F, 'NR==2{print $1}' outputs/scores/val_scored.csv)
RIGHT=$(awk -F, 'NR==2{print $2}' outputs/scores/val_scored.csv)
python scripts/infer.py --left "$LEFT" --right "$RIGHT"
```

The output prints score, threshold (0.351759), decision, calibrated confidence,
and per-pair latency in ms.

Batch mode (CSV with `left_path,right_path`):

```bash
python scripts/infer.py --batch outputs/pairs/test.csv
```

## 6. Run the Dockerized CLI

```bash
# Build (downloads + bakes the FaceNet weights into the image)
docker build -t face-verifier:v1.0-final .

# Single pair (mount the dataset cache so the image paths resolve)
docker run --rm \
    -v "$HOME/.cache/kagglehub:/root/.cache/kagglehub:ro" \
    face-verifier:v1.0-final \
    --left  /root/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0001.jpg \
    --right /root/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled/lfw-deepfunneled/Aaron_Sorkin/Aaron_Sorkin_0002.jpg

# Batch CSV
docker run --rm \
    -v "$HOME/.cache/kagglehub:/root/.cache/kagglehub:ro" \
    -v "$PWD/outputs:/app/outputs:ro" \
    face-verifier:v1.0-final \
    --batch /app/outputs/pairs/test.csv
```

## 7. Run the test suite

```bash
python -m pytest tests/ -v
```

74 tests should pass.

## 8. (Optional) Concurrency / load test

```bash
python scripts/load_test.py
```

Writes `outputs/load_test/runtime_summary.json`.

## 9. Where the final artifacts live

| Deliverable | Path |
|-------------|------|
| System Card (PDF) | `reports/system_card.pdf` |
| System Card (LaTeX source) | `reports/system_card.tex` |
| Profiling report (PDF) | `reports/profiling_report.pdf` |
| Profiling report (LaTeX source) | `reports/profiling_report.tex` |
| Reproducibility checklist | `reports/reproducibility_checklist.md` (this file) |
| Final config | `configs/m3.yaml` |
| Selected threshold | `outputs/threshold/selected_threshold.json` |
| Test metrics | `outputs/eval/test_metrics.json` |
| Profiling artifacts | `outputs/profiling/` |
| CLI entry point | `scripts/infer.py` |
| Dockerfile | `Dockerfile` |
| Final tag | `v1.0-final` |
