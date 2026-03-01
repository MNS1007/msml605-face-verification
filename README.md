# MSML 605 - Face Verification System

End-to-end face verification system built on the LFW (Labeled Faces in the Wild) dataset.
Given two face images, the system outputs whether they match (same identity or different),
along with a similarity score, calibrated confidence, and latency measurement.

**Milestone 1** builds the reproducible, deterministic foundation: dataset ingestion,
verification pair generation, and fast vectorized similarity scoring.

## Repo Layout

| Directory | Contents |
|-----------|----------|
| `src/` | Importable Python package (similarity module, etc.) |
| `scripts/` | CLI entrypoints: `ingest_lfw.py`, `make_pairs.py`, `bench_similarity.py` |
| `configs/` | YAML config files (seed, paths, split policy, benchmark params) |
| `tests/` | Unit tests and determinism checks |
| `notebooks/` | Optional exploration notebooks |
| `data/` | Downloaded dataset cache (**gitignored**) |
| `outputs/` | Generated artifacts — manifest, pairs, benchmark results (**gitignored**) |

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

# 3. Ingest LFW dataset (downloads data, creates deterministic splits and manifest)
python scripts/ingest_lfw.py --config configs/m1.yaml

# 4. Generate verification pairs
python scripts/make_pairs.py --config configs/m1.yaml

# 5. Run similarity benchmark (loop vs vectorized)
python scripts/bench_similarity.py --config configs/m1.yaml

# 6. Run tests
python -m pytest tests/ -v
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/manifest.json` | Dataset manifest (counts, seed, split policy, data source) |
| `outputs/pairs/train.csv` | Training verification pairs (left_path, right_path, label, split) |
| `outputs/pairs/val.csv` | Validation verification pairs |
| `outputs/pairs/test.csv` | Test verification pairs |
| `outputs/bench/benchmark_results.json` | Loop vs vectorized timing and correctness results |

## Determinism

- All random operations use seed `42` (configured in `configs/m1.yaml`).
- Dataset splits are deterministic via `sklearn.model_selection.train_test_split` with fixed `random_state`.
- Pair generation sorts candidates before output for consistent ordering.
- Rerunning any script with the same config produces identical outputs.
