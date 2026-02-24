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

### 1. Set up environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest LFW dataset

```bash
python scripts/ingest_lfw.py --config configs/m1.yaml
```

### 3. Generate verification pairs

```bash
python scripts/make_pairs.py --config configs/m1.yaml
```

### 4. Run similarity benchmark

```bash
python scripts/bench_similarity.py --config configs/m1.yaml
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## Outputs

| File | Location |
|------|----------|
| Dataset manifest | `outputs/manifest.json` |
| Pair splits | `outputs/pairs/{train,val,test}.csv` |
| Benchmark results | `outputs/bench/benchmark_results.json` |

## Determinism

- All random operations use seed `42` (configured in `configs/m1.yaml`).
- Dataset splits are deterministic: identities are sorted before splitting.
- Pair generation sorts candidates before sampling with the fixed seed.
- Rerunning any script with the same config produces identical outputs.
