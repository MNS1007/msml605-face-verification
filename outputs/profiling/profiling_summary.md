# Profiling Summary

## Environment

- **platform**: macOS-15.7.4-arm64-arm-64bit
- **python_version**: 3.12.0
- **machine**: arm64
- **processor**: arm
- **cpu_count_logical**: 8
- **torch_version**: 2.2.2
- **torch_num_threads**: 8
- **device_used**: cpu
- **cuda_available**: False
- **mps_available**: True
- **num_pairs_loaded**: 901
- **pairs_csv**: outputs/pairs/test.csv
- **num_single_pairs**: 25
- **warmup**: 3
- **repeat**: 5
- **timestamp**: 2026-05-04T19:03:56-0400

## Per-pair stage latency (single-pair inference path)

| Stage | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| preprocessing_per_pair | 7.01 | 6.758 | 8.149 | 6.062 | 8.506 |
| embedding_per_pair | 118.695 | 116.602 | 151.678 | 83.924 | 229.234 |
| scoring_per_pair | 0.175 | 0.143 | 0.369 | 0.119 | 0.566 |
| end_to_end_per_pair | 125.882 | 124.293 | 158.071 | 90.65 | 236.995 |

## Batch-size sensitivity (totals + per-pair amortized)

| batch_size | preprocessing_total_ms | embedding_total_ms | scoring_total_ms | end_to_end_total_ms | preprocessing_per_pair_ms | embedding_per_pair_ms | scoring_per_pair_ms | end_to_end_per_pair_ms | throughput_pairs_per_s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.602 | 112.446 | 0.131 | 120.181 | 7.602 | 112.446 | 0.131 | 120.181 | 8.32 |
| 2 | 14.985 | 182.327 | 0.149 | 197.463 | 7.493 | 91.164 | 0.074 | 98.732 | 10.13 |
| 4 | 30.952 | 347.567 | 0.164 | 378.684 | 7.738 | 86.892 | 0.041 | 94.671 | 10.56 |
| 8 | 51.93 | 605.718 | 0.248 | 657.899 | 6.491 | 75.715 | 0.031 | 82.237 | 12.16 |
| 16 | 116.215 | 6684.857 | 0.205 | 6801.28 | 7.263 | 417.804 | 0.013 | 425.08 | 2.35 |
| 32 | 212.286 | 9663.66 | 0.549 | 9876.497 | 6.634 | 301.989 | 0.017 | 308.641 | 3.24 |
