import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.inference import infer_pair


# -------------------------------------------------------------------------
# Load deterministic pairs (use your actual CSV)
# -------------------------------------------------------------------------
PAIRS_PATH = "outputs/pairs/test.csv"  # ✅ FIXED
THRESHOLD = 0.5

pairs_df = pd.read_csv(PAIRS_PATH)


def run_once(idx):
    row = pairs_df.iloc[idx % len(pairs_df)]

    img1 = row["left_path"]   
    img2 = row["right_path"] 

    result = infer_pair(img1, img2, THRESHOLD)
    return result["latency"]


# -------------------------------------------------------------------------
# Load test
# -------------------------------------------------------------------------
def main(num_requests=100, workers=4):
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        latencies = list(executor.map(run_once, range(num_requests)))

    total_time = time.time() - start
    latencies = np.array(latencies)

    print(f"Total requests: {num_requests}")
    print(f"Workers: {workers}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_requests / total_time:.2f} req/s")
    print(f"Avg latency: {latencies.mean():.4f}s")
    print(f"P95 latency: {np.percentile(latencies, 95):.4f}s")
    print(f"Min latency: {latencies.min():.4f}s")
    print(f"Max latency: {latencies.max():.4f}s")


if __name__ == "__main__":
    main()
