import os
import json
import csv
import uuid
import subprocess
from datetime import datetime


RUNS_DIR = os.path.join("outputs", "runs")
RUNS_CSV = os.path.join(RUNS_DIR, "runs.csv")


def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit = "unknown"
    return commit


def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def create_run(note="", config_name="m1.yaml", data_version="baseline"):
    os.makedirs(RUNS_DIR, exist_ok=True)

    run_id = generate_run_id()
    run_dir = os.path.join(RUNS_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    run_info = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "commit": get_git_commit(),
        "config": config_name,
        "data_version": data_version,
        "note": note,
    }

    # Save initial metadata
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    return run_id, run_dir, run_info


def log_metrics(run_id, run_dir, metrics_dict):
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def log_threshold(run_dir, threshold_data):
    with open(os.path.join(run_dir, "threshold.json"), "w") as f:
        json.dump(threshold_data, f, indent=2)


def log_artifact(run_dir, src_path, name=None):
    if not os.path.exists(src_path):
        return

    if name is None:
        name = os.path.basename(src_path)

    dst_path = os.path.join(run_dir, name)

    import shutil
    shutil.copy(src_path, dst_path)


def append_run_csv(run_info, metrics=None, threshold=None):
    os.makedirs(RUNS_DIR, exist_ok=True)

    row = {
        "run_id": run_info["run_id"],
        "timestamp": run_info["timestamp"],
        "commit": run_info["commit"],
        "config": run_info["config"],
        "data_version": run_info["data_version"],
        "note": run_info["note"],
    }

    if metrics:
        row.update({
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
        })

    if threshold:
        row["threshold"] = threshold

    file_exists = os.path.exists(RUNS_CSV)

    with open(RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
