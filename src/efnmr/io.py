import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def save_shot(run_dir: Path, stem: str, recording: np.ndarray, metadata: dict) -> None:
    """Save one shot as <stem>.npy (recording) and <stem>.json (metadata + schema version)."""
    npy_path = run_dir / f"{stem}.npy"
    json_path = run_dir / f"{stem}.json"
    np.save(npy_path, recording)
    meta = dict(metadata)
    meta["schema_version"] = 1
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_shot(run_dir: Path, stem: str) -> tuple[np.ndarray, dict]:
    """Load <stem>.npy and <stem>.json; raise ValueError on an unknown schema version."""
    npy_path = run_dir / f"{stem}.npy"
    json_path = run_dir / f"{stem}.json"
    recording = np.load(npy_path)
    with open(json_path) as f:
        meta = json.load(f)
    version = meta.get("schema_version")
    if version != 1:
        raise ValueError(f"{json_path}: unsupported schema_version {version!r}, expected 1")
    return recording, meta


def append_log(run_dir: Path, row: dict) -> None:
    """Append one row to <run_dir>/log.csv, writing the header on first use."""
    log_path = run_dir / "log.csv"
    is_new = not log_path.exists()
    if not is_new:
        with open(log_path, newline="") as f:
            header = next(csv.reader(f))
        if header != list(row):
            raise ValueError(f"{log_path}: columns {list(row)} don't match header {header}")

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def new_run_dir(base_dir: Path, experiment: str, now: datetime | None = None) -> Path:
    """Create and return base_dir/<YYYYMMDD-HHMMSS>_<experiment>/; doesnt reuses a folder."""
    now = now or datetime.now()
    run_dir = base_dir / f"{now:%Y%m%d-%H%M%S}_{experiment}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
