import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


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


def load_config(path: Path, site_path: Path | None = None) -> dict:
    """Load an experiment YAML, merged over site.yaml if given (experiment values win)."""
    config = {}
    if site_path is not None:
        with open(site_path) as f:
            config.update(yaml.safe_load(f) or {})
    with open(path) as f:
        config.update(yaml.safe_load(f) or {})
    return config


def git_commit(repo_dir: Path | None = None) -> str:
    """Current commit hash, with '-dirty' if there are uncommitted changes; 'unknown' outside git"""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{head}-dirty" if status.strip() else head
