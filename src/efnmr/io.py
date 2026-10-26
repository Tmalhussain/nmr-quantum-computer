import json
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
