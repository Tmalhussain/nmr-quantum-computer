import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from efnmr.io import (
    append_log,
    git_commit,
    load_config,
    load_shot,
    new_run_dir,
    save_shot,
)
from efnmr.sim import fid


def test_save_creates_both_files(tmp_path):
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    save_shot(tmp_path, "shot_0001", x, {"f0_hz": 1788.0})
    assert (tmp_path / "shot_0001.npy").exists()
    assert (tmp_path / "shot_0001.json").exists()


def test_caller_dict_unchanged(tmp_path):
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    meta = {"f0_hz": 1788.0}
    save_shot(tmp_path, "shot_0001", x, meta)
    assert "schema_version" not in meta


def test_round_trip(tmp_path):
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    save_shot(tmp_path, "shot_0001", x, {"f0_hz": 1788.0, "pulse_s": 0.02})
    loaded, meta = load_shot(tmp_path, "shot_0001")
    assert np.array_equal(loaded, x)
    assert loaded.dtype == np.float32
    assert meta["f0_hz"] == 1788.0
    assert meta["pulse_s"] == 0.02
    assert meta["schema_version"] == 1


def test_wrong_schema_raises(tmp_path):
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    save_shot(tmp_path, "shot_0001", x, {"f0_hz": 1788.0})
    with open(tmp_path / "shot_0001.json", "w") as f:
        json.dump({"schema_version": 99}, f)
    with pytest.raises(ValueError):
        load_shot(tmp_path, "shot_0001")


def test_log_header_written_once(tmp_path):
    append_log(tmp_path, {"shot": 1, "f_peak_hz": 1788.01})
    append_log(tmp_path, {"shot": 2, "f_peak_hz": 1788.03})
    lines = (tmp_path / "log.csv").read_text().splitlines()
    assert lines[0] == "shot,f_peak_hz"
    assert len(lines) == 3


def test_log_rows_in_order(tmp_path):
    append_log(tmp_path, {"shot": 1, "snr": 7.5})
    append_log(tmp_path, {"shot": 2, "snr": 8.1})
    with open(tmp_path / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["shot"] for r in rows] == ["1", "2"]
    assert rows[1]["snr"] == "8.1"


def test_log_column_mismatch_raises(tmp_path):
    append_log(tmp_path, {"shot": 1, "snr": 7.5})
    with pytest.raises(ValueError):
        append_log(tmp_path, {"shot": 2, "temp_c": 30.0})


def test_run_dir_name(tmp_path):
    run_dir = new_run_dir(tmp_path, "rabi", now=datetime(2026, 10, 21, 14, 30, 0))
    assert run_dir == tmp_path / "20261021-143000_rabi"
    assert run_dir.is_dir()


def test_run_dir_creates_parents(tmp_path):
    run_dir = new_run_dir(tmp_path / "data", "t1", now=datetime(2026, 10, 21, 9, 5, 7))
    assert run_dir == tmp_path / "data" / "20261021-090507_t1"
    assert run_dir.is_dir()


def test_run_dir_never_reused(tmp_path):
    when = datetime(2026, 10, 21, 14, 30, 0)
    new_run_dir(tmp_path, "rabi", now=when)
    with pytest.raises(FileExistsError):
        new_run_dir(tmp_path, "rabi", now=when)


def test_config_merge(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text("f0_hz: 1788.0\nt90_s: 0.020\n")
    exp = tmp_path / "rabi.yaml"
    exp.write_text("t90_s: 0.025\nrepeats: 4\n")
    config = load_config(exp, site_path=site)
    assert config == {"f0_hz": 1788.0, "t90_s": 0.025, "repeats": 4}


def test_config_empty_file(tmp_path):
    exp = tmp_path / "empty.yaml"
    exp.write_text("")
    assert load_config(exp) == {}


def test_git_commit_in_repo():
    repo = Path(__file__).parent
    commit = git_commit(repo)
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", commit)


def test_git_commit_outside_repo(tmp_path):
    assert git_commit(tmp_path) == "unknown"
