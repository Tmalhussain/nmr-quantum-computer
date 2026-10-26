import json
import csv

import numpy as np
import pytest

from efnmr.io import append_log, load_shot, save_shot
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
