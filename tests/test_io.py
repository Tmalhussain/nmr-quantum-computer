from efnmr.io import save_shot
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
