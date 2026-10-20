import numpy as np
import pytest

from efnmr.sim import fid


def test_length():
    s = fid(1788.0, 1.0, 1.0, 0.0)
    assert len(s) == 144000


def test_dtype():
    s = fid(1788.0, 1.0, 1.0, 0.0)
    assert s.dtype == np.float32


def test_initial_value():
    s = fid(f0_hz=1788.0, t2_star_s=100.0, amplitude=2.0, phase_rad=0.5)
    assert s[0] == pytest.approx(2.0 * np.cos(0.5), abs=1e-6)
