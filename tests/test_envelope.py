import numpy as np
import pytest

from efnmr.envelope import complex_amplitude, demodulate, fit_decay
from efnmr.sim import fid

FS_HZ = 48000


def test_demod_is_complex():
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    env = demodulate(x, fs_hz=FS_HZ, f0_hz=1788.0)
    assert np.iscomplexobj(env)
    assert len(env) == len(x)


def test_demod_magnitude():
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    env = demodulate(x, fs_hz=FS_HZ, f0_hz=1788.0)
    k = int(0.5 * FS_HZ)
    assert abs(env[k]) == pytest.approx(np.exp(-0.5 / 1.2), abs=1e-3)


def test_demod_phase():
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    env = demodulate(x, fs_hz=FS_HZ, f0_hz=1788.0)
    k = int(0.5 * FS_HZ)
    assert np.angle(env[k]) == pytest.approx(0.5, abs=1e-3)


def test_complex_amplitude_phase():
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    z = complex_amplitude(x, fs_hz=FS_HZ, f0_hz=1788.0)
    assert np.angle(z) == pytest.approx(0.5, abs=1e-3)


def test_complex_amplitude_magnitude():
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    z = complex_amplitude(x, fs_hz=FS_HZ, f0_hz=1788.0, t_from_s=0.1, t_to_s=0.3)
    expected = 1.2 * (np.exp(-0.1 / 1.2) - np.exp(-0.3 / 1.2)) / 0.2
    assert abs(z) == pytest.approx(expected, abs=1e-3)


def test_complex_amplitude_bad_window():
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    with pytest.raises(ValueError):
        complex_amplitude(x, fs_hz=FS_HZ, f0_hz=1788.0, t_from_s=2.9, t_to_s=3.5)


def test_fit_decay_noiseless():
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    env = demodulate(x, fs_hz=FS_HZ, f0_hz=1788.0)
    fit = fit_decay(env, fs_hz=FS_HZ)
    assert fit["t2_star"] == pytest.approx(1.2, rel=1e-3)


def test_fit_decay_noisy():
    rng = np.random.default_rng(0)
    x = fid(f0_hz=1788.0, t2_star_s=1.2, amplitude=1.0, phase_rad=0.5)
    x = x + rng.normal(0, 0.2, len(x))
    env = demodulate(x, fs_hz=FS_HZ, f0_hz=1788.0)
    fit = fit_decay(env, fs_hz=FS_HZ)
    assert abs(fit["t2_star"] - 1.2) < 3 * fit["t2_star_err"]
