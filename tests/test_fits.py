import numpy as np
import pytest

from efnmr.fits import fit_t1, t1_model

d = np.linspace(0.1, 2.5, 20)


def test_t1_model_at_zero():
    assert t1_model(0.0, a0=1.0, t1=2.0, c=7.0) == pytest.approx(8.0)


def test_t1_noiseles():
    amplitudes = t1_model(d, a0=1.0, t1=2.5, c=0.05)
    fit = fit_t1(d, amplitudes)
    assert fit["t1"] == pytest.approx(2.5, rel=1e-3)


def test_t1_noisy():
    rng = np.random.default_rng(0)
    amplitudes = t1_model(d, a0=1.0, t1=2.5, c=0.05) + rng.normal(0, 0.02, 20)
    fit = fit_t1(d, amplitudes)
    assert abs(fit["t1"] - 2.5) < 3 * fit["t1_err"]
