import numpy as np
import pytest

from efnmr.fits import (
    echo_model,
    fit_echo,
    fit_rabi,
    fit_rb,
    fit_t1,
    rabi_model,
    rb_model,
    t1_model,
)

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


PULSES_S = np.linspace(0.005, 0.100, 20)


def test_rabi_model_peak_at_t90():
    assert rabi_model(0.020, t90=0.020, tau=1e9, a0=1.0, c=0.0) == pytest.approx(1.0)


def test_rabi_noiseless():
    amplitudes = rabi_model(PULSES_S, t90=0.020, tau=0.3, a0=1.0, c=0.05)
    fit = fit_rabi(PULSES_S, amplitudes)
    assert fit["t90"] == pytest.approx(0.020, rel=1e-3)


def test_rabi_noisy():
    rng = np.random.default_rng(0)
    amplitudes = rabi_model(PULSES_S, t90=0.020, tau=0.3, a0=1.0, c=0.05)
    amplitudes = amplitudes + rng.normal(0, 0.02, 20)
    fit = fit_rabi(PULSES_S, amplitudes)
    assert abs(fit["t90"] - 0.020) < 3 * fit["t90_err"]


ECHO_TAUS_S = np.linspace(0.05, 1.5, 20)


def test_echo_model_at_zero():
    assert echo_model(0.0, a0=1.0, t2=2.0, c=0.05) == pytest.approx(1.05)


def test_echo_noiseless():
    amplitudes = echo_model(ECHO_TAUS_S, a0=1.0, t2=2.0, c=0.05)
    fit = fit_echo(ECHO_TAUS_S, amplitudes)

    assert fit["t2"] == pytest.approx(2.0, rel=1e-3)


def test_echo_noisy():
    rng = np.random.default_rng(0)
    amplitudes = echo_model(ECHO_TAUS_S, a0=1.0, t2=2.0, c=0.05) + rng.normal(0, 0.02, 20)
    fit = fit_echo(ECHO_TAUS_S, amplitudes)

    assert abs(fit["t2"] - 2.0) < 3 * fit["t2_err"]


RB_LENGTHS = np.array([1, 2, 4, 8, 16, 32, 64, 96, 128], dtype=float)


def test_rb_model_at_zero():
    assert rb_model(0.0, a=0.5, p=0.98, b=0.5) == pytest.approx(1.0)


def test_rb_noiseless():
    survival = rb_model(RB_LENGTHS, a=0.5, p=0.98, b=0.5)
    fit = fit_rb(RB_LENGTHS, survival)
    assert fit["p"] == pytest.approx(0.98, rel=1e-3)
    assert fit["r"] == pytest.approx(0.01, rel=1e-2)


def test_rb_noisy():
    rng = np.random.default_rng(0)
    survival = rb_model(RB_LENGTHS, a=0.5, p=0.98, b=0.5) + rng.normal(0, 0.01, len(RB_LENGTHS))
    fit = fit_rb(RB_LENGTHS, survival)
    assert abs(fit["p"] - 0.98) < 3 * fit["p_err"]
