import numpy as np
from scipy.signal import butter, sosfiltfilt

from efnmr.fits import fit_t1


def demodulate(x: np.ndarray, fs_hz: int, f0_hz: float, cutoff_hz: float = 50.0) -> np.ndarray:
    """Complex envelope of x at f0_hz: abs() is the amplitude, angle() the phase."""
    t = np.arange(len(x)) / fs_hz
    m = x * np.exp(-2j * np.pi * f0_hz * t)
    sos = butter(4, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    real = sosfiltfilt(sos, m.real)
    imag = sosfiltfilt(sos, m.imag)
    return 2 * (real + 1j * imag)


def complex_amplitude(
    x: np.ndarray,
    fs_hz: int,
    f0_hz: float,
    t_from_s: float = 0.05,
    t_to_s: float = 0.25,
    cutoff_hz: float = 50.0,
) -> complex:
    """Mean complex envelope of x over [t_from_s, t_to_s]: one readout value per shot."""
    env = demodulate(x, fs_hz, f0_hz, cutoff_hz)
    i_from = int(round(t_from_s * fs_hz))
    i_to = int(round(t_to_s * fs_hz))
    if i_from < 0 or i_to > len(x) or i_to <= i_from:
        raise ValueError(
            f"window {t_from_s}–{t_to_s} s is outside the {len(x) / fs_hz} s recording"
        )
    return complex(env[i_from:i_to].mean())


def fit_decay(
    enve: np.ndarray, fs_hz: int, t_from_s: float = 0.05, t_to_s: float = 2.9, step: int = 48
) -> dict:
    """Fit T2* to abs(envelope) over [t_from_s, t_to_s]; returns t2_star, a0, c and 1σ errors."""
    i_from = int(round(t_from_s * fs_hz))
    i_to = int(round(t_to_s * fs_hz))
    if i_from < 0 or i_to > len(enve) or i_to <= i_from:
        raise ValueError(
            f"window {t_from_s}–{t_to_s} s is outside the {len(enve) / fs_hz} s recording"
        )
    times = np.arange(i_from, i_to, step) / fs_hz
    mag = np.abs(enve[i_from:i_to:step])
    fit = fit_t1(times, mag)
    return {
        "t2_star": fit["t1"],
        "a0": fit["a0"],
        "c": fit["c"],
        "t2_star_err": fit["t1_err"],
        "a0_err": fit["a0_err"],
        "c_err": fit["c_err"],
    }
