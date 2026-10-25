import numpy as np
from scipy.optimize import curve_fit


def t1_model(d: np.ndarray, a0: float, t1: float, c: float) -> np.ndarray:
    return a0 * np.exp(-d / t1) + c


def fit_t1(d: np.ndarray, amplitudes: np.ndarray) -> dict:
    a0 = amplitudes[0] - amplitudes[-1]
    t1 = 1.0
    c = amplitudes[-1]
    low = [0.0, 0.0, 0.0]
    high = [np.inf, np.inf, np.inf]
    popt, pcov = curve_fit(t1_model, d, amplitudes, p0=(a0, t1, c), bounds=(low, high))
    perr = np.sqrt(np.diag(pcov))
    return {
        "a0": float(popt[0]),
        "t1": float(popt[1]),
        "c": float(popt[2]),
        "a0_err": float(perr[0]),
        "t1_err": float(perr[1]),
        "c_err": float(perr[2]),
    }


def rabi_model(t: np.ndarray, t90: float, tau: float, a0: float, c: float) -> np.ndarray:
    """Rabi oscillation: a0·|sin(πt / 2t90)|·exp(−t/tau) + c, t in seconds."""
    return a0 * np.abs(np.sin((np.pi * t) / (2 * t90))) * np.exp(-t / tau) + c


def fit_rabi(pulse_s: np.ndarray, amplitudes: np.ndarray) -> dict:
    """Fit a Rabi sweep; returns t90, tau, a0, c and their 1σ errors."""
    k = np.argmax(amplitudes[: len(amplitudes) // 2])
    t90 = pulse_s[k]
    tau = 5 * pulse_s[-1]
    a0 = amplitudes.max() - amplitudes.min()
    c = amplitudes.min()
    low = [1e-6, 1e-6, 0.0, 0.0]
    high = [np.inf] * 4
    popt, pcov = curve_fit(
        rabi_model, pulse_s, amplitudes, p0=(t90, tau, a0, c), bounds=(low, high)
    )
    perr = np.sqrt(np.diag(pcov))
    return {
        "t90": float(popt[0]),
        "tau": float(popt[1]),
        "a0": float(popt[2]),
        "c": float(popt[3]),
        "t90_err": float(perr[0]),
        "tau_err": float(perr[1]),
        "a0_err": float(perr[2]),
        "c_err": float(perr[3]),
    }
