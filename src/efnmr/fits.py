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
