import numpy as np


def amplitude_spectrum(
    x: np.ndarray, fs_hz: int, skip_s: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Hann windowed spectrum after skipping first skip_s seconds"""
    x = x[int(round(fs_hz * skip_s)) :]
    hann = np.hanning(len(x))
    tapered = hann * x
    amplitude = abs(np.fft.rfft(tapered)) * 2 / hann.sum()
    freqs = np.fft.rfftfreq(len(x), d=1 / fs_hz)
    return freqs, amplitude


def find_peak(
    freqs: np.ndarray, amplitude: np.ndarray, f_center_hz: float, half_width_hz: float = 20.0
) -> tuple[float, float]:
    """Find the vertex with parabolic interpolation on the log scaled amplitude"""
    idx = np.flatnonzero(
        (f_center_hz - half_width_hz < freqs) & (freqs < f_center_hz + half_width_hz)
    )
    k = idx[np.argmax(amplitude[idx])]
    a = np.log(amplitude[k - 1])
    b = np.log(amplitude[k])
    c = np.log(amplitude[k + 1])
    delta = 1 / 2 * (a - c) / (a - 2 * b + c)
    return float((k + delta) * (freqs[1] - freqs[0])), float(amplitude[k])
