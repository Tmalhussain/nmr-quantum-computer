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
