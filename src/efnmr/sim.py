import numpy as np


def fid(
    f0_hz: float,
    t2_star_s: float,
    amplitude: float,
    phase_rad: float,
    duration_s: float = 3.0,
    fs_hz: int = 48000,
) -> np.ndarray:
    """Simulate a free induction decay signal."""
    N = int(round(duration_s * fs_hz))
    k = np.arange(N)
    t = k / fs_hz
    s = amplitude * np.exp(-t / t2_star_s) * np.cos((2 * np.pi * f0_hz * t) + phase_rad)
    return s.astype(np.float32)
