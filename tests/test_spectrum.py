import numpy as np
import pytest

from efnmr.sim import fid
from efnmr.spectrum import amplitude_spectrum, average_spectra, find_peak, snr

FS_HZ = 48000


def test_shapes():
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    freqs, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ)
    assert len(freqs) == len(amplitude)


def test_frequency_axis_range():
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    freqs, _ = amplitude_spectrum(x, fs_hz=FS_HZ)

    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(FS_HZ / 2)


def test_peak_amplitude():
    x = fid(f0_hz=1788.0, t2_star_s=1000.0, amplitude=2.0, phase_rad=0.0)
    _, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ, skip_s=0.0)

    assert amplitude.max() == pytest.approx(2.0, rel=0.02)


def test_peak_frequency():
    x = fid(f0_hz=1788.0, t2_star_s=100.0, amplitude=1.0, phase_rad=0.0)
    freqs, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ)
    k = np.argmax(amplitude)
    assert freqs[k] == pytest.approx(1788.0, abs=0.5)


def test_skip_length():
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    _, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ, skip_s=0.02)
    n_used = 144000 - 960
    assert len(amplitude) == n_used // 2 + 1


def test_find_peak_on_bin():
    x = fid(f0_hz=1788.0, t2_star_s=100.0, amplitude=1.0, phase_rad=0.0)
    freqs, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ, skip_s=0.0)

    f_peak, _ = find_peak(freqs, amplitude, f_center_hz=1788.0)
    assert f_peak == pytest.approx(1788.0, abs=0.01)


def test_find_peak_off_bin():
    x = fid(f0_hz=1788.17, t2_star_s=100.0, amplitude=1.0, phase_rad=0.0)
    freqs, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ, skip_s=0.0)

    f_peak, _ = find_peak(freqs, amplitude, f_center_hz=1788.0)
    assert f_peak == pytest.approx(1788.17, abs=0.02)


def test_find_peak_stays_in_window():
    signal = fid(f0_hz=1788.0, t2_star_s=100.0, amplitude=1.0, phase_rad=0.0)
    strong = fid(f0_hz=1900.0, t2_star_s=100.0, amplitude=10.0, phase_rad=0.0)
    freqs, amplitude = amplitude_spectrum(signal + strong, fs_hz=FS_HZ, skip_s=0.0)

    f_peak, _ = find_peak(freqs, amplitude, f_center_hz=1788.0, half_width_hz=20.0)
    assert f_peak == pytest.approx(1788.0, abs=0.05)


def test_average_shape():
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    freqs, single = amplitude_spectrum(x, fs_hz=FS_HZ)
    freqs_avg, averaged = average_spectra([x, x, x, x], fs_hz=FS_HZ)

    assert averaged.shape == single.shape
    assert np.array_equal(freqs_avg, freqs)


def test_average_of_identical():
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    _, single = amplitude_spectrum(x, fs_hz=FS_HZ)
    _, averaged = average_spectra([x, x, x, x], fs_hz=FS_HZ)

    assert np.allclose(averaged, single)


def test_snr_clean_signal_is_high():
    rng = np.random.default_rng(0)
    x = fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=1.0, phase_rad=0.0)
    x = x + rng.normal(0, 0.001, len(x))
    freqs, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ)
    assert snr(freqs, amplitude, f_peak_hz=1788.0) > 100


def test_snr_pure_noise_is_low():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.0, 144000)
    freqs, amplitude = amplitude_spectrum(x, fs_hz=FS_HZ)
    assert snr(freqs, amplitude, f_peak_hz=1788.0) < 5


def test_averaging_improves_snr():
    shots = [
        fid(f0_hz=1788.0, t2_star_s=1.0, amplitude=0.05, phase_rad=0.0)
        + np.random.default_rng(seed).normal(0, 0.5, 144000)
        for seed in range(16)
    ]
    freqs, single = amplitude_spectrum(shots[0], fs_hz=FS_HZ)
    freqs_avg, averaged = average_spectra(shots, fs_hz=FS_HZ)
    snr_single = snr(freqs, single, f_peak_hz=1788.0)
    snr_avg = snr(freqs_avg, averaged, f_peak_hz=1788.0)
    assert snr_avg > 2 * snr_single
