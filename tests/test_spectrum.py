"""Regression tests for PSD estimation and occupied-bandwidth measurement."""

import numpy as np
import pytest

from ccsds_chain.spectrum import contiguous_bandwidth, null_to_null_bandwidth, welch_psd


class TestWelchPSD:
    def test_rejects_signal_shorter_than_one_segment(self):
        with pytest.raises(ValueError):
            welch_psd(np.zeros(10), nperseg=4096)

    def test_output_length_is_nperseg(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(4096 * 5) + 1j * rng.standard_normal(4096 * 5)
        psd = welch_psd(x, nperseg=4096)
        assert len(psd) == 4096

    def test_pure_tone_peaks_near_its_own_bin(self):
        n = 4096 * 8
        fs = 1.0
        f0 = 0.2  # cycles/sample
        t = np.arange(n)
        x = np.exp(2j * np.pi * f0 * t)
        psd = welch_psd(x, nperseg=4096)
        freqs = np.fft.fftshift(np.fft.fftfreq(4096, d=1 / fs))
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - f0) < 1 / 4096 * 2  # within a couple of FFT bins


def _synthetic_db_spectrum(nperseg, passband_bins, floor_db=-60.0):
    """A synthetic dB spectrum that is 0 dB inside +-passband_bins of DC and
    floor_db outside -- a stand-in for an ideal, symmetric RRC-like passband,
    used to test the bandwidth-measurement scan logic independent of any
    actual signal generation."""
    freqs = np.arange(-nperseg // 2, nperseg // 2)
    db = np.where(np.abs(freqs) <= passband_bins, 0.0, floor_db)
    return freqs.astype(float), db


class TestContiguousBandwidth:
    def test_measures_the_ideal_rectangular_passband(self):
        freqs, db = _synthetic_db_spectrum(4096, passband_bins=100)
        bw = contiguous_bandwidth(freqs, db, threshold_db=-3.0)
        assert bw == pytest.approx(200.0, abs=1.0)

    def test_ignores_short_dips_inside_the_passband(self):
        freqs, db = _synthetic_db_spectrum(4096, passband_bins=100)
        db = db.copy()
        db[2050] = -10.0  # a single noisy bin dipping below threshold
        bw = contiguous_bandwidth(freqs, db, threshold_db=-3.0, min_consecutive_below=3)
        assert bw == pytest.approx(200.0, abs=1.0)


class TestNullToNullBandwidth:
    def test_measures_symmetric_main_lobe(self):
        # A flat passband dropping straight to a flat floor has no actual
        # local minimum for the algorithm to lock onto (by design: it tracks
        # a running minimum until the PSD *rises* again) -- that's a
        # rectangular low-pass response, not a null-to-null main lobe, so it
        # is not representative of what null_to_null_bandwidth is measuring.
        # Model an RRC-like main lobe instead: flat top, roll-off down to a
        # genuine null, then a rise into the first side lobe -- and assert
        # the null is found exactly where this synthetic spectrum puts it.
        nperseg = 4096
        freqs = np.arange(-nperseg // 2, nperseg // 2).astype(float)
        passband_bins, null_bins, sidelobe_bins = 100, 150, 50
        af = np.abs(freqs)
        db = np.select(
            [af <= passband_bins, af <= null_bins, af <= null_bins + sidelobe_bins],
            [
                0.0,
                -80.0 * (af - passband_bins) / (null_bins - passband_bins),
                -80.0 + 40.0 * (af - null_bins) / sidelobe_bins,
            ],
            default=-40.0,
        )
        bw = null_to_null_bandwidth(freqs, db, entry_threshold_db=-20.0)
        assert bw == pytest.approx(2 * null_bins, abs=2.0)
