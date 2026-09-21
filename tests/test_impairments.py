"""Physical correctness tests for the transmitter impairment models
(impairments.py): frequency offset shifts a tone's spectrum by exactly the
configured amount, IQ imbalance produces a mirror-image tone at the
closed-form theoretical image rejection ratio, and phase noise is a
deterministic, cross-batch-consistent Wiener process."""

import numpy as np
import pytest

from ccsds_chain.impairments import PhaseNoiseGenerator, apply_frequency_offset, apply_iq_imbalance

FS = 1_000_000.0
N = 100_000


def _tone(freq_hz, n=N, fs=FS):
    t = np.arange(n) / fs
    return np.exp(1j * 2 * np.pi * freq_hz * t)


def _fft_peak_freq(iq, fs=FS):
    spec = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))
    return freqs[np.argmax(np.abs(spec))]


def _fft_at(iq, freq_hz, fs=FS):
    spec = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))
    return spec[np.argmin(np.abs(freqs - freq_hz))]


class TestFrequencyOffset:
    def test_disabled_is_exact_identity(self):
        tone = _tone(50_000.0)
        assert np.array_equal(apply_frequency_offset(tone, 0.0, FS), tone)

    def test_shifts_a_tone_by_exactly_the_configured_offset(self):
        f0, offset = 50_000.0, 10_000.0
        shifted = apply_frequency_offset(_tone(f0), offset, FS)
        assert _fft_peak_freq(shifted) == pytest.approx(f0 + offset, abs=1.0)

    def test_negative_offset_shifts_down(self):
        f0, offset = 50_000.0, -15_000.0
        shifted = apply_frequency_offset(_tone(f0), offset, FS)
        assert _fft_peak_freq(shifted) == pytest.approx(f0 + offset, abs=1.0)

    def test_exact_regardless_of_chunking(self):
        """A per-sample phase ramp computed from absolute position, not
        accumulated incrementally -- so splitting the same signal into
        chunks with the correct start_sample must reproduce the
        unchunked result exactly (no float accumulation drift)."""
        tone = _tone(50_000.0)
        whole = apply_frequency_offset(tone, 10_000.0, FS, start_sample=0)
        a = apply_frequency_offset(tone[:40_000], 10_000.0, FS, start_sample=0)
        b = apply_frequency_offset(tone[40_000:], 10_000.0, FS, start_sample=40_000)
        assert np.array_equal(whole, np.concatenate([a, b]))


class TestIQImbalance:
    def test_disabled_is_exact_identity(self):
        tone = _tone(50_000.0)
        assert np.allclose(apply_iq_imbalance(tone, 0.0, 0.0), tone)

    @pytest.mark.parametrize("gain_db,phase_deg", [(1.0, 5.0), (0.3, 1.0), (2.0, 10.0)])
    def test_image_rejection_ratio_matches_theory(self, gain_db, phase_deg):
        f0 = 50_000.0
        tone = _tone(f0)
        imbalanced = apply_iq_imbalance(tone, gain_db, phase_deg)
        wanted = abs(_fft_at(imbalanced, f0))
        image = abs(_fft_at(imbalanced, -f0))
        measured_irr_db = 20 * np.log10(wanted / image)

        g = 10 ** (gain_db / 20.0)
        phi = np.deg2rad(phase_deg)
        a = (1 + g * np.cos(phi) + 1j * g * np.sin(phi)) / 2
        b = (1 - g * np.cos(phi) + 1j * g * np.sin(phi)) / 2
        theory_irr_db = 20 * np.log10(abs(a) / abs(b))

        assert measured_irr_db == pytest.approx(theory_irr_db, abs=0.05)

    def test_zero_imbalance_leaves_no_image(self):
        f0 = 50_000.0
        imbalanced = apply_iq_imbalance(_tone(f0), 0.0, 0.0)
        assert abs(_fft_at(imbalanced, -f0)) < 1e-9


class TestPhaseNoise:
    def test_zero_linewidth_still_runs(self):
        # linewidth=0 means step_std=0: every increment is exactly 0, so
        # the process never rotates the signal at all.
        gen = PhaseNoiseGenerator(linewidth_hz=0.0, sample_rate=FS, seed=1)
        out = gen.apply(np.ones(1000, dtype=complex))
        assert np.allclose(out, np.ones(1000, dtype=complex))

    def test_deterministic_for_the_same_seed(self):
        gen1 = PhaseNoiseGenerator(linewidth_hz=100.0, sample_rate=FS, seed=42)
        gen2 = PhaseNoiseGenerator(linewidth_hz=100.0, sample_rate=FS, seed=42)
        out1 = gen1.apply(np.ones(10_000, dtype=complex))
        out2 = gen2.apply(np.ones(10_000, dtype=complex))
        assert np.array_equal(out1, out2)

    def test_different_seeds_diverge(self):
        gen1 = PhaseNoiseGenerator(linewidth_hz=100.0, sample_rate=FS, seed=1)
        gen2 = PhaseNoiseGenerator(linewidth_hz=100.0, sample_rate=FS, seed=2)
        out1 = gen1.apply(np.ones(10_000, dtype=complex))
        out2 = gen2.apply(np.ones(10_000, dtype=complex))
        assert not np.array_equal(out1, out2)

    def test_consistent_across_chunking_within_float_rounding(self):
        """State (phase + PRNG) carried across apply() calls: chunked
        output must match an unchunked call to float64 rounding noise --
        the same tolerance export_chain()'s own docstring documents for
        RRC/conv batching, not an exact-bit-for-bit contract."""
        whole_gen = PhaseNoiseGenerator(linewidth_hz=150.0, sample_rate=FS, seed=7)
        whole = whole_gen.apply(np.ones(100_000, dtype=complex))

        chunked_gen = PhaseNoiseGenerator(linewidth_hz=150.0, sample_rate=FS, seed=7)
        a = chunked_gen.apply(np.ones(40_000, dtype=complex))
        b = chunked_gen.apply(np.ones(60_000, dtype=complex))
        chunked = np.concatenate([a, b])

        assert np.allclose(whole, chunked, atol=1e-9)

    def test_output_stays_unit_magnitude(self):
        gen = PhaseNoiseGenerator(linewidth_hz=500.0, sample_rate=FS, seed=3)
        out = gen.apply(np.ones(5000, dtype=complex))
        assert np.allclose(np.abs(out), 1.0)

    def test_phase_variance_grows_with_linewidth(self):
        """Sanity check against the model's own defining property (Demir
        et al.): per-sample phase variance = 2*pi*linewidth/sample_rate,
        so a wider linewidth must produce a visibly larger phase spread
        over the same number of samples -- checked statistically (many
        independent generators), not as an exact value."""
        def final_phase_std(linewidth_hz, trials=200, n=2000):
            finals = []
            for seed in range(trials):
                gen = PhaseNoiseGenerator(linewidth_hz=linewidth_hz, sample_rate=FS, seed=seed)
                out = gen.apply(np.ones(n, dtype=complex))
                finals.append(np.unwrap(np.angle(out))[-1])
            return np.std(finals)

        narrow = final_phase_std(50.0)
        wide = final_phase_std(500.0)
        assert wide > narrow
