"""Physical correctness tests for the transmitter impairment models
(impairments.py): frequency offset shifts a tone's spectrum by exactly the
configured amount, IQ imbalance produces a mirror-image tone at the
closed-form theoretical image rejection ratio, phase noise is a
deterministic, cross-batch-consistent Wiener process, and PA nonlinearity
(Rapp AM-AM + AM-PM) saturates smoothly with the right asymptotic and
monotonicity properties."""

import numpy as np
import pytest

from ccsds_chain.impairments import (
    PhaseNoiseGenerator,
    apply_frequency_offset,
    apply_iq_imbalance,
    apply_pa_nonlinearity,
)

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


class TestPANonlinearity:
    def test_zero_input_passes_through(self):
        out = apply_pa_nonlinearity(np.zeros(10, dtype=complex), backoff_db=0.0, smoothness=3.0)
        assert np.allclose(out, 0.0)

    def test_small_signal_region_is_near_linear(self):
        """Well below saturation, the Rapp model's gain must be close to
        unity (the whole point of a soft-saturation curve: small signals
        pass through almost unaffected)."""
        a_sat = 10 ** (10.0 / 20.0)  # 10 dB backoff
        tiny = (a_sat * 0.01) * np.ones(10, dtype=complex)
        out = apply_pa_nonlinearity(tiny, backoff_db=10.0, smoothness=3.0)
        assert np.abs(out[0]) == pytest.approx(np.abs(tiny[0]), rel=1e-4)

    def test_gain_is_monotonically_decreasing_with_amplitude(self):
        """AM-AM compression: output/input gain must never increase as
        input amplitude grows -- a real amplifier never un-compresses."""
        r = np.linspace(0.01, 10.0, 200)
        out = apply_pa_nonlinearity(r.astype(complex), backoff_db=0.0, smoothness=3.0)
        gain = np.abs(out) / r
        assert np.all(np.diff(gain) <= 1e-12)

    def test_output_amplitude_asymptotically_saturates(self):
        """However large the input, output envelope must approach (never
        exceed) A_sat = 10^(backoff_db/20)."""
        a_sat = 10 ** (3.0 / 20.0)
        huge = np.array([1e6 + 0j])
        out = apply_pa_nonlinearity(huge, backoff_db=3.0, smoothness=3.0)
        assert np.abs(out[0]) == pytest.approx(a_sat, rel=1e-6)
        assert np.abs(out[0]) <= a_sat + 1e-9

    def test_never_amplifies(self):
        r = np.linspace(0.01, 20.0, 500)
        out = apply_pa_nonlinearity(r.astype(complex), backoff_db=-5.0, smoothness=2.0)
        assert np.all(np.abs(out) <= r + 1e-9)

    def test_am_pm_disabled_by_default_leaves_phase_unchanged(self):
        r = np.linspace(0.1, 5.0, 50).astype(complex)  # positive real -> phase 0
        out = apply_pa_nonlinearity(r, backoff_db=0.0, smoothness=3.0, am_pm_deg_per_db=0.0)
        assert np.allclose(np.angle(out), 0.0, atol=1e-9)

    def test_am_pm_rotates_proportionally_to_compression(self):
        r = np.array([3.0 + 0j])  # well into compression for a 0 dB backoff
        out_no_pm = apply_pa_nonlinearity(r, backoff_db=0.0, smoothness=3.0, am_pm_deg_per_db=0.0)
        out_pm = apply_pa_nonlinearity(r, backoff_db=0.0, smoothness=3.0, am_pm_deg_per_db=5.0)
        assert np.angle(out_no_pm)[0] == pytest.approx(0.0, abs=1e-9)
        assert np.angle(out_pm)[0] > 0.05  # visibly rotated
        # doubling the coefficient must roughly double the rotation (same
        # compression amount, linear deg/dB conversion)
        out_pm_2x = apply_pa_nonlinearity(r, backoff_db=0.0, smoothness=3.0, am_pm_deg_per_db=10.0)
        assert np.angle(out_pm_2x)[0] == pytest.approx(2 * np.angle(out_pm)[0], rel=1e-6)

    def test_higher_smoothness_gives_a_sharper_knee(self):
        """A larger Rapp `p` should compress less in the small-signal
        region (sharper, more limiter-like transition) for the same input
        just below saturation."""
        r = np.array([0.8 + 0j])  # just below a_sat=1.0 (0 dB backoff)
        soft = apply_pa_nonlinearity(r, backoff_db=0.0, smoothness=1.0)
        sharp = apply_pa_nonlinearity(r, backoff_db=0.0, smoothness=8.0)
        assert np.abs(sharp[0]) > np.abs(soft[0])
