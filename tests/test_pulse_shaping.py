"""Regression tests for RRC tap generation and pulse shaping."""

import numpy as np
import pytest

from ccsds_chain.pulse_shaping import RRCPulseShaper, matched_filter_sample, pulse_shape, rrc_taps


class TestRRCTaps:
    @pytest.mark.parametrize("alpha", [0.0, 0.2, 0.35, 0.5, 1.0])
    def test_symmetric_impulse_response(self, alpha):
        h = rrc_taps(alpha, span=8, sps=4)
        assert np.allclose(h, h[::-1], atol=1e-10)

    @pytest.mark.parametrize("alpha", [0.0, 0.35, 1.0])
    def test_unit_energy(self, alpha):
        h = rrc_taps(alpha, span=8, sps=4)
        assert np.isclose(np.sum(h**2), 1.0)

    def test_tap_count(self):
        span, sps = 8, 4
        h = rrc_taps(0.35, span, sps)
        assert len(h) == span * sps + 1

    def test_no_nans_at_the_singular_points(self):
        # ti == 0 and |4*alpha*ti| == 1 both hit an analytically-removable
        # 0/0 in the closed-form RRC formula and need their own branch;
        # this exercises both for a span/sps combination where they land
        # exactly on a sample.
        h = rrc_taps(0.5, span=8, sps=4)
        assert np.all(np.isfinite(h))


class TestPulseShapeStreamingEquivalence:
    """RRCPulseShaper's own contract: shape() over a sequence of chunks plus
    one final flush() must equal pulse_shape() on the concatenation."""

    @pytest.mark.parametrize("chunk_sizes", [
        [10],
        [1, 1, 1, 1],
        [3, 0, 7, 50, 1],
        [500, 500],
    ])
    def test_chunked_matches_single_call(self, chunk_sizes):
        sps = 4
        taps = rrc_taps(0.35, span=8, sps=sps)
        rng = np.random.default_rng(42)
        chunks = [
            (rng.integers(0, 2, size=n) * 2 - 1).astype(np.complex128) / np.sqrt(2.0)
            for n in chunk_sizes
        ]
        whole = np.concatenate(chunks)

        expected = pulse_shape(whole, sps, taps)

        shaper = RRCPulseShaper(sps, taps)
        parts = [shaper.shape(c) for c in chunks]
        parts.append(shaper.flush())
        actual = np.concatenate(parts)

        assert np.allclose(actual, expected)

    def test_output_length(self):
        sps = 4
        taps = rrc_taps(0.35, span=8, sps=sps)
        symbols = np.ones(20, dtype=np.complex128)
        out = pulse_shape(symbols, sps, taps)
        assert len(out) == len(symbols) * sps + len(taps) - 1

    def test_progress_callback_reaches_one(self):
        # Larger than pulse_shaping._CHUNK_SYMBOLS so the callback fires more
        # than once, exercising the chunked progress-reporting loop itself.
        sps = 4
        taps = rrc_taps(0.35, span=8, sps=sps)
        symbols = np.ones(450_000, dtype=np.complex128)
        seen = []
        pulse_shape(symbols, sps, taps, progress_callback=seen.append)
        assert len(seen) > 1
        assert seen[-1] == pytest.approx(1.0)


class TestMatchedFilterSample:
    """A single transmit-side RRC pass (pulse_shape()) does not by itself
    give a clean, ISI-free constellation at symbol-spaced samples -- only
    the cascade of transmit + matched receive RRC filters forms the full
    Nyquist raised-cosine pulse with that property. matched_filter_sample()
    applies that second (matched) pass so the GUI's constellation plot
    reflects the actual (possibly impaired) IQ, not just the ideal
    pre-pulse-shaping symbols."""

    def _random_qpsk_symbols(self, n, seed=1):
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=(n, 2)) * 2 - 1
        return (bits[:, 0] + 1j * bits[:, 1]).astype(np.complex128) / np.sqrt(2.0)

    def test_recovers_original_symbols_closely_with_no_impairment(self):
        sps, span, alpha = 4, 8, 0.35
        taps = rrc_taps(alpha, span, sps)
        symbols = self._random_qpsk_symbols(200)
        iq = pulse_shape(symbols, sps, taps)

        recovered = matched_filter_sample(iq, len(symbols), sps, taps)
        # Away from the filter's own start/end transient (a handful of
        # symbols at each edge), recovered points must land close to the
        # original symbols -- this is the textbook zero-ISI property of a
        # matched-filter cascade, not an approximation.
        assert np.allclose(recovered[20:-20], symbols[20:-20], atol=0.05)

    def test_single_rrc_pass_alone_does_not_recover_symbols(self):
        """The bug this function exists to fix: sampling the transmit-only
        `iq` directly (no matched filter) at the same symbol-spaced
        instants leaves a large residual even with zero impairments --
        confirming the matched filter is doing real work, not a no-op."""
        sps, span, alpha = 4, 8, 0.35
        taps = rrc_taps(alpha, span, sps)
        symbols = self._random_qpsk_symbols(200)
        iq = pulse_shape(symbols, sps, taps)

        group_delay = (len(taps) - 1) // 2  # single-pass group delay
        idx = group_delay + np.arange(len(symbols)) * sps
        idx = idx[idx < len(iq)]
        unmatched = iq[idx]

        assert np.max(np.abs(unmatched[20:-20] - symbols[20:len(unmatched) - 20])) > 0.3

    def test_output_length_drops_symbols_beyond_the_covered_signal(self):
        sps, span, alpha = 4, 8, 0.35
        taps = rrc_taps(alpha, span, sps)
        symbols = self._random_qpsk_symbols(50)
        iq = pulse_shape(symbols, sps, taps)

        # Asking for far more symbols than iq actually covers must not
        # over-run the array -- just return however many decision points
        # actually fit.
        recovered = matched_filter_sample(iq, 10_000, sps, taps)
        assert len(recovered) < 10_000
        assert len(recovered) > 0

    def test_reflects_a_frequency_offset_as_phase_rotation(self):
        sps, span, alpha = 4, 8, 0.35
        symbol_rate = 1_000_000.0
        sample_rate = symbol_rate * sps
        taps = rrc_taps(alpha, span, sps)
        symbols = np.ones(500, dtype=np.complex128) / np.sqrt(2.0)  # constant symbol, easiest to check rotation on
        iq = pulse_shape(symbols, sps, taps)

        offset_hz = 20_000.0
        n = np.arange(len(iq))
        rotated_iq = iq * np.exp(1j * 2 * np.pi * offset_hz * n / sample_rate)

        plain = matched_filter_sample(iq, len(symbols), sps, taps)
        rotated = matched_filter_sample(rotated_iq, len(symbols), sps, taps)

        # A constant symbol stream's decision points should stay put
        # without an offset (allowing for edge transient) ...
        assert np.std(np.angle(plain[20:-20])) < 0.05
        # ... and visibly spin once one is applied.
        assert np.std(np.angle(rotated[20:-20])) > 0.5
