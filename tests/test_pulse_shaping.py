"""Regression tests for RRC tap generation and pulse shaping."""

import numpy as np
import pytest

from ccsds_chain.pulse_shaping import RRCPulseShaper, pulse_shape, rrc_taps


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
