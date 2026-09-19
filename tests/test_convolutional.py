"""Regression tests for the CCSDS K=7 convolutional encoder and puncturing."""

from fractions import Fraction

import numpy as np
import pytest

from ccsds_chain.convolutional import (
    PUNCTURE_PATTERNS,
    ConvEncoder,
    conv_encode,
)


class TestBaseRateHalf:
    def test_all_zero_input_gives_all_zero_g1(self):
        bits = np.zeros(50, dtype=np.uint8)
        out = conv_encode(bits, invert_g2=False, rate="1/2")
        assert np.all(out[0::2] == 0)  # g1
        assert np.all(out[1::2] == 0)  # g2, uninverted

    def test_invert_g2_flips_only_g2(self):
        bits = np.zeros(50, dtype=np.uint8)
        uninverted = conv_encode(bits, invert_g2=False, rate="1/2")
        inverted = conv_encode(bits, invert_g2=True, rate="1/2")
        assert np.array_equal(inverted[0::2], uninverted[0::2])
        assert np.array_equal(inverted[1::2], 1 - uninverted[1::2])

    def test_output_length_is_double_input_at_rate_half(self):
        bits = np.random.default_rng(0).integers(0, 2, size=123, dtype=np.uint8)
        out = conv_encode(bits, rate="1/2")
        assert len(out) == 2 * len(bits)

    def test_deterministic(self):
        bits = np.random.default_rng(1).integers(0, 2, size=200, dtype=np.uint8)
        assert np.array_equal(conv_encode(bits, rate="1/2"), conv_encode(bits, rate="1/2"))


class TestPuncturing:
    @pytest.mark.parametrize("rate", ["2/3", "3/4", "5/6", "7/8"])
    def test_pattern_labels_match_the_actual_kept_fraction(self, rate):
        """The rate string itself (e.g. "2/3") must equal the code rate the
        puncture pattern actually implements: period input bits in, and
        sum(c1)+sum(c2) output bits kept, per period."""
        c1, c2 = PUNCTURE_PATTERNS[rate]
        period = len(c1)
        assert len(c2) == period
        kept = sum(c1) + sum(c2)
        p, q = (int(x) for x in rate.split("/"))
        assert Fraction(p, q) == Fraction(period, kept)

    @pytest.mark.parametrize("rate", ["2/3", "3/4", "5/6", "7/8"])
    def test_output_length_matches_pattern_over_whole_periods(self, rate):
        c1, _ = PUNCTURE_PATTERNS[rate]
        period = len(c1)
        n_periods = 10
        bits = np.random.default_rng(2).integers(0, 2, size=period * n_periods, dtype=np.uint8)
        out = conv_encode(bits, rate=rate)
        c1_pat, c2_pat = PUNCTURE_PATTERNS[rate]
        kept_per_period = sum(c1_pat) + sum(c2_pat)
        assert len(out) == kept_per_period * n_periods

    @pytest.mark.parametrize("rate", ["2/3", "3/4", "5/6", "7/8"])
    def test_invert_g2_is_ignored_for_punctured_rates(self, rate):
        bits = np.random.default_rng(3).integers(0, 2, size=64, dtype=np.uint8)
        a = conv_encode(bits, invert_g2=True, rate=rate)
        b = conv_encode(bits, invert_g2=False, rate=rate)
        assert np.array_equal(a, b)

    def test_unsupported_rate_raises(self):
        with pytest.raises(ValueError):
            conv_encode(np.zeros(10, dtype=np.uint8), rate="9/10")


class TestConvEncoderStatefulEquivalence:
    """ConvEncoder's own contract (its class docstring): encoding chunks one
    after another must be bit-for-bit identical to a single conv_encode()
    call on the concatenation of those chunks."""

    @pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4", "5/6", "7/8"])
    @pytest.mark.parametrize("chunk_sizes", [
        [17],
        [1, 1, 1, 1, 1],
        [5, 0, 3, 40, 1, 7],
        [200, 200, 200],
        [1000],
    ])
    def test_chunked_matches_single_call(self, rate, chunk_sizes):
        rng = np.random.default_rng(hash((rate, tuple(chunk_sizes))) % (2**32))
        chunks = [rng.integers(0, 2, size=n, dtype=np.uint8) for n in chunk_sizes]
        whole = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)

        expected = conv_encode(whole, invert_g2=True, rate=rate)

        encoder = ConvEncoder(invert_g2=True, rate=rate)
        actual = np.concatenate([encoder.encode(c) for c in chunks]) if chunks else np.empty(0, dtype=np.uint8)

        assert np.array_equal(actual, expected)

    def test_rejects_unsupported_rate(self):
        with pytest.raises(ValueError):
            ConvEncoder(rate="9/10")

    def test_empty_chunk_returns_empty(self):
        encoder = ConvEncoder()
        assert len(encoder.encode(np.empty(0, dtype=np.uint8))) == 0
