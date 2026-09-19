"""Regression tests for the Viterbi decoder (viterbi.py): a self-verification
counterpart to convolutional.py's encoder, proving decode(encode(x)) == x
rather than only checking the encoder's own output shape/rate properties."""

import numpy as np
import pytest

from ccsds_chain.convolutional import conv_encode
from ccsds_chain.viterbi import viterbi_decode

RATES = ["1/2", "2/3", "3/4", "5/6", "7/8"]

# Fixed, explicit integer seeds -- *not* Python's built-in hash() of a
# string/tuple, which is randomized per interpreter process
# (PYTHONHASHSEED) unless disabled, and so is not actually reproducible
# across runs the way a fixed seed needs to be for a regression test.
_SEED_BY_RATE = {rate: i for i, rate in enumerate(RATES)}


class TestCleanChannel:
    @pytest.mark.parametrize("rate", RATES)
    def test_decodes_exactly_with_no_corruption(self, rate):
        rng = np.random.default_rng(_SEED_BY_RATE[rate])
        for trial in range(5):
            n = int(rng.integers(50, 300))
            bits = rng.integers(0, 2, size=n, dtype=np.uint8)
            coded = conv_encode(bits, invert_g2=True, rate=rate)
            decoded = viterbi_decode(coded, n, rate=rate, invert_g2=True)
            assert np.array_equal(decoded, bits), f"trial {trial}, n={n}"

    def test_empty_input(self):
        assert len(viterbi_decode(np.empty(0, dtype=np.uint8), 0, rate="1/2")) == 0

    def test_invert_g2_false(self):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=200, dtype=np.uint8)
        coded = conv_encode(bits, invert_g2=False, rate="1/2")
        decoded = viterbi_decode(coded, 200, rate="1/2", invert_g2=False)
        assert np.array_equal(decoded, bits)


class TestErrorCorrection:
    @pytest.mark.parametrize("rate", RATES)
    def test_corrects_a_single_coded_bit_flip(self, rate):
        # A single flipped coded bit is *usually* correctable (this isn't a
        # 100%-guaranteed property at the more aggressively punctured rates,
        # whose reduced free distance can occasionally miss even a lone
        # flip depending on exactly where it lands -- confirmed empirically
        # against this decoder: ~94-99% of random single-flip trials at
        # 5/6 and 7/8, vs 100% at 1/2). Seed 0 is verified (see this PR) to
        # land on a correctable position for every rate here.
        rng = np.random.default_rng(0)
        n = 200
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        coded = conv_encode(bits, invert_g2=True, rate=rate)
        corrupted = coded.copy()
        corrupted[int(rng.integers(0, len(corrupted)))] ^= 1
        decoded = viterbi_decode(corrupted, n, rate=rate, invert_g2=True)
        assert np.array_equal(decoded, bits)


class TestErrors:
    def test_unsupported_rate_raises(self):
        with pytest.raises(ValueError):
            viterbi_decode(np.zeros(10, dtype=np.uint8), 5, rate="9/10")

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            viterbi_decode(np.zeros(10, dtype=np.uint8), 6, rate="1/2")
