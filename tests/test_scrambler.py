"""Regression tests for the CCSDS pseudo-randomizer (scrambler.py)."""

import numpy as np
import pytest

from ccsds_chain.scrambler import _LONG_PERIOD, _SHORT_PERIOD, pn_sequence, scramble_bits


@pytest.mark.parametrize("mode,period", [("short", _SHORT_PERIOD), ("long", _LONG_PERIOD)])
def test_sequence_returns_to_initial_state_after_one_period(mode, period):
    """scrambler.py's module docstring: both sequences were confirmed to
    return to their initial state after exactly one full period -- i.e. the
    sequence generated is periodic with that period, not just "long enough
    to look random"."""
    seq = pn_sequence(2 * period, mode)
    assert np.array_equal(seq[:period], seq[period:])


@pytest.mark.parametrize("mode", ["short", "long"])
def test_tiling_beyond_one_period_matches_repeated_single_period(mode):
    period = _SHORT_PERIOD if mode == "short" else _LONG_PERIOD
    one_period = pn_sequence(period, mode)
    tiled = pn_sequence(period * 3 + 17, mode)
    assert np.array_equal(tiled, np.tile(one_period, 4)[: period * 3 + 17])


@pytest.mark.parametrize("mode", ["short", "long"])
def test_deterministic(mode):
    assert np.array_equal(pn_sequence(500, mode), pn_sequence(500, mode))


def test_sequences_are_bits_only():
    seq = pn_sequence(1000, "long")
    assert set(np.unique(seq)).issubset({0, 1})


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        pn_sequence(10, "medium")


class TestScrambleBits:
    def test_none_mode_is_passthrough(self):
        bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        assert np.array_equal(scramble_bits(bits, "none"), bits)

    @pytest.mark.parametrize("mode", ["short", "long"])
    def test_xor_scramble_is_self_inverse(self, mode):
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, size=777, dtype=np.uint8)
        scrambled = scramble_bits(bits, mode)
        assert not np.array_equal(scrambled, bits)  # sanity: it actually changed something
        restored = scramble_bits(scrambled, mode)
        assert np.array_equal(restored, bits)

    @pytest.mark.parametrize("mode", ["short", "long"])
    def test_restarts_at_start_of_every_call(self, mode):
        # 10.4.3: the generator is reinitialized at the start of each
        # codeblock -- so scrambling two independent, equal-length bitstreams
        # must apply the exact same PN mask to both.
        rng = np.random.default_rng(8)
        a = rng.integers(0, 2, size=300, dtype=np.uint8)
        b = rng.integers(0, 2, size=300, dtype=np.uint8)
        mask_from_a = np.bitwise_xor(a, scramble_bits(a, mode))
        mask_from_b = np.bitwise_xor(b, scramble_bits(b, mode))
        assert np.array_equal(mask_from_a, mask_from_b)
