"""Regression tests for NRZ-L bipolar mapping and QPSK/BPSK symbol mapping."""

import numpy as np
import pytest

from ccsds_chain.mapping import BITS_PER_SYMBOL, bits_to_nrzl, bpsk_map, map_symbols, qpsk_gray_map


def test_bits_per_symbol_table():
    assert BITS_PER_SYMBOL == {"QPSK": 2, "BPSK": 1}


def test_bits_to_nrzl():
    bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    assert np.array_equal(bits_to_nrzl(bits), np.array([-1.0, 1.0, -1.0, 1.0, 1.0]))


class TestQPSK:
    def test_odd_length_rejected(self):
        with pytest.raises(ValueError):
            qpsk_gray_map(np.array([1.0, -1.0, 1.0]))

    def test_pairing_and_unit_energy(self):
        bipolar = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        symbols = qpsk_gray_map(bipolar)
        expected = np.array([1 - 1j, -1 + 1j]) / np.sqrt(2.0)
        assert np.allclose(symbols, expected)
        assert np.allclose(np.abs(symbols), 1.0)  # unit average symbol energy

    def test_all_four_constellation_points_reachable(self):
        bipolar = np.array([1, 1, 1, -1, -1, 1, -1, -1], dtype=np.float64)
        symbols = qpsk_gray_map(bipolar)
        scale = 1 / np.sqrt(2.0)
        expected_points = {(1 + 1j) * scale, (1 - 1j) * scale, (-1 + 1j) * scale, (-1 - 1j) * scale}
        for s in symbols:
            assert any(np.isclose(s, p) for p in expected_points)


class TestBPSK:
    def test_maps_bipolar_directly_with_zero_q(self):
        bipolar = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        symbols = bpsk_map(bipolar)
        assert np.array_equal(symbols, np.array([1 + 0j, -1 + 0j, 1 + 0j]))
        assert np.allclose(np.abs(symbols), 1.0)


class TestDispatch:
    def test_qpsk(self):
        bipolar = np.array([1.0, -1.0], dtype=np.float64)
        assert np.array_equal(map_symbols(bipolar, "QPSK"), qpsk_gray_map(bipolar))

    def test_bpsk(self):
        bipolar = np.array([1.0, -1.0], dtype=np.float64)
        assert np.array_equal(map_symbols(bipolar, "BPSK"), bpsk_map(bipolar))

    def test_unknown_modulation_raises(self):
        with pytest.raises(NotImplementedError):
            map_symbols(np.array([1.0]), "16QAM")
