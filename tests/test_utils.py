"""Regression tests for bit/byte helpers and raw IQ file I/O (utils.py)."""

import numpy as np
import pytest

from ccsds_chain.utils import (
    INT16_FULL_SCALE,
    bytes_to_bits,
    detect_cadu_length,
    find_all_cadu_positions,
    find_cadu_sync,
    generate_payload,
    normalize_peak,
    pack_iq_interleaved,
    resample_ratio,
    unpack_iq_interleaved,
)


def test_bytes_to_bits_is_msb_first():
    # CCSDS transmits MSB of each octet first.
    assert np.array_equal(bytes_to_bits(bytes([0b10000001])), np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8))


def test_bytes_to_bits_round_trips_with_packbits():
    data = bytes(range(256))
    bits = bytes_to_bits(data)
    assert len(bits) == len(data) * 8
    assert np.packbits(bits).tobytes() == data


class TestGeneratePayload:
    def test_pseudo_random_is_reproducible_with_same_seed(self):
        a = generate_payload(3, 100, seed=42)
        b = generate_payload(3, 100, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_payload(3, 100, seed=1)
        b = generate_payload(3, 100, seed=2)
        assert a != b

    def test_size_is_n_cadu_times_frame_bytes(self):
        assert len(generate_payload(5, 37, seed=0)) == 5 * 37

    def test_source_bytes_used_verbatim_and_padded(self):
        source = bytes(range(10))
        out = generate_payload(1, 20, source_bytes=source)
        assert out[:10] == source
        assert out[10:] == bytes(10)

    def test_source_bytes_truncated_when_longer_than_needed(self):
        source = bytes(range(50))
        out = generate_payload(1, 10, source_bytes=source)
        assert out == source[:10]

    def test_source_path_used_and_padded(self, tmp_path):
        f = tmp_path / "payload.bin"
        f.write_bytes(bytes(range(5)))
        out = generate_payload(1, 10, source_path=str(f))
        assert out == bytes(range(5)) + bytes(5)


class TestCaduSync:
    def test_find_cadu_sync_locates_first_asm(self):
        asm = bytes.fromhex("1ACFFC1D")
        data = b"\x00\x00\x00" + asm + b"\xAA" * 20
        assert find_cadu_sync(data, asm) == 3

    def test_find_cadu_sync_raises_when_absent(self):
        asm = bytes.fromhex("1ACFFC1D")
        with pytest.raises(ValueError):
            find_cadu_sync(b"\x00" * 50, asm)

    def test_find_all_cadu_positions(self):
        asm = bytes.fromhex("1ACFFC1D")
        cadu = asm + b"\x00" * 20
        data = b"junk" + cadu + b"wrapper" + cadu
        positions = find_all_cadu_positions(data, asm)
        assert len(positions) == 2
        assert data[positions[0]:positions[0] + len(asm)] == asm
        assert data[positions[1]:positions[1] + len(asm)] == asm

    def test_find_all_cadu_positions_empty_when_absent(self):
        asm = bytes.fromhex("1ACFFC1D")
        assert find_all_cadu_positions(b"\x00" * 30, asm) == []

    def test_detect_cadu_length_measures_gap_between_asms(self):
        asm = bytes.fromhex("1ACFFC1D")
        data = asm + b"\x00" * 16 + asm + b"\x00" * 16
        length = detect_cadu_length(data, asm, first_asm_offset=0)
        assert length == len(asm) + 16

    def test_detect_cadu_length_none_when_no_second_occurrence(self):
        asm = bytes.fromhex("1ACFFC1D")
        data = asm + b"\x00" * 16
        assert detect_cadu_length(data, asm, first_asm_offset=0) is None


class TestNormalizePeak:
    def test_scales_to_requested_peak(self):
        iq = np.array([1 + 2j, -3 + 1j, 0.5 - 0.5j])
        out = normalize_peak(iq, peak=0.9)
        current_peak = max(np.abs(out.real).max(), np.abs(out.imag).max())
        assert np.isclose(current_peak, 0.9)

    def test_all_zero_signal_is_unaffected(self):
        iq = np.zeros(10, dtype=complex)
        assert np.array_equal(normalize_peak(iq, peak=0.9), iq)


class TestResampleRatio:
    def test_reduces_to_lowest_terms(self):
        # 10 MHz target from a 8 MHz source -> 5/4 after dividing by gcd=2e6.
        up, down = resample_ratio(8_000_000, 10_000_000)
        assert (up, down) == (5, 4)

    def test_equal_rates_give_1_1(self):
        assert resample_ratio(1_785_000, 1_785_000) == (1, 1)

    def test_rounds_near_integer_rates(self):
        # Real sample rates are effectively integers; a tiny float fuzz
        # shouldn't change the reduced ratio.
        up, down = resample_ratio(8_000_000.0000001, 10_000_000.0)
        assert (up, down) == (5, 4)


class TestIQPacking:
    @pytest.mark.parametrize("dtype", ["float32", "int16"])
    def test_round_trip(self, dtype):
        iq = np.array([0.1 + 0.2j, -0.5 - 0.5j, 0.9 - 0.1j, 0.0 + 0.0j])
        packed = pack_iq_interleaved(iq, dtype)
        unpacked = unpack_iq_interleaved(packed, dtype)
        tol = 2.0 / INT16_FULL_SCALE if dtype == "int16" else 1e-6
        assert np.allclose(unpacked, iq, atol=tol)

    def test_int16_clips_out_of_range_values(self):
        iq = np.array([2.0 + 2.0j, -2.0 - 2.0j])
        packed = pack_iq_interleaved(iq, "int16")
        raw = np.frombuffer(packed, dtype="<i2")
        assert raw[0] == 2047 and raw[1] == 2047
        assert raw[2] == -2048 and raw[3] == -2048

    def test_pack_rejects_unsupported_dtype(self):
        with pytest.raises(ValueError):
            pack_iq_interleaved(np.array([1 + 1j]), "float64")

    def test_unpack_rejects_unsupported_dtype(self):
        with pytest.raises(ValueError):
            unpack_iq_interleaved(b"\x00\x00\x00\x00", "float64")

    def test_interleaving_order_is_i_then_q(self):
        iq = np.array([1.0 + 2.0j])
        packed = pack_iq_interleaved(iq, "float32")
        values = np.frombuffer(packed, dtype=np.float32)
        assert list(values) == [1.0, 2.0]
