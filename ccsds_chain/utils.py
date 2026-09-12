"""Bit/byte helpers and raw IQ file I/O shared by the CCSDS signal chain."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def bytes_to_bits(data: bytes) -> np.ndarray:
    """MSB-first bit unpacking (CCSDS transmits the most significant bit of
    each octet first)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="big")


def generate_payload(n_cadu: int, frame_bytes: int, source_path: str | None = None,
                      source_bytes: bytes | None = None, seed: int | None = 42) -> bytes:
    """Build the concatenated data-zone payload for `n_cadu` CADUs.

    With neither `source_path` nor `source_bytes`, generates reproducible
    pseudo-random test data. With a source (file path, or raw bytes already
    read e.g. from a GUI upload), real Transfer Frame bytes are used
    sequentially; if shorter than needed it is zero-padded (not looped, to
    avoid silently repeating frames).
    """
    total_bytes = n_cadu * frame_bytes

    if source_bytes is not None:
        data = source_bytes[:total_bytes]
        if len(data) < total_bytes:
            data = data + bytes(total_bytes - len(data))
        return data

    if source_path is None:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=total_bytes, dtype=np.uint8).tobytes()

    with open(source_path, "rb") as f:
        data = f.read(total_bytes)
    if len(data) < total_bytes:
        data = data + bytes(total_bytes - len(data))
    return data


def normalize_peak(iq: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale complex samples so the largest |I| or |Q| excursion equals
    `peak` (default 0.9, leaving headroom against clipping on playback)."""
    current_peak = max(np.abs(iq.real).max(), np.abs(iq.imag).max())
    if current_peak == 0:
        return iq
    return iq * (peak / current_peak)


def resample_ratio(source_fs: float, target_fs: float) -> tuple[int, int]:
    """Smallest exact integer (up, down) ratio between two sample rates,
    rounded to the nearest Hz first (real sample rates are always
    effectively integers)."""
    source_hz, target_hz = round(source_fs), round(target_fs)
    step = gcd(source_hz, target_hz)
    return target_hz // step, source_hz // step


def resample_iq(iq: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
    """Resample complex samples to an exact target sample rate, via
    polyphase resampling (`scipy.signal.resample_poly`) at the smallest
    exact integer up/down ratio between the two."""
    up, down = resample_ratio(source_fs, target_fs)
    return resample_poly(iq, up, down)


INT16_FULL_SCALE = 2047  # 12 significant bits (RF-Catcher/TestTree format), LSB-aligned in the 16-bit word


def pack_iq_interleaved(iq: np.ndarray, dtype: str = "float32") -> bytes:
    """Pack complex samples as raw interleaved bytes, no header: I0, Q0,
    I1, Q1, .... `dtype` is "float32" (range [-1, +1]) or "int16" (RF-Catcher
    format: little-endian, 12 significant bits in two's complement, LSB-
    aligned, range [-2048, 2047]; samples should already be normalized to at
    most unit magnitude, e.g. via `normalize_peak`)."""
    n = len(iq)
    if dtype == "float32":
        interleaved = np.empty(2 * n, dtype=np.float32)
        interleaved[0::2] = iq.real.astype(np.float32)
        interleaved[1::2] = iq.imag.astype(np.float32)
    elif dtype == "int16":
        interleaved = np.empty(2 * n, dtype="<i2")
        interleaved[0::2] = np.clip(np.round(iq.real * INT16_FULL_SCALE), -2048, 2047)
        interleaved[1::2] = np.clip(np.round(iq.imag * INT16_FULL_SCALE), -2048, 2047)
    else:
        raise ValueError(f"unsupported IQ output dtype {dtype!r} (expected 'float32' or 'int16')")
    return interleaved.tobytes()


def unpack_iq_interleaved(data: bytes, dtype: str = "float32") -> np.ndarray:
    """Inverse of `pack_iq_interleaved`."""
    if dtype == "float32":
        raw = np.frombuffer(data, dtype=np.float32)
        return raw[0::2] + 1j * raw[1::2]
    if dtype == "int16":
        raw = np.frombuffer(data, dtype="<i2")
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex128) / INT16_FULL_SCALE
    raise ValueError(f"unsupported IQ input dtype {dtype!r} (expected 'float32' or 'int16')")
