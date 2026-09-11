"""Bit/byte helpers and raw IQ file I/O shared by the CCSDS signal chain."""

import numpy as np


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


def write_iq_interleaved_float32(path: str, iq: np.ndarray) -> None:
    """Write complex samples as raw interleaved float32: I0,Q0,I1,Q1,..."""
    interleaved = np.empty(2 * len(iq), dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    interleaved.tofile(path)


def read_iq_interleaved_float32(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    return raw[0::2] + 1j * raw[1::2]
