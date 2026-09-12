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


def iq_to_int16_interleaved(iq: np.ndarray, full_scale: int = 2047,
                             headroom_db: float = 1.0) -> np.ndarray:
    """Quantize complex samples to the RF-Catcher (TestTree) raw IQ format:
    little-endian int16 per component with 12 significant bits in two's
    complement (values in [-2048, 2047]), interleaved I0,Q0,I1,Q1,....

    Samples are normalized to their peak magnitude and scaled to
    `full_scale` with `headroom_db` dB of headroom before rounding, to use
    the available 12-bit dynamic range without clipping. Returns a `<i2`
    array ready to be written or serialized to bytes.
    """
    peak = np.max(np.abs(iq))
    if peak == 0:
        raise ValueError("all-zero IQ signal, cannot normalize for int16 export")
    scale = (full_scale / peak) * 10 ** (-headroom_db / 20)

    i = np.clip(np.round(iq.real * scale), -2048, 2047)
    q = np.clip(np.round(iq.imag * scale), -2048, 2047)

    interleaved = np.empty(2 * len(iq), dtype="<i2")
    interleaved[0::2] = i
    interleaved[1::2] = q
    return interleaved


def write_iq_interleaved_int16(path: str, iq: np.ndarray, full_scale: int = 2047,
                                headroom_db: float = 1.0) -> None:
    """Write complex samples to `path` in the RF-Catcher raw IQ format
    (no header, uncompressed, unencrypted) -- see `iq_to_int16_interleaved`."""
    iq_to_int16_interleaved(iq, full_scale, headroom_db).tofile(path)


def read_iq_interleaved_int16(path: str) -> np.ndarray:
    """Read a raw IQ file in the RF-Catcher format written by
    `write_iq_interleaved_int16` back into a complex array."""
    raw = np.fromfile(path, dtype="<i2")
    return raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
