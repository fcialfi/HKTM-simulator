"""Bit/byte helpers and raw IQ file I/O shared by the CCSDS signal chain."""

from math import gcd
from typing import Optional

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


def find_cadu_sync(data: bytes, asm: bytes) -> int:
    """Return the byte offset of the first occurrence of `asm` in `data` --
    i.e. the first genuine CADU boundary in a real/captured CADU stream.

    Unlike a stream this tool built itself (where CADU 0 always starts at
    byte 0), a real captured or exported CADU file is not guaranteed to
    begin exactly on a CADU boundary: it may be preceded by unframed idle
    line-fill, or simply be an excerpt starting mid-stream. Byte-aligned
    search for the (byte-aligned, never-coded) ASM is how a real frame
    synchronizer locates the first frame too.

    Raises ValueError if `asm` does not appear anywhere in `data`.
    """
    offset = data.find(asm)
    if offset < 0:
        raise ValueError(
            f"ASM pattern {asm.hex()} not found anywhere in the input data: "
            "cannot locate a CADU boundary to synchronize to."
        )
    return offset


def detect_cadu_length(data: bytes, asm: bytes, first_asm_offset: int) -> Optional[int]:
    """Measure the real CADU length (ASM + coded data zone) directly from
    the data, as the byte distance from `first_asm_offset` to the *next*
    occurrence of `asm` -- rather than trusting a length computed from the
    tool's configured RS/interleave settings.

    This matters because a real/captured CADU stream is not guaranteed to
    use this tool's exact RS(255,*) interleaved framing: extra fields, a
    different interleave depth, CCSDS "virtual fill" (section 11), or a
    project-specific envelope can all change the true CADU length in ways
    the configured settings alone can't predict. The ASM's own design
    (CCSDS 131.0-B-5 4.7-4.8) makes a false-positive match inside coded,
    effectively-random data astronomically unlikely, so the distance
    between two consecutive real matches is a reliable measurement of the
    actual frame length -- independent of whatever E/interleave-depth is
    selected in the UI.

    Returns None if no second occurrence is found (e.g. the source holds
    only one CADU), in which case the caller should fall back to the
    length computed from its configured settings.
    """
    next_offset = data.find(asm, first_asm_offset + len(asm))
    if next_offset < 0:
        return None
    return next_offset - first_asm_offset


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
        # In-place round/clip (instead of chaining np.round(np.clip(...))) keeps
        # only one extra float64 buffer alive at a time -- for very large exports
        # (hundreds of millions of samples) the naive chained version briefly
        # allocates three such buffers per I/Q leg and can exhaust RAM.
        scaled = iq.real * INT16_FULL_SCALE
        np.round(scaled, out=scaled)
        np.clip(scaled, -2048, 2047, out=scaled)
        interleaved[0::2] = scaled
        scaled = iq.imag * INT16_FULL_SCALE
        np.round(scaled, out=scaled)
        np.clip(scaled, -2048, 2047, out=scaled)
        interleaved[1::2] = scaled
        del scaled
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
