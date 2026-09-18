"""CCSDS K=7 convolutional encoder (polynomials 171/133 octal), rate 1/2
with optional puncturing to 2/3, 3/4, 5/6, or 7/8 (CCSDS 131.0-B-5 3.3-3.4).

Continuous encoder: the shift register starts in the all-zero state once,
at the beginning of the whole bitstream (not reset per CADU), matching how
a physical convolutional coder runs continuously across the channel.
"""

import numpy as np

K = 7
G1 = 0o171  # 1111001
G2 = 0o133  # 1011011

_PARITY_LUT = np.array([bin(x).count("1") & 1 for x in range(1 << K)], dtype=np.uint8)

# Table 3-1: puncture code patterns, 1 = transmitted symbol, 0 = punctured.
PUNCTURE_PATTERNS = {
    "1/2": None,
    "2/3": ((1, 0), (1, 1)),
    "3/4": ((1, 0, 1), (1, 1, 0)),
    "5/6": ((1, 0, 1, 0, 1), (1, 1, 0, 1, 0)),
    "7/8": ((1, 0, 0, 0, 1, 0, 1), (1, 1, 1, 1, 0, 1, 0)),
}


_WEIGHTS = (1 << np.arange(K)).astype(np.uint8)  # column c (oldest..newest) -> bit c

# Chunk size for _base_rate_half_encode: each chunk briefly materializes a
# (chunk_size, K) array, so this bounds peak memory independently of the
# total bitstream length (a multi-million-CADU export would otherwise blow
# up the single sliding_window_view() below to tens of GB, see #12).
_CHUNK_BITS = 2_000_000


def _rate_half_encode_from_history(bits: np.ndarray, invert_g2: bool, history: np.ndarray) -> tuple:
    """Returns (g1, g2, new_history): one output bit per input bit, plus the
    K-1 trailing bits to carry as shift-register state into a *later* call
    on the next chunk of the same continuous bitstream.

    Processed in fixed-size sub-chunks internally so peak memory stays
    bounded regardless of len(bits), same as before this was split out for
    reuse by the stateful `ConvEncoder` below.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    g1 = np.empty(n, dtype=np.uint8)
    g2 = np.empty(n, dtype=np.uint8)

    for start in range(0, n, _CHUNK_BITS):
        end = min(start + _CHUNK_BITS, n)
        chunk = bits[start:end]
        padded = np.concatenate([history, chunk])
        windows = np.lib.stride_tricks.sliding_window_view(padded, K)  # oldest..newest per row

        full = (windows * _WEIGHTS).sum(axis=1, dtype=np.uint8)

        g1[start:end] = _PARITY_LUT[full & G1]
        g2[start:end] = _PARITY_LUT[full & G2]
        history = padded[-(K - 1):]

    if invert_g2:
        g2 = 1 - g2
    return g1, g2, history


def _puncture(g1: np.ndarray, g2: np.ndarray, rate: str, phase: int) -> tuple:
    """Interleave and puncture (g1, g2) per Table 3-1, starting at puncture
    pattern position `phase` (so a later call can resume mid-period for a
    continuous punctured stream split across chunks). Returns (out, new_phase)."""
    pattern = PUNCTURE_PATTERNS[rate]
    n = len(g1)
    if pattern is None:
        out = np.empty(2 * n, dtype=np.uint8)
        out[0::2] = g1
        out[1::2] = g2
        return out, 0

    c1_pattern, c2_pattern = pattern
    period = len(c1_pattern)
    idx = (np.arange(n) + phase) % period
    c1_mask = np.array(c1_pattern, dtype=bool)[idx]
    c2_mask = np.array(c2_pattern, dtype=bool)[idx]

    interleaved = np.empty(2 * n, dtype=np.uint8)
    interleaved[0::2] = g1
    interleaved[1::2] = g2
    keep = np.empty(2 * n, dtype=bool)
    keep[0::2] = c1_mask
    keep[1::2] = c2_mask
    return interleaved[keep], (phase + n) % period


def conv_encode(bits: np.ndarray, invert_g2: bool = True, rate: str = "1/2") -> np.ndarray:
    """Convolutional encode at the given code rate.

    `invert_g2=True` matches the CCSDS convention of complementing the G2
    (133 octal) output symbol; this only applies to the base rate-1/2 code
    -- per 3.4.1(5), punctured codes use no symbol inversion, so
    `invert_g2` is ignored (forced off) whenever `rate != "1/2"`.
    """
    if rate not in PUNCTURE_PATTERNS:
        raise ValueError(f"unsupported convolutional code rate {rate!r}")

    g1, g2, _ = _rate_half_encode_from_history(bits, invert_g2 and rate == "1/2", np.zeros(K - 1, dtype=np.uint8))
    out, _ = _puncture(g1, g2, rate, 0)
    return out


class ConvEncoder:
    """Stateful counterpart to `conv_encode()`, for encoding a single
    continuous bitstream a chunk at a time (e.g. batched/streaming export):
    carries the shift-register history and puncture-pattern phase across
    `encode()` calls, so encoding chunks one after another produces output
    bit-for-bit identical to a single `conv_encode()` call on the
    concatenation of those same chunks -- with peak memory bounded by the
    chunk size instead of the whole bitstream.
    """

    def __init__(self, invert_g2: bool = True, rate: str = "1/2"):
        if rate not in PUNCTURE_PATTERNS:
            raise ValueError(f"unsupported convolutional code rate {rate!r}")
        self.rate = rate
        self.invert_g2 = invert_g2 and rate == "1/2"
        self._history = np.zeros(K - 1, dtype=np.uint8)
        self._phase = 0

    def encode(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) == 0:
            return np.empty(0, dtype=np.uint8)
        g1, g2, self._history = _rate_half_encode_from_history(bits, self.invert_g2, self._history)
        out, self._phase = _puncture(g1, g2, self.rate, self._phase)
        return out
