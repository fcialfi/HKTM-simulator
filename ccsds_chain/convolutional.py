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


def _base_rate_half_encode(bits: np.ndarray, invert_g2: bool) -> tuple:
    """Returns (g1, g2) arrays, one output bit per input bit."""
    bits = np.asarray(bits, dtype=np.uint8)
    padded = np.concatenate([np.zeros(K - 1, dtype=np.uint8), bits])
    windows = np.lib.stride_tricks.sliding_window_view(padded, K)  # oldest..newest per row

    weights = (1 << np.arange(K)).astype(np.uint32)  # column c (oldest..newest) -> bit c
    full = (windows.astype(np.uint32) * weights).sum(axis=1)

    g1 = _PARITY_LUT[full & G1]
    g2 = _PARITY_LUT[full & G2]
    if invert_g2:
        g2 = 1 - g2
    return g1, g2


def conv_encode(bits: np.ndarray, invert_g2: bool = True, rate: str = "1/2") -> np.ndarray:
    """Convolutional encode at the given code rate.

    `invert_g2=True` matches the CCSDS convention of complementing the G2
    (133 octal) output symbol; this only applies to the base rate-1/2 code
    -- per 3.4.1(5), punctured codes use no symbol inversion, so
    `invert_g2` is ignored (forced off) whenever `rate != "1/2"`.
    """
    if rate not in PUNCTURE_PATTERNS:
        raise ValueError(f"unsupported convolutional code rate {rate!r}")

    g1, g2 = _base_rate_half_encode(bits, invert_g2 and rate == "1/2")

    pattern = PUNCTURE_PATTERNS[rate]
    if pattern is None:
        out = np.empty(2 * len(bits), dtype=np.uint8)
        out[0::2] = g1
        out[1::2] = g2
        return out

    c1_pattern, c2_pattern = pattern
    period = len(c1_pattern)
    n = len(bits)
    idx = np.arange(n) % period
    c1_mask = np.array(c1_pattern, dtype=bool)[idx]
    c2_mask = np.array(c2_pattern, dtype=bool)[idx]

    interleaved = np.empty(2 * n, dtype=np.uint8)
    interleaved[0::2] = g1
    interleaved[1::2] = g2
    keep = np.empty(2 * n, dtype=bool)
    keep[0::2] = c1_mask
    keep[1::2] = c2_mask
    return interleaved[keep]
