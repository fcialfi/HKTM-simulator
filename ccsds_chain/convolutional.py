"""CCSDS rate-1/2, K=7 convolutional encoder (polynomials 171/133 octal).

Continuous encoder: the shift register starts in the all-zero state once,
at the beginning of the whole bitstream (not reset per CADU), matching how
a physical convolutional coder runs continuously across the channel.
"""

import numpy as np

K = 7
G1 = 0o171  # 1111001
G2 = 0o133  # 1011011

_PARITY_LUT = np.array([bin(x).count("1") & 1 for x in range(1 << K)], dtype=np.uint8)


def conv_encode(bits: np.ndarray, invert_g2: bool = True) -> np.ndarray:
    """Rate-1/2 convolutional encode. Returns interleaved [g1_0,g2_0,g1_1,g2_1,...].

    `invert_g2=True` matches the CCSDS convention of complementing the G2
    (133 octal) output symbol.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    padded = np.concatenate([np.zeros(K - 1, dtype=np.uint8), bits])
    windows = np.lib.stride_tricks.sliding_window_view(padded, K)  # oldest..newest per row

    weights = (1 << np.arange(K)).astype(np.uint32)  # column c (oldest..newest) -> bit c
    full = (windows.astype(np.uint32) * weights).sum(axis=1)

    g1 = _PARITY_LUT[full & G1]
    g2 = _PARITY_LUT[full & G2]
    if invert_g2:
        g2 = 1 - g2

    out = np.empty(2 * len(bits), dtype=np.uint8)
    out[0::2] = g1
    out[1::2] = g2
    return out
