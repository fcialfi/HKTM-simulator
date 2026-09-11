"""CCSDS pseudo-randomizer: h(x) = 1 + x^3 + x^5 + x^7 + x^8, seed 0xFF.

Implemented as a Fibonacci LFSR (period 255 for this primitive octic
polynomial). NOTE: the exact bit-order/output-tap convention has not been
cross-checked against the CCSDS 131.0-B-2 reference sequence table -- see
README TODO before relying on this for a real receiver test.
"""

import numpy as np

SEED = 0xFF
_TAP_DEGREES = (8, 7, 5, 3)


def _one_period(seed: int = SEED) -> np.ndarray:
    taps_mask = 0
    for t in _TAP_DEGREES:
        taps_mask |= 1 << (t - 1)

    reg = seed & 0xFF
    out = np.empty(255, dtype=np.uint8)
    for i in range(255):
        out[i] = (reg >> 7) & 1
        fb = bin(reg & taps_mask).count("1") & 1
        reg = ((reg << 1) | fb) & 0xFF
    if reg != (seed & 0xFF):
        raise RuntimeError("scrambler LFSR did not return to seed after 255 steps "
                            "(polynomial/tap convention is not maximal-length as configured)")
    return out


def ccsds_pn_sequence(n_bits: int, seed: int = SEED) -> np.ndarray:
    period = _one_period(seed)
    reps = int(np.ceil(n_bits / len(period)))
    return np.tile(period, reps)[:n_bits]


def apply_scrambler(bipolar: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Multiplicative scrambling in the bipolar (NRZ-L) domain: equivalent
    to XOR-scrambling the bits before NRZ-L mapping."""
    pn = ccsds_pn_sequence(len(bipolar), seed)
    pn_bipolar = 1 - 2 * pn.astype(np.float64)
    return bipolar * pn_bipolar
