"""NRZ-L bipolar mapping and Gray-coded QPSK symbol mapping."""

import numpy as np


def bits_to_nrzl(bits: np.ndarray) -> np.ndarray:
    """NRZ-L, direct bipolar (no differential encoding): bit 1 -> +1, bit 0 -> -1."""
    return 2.0 * bits.astype(np.float64) - 1.0


def qpsk_gray_map(bipolar: np.ndarray) -> np.ndarray:
    """Pair up consecutive bipolar samples into I/Q. Each axis independently
    carries one bit (+-1), which is inherently Gray-coded for QPSK (adjacent
    constellation points differ by exactly one bit). Normalized to unit
    average symbol energy (|I|=|Q|=1/sqrt(2))."""
    if len(bipolar) % 2 != 0:
        raise ValueError("bipolar stream length must be even for QPSK pairing")
    i = bipolar[0::2]
    q = bipolar[1::2]
    return (i + 1j * q) / np.sqrt(2.0)
