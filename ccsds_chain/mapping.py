"""NRZ-L bipolar mapping and Gray-coded QPSK/BPSK symbol mapping."""

import numpy as np

# Bits carried by one symbol, per modulation -- the single source of truth
# used throughout the chain (rate derivation in the GUI/CLI, symbol-count
# sizing in pipeline.py, and the bit-per-symbol pairing/carry logic in
# export_chain()'s batched mapping).
BITS_PER_SYMBOL = {"QPSK": 2, "BPSK": 1}


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


def bpsk_map(bipolar: np.ndarray) -> np.ndarray:
    """One bipolar sample (+-1) per symbol, carried on I only (Q=0) --
    CCSDS 131.0-B-5's base rate-1/2 convolutional code (and its G2
    inversion, 3.3.1(5)) is specified against this modulation. Already at
    unit symbol energy (|symbol|=1), matching qpsk_gray_map()'s
    normalization -- no separate scaling needed."""
    return bipolar.astype(np.complex128)


def map_symbols(bipolar: np.ndarray, modulation: str) -> np.ndarray:
    """Dispatch to the symbol mapping for `modulation` ("QPSK" or "BPSK")."""
    if modulation == "QPSK":
        return qpsk_gray_map(bipolar)
    if modulation == "BPSK":
        return bpsk_map(bipolar)
    raise NotImplementedError(f"modulation {modulation!r} not implemented (expected 'QPSK' or 'BPSK')")
