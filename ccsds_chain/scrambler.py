"""CCSDS pseudo-randomizer (131.0-B-5 section 10).

Two sequences are specified:
- "long": 131071-bit period, h(x) = x^17+x^14+1 (10.4.1) -- the preferred
  default as of Issue 5 (Sept 2023), added to avoid spectral spikes at
  high data rates.
- "short": 255-bit period, h(x) = x^8+x^7+x^5+x^3+1 (10.4.2) -- kept only
  for backward compatibility with legacy systems.

Both are implemented as a single generic Fibonacci LFSR (see `_lfsr_bits`).
The (feedback tap positions, output tap position) pair for each sequence
was not derived from the standard's block diagrams (figures 10-2/10-3),
which leave the register-index-to-polynomial-degree mapping ambiguous on
the page; instead it was found by brute-force search over all tap/output/
shift-direction conventions for a register of the right size, keeping only
the configuration(s) that reproduce the *first 40 bits given in the
standard itself* (10.4.3, note 2) bit-for-bit. Both sequences were then
independently confirmed to return to their initial state after exactly one
full period (255 / 131071 steps), confirming they are the intended
maximal-length sequences.
"""

import numpy as np

_SHORT_SEED = [1] * 8
_SHORT_TAPS = (0, 3, 5, 7)
_SHORT_OUTPUT_POS = 0
_SHORT_PERIOD = 255

_LONG_SEED = [int(c) for c in "11000111000111000"]
_LONG_TAPS = (0, 14)
_LONG_OUTPUT_POS = 2
_LONG_PERIOD = 131071


def _lfsr_bits(n_bits: int, seed: list[int], taps: tuple[int, ...], output_pos: int) -> np.ndarray:
    """Fibonacci LFSR: each step outputs reg[output_pos], then XORs the tap
    positions into a new bit shifted in from the right (register shifts
    left)."""
    reg = list(seed)
    out = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        out[i] = reg[output_pos]
        fb = 0
        for t in taps:
            fb ^= reg[t]
        reg = reg[1:] + [fb]
    return out


def pn_sequence(n_bits: int, mode: str) -> np.ndarray:
    """`n_bits` of the CCSDS pseudo-random sequence, repeating the
    underlying period as needed. `mode` is "long" (131071 bits) or "short"
    (255 bits, legacy)."""
    if mode == "long":
        seed, taps, out_pos, period = _LONG_SEED, _LONG_TAPS, _LONG_OUTPUT_POS, _LONG_PERIOD
    elif mode == "short":
        seed, taps, out_pos, period = _SHORT_SEED, _SHORT_TAPS, _SHORT_OUTPUT_POS, _SHORT_PERIOD
    else:
        raise ValueError(f"unknown randomizer mode {mode!r} (expected 'long' or 'short')")

    if n_bits <= period:
        return _lfsr_bits(n_bits, seed, taps, out_pos)
    one_period = _lfsr_bits(period, seed, taps, out_pos)
    reps = int(np.ceil(n_bits / period))
    return np.tile(one_period, reps)[:n_bits]


def scramble_bits(bits: np.ndarray, mode: str) -> np.ndarray:
    """XOR-scramble a bitstream with the CCSDS pseudo-random sequence,
    restarting the sequence at the start of `bits` (10.4.3: the generator
    is reinitialized at the start of each codeblock, codeword, or Transfer
    Frame). `mode="none"` returns `bits` unchanged."""
    if mode == "none":
        return bits
    pn = pn_sequence(len(bits), mode)
    return np.bitwise_xor(bits.astype(np.uint8), pn)
