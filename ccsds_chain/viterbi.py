"""Hard-decision Viterbi decoder for the CCSDS K=7 convolutional code
(convolutional.py), for self-verification loopback: decode what conv_encode()
produced and confirm the original message bits come back out.

The trellis's per-state transition table (which (g1,g2) pair a given state
outputs for a given input bit, and the resulting next state) is built by
*calling* convolutional._rate_half_encode_from_history() for every
(state, input) combination, rather than re-deriving the G1/G171 and
G2/133-octal tap positions independently here. This guarantees the decoder's
trellis is structurally consistent with whatever the encoder actually does
-- including if that encoder were ever revised -- instead of risking a
transcription mismatch between two hand-written copies of the same taps.
"""

import numpy as np

from .convolutional import K, PUNCTURE_PATTERNS, _rate_half_encode_from_history

_N_STATES = 1 << (K - 1)  # 64


def _build_trellis():
    """(next_state, output) tables, each shaped (_N_STATES, 2): output[s][b]
    is the (g1, g2) pair (as a 2-bit int, g1 in bit 1, g2 in bit 0) the
    encoder emits from state `s` on input bit `b`; next_state[s][b] is the
    resulting state. Computed once at import time (pure function of K/G1/G2,
    no run-time parameters)."""
    next_state = np.zeros((_N_STATES, 2), dtype=np.int64)
    output = np.zeros((_N_STATES, 2), dtype=np.uint8)
    for state in range(_N_STATES):
        history = np.array([(state >> i) & 1 for i in range(K - 1)], dtype=np.uint8)
        for bit in (0, 1):
            g1, g2, new_history = _rate_half_encode_from_history(
                np.array([bit], dtype=np.uint8), invert_g2=False, history=history)
            next_state[state, bit] = sum(int(new_history[i]) << i for i in range(K - 1))
            output[state, bit] = (int(g1[0]) << 1) | int(g2[0])
    return next_state, output


_NEXT_STATE, _OUTPUT = _build_trellis()


def _build_predecessors():
    """Inverse of _NEXT_STATE: for each state s, its two (predecessor
    state, input bit) pairs -- every state in this trellis has exactly two
    predecessors, the shift-register structure's own guarantee, checked
    here rather than assumed. Lets the decode loop below look up, for each
    of the 64 states, both incoming branches directly instead of scanning
    all 128 (state, bit) source combinations every step."""
    pred_state = np.zeros((_N_STATES, 2), dtype=np.int64)
    pred_bit = np.zeros((_N_STATES, 2), dtype=np.int64)
    fill = np.zeros(_N_STATES, dtype=np.int64)
    for state in range(_N_STATES):
        for bit in (0, 1):
            nxt = int(_NEXT_STATE[state, bit])
            slot = fill[nxt]
            if slot >= 2:
                raise RuntimeError(f"state {nxt} has more than 2 predecessors -- trellis is not a valid binary shift register")
            pred_state[nxt, slot] = state
            pred_bit[nxt, slot] = bit
            fill[nxt] += 1
    if np.any(fill != 2):
        raise RuntimeError("not every state has exactly 2 predecessors -- trellis is not a valid binary shift register")
    pred_output = _OUTPUT[pred_state, pred_bit]  # (64, 2), reuses the (state,bit)->output table
    return pred_state, pred_bit, pred_output


_PRED_STATE, _PRED_BIT, _PRED_OUTPUT = _build_predecessors()
_PRED_G1 = (_PRED_OUTPUT >> 1) & 1  # (64, 2)
_PRED_G2 = _PRED_OUTPUT & 1  # (64, 2)
_STATE_RANGE = np.arange(_N_STATES)


def depuncture(coded_bits: np.ndarray, n_bits: int, rate: str) -> tuple:
    """Inverse of convolutional._puncture(): reinserts an erasure placeholder
    at every (g1, g2) position the encoder's puncturing dropped, restoring a
    full 2*n_bits-long stream. Returns (bits, present), both length
    2*n_bits: bits[2i]/present[2i] is g1 for input-bit i (present=False at a
    punctured position, where bits[...] is a meaningless placeholder), and
    similarly bits[2i+1]/present[2i+1] for g2.

    `n_bits` must be the exact number of *message* bits originally encoded
    (mirroring how conv_encode() needs len(bits) implicitly) -- decoding a
    punctured stream can't otherwise tell how many input bits it covers
    without also knowing the puncture ratio and rounding behavior at the
    boundary.
    """
    if rate not in PUNCTURE_PATTERNS:
        raise ValueError(f"unsupported convolutional code rate {rate!r}")
    if PUNCTURE_PATTERNS[rate] is None:
        if len(coded_bits) != 2 * n_bits:
            raise ValueError(f"expected {2 * n_bits} coded bits at rate 1/2, got {len(coded_bits)}")
        return np.asarray(coded_bits, dtype=np.uint8), np.ones(2 * n_bits, dtype=bool)

    # Reconstruct the exact same keep-mask _puncture() would have produced
    # for n_bits input bits starting at phase 0 (a one-shot conv_encode()
    # call, which is what a whole-burst decode mirrors -- not the
    # phase-carrying ConvEncoder streaming path).
    c1_pattern, c2_pattern = PUNCTURE_PATTERNS[rate]
    period = len(c1_pattern)
    idx = np.arange(n_bits) % period
    c1_mask = np.array(c1_pattern, dtype=bool)[idx]
    c2_mask = np.array(c2_pattern, dtype=bool)[idx]
    keep = np.empty(2 * n_bits, dtype=bool)
    keep[0::2] = c1_mask
    keep[1::2] = c2_mask

    if len(coded_bits) != int(keep.sum()):
        raise ValueError(
            f"expected {int(keep.sum())} coded bits at rate {rate} for {n_bits} message bits, "
            f"got {len(coded_bits)}"
        )
    bits = np.zeros(2 * n_bits, dtype=np.uint8)
    bits[keep] = coded_bits
    return bits, keep


def viterbi_decode(coded_bits: np.ndarray, n_bits: int, rate: str = "1/2", invert_g2: bool = True) -> np.ndarray:
    """Decode `coded_bits` (conv_encode()'s output) back to the original
    `n_bits` message bits.

    Standard hard-decision Viterbi over the 64-state trellis: the encoder
    always starts at the all-zero state (conv_encode() never resets/flushes
    mid-burst, matching a continuously-running physical coder), so path
    metrics start at state 0 only; since the encoder is also never flushed
    back to state 0 at the *end* of the burst, traceback starts from
    whichever final state has the lowest accumulated metric, not state 0.

    A punctured position (rate != "1/2") contributes 0 to every branch's
    metric (an erasure: neither hypothesis is penalized for a bit that was
    never actually transmitted), exactly undoing what puncturing removed.
    """
    invert_g2 = invert_g2 and rate == "1/2"  # matches conv_encode()'s own rule (3.4.1(5))
    if n_bits == 0:
        return np.empty(0, dtype=np.uint8)

    bits, present = depuncture(coded_bits, n_bits, rate)
    g1_bits, g1_present = bits[0::2], present[0::2]
    g2_bits, g2_present = bits[1::2], present[1::2]
    if invert_g2:
        g2_bits = np.where(g2_present, 1 - g2_bits, g2_bits)

    # path_metric[s] = best accumulated Hamming distance to reach state s;
    # inf everywhere but the guaranteed initial state 0.
    path_metric = np.full(_N_STATES, np.inf)
    path_metric[0] = 0.0
    # backptr[t, s, :] = (predecessor state, input bit) on the surviving
    # path into state s at time t -- filled in one vectorized step per t
    # (over all 64 states/2 predecessor slots at once) rather than a
    # Python-level loop over states, which for a whole-burst decode of a
    # real CADU stream (thousands of bits) is otherwise far too slow.
    backptr = np.empty((n_bits, _N_STATES, 2), dtype=np.int64)

    for t in range(n_bits):
        # Branch-metric contribution from each of the 2 predecessor slots
        # of every state, shaped (64, 2); an absent (punctured) bit
        # contributes 0 regardless of hypothesis (an erasure).
        dist = np.zeros((_N_STATES, 2), dtype=np.float64)
        if g1_present[t]:
            dist += (_PRED_G1 != int(g1_bits[t]))
        if g2_present[t]:
            dist += (_PRED_G2 != int(g2_bits[t]))
        candidates = path_metric[_PRED_STATE] + dist  # (64, 2)
        best_slot = np.argmin(candidates, axis=1)  # (64,)
        path_metric = candidates[_STATE_RANGE, best_slot]
        backptr[t, :, 0] = _PRED_STATE[_STATE_RANGE, best_slot]
        backptr[t, :, 1] = _PRED_BIT[_STATE_RANGE, best_slot]

    state = int(np.argmin(path_metric))
    decoded = np.empty(n_bits, dtype=np.uint8)
    for t in range(n_bits - 1, -1, -1):
        prev_state, bit = backptr[t, state]
        decoded[t] = bit
        state = int(prev_state)
    return decoded
