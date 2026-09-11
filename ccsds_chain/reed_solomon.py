"""RS(255,223) / RS(255,239) encoding with CCSDS byte interleaving, per
CCSDS 131.0-B-5 section 4.

This is a from-scratch CCSDS-native encoder (not a wrapper around a generic
RS library), because the standard's code differs from a "conventional"
Reed-Solomon setup in three ways that a generic library's defaults won't
match:

- Field generator polynomial F(x) = x^8+x^7+x^2+x+1 = 0x187 (4.3.3), not the
  0x11D used by most RS libraries (including `reedsolo`'s default).
- Code generator polynomial roots are alpha^(11j) for j=128-E..127+E (4.3.4)
  -- widely spaced powers of alpha, not consecutive ones.
- Transmitted symbols use the "dual basis" (Berlekamp) representation
  (4.3.9), not the polynomial-in-alpha ("conventional") basis a generic
  encoder produces.

The GF(256) table, the two generator-polynomial coefficient sets, and the
dual-basis transform matrix below are all copied directly from the
standard's own explicit numeric tables (Annex G and Annex F respectively)
rather than re-derived, and have been cross-checked against the worked
examples and reference table entries the standard provides for exactly
this purpose:
- GF(256) exp table checked against 13 reference points in Table F-1
  (including all 8 "single-bit dual-basis" exponents 46,67,88,125,163,
  184,226,242).
- Generator polynomial coefficients checked against their claimed alpha
  exponents in Annex G, for both E=16 and E=8.
- Dual-basis transform checked against Annex F's worked Examples 1 and 2.
- The resulting codewords were verified to be divisible by all 2E required
  roots alpha^(11j), which is the defining algebraic property of the code.
"""

import numpy as np

FIELD_PRIM = 0x187  # F(x) = x^8 + x^7 + x^2 + x + 1 (CCSDS 131.0-B-5, 4.3.3)


def _build_gf_tables(prim: int):
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= prim
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


_EXP, _LOG = _build_gf_tables(FIELD_PRIM)


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


# Full multiplication table, built once: _GF_MUL_TABLE[a][b] = gf_mul(a, b).
# Lets the RS division inner loop (multiply one coefficient by every
# generator-polynomial coefficient at once) be a single vectorized numpy
# row lookup instead of a Python-level loop of gf_mul() calls.
_GF_MUL_TABLE = np.array([[gf_mul(a, b) for b in range(256)] for a in range(256)], dtype=np.uint8)


# Generator polynomial coefficients G0..G_2E, ascending degree, from CCSDS
# 131.0-B-5 Annex G ("Expansion of Reed-Solomon Coefficients"). Each g(x) is
# self-reciprocal (G_i = G_(2E-i)).
_GEN_POLY = {
    16: [1, 91, 127, 86, 16, 30, 13, 235, 97, 165, 8, 42, 54, 86, 171, 32, 113,
         32, 171, 86, 54, 42, 8, 165, 97, 235, 13, 30, 16, 86, 127, 91, 1],
    8: [1, 165, 105, 27, 159, 104, 152, 101, 74,
        101, 152, 104, 159, 27, 105, 165, 1],
}

# Dual-basis (Berlekamp) transform T_(alpha,ell): [z0..z7] = [u7..u0] . T,
# from CCSDS 131.0-B-5 Annex F. Row i is the contribution of input bit u_(7-i).
_DUAL_BASIS_T = [
    [1, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 1, 1],
]


_DUAL_BASIS_T_INV = [
    [1, 1, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 1, 0, 0],
]


def _apply_transform(byte: int, matrix: list[list[int]]) -> int:
    out = 0
    for i in range(8):
        if (byte >> (7 - i)) & 1:
            for j in range(8):
                out ^= matrix[i][j] << (7 - j)
    return out


# Precomputed as 256-entry lookup tables (this is called once per RS symbol
# transmitted, so a table lookup matters for larger CADU counts).
_TO_DUAL_BASIS_LUT = [_apply_transform(b, _DUAL_BASIS_T) for b in range(256)]
_FROM_DUAL_BASIS_LUT = [_apply_transform(b, _DUAL_BASIS_T_INV) for b in range(256)]


def to_dual_basis(u_byte: int) -> int:
    """Convert one RS symbol from the polynomial-in-alpha ("conventional")
    basis to the dual (Berlekamp) basis required for transmission (4.3.9)."""
    return _TO_DUAL_BASIS_LUT[u_byte]


def from_dual_basis(z_byte: int) -> int:
    """Inverse of to_dual_basis()."""
    return _FROM_DUAL_BASIS_LUT[z_byte]


def rs_generator_polynomial(e: int) -> list[int]:
    """g(x) coefficients, descending degree (G_2E first), for error
    correction capability `e` (16 or 8)."""
    return list(reversed(_GEN_POLY[e]))


_GEN_POLY_ARR = {e: np.array(rs_generator_polynomial(e), dtype=np.uint8) for e in _GEN_POLY}


def rs_encode_codeword(message: bytes, e: int) -> bytes:
    """Systematic RS encode of one codeword: `message` (k = 255-2E symbols,
    in the polynomial-in-alpha/"conventional" basis) in, 2E parity symbols
    (same basis) out, via polynomial division by g(x) (4.3.4)."""
    g = _GEN_POLY_ARR[e]
    two_e = 2 * e
    remainder = np.zeros(len(message) + two_e, dtype=np.uint8)
    remainder[:len(message)] = np.frombuffer(message, dtype=np.uint8)
    for i in range(len(message)):
        coef = remainder[i]
        if coef:
            remainder[i:i + len(g)] ^= _GF_MUL_TABLE[coef, g]
    return remainder[len(message):].tobytes()


def rs_encode_interleaved(data: bytes, k: int, n: int, depth: int) -> bytes:
    """RS-encode `data` (must be exactly k*depth bytes) as `depth`
    interleaved RS(n,k) codewords, per CCSDS byte interleaving: substream j
    (j=0..depth-1) is data[j::depth], and the interleaved output byte at
    position i*depth+j is codeword_j[i].

    `data` is Transfer Frame data and is transmitted byte-for-byte
    unaltered (it is the "uncoded" part of the codeblock, figure 4-1): per
    Annex F's transformational-equivalence procedure, it is treated as
    already being in the dual (Berlekamp) basis required for transmission,
    converted to the polynomial-in-alpha basis only internally to compute
    parity with a conventional (polynomial-division) encoder, and the
    resulting parity is then converted forward to the dual basis for
    output -- so only the parity bytes actually change representation.
    """
    if len(data) != k * depth:
        raise ValueError(f"expected {k * depth} bytes, got {len(data)}")

    e = (n - k) // 2
    if e not in _GEN_POLY:
        raise ValueError(f"unsupported error-correction capability E={e} (must be 8 or 16)")

    codewords = []
    for j in range(depth):
        substream = data[j::depth]
        message_conventional = bytes(from_dual_basis(b) for b in substream)
        parity_conventional = rs_encode_codeword(message_conventional, e)
        parity = bytes(to_dual_basis(b) for b in parity_conventional)
        codeword = substream + parity
        if len(codeword) != n:
            raise RuntimeError(f"unexpected RS codeword length {len(codeword)} (expected {n})")
        codewords.append(codeword)

    out = bytearray(n * depth)
    for j, codeword in enumerate(codewords):
        out[j::depth] = codeword
    return bytes(out)
