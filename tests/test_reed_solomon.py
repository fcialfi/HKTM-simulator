"""Regression tests for the CCSDS-native RS(255,*) encoder.

These do not re-derive ground truth from the standard's own tables (that
cross-check is described, and was done by hand, in reed_solomon.py's module
docstring) -- instead they turn the *algebraic properties* that the module
docstring claims into automated checks, so a future edit that silently
breaks GF(256) arithmetic, the dual-basis transform, or the generator
polynomials is caught by CI rather than only by eyeballing a spectrum plot.
"""

import numpy as np
import pytest

from ccsds_chain.reed_solomon import (
    _EXP,
    _GEN_POLY,
    _LOG,
    from_dual_basis,
    gf_mul,
    rs_decode_codeword,
    rs_decode_interleaved,
    rs_encode_codeword,
    rs_encode_interleaved,
    rs_generator_polynomial,
    to_dual_basis,
)


def gf_poly_eval(coeffs_desc, x):
    """Evaluate a GF(256) polynomial (coefficients highest-degree first, the
    same left-to-right convention `rs_encode_codeword` uses) at `x`, via
    Horner's method."""
    result = 0
    for c in coeffs_desc:
        result = gf_mul(result, x) ^ c
    return result


class TestGF256Tables:
    def test_exp_log_are_inverses(self):
        assert _EXP[0] == 1
        assert _LOG[1] == 0
        for a in range(1, 256):
            assert _EXP[_LOG[a]] == a
        for i in range(255):
            assert _LOG[_EXP[i]] == i

    def test_exp_table_periodic_with_period_255(self):
        for i in range(255):
            assert _EXP[i] == _EXP[i + 255]

    def test_alpha_is_a_primitive_element(self):
        # exp[0..254] must be a permutation of 1..255, i.e. alpha (whose
        # powers fill exp[]) actually generates the whole multiplicative
        # group -- the defining property of a *primitive* field polynomial.
        assert sorted(_EXP[:255]) == list(range(1, 256))

    def test_gf_mul_properties(self):
        rng = np.random.default_rng(0)
        for a, b in rng.integers(0, 256, size=(200, 2)):
            a, b = int(a), int(b)
            assert gf_mul(a, b) == gf_mul(b, a)
        for a in range(256):
            assert gf_mul(a, 0) == 0
            assert gf_mul(a, 1) == a


class TestDualBasisTransform:
    def test_round_trip_is_identity_for_every_byte(self):
        # to_dual_basis/from_dual_basis are two fixed 8x8 GF(2) matrices;
        # this holds iff they are genuinely inverse matrices, independent of
        # any value copied from the standard.
        for b in range(256):
            assert from_dual_basis(to_dual_basis(b)) == b
            assert to_dual_basis(from_dual_basis(b)) == b

    def test_zero_maps_to_zero(self):
        # Both transforms are linear (no additive constant), so 0 is fixed.
        assert to_dual_basis(0) == 0
        assert from_dual_basis(0) == 0

    def test_to_dual_basis_is_a_bijection(self):
        assert sorted(to_dual_basis(b) for b in range(256)) == list(range(256))


class TestGeneratorPolynomial:
    @pytest.mark.parametrize("e", [8, 16])
    def test_is_self_reciprocal(self, e):
        # CCSDS 131.0-B-5 Annex G: g(x) is self-reciprocal, G_i = G_(2E-i).
        coeffs = _GEN_POLY[e]
        assert coeffs == list(reversed(coeffs))

    @pytest.mark.parametrize("e", [8, 16])
    def test_has_the_right_degree(self, e):
        assert len(rs_generator_polynomial(e)) == 2 * e + 1
        assert rs_generator_polynomial(e)[0] == 1  # monic, leading coefficient
        assert rs_generator_polynomial(e)[-1] == 1  # self-reciprocal -> constant term also 1


class TestSystematicEncoding:
    @pytest.mark.parametrize("e", [8, 16])
    def test_codeword_is_divisible_by_every_required_root(self, e):
        """The defining algebraic property of the code (4.3.4): the
        systematic codeword must be a multiple of g(x), whose roots are
        alpha^(11j) for j = 128-E .. 127+E."""
        rng = np.random.default_rng(1234 + e)
        k = 255 - 2 * e
        message = bytes(int(b) for b in rng.integers(0, 256, size=k, dtype=np.uint8))
        parity = rs_encode_codeword(message, e)
        assert len(parity) == 2 * e

        codeword = list(message) + list(parity)  # descending degree, x^(n-1) .. x^0
        for j in range(128 - e, 128 + e):
            root = _EXP[(11 * j) % 255]
            assert gf_poly_eval(codeword, root) == 0, f"codeword not divisible by root alpha^{11 * j}"

    def test_zero_message_has_zero_parity(self):
        # The all-zero codeword is always a valid codeword of a linear code.
        for e in (8, 16):
            k = 255 - 2 * e
            parity = rs_encode_codeword(bytes(k), e)
            assert parity == bytes(2 * e)

    def test_rejects_wrong_length_data(self):
        with pytest.raises(ValueError):
            rs_encode_interleaved(bytes(10), k=223, n=255, depth=5)

    def test_rejects_unsupported_error_correction_capability(self):
        # k*depth must stay consistent with n so that e=(n-k)//2 lands on an
        # unsupported value (e.g. e=4) to exercise this guard.
        with pytest.raises(ValueError):
            rs_encode_interleaved(bytes(247 * 2), k=247, n=255, depth=2)


class TestInterleaving:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 8])
    def test_interleaved_output_size(self, depth):
        k, n = 223, 255
        data = bytes((i * 7) % 256 for i in range(k * depth))
        out = rs_encode_interleaved(data, k, n, depth)
        assert len(out) == n * depth

    def test_data_zone_passes_through_unaltered(self):
        # data[j::depth] must reappear byte-for-byte as the first k symbols
        # of interleaved codeword j (4.3.4: "uncoded" part of the codeblock).
        k, n, depth = 223, 255, 5
        data = bytes((i * 13 + 1) % 256 for i in range(k * depth))
        out = rs_encode_interleaved(data, k, n, depth)
        for j in range(depth):
            substream = data[j::depth]
            codeword = out[j::depth]
            assert codeword[:k] == substream

    def test_deterministic(self):
        k, n, depth = 223, 255, 5
        data = bytes((i * 17) % 256 for i in range(k * depth))
        assert rs_encode_interleaved(data, k, n, depth) == rs_encode_interleaved(data, k, n, depth)

    @pytest.mark.parametrize("depth", [1, 2, 5])
    def test_each_interleaved_codeword_is_a_valid_rs_codeword(self, depth):
        """End-to-end check tying interleaving back to the root-divisibility
        property, in the representation actually transmitted (dual basis):
        each de-interleaved, dual-basis codeword must convert back to a
        conventional-basis codeword divisible by every required root."""
        k, n, e = 223, 255, 16
        rng = np.random.default_rng(99)
        data = bytes(int(b) for b in rng.integers(0, 256, size=k * depth, dtype=np.uint8))
        out = rs_encode_interleaved(data, k, n, depth)
        for j in range(depth):
            codeword_dual = out[j::depth]
            codeword_conventional = [from_dual_basis(b) for b in codeword_dual]
            for jroot in range(128 - e, 128 + e):
                root = _EXP[(11 * jroot) % 255]
                assert gf_poly_eval(codeword_conventional, root) == 0


class TestDecoding:
    """Self-verification loopback for the codeword-level encoder: proves
    rs_decode_codeword actually corrects real injected errors (up to E per
    codeword), not just that a zero-error codeword passes through -- the
    exact property the module docstring's decoding-comments section
    describes needing this kind of check for."""

    @pytest.mark.parametrize("e", [8, 16])
    def test_decodes_exactly_with_no_errors(self, e):
        rng = np.random.default_rng(1000 + e)
        k = 255 - 2 * e
        message = bytes(int(b) for b in rng.integers(0, 256, size=k, dtype=np.uint8))
        codeword = message + rs_encode_codeword(message, e)
        assert rs_decode_codeword(codeword, e) == message

    @pytest.mark.parametrize("e", [8, 16])
    def test_corrects_every_error_count_up_to_e(self, e):
        rng = np.random.default_rng(2000 + e)
        k = 255 - 2 * e
        for n_errors in range(0, e + 1):
            message = bytes(int(b) for b in rng.integers(0, 256, size=k, dtype=np.uint8))
            codeword = bytearray(message + rs_encode_codeword(message, e))
            positions = rng.choice(255, size=n_errors, replace=False)
            for pos in positions:
                codeword[pos] ^= int(rng.integers(1, 256))
            decoded = rs_decode_codeword(bytes(codeword), e)
            assert decoded == message, f"E={e}, {n_errors} injected errors"

    @pytest.mark.parametrize("depth", [1, 2, 5, 8])
    def test_interleaved_round_trip_with_up_to_e_errors_per_codeword(self, depth):
        rng = np.random.default_rng(3000 + depth)
        k, n, e = 223, 255, 16
        data = bytes(int(b) for b in rng.integers(0, 256, size=k * depth, dtype=np.uint8))
        encoded = bytearray(rs_encode_interleaved(data, k, n, depth))
        for j in range(depth):
            n_errors = int(rng.integers(0, e + 1))
            positions = rng.choice(n, size=n_errors, replace=False)
            for pos in positions:
                idx = pos * depth + j
                encoded[idx] ^= int(rng.integers(1, 256))
        decoded = rs_decode_interleaved(bytes(encoded), k, n, depth)
        assert decoded == data

    def test_uncorrectable_codeword_raises_rather_than_miscorrects(self):
        # A fixed seed known (empirically, see the PR this landed in) to
        # inject more errors (E+1) than RS(255,223) can correct and produce
        # a locator polynomial degree that Berlekamp-Massey/Chien-search
        # detect as inconsistent, rather than one of the rarer cases where
        # an over-limit pattern happens to look like a valid <=E-error one.
        rng = np.random.default_rng(9)
        e = 8
        k = 255 - 2 * e
        message = bytes(int(b) for b in rng.integers(0, 256, size=k, dtype=np.uint8))
        codeword = bytearray(message + rs_encode_codeword(message, e))
        positions = rng.choice(255, size=e + 1, replace=False)
        for pos in positions:
            codeword[pos] ^= int(rng.integers(1, 256))
        with pytest.raises(ValueError):
            rs_decode_codeword(bytes(codeword), e)

    def test_unsupported_e_raises(self):
        with pytest.raises(ValueError):
            rs_decode_codeword(bytes(255), e=4)
