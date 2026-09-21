"""Integration/regression tests for the chain orchestration (pipeline.py).

Keeps n_cadu small everywhere so the suite stays fast; the batched-export
tests force tiny batches via monkeypatching so the cross-batch state
handling (the actual bug-prone part, per pipeline.py's own comments and
issue #12) is exercised even with a handful of CADUs.
"""

import numpy as np
import pytest

import ccsds_chain.pipeline as pipeline
from ccsds_chain.pipeline import ChainParams, export_chain, run_chain
from ccsds_chain.utils import normalize_peak, pack_iq_interleaved, resample_iq


def _small_params(**overrides):
    defaults = dict(n_cadu=5, sps=4, rrc_span=8, seed=1)
    defaults.update(overrides)
    return ChainParams(**defaults)


class TestChainParams:
    def test_rs_k_property(self):
        p = ChainParams(rs_n=255, rs_e=16)
        assert p.rs_k == 255 - 32
        p8 = ChainParams(rs_n=255, rs_e=8)
        assert p8.rs_k == 255 - 16


class TestValidation:
    def test_unknown_modulation_rejected(self):
        with pytest.raises(NotImplementedError):
            run_chain(_small_params(modulation="16QAM"))

    def test_unknown_encoding_rejected(self):
        with pytest.raises(NotImplementedError):
            run_chain(_small_params(encoding="Manchester"))

    def test_unknown_input_format_rejected(self):
        with pytest.raises(NotImplementedError):
            run_chain(_small_params(input_format="frame"))


class TestRunChainSmoke:
    @pytest.mark.parametrize("modulation", ["QPSK", "BPSK"])
    @pytest.mark.parametrize("fec_rs,fec_conv", [(True, True), (False, False), (True, False), (False, True)])
    def test_produces_consistent_shapes(self, modulation, fec_rs, fec_conv):
        p = _small_params(modulation=modulation, fec_rs=fec_rs, fec_conv=fec_conv)
        result = run_chain(p)
        assert result.meta["n_symbols"] == len(result.symbols)
        assert result.meta["n_iq_samples"] == len(result.iq)
        # pulse_shape() appends the RRC filter's group-delay tail
        # (len(taps) - 1 samples) beyond the upsampled symbol stream.
        n_taps = p.rrc_span * p.sps + 1
        assert len(result.iq) == len(result.symbols) * p.sps + (n_taps - 1)
        assert result.cadu_bytes == len(p.asm) + (p.rs_n if fec_rs else p.rs_k) * p.interleave_depth

    def test_deterministic_for_same_seed(self):
        p = _small_params()
        a = run_chain(p)
        b = run_chain(p)
        assert np.array_equal(a.iq, b.iq)

    def test_progress_callback_reaches_one(self):
        seen = []
        run_chain(_small_params(n_cadu=50), progress_callback=lambda frac, msg: seen.append(frac))
        assert seen[-1] == pytest.approx(1.0)


class TestCaduInputSync:
    """Exercises _prepare_payload()'s real-capture synchronization logic:
    leading junk, per-record wrapper framing between CADUs, and the
    too-short-gap guard -- the exact scenario pipeline.py's docstrings
    describe as having caused a real, silent framing bug (issue referenced
    in _PayloadPrep's docstring) before this logic existed."""

    @staticmethod
    def _build_source(unit_bytes, asm, n_cadu, leading_junk=0, wrapper_bytes=0):
        # All-zero CADU bodies and wrapper bytes never accidentally contain
        # the (non-zero) ASM pattern, so every ASM occurrence found is
        # exactly the ones this helper places on purpose.
        junk = bytes(leading_junk)
        cadu = asm + bytes(unit_bytes - len(asm))
        wrapper = bytes(wrapper_bytes)
        return junk + wrapper.join([cadu] * n_cadu) if n_cadu > 1 else junk + cadu

    def test_sync_skips_leading_junk_and_wrapper_bytes(self):
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1)
        _, unit_bytes, _, _ = pipeline._unit_bytes(p)
        source = self._build_source(unit_bytes, p.asm, n_cadu=4, leading_junk=17, wrapper_bytes=6)

        prep = pipeline._prepare_payload(
            ChainParams(**{**p.__dict__, "payload_bytes": source, "n_cadu": 4})
        )
        assert prep.sync_skipped_bytes == 17
        assert prep.n_synced_cadu == 4
        assert prep.cadu_wrapper_min_bytes == 6
        assert prep.cadu_wrapper_max_bytes == 6

    def test_cadu_bodies_extracted_ignore_wrapper_content(self):
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1)
        _, unit_bytes, _, _ = pipeline._unit_bytes(p)
        cadu_body = p.asm + bytes((i % 251) for i in range(unit_bytes - len(p.asm)))
        source = b"\xff" * 5 + cadu_body + b"\xee" * 9 + cadu_body

        full_p = ChainParams(**{**p.__dict__, "payload_bytes": source, "n_cadu": 2})
        prep = pipeline._prepare_payload(full_p)
        assert pipeline._cadu_bits(full_p, prep, 0).tobytes() is not None  # doesn't raise
        from ccsds_chain.utils import bytes_to_bits
        assert np.array_equal(pipeline._cadu_bits(full_p, prep, 0), bytes_to_bits(cadu_body))
        assert np.array_equal(pipeline._cadu_bits(full_p, prep, 1), bytes_to_bits(cadu_body))

    def test_gap_shorter_than_unit_bytes_raises(self):
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1)
        _, unit_bytes, _, _ = pipeline._unit_bytes(p)
        cadu = p.asm + bytes(unit_bytes - len(p.asm))
        # Second ASM arrives one byte too early -- shorter than a real CADU
        # could be at this configuration.
        source = cadu[:-1] + p.asm + bytes(unit_bytes - len(p.asm))
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": source, "n_cadu": 2})
        with pytest.raises(ValueError, match="only .* bytes"):
            pipeline._prepare_payload(full_p)


class TestCaduInputRandomizer:
    """CADU input isn't assumed to already be scrambled: `randomizer` must
    keep controlling whether the RS-coded region gets (re-)scrambled even
    in "cadu" input_format, the same as in "transfer_frame" mode. Before
    this was fixed, CADU input unconditionally forced the randomizer off
    (`pn = None` regardless of `p.randomizer`), so a captured/decoded CADU
    source that was already de-scrambled (e.g. by the capture instrument's
    own frame sync) could never be re-scrambled for retransmission -- a
    real receiver's own descrambler would then corrupt every frame."""

    @staticmethod
    def _source(p, body_byte=0x42):
        _, unit_bytes, _, _ = pipeline._unit_bytes(p)
        cadu = p.asm + bytes([body_byte]) * (unit_bytes - len(p.asm))
        return cadu, unit_bytes

    def test_randomizer_none_leaves_cadu_bits_unchanged(self):
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1, randomizer="none")
        cadu, _ = self._source(p)
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": cadu, "n_cadu": 1})
        prep = pipeline._prepare_payload(full_p)
        from ccsds_chain.utils import bytes_to_bits
        assert np.array_equal(pipeline._cadu_bits(full_p, prep, 0), bytes_to_bits(cadu))

    @pytest.mark.parametrize("randomizer", ["short", "long"])
    def test_randomizer_scrambles_rs_region_not_asm(self, randomizer):
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1, randomizer=randomizer)
        cadu, _ = self._source(p)
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": cadu, "n_cadu": 1})
        prep = pipeline._prepare_payload(full_p)
        bits = pipeline._cadu_bits(full_p, prep, 0)
        from ccsds_chain.utils import bytes_to_bits
        asm_bit_len = len(p.asm) * 8
        original_bits = bytes_to_bits(cadu)
        assert np.array_equal(bits[:asm_bit_len], original_bits[:asm_bit_len])  # ASM untouched
        assert not np.array_equal(bits[asm_bit_len:], original_bits[asm_bit_len:])  # RS region scrambled

    def test_meta_reports_actual_randomizer_not_forced_none(self):
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1, randomizer="long")
        cadu, _ = self._source(p)
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": cadu, "n_cadu": 1})
        result = run_chain(full_p)
        assert result.meta["randomizer"] == "long"

    def test_scrambling_is_reversible_via_the_same_pn_sequence(self):
        """Round-trip sanity check: XOR-ing an already-scrambled CADU's
        RS region with the same PN sequence again recovers the original
        bytes -- confirms _cadu_bits() scrambles with a plain XOR (as
        CCSDS section 10 specifies), not something order-dependent."""
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1, randomizer="long")
        cadu, _ = self._source(p, body_byte=0x99)
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": cadu, "n_cadu": 1})
        prep = pipeline._prepare_payload(full_p)
        scrambled_bits = pipeline._cadu_bits(full_p, prep, 0)

        # Feed the now-scrambled CADU back in as a fresh "already on-air"
        # source with the same randomizer: scrambling it again must undo
        # the first pass and recover the original bytes.
        scrambled_cadu = np.packbits(scrambled_bits, bitorder="big").tobytes()
        full_p2 = ChainParams(**{**p.__dict__, "payload_bytes": scrambled_cadu, "n_cadu": 1})
        prep2 = pipeline._prepare_payload(full_p2)
        round_tripped_bits = pipeline._cadu_bits(full_p2, prep2, 0)
        from ccsds_chain.utils import bytes_to_bits
        assert np.array_equal(round_tripped_bits, bytes_to_bits(cadu))


class TestExportChainMatchesRunChain:
    """export_chain()'s own docstring claim: for int16 output, batched
    streaming export is exactly byte-for-byte identical to packing
    run_chain()'s whole-array result. Forces a tiny batch size so every
    CADU is its own batch, stressing the conv-encoder/RRC-shaper state
    carried across batches -- not just the (trivial) single-batch case a
    small n_cadu would otherwise hit by default."""

    @pytest.mark.parametrize("modulation,conv_rate", [("QPSK", "1/2"), ("BPSK", "1/2"), ("QPSK", "3/4")])
    def test_forced_tiny_batches_match_whole_array_run(self, tmp_path, monkeypatch, modulation, conv_rate):
        monkeypatch.setattr(pipeline, "_EXPORT_BATCH_TARGET_BYTES", 1)  # forces batch_n_cadu == 1

        p = _small_params(n_cadu=6, modulation=modulation, conv_rate=conv_rate)
        expected = pack_iq_interleaved(normalize_peak(run_chain(p).iq, peak=0.9), "int16")

        out_path = tmp_path / "out.raw"
        export_chain(p, str(out_path), output_dtype="int16", peak=0.9)
        assert out_path.read_bytes() == expected

    def test_resampled_export_matches_manual_resample_of_run_chain(self, tmp_path):
        p = _small_params(n_cadu=4)
        native_fs = p.symbol_rate * p.sps
        target_fs = native_fs / 2  # exact 1:2 ratio, cheap to resample

        result = run_chain(p)
        expected_resampled = resample_iq(normalize_peak(result.iq, peak=0.9), native_fs, target_fs)
        expected = pack_iq_interleaved(expected_resampled, "int16")

        out_path = tmp_path / "out.raw"
        export_chain(p, str(out_path), output_dtype="int16", peak=0.9, target_fs=target_fs)
        assert out_path.read_bytes() == expected

    def test_refuses_infeasible_resample_upfront(self, tmp_path, monkeypatch):
        # Exercise the upfront memory guard without actually needing a
        # multi-GB export: shrink the guard threshold itself instead of
        # trying to build a real oversized job.
        monkeypatch.setattr(pipeline, "_RESAMPLE_MEMORY_GUARD_BYTES", 1)
        p = _small_params(n_cadu=1)
        native_fs = p.symbol_rate * p.sps
        with pytest.raises(MemoryError):
            export_chain(p, str(tmp_path / "out.raw"), target_fs=native_fs / 2)
