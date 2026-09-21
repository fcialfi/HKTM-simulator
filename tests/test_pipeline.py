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


class TestVirtualChannelFraming:
    """`vcid_list` builds a real CCSDS TM primary header (see
    transfer_frame.py) into synthetic Transfer Frames, so a receiver's
    Virtual Channel identification/routing can be validated against a
    known, controlled assignment -- round-robined across generated
    frames, with a VCID's repeat count in the list weighting its share."""

    def test_rejected_with_cadu_input(self):
        with pytest.raises(ValueError, match="input_format='transfer_frame'"):
            run_chain(_small_params(input_format="cadu", vcid_list=[0]))

    def test_rejected_with_a_real_payload_file(self):
        p = _small_params(vcid_list=[0], payload_bytes=bytes(100_000))
        with pytest.raises(ValueError, match="real payload"):
            run_chain(p)

    def test_rejected_for_out_of_range_vcid(self):
        with pytest.raises(ValueError, match="0-7"):
            run_chain(_small_params(vcid_list=[8]))

    def test_rejected_for_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            run_chain(_small_params(vcid_list=[]))

    def test_disabled_by_default(self):
        result = run_chain(_small_params())
        assert result.meta["vcid_list"] is None
        assert result.meta["spacecraft_id"] is None

    def test_headers_land_in_the_transmitted_cadus_with_correct_assignment(self):
        """Systematic RS encoding means the frame's leading bytes (the
        header, when vcid_list is set) are transmitted byte-for-byte
        unaltered right after the ASM -- so the actual CADU bytes this
        chain would put on the air can be checked directly against what
        transfer_frame.py predicts for each frame index."""
        from ccsds_chain.transfer_frame import build_primary_header, vcid_schedule_counts
        from ccsds_chain.utils import bits_to_bytes

        pattern = [0, 0, 1, 2]
        p = _small_params(n_cadu=8, vcid_list=pattern, spacecraft_id=0x123,
                           fec_rs=True, randomizer="none")
        prep = pipeline._prepare_payload(p)
        for i in range(8):
            cadu = bits_to_bytes(pipeline._cadu_bits(p, prep, i))
            header_in_cadu = cadu[len(p.asm):len(p.asm) + 6]
            vcid, vc_frame_count = vcid_schedule_counts(pattern, i)
            expected = build_primary_header(
                scid=p.spacecraft_id, vcid=vcid,
                mc_frame_count=i % 256, vc_frame_count=vc_frame_count % 256,
            )
            assert header_in_cadu == expected

    def test_meta_reports_vcid_list_and_spacecraft_id(self):
        p = _small_params(vcid_list=[0, 1, 2], spacecraft_id=0x2AA)
        result = run_chain(p)
        assert result.meta["vcid_list"] == [0, 1, 2]
        assert result.meta["spacecraft_id"] == 0x2AA

    def test_export_chain_matches_run_chain_with_vcid_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "_EXPORT_BATCH_TARGET_BYTES", 1)  # forces batch_n_cadu == 1
        p = _small_params(n_cadu=6, vcid_list=[0, 1, 2], spacecraft_id=0x123)
        expected = pack_iq_interleaved(normalize_peak(run_chain(p).iq, peak=0.9), "int16")

        out_path = tmp_path / "out.raw"
        export_chain(p, str(out_path), output_dtype="int16", peak=0.9)
        assert out_path.read_bytes() == expected


class TestDeterministicCorruption:
    """`corrupt_rs_symbols` flips an exact, reproducible number of RS
    symbols within one interleaved codeword of targeted CADUs -- for
    boundary-testing a receiver's RS decoder against its declared
    correction capability E, and for validating per-VC FER accounting --
    complementary to a replayer's AWGN, which can't guarantee hitting a
    precise per-codeword error count."""

    def test_disabled_by_default(self):
        result = run_chain(_small_params())
        assert result.meta["corrupt_rs_symbols"] is None

    def test_rejected_without_rs_coded_region(self):
        with pytest.raises(ValueError, match="RS-coded region"):
            run_chain(_small_params(fec_rs=False, corrupt_rs_symbols=1, corrupt_cadu_indices=[0]))

    def test_rejected_for_out_of_range_symbol_count(self):
        with pytest.raises(ValueError, match="corrupt_rs_symbols"):
            run_chain(_small_params(corrupt_rs_symbols=999, corrupt_cadu_indices=[0]))

    def test_rejected_for_out_of_range_codeword_index(self):
        with pytest.raises(ValueError, match="corrupt_codeword_index"):
            run_chain(_small_params(corrupt_rs_symbols=1, corrupt_cadu_indices=[0], corrupt_codeword_index=5))

    def test_rejected_without_a_target(self):
        with pytest.raises(ValueError, match="corrupt_cadu_indices"):
            run_chain(_small_params(corrupt_rs_symbols=1))

    def test_corrupt_vc_rejected_without_vcid_list(self):
        with pytest.raises(ValueError, match="vcid_list"):
            run_chain(_small_params(corrupt_rs_symbols=1, corrupt_vc=0))

    def test_corrupt_vc_rejected_for_out_of_range_vcid(self):
        with pytest.raises(ValueError, match="0-7"):
            run_chain(_small_params(vcid_list=[0], corrupt_rs_symbols=1, corrupt_vc=8))

    def test_exactly_e_errors_still_decodes_perfectly(self):
        """The whole point of the feature: hitting the RS decoder's
        declared correction capability exactly, on demand -- something a
        statistical noise sweep can't reliably guarantee."""
        from ccsds_chain.reed_solomon import rs_decode_interleaved
        from ccsds_chain.utils import bits_to_bytes

        p = _small_params(n_cadu=1, rs_e=16, rs_n=255, interleave_depth=5,
                           corrupt_rs_symbols=16, corrupt_codeword_index=0, corrupt_cadu_indices=[0])
        prep = pipeline._prepare_payload(p)
        cadu = bits_to_bytes(pipeline._cadu_bits(p, prep, 0))
        rs_region = cadu[len(p.asm):]
        decoded = rs_decode_interleaved(rs_region, p.rs_k, p.rs_n, p.interleave_depth)
        assert decoded == prep.payload[:prep.frame_bytes]

    def test_e_plus_one_errors_is_uncorrectable(self):
        from ccsds_chain.reed_solomon import rs_decode_interleaved
        from ccsds_chain.utils import bits_to_bytes

        p = _small_params(n_cadu=1, rs_e=16, rs_n=255, interleave_depth=5,
                           corrupt_rs_symbols=17, corrupt_codeword_index=0, corrupt_cadu_indices=[0])
        prep = pipeline._prepare_payload(p)
        cadu = bits_to_bytes(pipeline._cadu_bits(p, prep, 0))
        rs_region = cadu[len(p.asm):]
        with pytest.raises(ValueError, match="uncorrectable"):
            rs_decode_interleaved(rs_region, p.rs_k, p.rs_n, p.interleave_depth)

    def test_only_targeted_cadu_indices_are_corrupted(self):
        from ccsds_chain.reed_solomon import rs_decode_interleaved
        from ccsds_chain.utils import bits_to_bytes

        p = _small_params(n_cadu=4, rs_e=16, rs_n=255, interleave_depth=5,
                           corrupt_rs_symbols=17, corrupt_codeword_index=0, corrupt_cadu_indices=[2])
        prep = pipeline._prepare_payload(p)
        for i in range(4):
            cadu = bits_to_bytes(pipeline._cadu_bits(p, prep, i))
            rs_region = cadu[len(p.asm):]
            if i == 2:
                with pytest.raises(ValueError, match="uncorrectable"):
                    rs_decode_interleaved(rs_region, p.rs_k, p.rs_n, p.interleave_depth)
            else:
                rs_decode_interleaved(rs_region, p.rs_k, p.rs_n, p.interleave_depth)  # must not raise

    def test_only_targeted_codeword_is_corrupted(self):
        """The other interleave_depth-1 codewords in the same CADU stay
        clean -- corruption is scoped to corrupt_codeword_index alone."""
        from ccsds_chain.reed_solomon import from_dual_basis, rs_decode_codeword
        from ccsds_chain.utils import bits_to_bytes

        p = _small_params(n_cadu=1, rs_e=16, rs_n=255, interleave_depth=5,
                           corrupt_rs_symbols=17, corrupt_codeword_index=0, corrupt_cadu_indices=[0])
        prep = pipeline._prepare_payload(p)
        cadu = bits_to_bytes(pipeline._cadu_bits(p, prep, 0))
        rs_region = cadu[len(p.asm):]
        for j in range(1, p.interleave_depth):
            codeword_conventional = bytes(from_dual_basis(b) for b in rs_region[j::p.interleave_depth])
            rs_decode_codeword(codeword_conventional, p.rs_e)  # must not raise

    def test_corrupt_vc_targets_only_matching_virtual_channel(self):
        from ccsds_chain.reed_solomon import rs_decode_interleaved
        from ccsds_chain.transfer_frame import vcid_schedule_counts
        from ccsds_chain.utils import bits_to_bytes

        pattern = [0, 1, 2]
        p = _small_params(n_cadu=6, rs_e=16, rs_n=255, interleave_depth=5,
                           vcid_list=pattern, spacecraft_id=0x123,
                           corrupt_rs_symbols=17, corrupt_codeword_index=0, corrupt_vc=1)
        prep = pipeline._prepare_payload(p)
        for i in range(6):
            cadu = bits_to_bytes(pipeline._cadu_bits(p, prep, i))
            rs_region = cadu[len(p.asm):]
            vcid, _ = vcid_schedule_counts(pattern, i)
            if vcid == 1:
                with pytest.raises(ValueError, match="uncorrectable"):
                    rs_decode_interleaved(rs_region, p.rs_k, p.rs_n, p.interleave_depth)
            else:
                rs_decode_interleaved(rs_region, p.rs_k, p.rs_n, p.interleave_depth)  # must not raise

    def test_deterministic_across_independent_calls(self):
        p = _small_params(n_cadu=1, corrupt_rs_symbols=16, corrupt_cadu_indices=[0])
        prep = pipeline._prepare_payload(p)
        bits_a = pipeline._cadu_bits(p, prep, 0)
        bits_b = pipeline._cadu_bits(p, prep, 0)
        assert np.array_equal(bits_a, bits_b)

    def test_meta_reports_corruption_settings(self):
        p = _small_params(corrupt_rs_symbols=5, corrupt_codeword_index=2, corrupt_cadu_indices=[0, 1])
        result = run_chain(p)
        assert result.meta["corrupt_rs_symbols"] == 5
        assert result.meta["corrupt_codeword_index"] == 2
        assert result.meta["corrupt_cadu_indices"] == [0, 1]

    def test_export_chain_matches_run_chain_with_corruption(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "_EXPORT_BATCH_TARGET_BYTES", 1)  # forces batch_n_cadu == 1
        p = _small_params(n_cadu=6, corrupt_rs_symbols=16, corrupt_cadu_indices=[1, 3, 5])
        expected = pack_iq_interleaved(normalize_peak(run_chain(p).iq, peak=0.9), "int16")

        out_path = tmp_path / "out.raw"
        export_chain(p, str(out_path), output_dtype="int16", peak=0.9)
        assert out_path.read_bytes() == expected

    def test_works_with_cadu_input(self):
        """corrupt_rs_symbols also applies to input_format='cadu' (already
        RS-coded by construction, per _validate_chain_params)."""
        from ccsds_chain.reed_solomon import rs_decode_interleaved
        from ccsds_chain.utils import bits_to_bytes

        p = _small_params(n_cadu=1, input_format="cadu", rs_e=16, rs_n=255, interleave_depth=5,
                           corrupt_rs_symbols=17, corrupt_codeword_index=0, corrupt_cadu_indices=[0])
        prep = pipeline._prepare_payload(p)
        cadu = bits_to_bytes(pipeline._cadu_bits(p, prep, 0))
        rs_region = cadu[len(p.asm):]
        with pytest.raises(ValueError, match="uncorrectable"):
            rs_decode_interleaved(rs_region, p.rs_k, p.rs_n, p.interleave_depth)


class TestTransmitterImpairments:
    """freq_offset_hz / iq_gain_imbalance_db / iq_phase_imbalance_deg /
    phase_noise_linewidth_hz model a real transmitter's own imperfections
    (LO offset, IQ modulator mismatch, oscillator phase noise), applied to
    the pulse-shaped IQ -- distinct from a replayer's AWGN, which
    characterizes receiver sensitivity rather than transmitter realism."""

    def test_disabled_by_default_matches_pre_feature_output_bit_for_bit(self):
        p = _small_params()
        result = run_chain(p)
        assert result.meta["freq_offset_hz"] is None
        assert result.meta["iq_gain_imbalance_db"] is None
        assert result.meta["iq_phase_imbalance_deg"] is None
        assert result.meta["phase_noise_linewidth_hz"] is None

        p_explicit_zero = _small_params(freq_offset_hz=0.0, iq_gain_imbalance_db=0.0,
                                         iq_phase_imbalance_deg=0.0, phase_noise_linewidth_hz=0.0)
        assert np.array_equal(result.iq, run_chain(p_explicit_zero).iq)

    def test_rejects_negative_linewidth(self):
        with pytest.raises(ValueError, match="phase_noise_linewidth_hz"):
            run_chain(_small_params(phase_noise_linewidth_hz=-1.0))

    def test_frequency_offset_changes_the_signal(self):
        p_plain = _small_params(n_cadu=20)
        p_offset = _small_params(n_cadu=20, freq_offset_hz=50_000.0)
        assert not np.array_equal(run_chain(p_plain).iq, run_chain(p_offset).iq)

    def test_iq_imbalance_changes_the_signal(self):
        p_plain = _small_params(n_cadu=20)
        p_imbalanced = _small_params(n_cadu=20, iq_gain_imbalance_db=1.0, iq_phase_imbalance_deg=5.0)
        assert not np.array_equal(run_chain(p_plain).iq, run_chain(p_imbalanced).iq)

    def test_phase_noise_changes_the_signal(self):
        p_plain = _small_params(n_cadu=20)
        p_noisy = _small_params(n_cadu=20, phase_noise_linewidth_hz=500.0)
        assert not np.array_equal(run_chain(p_plain).iq, run_chain(p_noisy).iq)

    def test_meta_reports_configured_values(self):
        p = _small_params(freq_offset_hz=1234.0, iq_gain_imbalance_db=0.5,
                           iq_phase_imbalance_deg=2.0, phase_noise_linewidth_hz=300.0)
        meta = run_chain(p).meta
        assert meta["freq_offset_hz"] == 1234.0
        assert meta["iq_gain_imbalance_db"] == 0.5
        assert meta["iq_phase_imbalance_deg"] == 2.0
        assert meta["phase_noise_linewidth_hz"] == 300.0

    def test_export_chain_matches_run_chain_with_all_impairments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "_EXPORT_BATCH_TARGET_BYTES", 1)  # forces batch_n_cadu == 1
        p = _small_params(n_cadu=6, freq_offset_hz=5_000.0, iq_gain_imbalance_db=0.8,
                           iq_phase_imbalance_deg=3.0, phase_noise_linewidth_hz=200.0, impairment_seed=99)
        expected = pack_iq_interleaved(normalize_peak(run_chain(p).iq, peak=0.9), "int16")

        out_path = tmp_path / "out.raw"
        export_chain(p, str(out_path), output_dtype="int16", peak=0.9)
        assert out_path.read_bytes() == expected


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
