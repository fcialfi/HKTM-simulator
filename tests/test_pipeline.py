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


class TestAsmFramedInputRandomizer:
    """The two ASM-framed input formats differ only in scrambling:
    "asm_frame" is ASM + a Transfer Frame that is NOT scrambled yet (what a
    ground station such as Cortex hands back once its per-record overhead
    is stripped), so the block after the ASM gets scrambled here -- never
    the ASM; "cadu" is already exactly as on the air, so it is used
    verbatim. Each enforces the matching `randomizer`."""

    @staticmethod
    def _source(p, body_byte=0x42):
        _, unit_bytes, _, _ = pipeline._unit_bytes(p)
        cadu = p.asm + bytes([body_byte]) * (unit_bytes - len(p.asm))
        return cadu, unit_bytes

    def test_record_is_1279_bytes_at_default_interleave_depth(self):
        for input_format, randomizer in [("asm_frame", "long"), ("cadu", "none")]:
            p = ChainParams(input_format=input_format, randomizer=randomizer)
            assert pipeline._unit_bytes(p)[1] == 1279

    def test_extra_bytes_after_each_record_are_skipped(self):
        p = _small_params(input_format="asm_frame", randomizer="long")
        cadu, unit_bytes = self._source(p)
        trailer = b"\xee" * 37  # receiver-added overhead, never transmitted
        source = cadu + trailer + cadu + trailer
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": source, "n_cadu": 2})
        prep = pipeline._prepare_payload(full_p)
        assert prep.n_synced_cadu == 2
        assert prep.cadu_wrapper_min_bytes == 37
        assert np.array_equal(pipeline._cadu_bits(full_p, prep, 0), pipeline._cadu_bits(full_p, prep, 1))

    def test_cadu_is_used_verbatim(self):
        p = _small_params(input_format="cadu", rs_e=8, interleave_depth=1, randomizer="none")
        cadu, _ = self._source(p)
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": cadu, "n_cadu": 1})
        prep = pipeline._prepare_payload(full_p)
        from ccsds_chain.utils import bytes_to_bits
        assert np.array_equal(pipeline._cadu_bits(full_p, prep, 0), bytes_to_bits(cadu))

    @pytest.mark.parametrize("randomizer", ["short", "long"])
    def test_asm_frame_scrambles_frame_not_asm(self, randomizer):
        from ccsds_chain.scrambler import pn_sequence
        from ccsds_chain.utils import bytes_to_bits
        p = _small_params(input_format="asm_frame", rs_e=8, interleave_depth=1, randomizer=randomizer)
        cadu, unit_bytes = self._source(p)
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": cadu, "n_cadu": 1})
        prep = pipeline._prepare_payload(full_p)
        bits = pipeline._cadu_bits(full_p, prep, 0)
        asm_bit_len = len(p.asm) * 8
        original_bits = bytes_to_bits(cadu)
        assert np.array_equal(bits[:asm_bit_len], original_bits[:asm_bit_len])  # ASM untouched
        pn = pn_sequence((unit_bytes - len(p.asm)) * 8, randomizer)
        assert np.array_equal(bits[asm_bit_len:], original_bits[asm_bit_len:] ^ pn)  # frame scrambled

    def test_asm_frame_rejects_no_randomizer(self):
        with pytest.raises(ValueError, match="NOT scrambled"):
            run_chain(_small_params(input_format="asm_frame", randomizer="none"))

    @pytest.mark.parametrize("randomizer", ["short", "long"])
    def test_cadu_rejects_randomizer(self, randomizer):
        with pytest.raises(ValueError, match="already exactly as on the air"):
            run_chain(_small_params(input_format="cadu", randomizer=randomizer))

    def test_meta_reports_randomizer(self):
        p = _small_params(input_format="asm_frame", rs_e=8, interleave_depth=1, randomizer="long")
        cadu, _ = self._source(p)
        full_p = ChainParams(**{**p.__dict__, "payload_bytes": cadu, "n_cadu": 1})
        result = run_chain(full_p)
        assert result.meta["randomizer"] == "long"
        assert result.meta["input_format"] == "asm_frame"

    def test_asm_frame_output_equals_cadu_of_prescrambled_input(self):
        """Scrambling an "asm_frame" source here must give exactly the
        signal of the same data pre-scrambled and fed as an on-air "cadu"."""
        from ccsds_chain.scrambler import pn_sequence
        p = _small_params(input_format="asm_frame", rs_e=8, interleave_depth=1, randomizer="long")
        cadu, unit_bytes = self._source(p, body_byte=0x99)
        pn_bytes = np.packbits(pn_sequence((unit_bytes - len(p.asm)) * 8, "long")).tobytes()
        on_air = p.asm + bytes(a ^ b for a, b in zip(cadu[len(p.asm):], pn_bytes))
        a = run_chain(ChainParams(**{**p.__dict__, "payload_bytes": cadu * 3, "n_cadu": 3}))
        b = run_chain(ChainParams(**{**p.__dict__, "input_format": "cadu", "randomizer": "none",
                                     "payload_bytes": on_air * 3, "n_cadu": 3}))
        assert np.array_equal(a.iq, b.iq)


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

    def test_iq_imbalance_produces_a_visible_image_once_offset_from_zero_hz(self):
        """Regression test for an ordering bug: this chain's 0 Hz *is* the
        transmitter's intended RF center frequency, so apply_iq_imbalance()'s
        mirror-image term only separates visibly from the wanted signal once
        freq_offset_hz has displaced it away from 0 Hz first (see
        _apply_impairments()'s docstring) -- applying imbalance to a signal
        still symmetric about 0 Hz (this project's random-data QPSK, with no
        offset) folds the "image" invisibly back onto the same band. Checks
        this against the actual run_chain() output, not just the impairment
        functions in isolation (which are correct either way -- this is
        about *pipeline ordering*, not the per-function math)."""
        n_cadu = 30
        freq_offset = 1_500_000.0

        def psd_db(iq, fs):
            spec = np.fft.fftshift(np.fft.fft(iq * np.hanning(len(iq))))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))
            p_db = 20 * np.log10(np.abs(spec) + 1e-12)
            return freqs, p_db - p_db.max()

        p_offset_only = _small_params(n_cadu=n_cadu, freq_offset_hz=freq_offset)
        result = run_chain(p_offset_only)
        fs = result.sample_rate
        freqs, psd_offset_only = psd_db(result.iq, fs)

        p_offset_and_imbalance = _small_params(
            n_cadu=n_cadu, freq_offset_hz=freq_offset,
            iq_gain_imbalance_db=3.0, iq_phase_imbalance_deg=15.0,
        )
        _, psd_with_imbalance = psd_db(run_chain(p_offset_and_imbalance).iq, fs)

        image_band = (freqs >= -1.8e6) & (freqs < -1.2e6)  # mirror of the +1.5 MHz wanted band
        wanted_band = (freqs >= 1.2e6) & (freqs < 1.8e6)

        image_rise_db = psd_with_imbalance[image_band].mean() - psd_offset_only[image_band].mean()
        wanted_change_db = psd_with_imbalance[wanted_band].mean() - psd_offset_only[wanted_band].mean()

        assert image_rise_db > 20, f"expected a clearly visible image (>20 dB rise), got {image_rise_db:.1f} dB"
        assert abs(wanted_change_db) < 1, f"wanted band should stay ~unchanged, moved {wanted_change_db:.1f} dB"

    def test_pa_disabled_by_default(self):
        result = run_chain(_small_params())
        assert result.meta["pa_backoff_db"] is None
        assert result.meta["pa_smoothness"] is None
        assert result.meta["pa_am_pm_deg_per_db"] is None

    def test_pa_rejects_non_positive_smoothness(self):
        with pytest.raises(ValueError, match="pa_smoothness"):
            run_chain(_small_params(pa_backoff_db=0.0, pa_smoothness=0.0))

    def test_pa_changes_the_signal(self):
        p_plain = _small_params(n_cadu=20)
        p_pa = _small_params(n_cadu=20, pa_backoff_db=0.0, pa_smoothness=3.0)
        assert not np.array_equal(run_chain(p_plain).iq, run_chain(p_pa).iq)

    def test_pa_meta_reports_configured_values(self):
        p = _small_params(pa_backoff_db=-3.0, pa_smoothness=2.5, pa_am_pm_deg_per_db=4.0)
        meta = run_chain(p).meta
        assert meta["pa_backoff_db"] == -3.0
        assert meta["pa_smoothness"] == 2.5
        assert meta["pa_am_pm_deg_per_db"] == 4.0

    def test_pa_produces_visible_spectral_regrowth(self):
        """The one qualitatively different signature among these
        impairments: a real nonlinearity generates energy outside the
        signal's own occupied bandwidth (odd-order intermodulation
        products), unlike the other three (pure rotation or a linear
        image term that stays within the existing band shape)."""
        n_cadu = 30

        def psd_db(iq, fs):
            spec = np.fft.fftshift(np.fft.fft(iq * np.hanning(len(iq))))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))
            p_db = 20 * np.log10(np.abs(spec) + 1e-12)
            return freqs, p_db - p_db.max()

        p_plain = _small_params(n_cadu=n_cadu)
        result = run_chain(p_plain)
        fs = result.sample_rate
        freqs, psd_plain = psd_db(result.iq, fs)

        p_pa = _small_params(n_cadu=n_cadu, pa_backoff_db=-6.0, pa_smoothness=3.0)
        _, psd_pa = psd_db(run_chain(p_pa).iq, fs)

        shoulder = (freqs >= 1.5e6) & (freqs < 2.5e6)  # just outside the ~1.2 MHz occupied half-bandwidth
        main_lobe = (freqs >= -1.0e6) & (freqs < 1.0e6)

        shoulder_rise_db = psd_pa[shoulder].mean() - psd_plain[shoulder].mean()
        main_change_db = psd_pa[main_lobe].mean() - psd_plain[main_lobe].mean()

        assert shoulder_rise_db > 15, f"expected clear spectral regrowth (>15 dB rise), got {shoulder_rise_db:.1f} dB"
        assert abs(main_change_db) < 1, f"main lobe should stay ~unchanged, moved {main_change_db:.1f} dB"

    def test_pa_never_increases_peak_amplitude(self):
        p_plain = _small_params(n_cadu=20)
        p_pa = _small_params(n_cadu=20, pa_backoff_db=-6.0, pa_smoothness=3.0)
        assert np.abs(run_chain(p_pa).iq).max() <= np.abs(run_chain(p_plain).iq).max()

    def test_export_chain_matches_run_chain_with_pa_nonlinearity(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline, "_EXPORT_BATCH_TARGET_BYTES", 1)  # forces batch_n_cadu == 1
        p = _small_params(n_cadu=6, pa_backoff_db=-3.0, pa_smoothness=2.5, pa_am_pm_deg_per_db=3.0)
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
