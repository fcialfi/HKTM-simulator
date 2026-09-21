"""Regression tests for the CCSDS TM Transfer Frame primary header builder
(transfer_frame.py) -- bit-packing correctness against CCSDS 132.0-B-3
Table 4-1, and the round-robin Virtual Channel scheduling helper."""

import pytest

from ccsds_chain.transfer_frame import (
    FIRST_HEADER_POINTER_NO_PACKET,
    PRIMARY_HEADER_BYTES,
    SEGMENT_LENGTH_ID_UNSEGMENTED,
    build_primary_header,
    vcid_schedule_counts,
)


def _unpack(header: bytes) -> dict:
    """Inverse of build_primary_header(), for asserting on individual
    fields rather than raw bytes."""
    assert len(header) == PRIMARY_HEADER_BYTES
    word0 = (header[0] << 8) | header[1]
    status = (header[4] << 8) | header[5]
    return {
        "tfvn": (word0 >> 14) & 0b11,
        "scid": (word0 >> 4) & 0x3FF,
        "vcid": (word0 >> 1) & 0b111,
        "ocf_flag": bool(word0 & 1),
        "mc_frame_count": header[2],
        "vc_frame_count": header[3],
        "secondary_header_flag": bool((status >> 15) & 1),
        "sync_flag": bool((status >> 14) & 1),
        "packet_order_flag": bool((status >> 13) & 1),
        "segment_length_id": (status >> 11) & 0b11,
        "first_header_pointer": status & 0x7FF,
    }


class TestBuildPrimaryHeader:
    def test_length_is_six_octets(self):
        assert len(build_primary_header(scid=0, vcid=0, mc_frame_count=0, vc_frame_count=0)) == PRIMARY_HEADER_BYTES

    def test_round_trips_every_field(self):
        header = build_primary_header(
            scid=0x123, vcid=5, mc_frame_count=0xAB, vc_frame_count=0xCD,
            ocf_flag=True, secondary_header_flag=True, sync_flag=True,
            packet_order_flag=True, segment_length_id=0b10, first_header_pointer=0x321,
        )
        fields = _unpack(header)
        assert fields == {
            "tfvn": 0, "scid": 0x123, "vcid": 5, "ocf_flag": True,
            "mc_frame_count": 0xAB, "vc_frame_count": 0xCD,
            "secondary_header_flag": True, "sync_flag": True, "packet_order_flag": True,
            "segment_length_id": 0b10, "first_header_pointer": 0x321,
        }

    def test_defaults_match_unsegmented_idle_data_convention(self):
        fields = _unpack(build_primary_header(scid=1, vcid=1, mc_frame_count=0, vc_frame_count=0))
        assert fields["ocf_flag"] is False
        assert fields["secondary_header_flag"] is False
        assert fields["sync_flag"] is False
        assert fields["packet_order_flag"] is False
        assert fields["segment_length_id"] == SEGMENT_LENGTH_ID_UNSEGMENTED
        assert fields["first_header_pointer"] == FIRST_HEADER_POINTER_NO_PACKET

    def test_tfvn_is_always_zero(self):
        # No parameter exists to set it otherwise -- TFVN=0 is fixed by the
        # standard for this version of the TM Transfer Frame.
        fields = _unpack(build_primary_header(scid=0, vcid=0, mc_frame_count=0, vc_frame_count=0))
        assert fields["tfvn"] == 0

    @pytest.mark.parametrize("field,bad_value", [
        ("scid", 1024), ("scid", -1),
        ("vcid", 8), ("vcid", -1),
        ("mc_frame_count", 256), ("mc_frame_count", -1),
        ("vc_frame_count", 256), ("vc_frame_count", -1),
        ("segment_length_id", 4), ("segment_length_id", -1),
        ("first_header_pointer", 2048), ("first_header_pointer", -1),
    ])
    def test_out_of_range_field_rejected(self, field, bad_value):
        kwargs = dict(scid=0, vcid=0, mc_frame_count=0, vc_frame_count=0)
        kwargs[field] = bad_value
        with pytest.raises(ValueError):
            build_primary_header(**kwargs)

    def test_field_boundaries_accepted(self):
        # Max value of every field simultaneously -- confirms no field
        # silently overflows into its neighbor's bits. TFVN is fixed at 0
        # (its own top 2 bits), so byte 0 tops out at 0x3F, not 0xFF, even
        # with every other field maxed.
        header = build_primary_header(
            scid=0x3FF, vcid=0b111, mc_frame_count=0xFF, vc_frame_count=0xFF,
            ocf_flag=True, secondary_header_flag=True, sync_flag=True,
            packet_order_flag=True, segment_length_id=0b11, first_header_pointer=0x7FF,
        )
        assert header == bytes([0x3F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])


class TestVcidScheduleCounts:
    def test_single_vcid_increments_every_frame(self):
        counts = [vcid_schedule_counts([3], i) for i in range(5)]
        assert counts == [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]

    def test_round_robin_assigns_in_pattern_order(self):
        pattern = [0, 1, 2]
        vcids = [vcid_schedule_counts(pattern, i)[0] for i in range(9)]
        assert vcids == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    def test_repeated_vcid_gets_proportionally_more_frames(self):
        pattern = [0, 0, 1]  # VC 0 should get twice VC 1's frames
        vcids = [vcid_schedule_counts(pattern, i)[0] for i in range(9)]
        assert vcids.count(0) == 6
        assert vcids.count(1) == 3

    def test_per_vc_frame_count_is_that_vcs_own_sequential_index(self):
        pattern = [0, 0, 1]
        counts = [vcid_schedule_counts(pattern, i) for i in range(9)]
        vc0_counts = [c for v, c in counts if v == 0]
        vc1_counts = [c for v, c in counts if v == 1]
        assert vc0_counts == list(range(6))
        assert vc1_counts == list(range(3))

    def test_pure_function_of_index_no_shared_state(self):
        # Same result whether queried in order or out of order -- confirms
        # there's no hidden mutable counter across calls (required for
        # run_chain()/export_chain() to agree regardless of batching).
        pattern = [0, 1, 0, 2]
        in_order = [vcid_schedule_counts(pattern, i) for i in range(20)]
        out_of_order = {i: vcid_schedule_counts(pattern, i) for i in reversed(range(20))}
        assert in_order == [out_of_order[i] for i in range(20)]
