"""CCSDS TM Transfer Frame primary header construction (CCSDS 132.0-B-3,
"TM Space Data Link Protocol", section 4.1.2) -- a different part of the
standard from the channel-coding layer (131.0-B-5) the rest of this project
implements.

Built to validate that a receiver correctly *identifies* and *routes* each
Transfer Frame to its assigned Virtual Channel: this tool otherwise only
ever generates a single, undifferentiated stream of synthetic payload
bytes (see README.md's Limitations), which is enough to exercise channel
coding/modulation but says nothing about a receiver's Virtual Channel
demultiplexing. Only the primary header is built (6 fixed octets); no
secondary header, and the data field beyond it stays whatever synthetic
payload the caller already generates (there is no real CCSDS Space Packet
structure inside it) -- entirely sufficient for a receiver to identify and
route frames by Virtual Channel, which is the one thing this module exists
to test.
"""

TFVN = 0  # TM Transfer Frame Version Number: fixed at 0 (binary 00) for this version of the standard (4.1.2.2.2)

PRIMARY_HEADER_BYTES = 6

# Segment Length ID (4.1.2.7.3): "11" is the fixed value used whenever the
# Transfer Frame Data Field is not segmented (i.e. holds Packets/Idle Data
# directly, never a Packet segment) -- the only case this module builds.
SEGMENT_LENGTH_ID_UNSEGMENTED = 0b11

# First Header Pointer (4.1.2.7.4) special value meaning "this Transfer
# Frame's Data Field contains no Packet header" (i.e. pure filler/Idle
# Data, exactly what this tool's synthetic payload is) -- used as the
# default rather than claiming a Packet starts at a specific offset that
# doesn't actually exist.
FIRST_HEADER_POINTER_NO_PACKET = 0b11111111110


def build_primary_header(
    scid: int,
    vcid: int,
    mc_frame_count: int,
    vc_frame_count: int,
    ocf_flag: bool = False,
    secondary_header_flag: bool = False,
    sync_flag: bool = False,
    packet_order_flag: bool = False,
    segment_length_id: int = SEGMENT_LENGTH_ID_UNSEGMENTED,
    first_header_pointer: int = FIRST_HEADER_POINTER_NO_PACKET,
) -> bytes:
    """Pack the fixed 6-octet TM Transfer Frame primary header (4.1.2),
    MSB-first, exactly as CCSDS 132.0-B-3 Table 4-1 lays it out:

    Octet 1-2 (16 bits): TFVN(2) | Spacecraft ID(10) | Virtual Channel ID(3) | OCF Flag(1)
    Octet 3:             Master Channel Frame Count(8)
    Octet 4:             Virtual Channel Frame Count(8)
    Octet 5-6 (16 bits): Secondary Header Flag(1) | Sync Flag(1) | Packet Order Flag(1)
                          | Segment Length ID(2) | First Header Pointer(11)
    """
    if not (0 <= scid < 2 ** 10):
        raise ValueError(f"scid must fit in 10 bits (0-1023), got {scid}")
    if not (0 <= vcid < 2 ** 3):
        raise ValueError(f"vcid must fit in 3 bits (0-7), got {vcid}")
    if not (0 <= mc_frame_count < 2 ** 8):
        raise ValueError(f"mc_frame_count must fit in 8 bits (0-255), got {mc_frame_count}")
    if not (0 <= vc_frame_count < 2 ** 8):
        raise ValueError(f"vc_frame_count must fit in 8 bits (0-255), got {vc_frame_count}")
    if not (0 <= segment_length_id < 2 ** 2):
        raise ValueError(f"segment_length_id must fit in 2 bits (0-3), got {segment_length_id}")
    if not (0 <= first_header_pointer < 2 ** 11):
        raise ValueError(f"first_header_pointer must fit in 11 bits (0-2047), got {first_header_pointer}")

    word0 = (TFVN & 0b11) << 14 | (scid & 0x3FF) << 4 | (vcid & 0b111) << 1 | (1 if ocf_flag else 0)
    status = (
        (1 if secondary_header_flag else 0) << 15
        | (1 if sync_flag else 0) << 14
        | (1 if packet_order_flag else 0) << 13
        | (segment_length_id & 0b11) << 11
        | (first_header_pointer & 0x7FF)
    )
    return bytes([
        (word0 >> 8) & 0xFF, word0 & 0xFF,
        mc_frame_count & 0xFF,
        vc_frame_count & 0xFF,
        (status >> 8) & 0xFF, status & 0xFF,
    ])


def vcid_schedule_counts(vcid_list: list[int], i: int) -> tuple[int, int]:
    """For round-robin Virtual Channel assignment `vcid_list[i % len(vcid_list)]`
    (frame index `i`, 0-based), return `(vcid, vc_frame_count)`: the VCID
    assigned to frame `i`, and how many frames already carried that same
    VCID *before* frame `i` (i.e. this frame's 0-based sequence number
    within its own Virtual Channel) -- a pure function of `i` and the
    pattern alone, so it needs no counter state carried between calls
    (every `_cadu_bits(p, prep, i)` call is independent, by design, so
    both the whole-array `run_chain()` and the batched `export_chain()`
    compute the exact same header for the same `i` either way).

    A VCID repeated in `vcid_list` is assigned proportionally more often
    -- e.g. [0, 0, 1] gives Virtual Channel 0 twice as many frames as
    Virtual Channel 1 -- rather than needing a separate weighting scheme.
    """
    period = len(vcid_list)
    pos = i % period
    vcid = vcid_list[pos]
    full_cycles = i // period
    occurrences_per_cycle = vcid_list.count(vcid)
    occurrences_before_this_cycle = vcid_list[:pos].count(vcid)
    vc_frame_count = full_cycles * occurrences_per_cycle + occurrences_before_this_cycle
    return vcid, vc_frame_count
