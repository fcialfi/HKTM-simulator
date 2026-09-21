"""Shared chain orchestration, used by both the CLI (generate_signal.py) and
the GUI (app.py) so the two never drift apart."""

from dataclasses import dataclass, field
from typing import Callable, Optional
import os
import tempfile
import time

import numpy as np

from .reed_solomon import rs_encode_interleaved
from .convolutional import conv_encode, ConvEncoder
from .scrambler import pn_sequence
from .mapping import BITS_PER_SYMBOL, bits_to_nrzl, map_symbols
from .pulse_shaping import rrc_taps, pulse_shape, RRCPulseShaper
from .transfer_frame import PRIMARY_HEADER_BYTES, build_primary_header, vcid_schedule_counts
from .impairments import apply_frequency_offset, apply_iq_imbalance, PhaseNoiseGenerator
from .utils import (bytes_to_bits, find_all_cadu_positions, find_cadu_sync, generate_payload,
                     pack_iq_interleaved, resample_iq)

ASM = bytes.fromhex("1ACFFC1D")


@dataclass
class ChainParams:
    modulation: str = "QPSK"
    encoding: str = "NRZ-L"
    bit_rate: int = 3_570_000
    symbol_rate: int = 1_785_000
    sps: int = 4
    rrc_alpha: float = 0.35
    rrc_span: int = 8
    rs_e: int = 16  # error-correction capability, in symbols: 8 or 16 (12.5)
    rs_n: int = 255
    interleave_depth: int = 5  # 1, 2, 3, 4, 5, or 8 (12.5)
    fec_rs: bool = True
    fec_conv: bool = True
    conv_rate: str = "1/2"  # 1/2, 2/3, 3/4, 5/6, or 7/8 (12.4)
    conv_invert_g2: bool = True  # only applies at rate 1/2 (3.4.1(5): punctured codes use no inversion)
    randomizer: str = "none"  # "none", "short" (255-bit, legacy), "long" (131071-bit, default per 12.3)
                               # -- also applies in "cadu" input_format (see below): a captured/decoded
                               # CADU source is often already de-scrambled, so this still controls
                               # whether the RS-coded region gets (re-)scrambled before transmission.
    asm: bytes = ASM
    input_format: str = "transfer_frame"  # "transfer_frame" (raw, gets RS/randomizer/ASM applied here)
                                           # or "cadu" (already ASM+RS-encoded: used as-is, no RS/ASM
                                           # re-encoding -- but `randomizer` above still applies to it)
    n_cadu: int = 100
    payload_source: str | None = None
    payload_bytes: bytes | None = None
    seed: int = 42
    vcid_list: list[int] | None = None  # None = no synthetic Transfer Frame header (default,
                                         # unchanged behavior). A list of Virtual Channel IDs
                                         # (0-7) round-robined across generated frames -- builds
                                         # a real 6-octet CCSDS TM primary header (132.0-B-3
                                         # 4.1.2) with the per-frame VCID from this list, so a
                                         # receiver's Virtual Channel identification/routing can
                                         # be validated against a known assignment. Repeat a VCID
                                         # to give it proportionally more frames (e.g. [0, 0, 1]
                                         # gives VC 0 twice VC 1's share). Only applies to
                                         # "transfer_frame" input with synthetic (pseudo-random)
                                         # payload -- never to an uploaded real Transfer Frame
                                         # file, which already carries its own real header.
    spacecraft_id: int = 0x123  # 10-bit SCID (0-1023) used in the synthetic primary header
                                 # above, when vcid_list is set; otherwise unused.
    corrupt_rs_symbols: int = 0  # 0 = disabled (default). Otherwise, the exact number of RS
                                  # symbols (bytes) to flip within one interleaved codeword
                                  # (see corrupt_codeword_index) of every targeted CADU (see
                                  # corrupt_cadu_indices/corrupt_vc below) -- deterministic,
                                  # exact-per-codeword-count error injection, unlike a
                                  # replayer's AWGN (which characterizes statistical BER/FER
                                  # vs Eb/N0 well, but can't guarantee hitting a precise error
                                  # count in one specific codeword). Meant for boundary-testing
                                  # a receiver's RS decoder against its declared correction
                                  # capability E (rs_e): exactly E symbol errors must still
                                  # decode perfectly; E+1 must fail (or be flagged), never
                                  # silently miscorrect. Needs an actual RS-coded region to
                                  # corrupt: requires fec_rs=True, or input_format="cadu"
                                  # (already RS-coded by construction).
    corrupt_codeword_index: int = 0  # which of the `interleave_depth` interleaved RS
                                      # codewords to target (0-based); only meaningful when
                                      # corrupt_rs_symbols > 0.
    corrupt_cadu_indices: list[int] | None = None  # explicit 0-based CADU indices to corrupt.
                                                     # Combines with corrupt_vc (either match
                                                     # corrupts that CADU); at least one of the
                                                     # two must be set when corrupt_rs_symbols > 0.
    corrupt_vc: int | None = None  # corrupt every CADU assigned to this Virtual Channel ID
                                    # (requires vcid_list) -- e.g. to verify a receiver's
                                    # per-VC FER accounting attributes errors to the right
                                    # channel and leaves other VCs' counts untouched.
    corrupt_seed: int = 777  # seed for choosing which symbol positions (and XOR values) get
                              # flipped -- deterministic and reproducible across runs, and
                              # combined with each CADU's own index (never carried as mutable
                              # state), so run_chain() and export_chain() (batched/streaming)
                              # always compute the exact same corruption for the same CADU.
    freq_offset_hz: float = 0.0  # 0 = disabled. Constant residual LO frequency offset applied
                                  # to the pulse-shaped IQ (ccsds_chain/impairments.py) -- for
                                  # validating a receiver's carrier-recovery loop actually
                                  # acquires/tracks a real (imperfect) transmitter's LO, not
                                  # just a perfectly on-frequency signal.
    iq_gain_imbalance_db: float = 0.0  # 0 = disabled. IQ modulator gain mismatch between the I
                                        # and Q branches, in dB -- produces a mirror-image tone
                                        # (see impairments.apply_iq_imbalance).
    iq_phase_imbalance_deg: float = 0.0  # 0 = disabled. IQ modulator phase deviation from ideal
                                          # 90 degree I/Q separation, in degrees -- combines with
                                          # iq_gain_imbalance_db in the same mirror-image model.
    phase_noise_linewidth_hz: float = 0.0  # 0 = disabled. Free-running-oscillator single-
                                            # sideband 3 dB linewidth, in Hz -- generates Wiener
                                            # (random-walk) phase noise on the IQ (see
                                            # impairments.PhaseNoiseGenerator), for validating a
                                            # receiver's tolerance to constellation smearing from
                                            # a real (non-ideal) transmitter LO.
    impairment_seed: int = 2718  # seed for the phase noise random walk -- deterministic and
                                  # reproducible across runs; independent of corrupt_seed above.

    @property
    def rs_k(self) -> int:
        return self.rs_n - 2 * self.rs_e


@dataclass
class ChainResult:
    iq: np.ndarray
    symbols: np.ndarray
    frame_bytes: int
    cadu_bytes: int
    sample_rate: float
    elapsed: float
    meta: dict = field(default_factory=dict)


# Rough, fixed weights for run_chain()'s progress_callback, from benchmarking
# a representative run: the pure-Python per-CADU RS/ASM loop and the RRC
# pulse shaping dominate the runtime, so those two get fine-grained
# incremental progress; convolutional encoding + QPSK mapping are fast
# enough (vectorized numpy) to just report as a single jump between them.
_PROGRESS_RS_ASM_WEIGHT = 0.55
_PROGRESS_CONV_MAP_WEIGHT = 0.12
_PROGRESS_PULSE_WEIGHT = 1.0 - _PROGRESS_RS_ASM_WEIGHT - _PROGRESS_CONV_MAP_WEIGHT


@dataclass
class _PayloadPrep:
    """Shared setup result from `_prepare_payload()`, consumed by both
    `run_chain()` (whole-array) and `export_chain()` (batched/streaming)."""
    payload: bytes
    unit_bytes: int
    frame_bytes: int
    rs_block_bytes: int
    is_cadu_input: bool
    has_real_source: bool
    n_synced_cadu: int
    sync_skipped_bytes: int
    asm_bits: np.ndarray
    pn: Optional[np.ndarray]
    # Set only for CADU input from a real source: the byte offset of every
    # real ASM found in `raw_synced` that has at least `unit_bytes` bytes
    # remaining after it (CADU i is always exactly `raw_synced[cadu_
    # positions[i] : cadu_positions[i] + unit_bytes]` -- the configured
    # RS-E/interleave-depth length, never measured from ASM spacing: a real
    # capture wraps each CADU in instrument-specific framing (e.g. a fixed
    # header + CADU + postamble per record) that is not part of the CADU
    # and must never be fed into convolutional coding as if it were. The
    # ASM is only used to find where each CADU *starts* -- see
    # `_prepare_payload()`.
    cadu_positions: Optional[list] = None
    raw_synced: Optional[bytes] = None
    # Byte span between the end of one CADU and the start of the next real
    # ASM (i.e. whatever per-record framing the source wraps CADUs in),
    # for meta/UI reporting only -- not used by any encoding logic.
    cadu_wrapper_min_bytes: Optional[int] = None
    cadu_wrapper_max_bytes: Optional[int] = None
    # Precomputed once from ChainParams.corrupt_cadu_indices (a plain list,
    # the convenient public/CLI/GUI shape) so _cadu_bits()'s per-CADU
    # membership check -- run for every CADU, including ones never
    # corrupted -- is O(1) instead of an O(len(list)) scan each time.
    corrupt_cadu_index_set: Optional[frozenset] = None


def _validate_chain_params(p: ChainParams) -> None:
    if p.modulation not in BITS_PER_SYMBOL:
        raise NotImplementedError(f"modulation {p.modulation!r} not implemented (expected 'QPSK' or 'BPSK')")
    if p.encoding != "NRZ-L":
        raise NotImplementedError(f"encoding {p.encoding!r} not implemented (baseline: NRZ-L)")
    if p.input_format not in ("transfer_frame", "cadu"):
        raise NotImplementedError(f"input_format {p.input_format!r} not implemented (expected 'transfer_frame' or 'cadu')")
    if p.vcid_list is not None:
        if p.input_format != "transfer_frame":
            raise ValueError("vcid_list (synthetic Transfer Frame headers) only applies to input_format='transfer_frame'")
        if p.payload_source is not None or p.payload_bytes is not None:
            raise ValueError(
                "vcid_list (synthetic Transfer Frame headers) can't be combined with a real "
                "payload file/bytes: it would overwrite the first 6 bytes of your real data "
                "with a synthetic header. Use it only with pseudo-random payload."
            )
        if len(p.vcid_list) == 0:
            raise ValueError("vcid_list must not be empty")
        for vcid in p.vcid_list:
            if not (0 <= vcid < 8):
                raise ValueError(f"vcid_list values must be 3-bit Virtual Channel IDs (0-7), got {vcid}")
        if not (0 <= p.spacecraft_id < 1024):
            raise ValueError(f"spacecraft_id must fit in 10 bits (0-1023), got {p.spacecraft_id}")
    if p.corrupt_rs_symbols > 0:
        if not p.fec_rs and p.input_format != "cadu":
            raise ValueError(
                "corrupt_rs_symbols needs an actual RS-coded region to corrupt: enable "
                "fec_rs, or use input_format='cadu' (already RS-coded by construction)"
            )
        if not (0 <= p.corrupt_rs_symbols <= p.rs_n):
            raise ValueError(f"corrupt_rs_symbols must be between 0 and rs_n={p.rs_n}, got {p.corrupt_rs_symbols}")
        if not (0 <= p.corrupt_codeword_index < p.interleave_depth):
            raise ValueError(
                f"corrupt_codeword_index must be a valid interleaved codeword index "
                f"(0-{p.interleave_depth - 1}), got {p.corrupt_codeword_index}"
            )
        if p.corrupt_cadu_indices is None and p.corrupt_vc is None:
            raise ValueError(
                "corrupt_rs_symbols requires corrupt_cadu_indices and/or corrupt_vc, to say "
                "which CADUs to corrupt"
            )
        if p.corrupt_vc is not None:
            if p.vcid_list is None:
                raise ValueError("corrupt_vc requires vcid_list to be set (it targets CADUs by their assigned Virtual Channel)")
            if not (0 <= p.corrupt_vc < 8):
                raise ValueError(f"corrupt_vc must be a 3-bit Virtual Channel ID (0-7), got {p.corrupt_vc}")
        if p.corrupt_cadu_indices is not None:
            for idx in p.corrupt_cadu_indices:
                if idx < 0:
                    raise ValueError(f"corrupt_cadu_indices must be non-negative, got {idx}")
    if p.phase_noise_linewidth_hz < 0:
        raise ValueError(f"phase_noise_linewidth_hz must be non-negative, got {p.phase_noise_linewidth_hz}")


def _unit_bytes(p: ChainParams) -> tuple:
    """Cheap, data-independent computation of (frame_bytes, unit_bytes,
    rs_block_bytes, is_cadu_input) -- split out from `_prepare_payload()`
    so `export_chain()` can size/guard a batched run (and refuse an
    infeasible resample) *before* generating or reading any actual
    payload, which for a pathologically large `n_cadu` can itself be a
    multi-GB allocation."""
    frame_bytes = p.rs_k * p.interleave_depth
    is_cadu_input = p.input_format == "cadu"
    rs_block_bytes = p.rs_n * p.interleave_depth if (p.fec_rs or is_cadu_input) else frame_bytes
    unit_bytes = len(p.asm) + rs_block_bytes if is_cadu_input else frame_bytes
    return frame_bytes, unit_bytes, rs_block_bytes, is_cadu_input


def _prepare_payload(p: ChainParams) -> _PayloadPrep:
    """Validate params and build the concatenated payload for `p.n_cadu`
    CADUs, synchronizing to the first genuine CADU boundary first when
    `p.input_format == "cadu"` and a real source is given.

    Build each CADU = ASM + RS-encoded codeblock, per CCSDS 131.0-B-3's
    "Overall Structure of Channel Coding": RS encode -> pseudo-randomize
    (the codeblock only, never the ASM) -> attach ASM (this forms the
    CADU) -> convolutionally encode the *stream of CADUs*. So the ASM IS
    convolutionally coded along with the rest: a convolutionally-coded
    stream is continuously Viterbi-decoded at the receiver (no framing
    needed to decode it), and frame sync is recovered by correlating for
    the literal ASM pattern in the *decoded* bitstream, not before
    decoding. The pseudo-randomizer is reinitialized for each CADU's
    codeblock (not carried over between CADUs) -- since every CADU's
    codeblock is the same length, it is always reinitialized to the same
    state, so the PN sequence is generated once and reused rather than
    regenerated (an O(bits) pure-Python LFSR) on every iteration.

    When `input_format == "cadu"`, the caller is supplying data that is
    *already* a stream of complete CADUs (ASM + RS-encoded) -- e.g.
    captured or previously generated CADUs, not raw Transfer Frames. RS
    encoding and the ASM prepend are always skipped in this mode: applying
    them again would double-encode and prefix a second ASM in front of data
    that already has one.

    Whether the pseudo-randomizer is *also* skipped depends on `p.
    randomizer`, exactly as in `transfer_frame` mode -- it is not assumed
    to be a no-op just because the input is already-formed CADUs. A
    genuinely raw capture (bytes as they would appear on the air right
    before convolutional coding) already carries the scrambling, and
    `randomizer` should be left at "none". But a captured/decoded CADU
    source (e.g. exported from an instrument that does its own frame sync
    and hands back clean bytes) is commonly *de*-scrambled as part of that
    decoding -- confirmed against a real capture, whose CADU payload
    contained a plainly readable ASCII string (a firmware version tag)
    that a genuinely scrambled/RS-coded byte stream could never produce.
    Retransmitting such already-descrambled bytes without re-scrambling
    them produces a signal a real, CCSDS-conformant receiver's descrambler
    will XOR against the PN sequence anyway, corrupting every frame after
    the ASM -- `randomizer` must then be set to match what the original
    transmission used, so the RS-coded region gets scrambled again here.

    A real/captured CADU file is also not guaranteed to be a bare
    back-to-back stream of CADUs: a capture instrument commonly wraps each
    CADU in its own record framing (e.g. RF-Catcher/TestTree's CRT format:
    a fixed-size header, the CADU, then a short postamble, repeated per
    record) -- confirmed against a real captured pass, where treating the
    distance between consecutive ASMs as the CADU length (as an earlier
    version of this code did) silently folded 69 bytes of that per-record
    header+postamble into what was fed to the convolutional encoder as if
    it were coded CADU data. The CADU itself is always exactly `unit_bytes`
    long -- the length the configured RS-E/interleave-depth predicts, never
    measured from ASM spacing -- and the ASM is used only to find where
    each CADU *starts* (also handling leading junk before the first one,
    and an excerpt starting mid-stream): CADU i is `raw_synced[cadu_
    positions[i] : cadu_positions[i] + unit_bytes]`, and whatever bytes lie
    between that and the next real ASM (the instrument's own per-record
    framing, if any) are simply skipped.
    """
    _validate_chain_params(p)
    frame_bytes, unit_bytes, rs_block_bytes, is_cadu_input = _unit_bytes(p)
    asm_bits = bytes_to_bits(p.asm)

    has_real_source = p.payload_source is not None or p.payload_bytes is not None
    sync_skipped_bytes = 0
    n_synced_cadu = 0
    cadu_positions = None
    raw_synced = None
    cadu_wrapper_min_bytes = None
    cadu_wrapper_max_bytes = None
    if is_cadu_input and has_real_source:
        # Read the raw source in full (not just the n_cadu*unit_bytes we
        # ultimately need) so there's enough of it to search past any
        # leading junk, locate every real ASM, and still have n_cadu whole
        # CADUs after the first one.
        raw = p.payload_bytes if p.payload_bytes is not None else open(p.payload_source, "rb").read()
        sync_skipped_bytes = find_cadu_sync(raw, p.asm)
        raw_synced = raw[sync_skipped_bytes:]

        positions = find_all_cadu_positions(raw_synced, p.asm)
        for j in range(len(positions) - 1):
            gap = positions[j + 1] - positions[j]
            if gap < unit_bytes:
                raise ValueError(
                    f"CADU length from the configured RS error correction E={p.rs_e} and "
                    f"interleave depth I={p.interleave_depth} is {unit_bytes} bytes, but the "
                    f"next ASM in the source is only {gap} bytes after this one (byte offset "
                    f"{sync_skipped_bytes + positions[j]} in the source). Check that they match "
                    "how these CADUs were actually built."
                )
        if len(positions) >= 2:
            wrapper_gaps = [positions[j + 1] - (positions[j] + unit_bytes) for j in range(len(positions) - 1)]
            cadu_wrapper_min_bytes = min(wrapper_gaps)
            cadu_wrapper_max_bytes = max(wrapper_gaps)

        # Only positions with a full CADU's worth of bytes remaining are
        # usable -- e.g. the very last ASM in a file that ends mid-CADU.
        cadu_positions = [pos for pos in positions if pos + unit_bytes <= len(raw_synced)]
        n_synced_cadu = len(cadu_positions)

        payload = b""  # unused in this branch -- _cadu_bits() reads cadu_positions/raw_synced instead
    else:
        payload = generate_payload(p.n_cadu, unit_bytes, p.payload_source, p.payload_bytes, p.seed)

    # Applies in both input_format modes -- see the "cadu" branch note above
    # for why CADU input isn't assumed to already be scrambled.
    pn = pn_sequence(rs_block_bytes * 8, p.randomizer) if p.randomizer != "none" else None

    corrupt_cadu_index_set = frozenset(p.corrupt_cadu_indices) if p.corrupt_cadu_indices is not None else None

    return _PayloadPrep(payload=payload, unit_bytes=unit_bytes, frame_bytes=frame_bytes,
                         rs_block_bytes=rs_block_bytes, is_cadu_input=is_cadu_input,
                         has_real_source=has_real_source, n_synced_cadu=n_synced_cadu,
                         sync_skipped_bytes=sync_skipped_bytes, asm_bits=asm_bits, pn=pn,
                         corrupt_cadu_index_set=corrupt_cadu_index_set,
                         cadu_positions=cadu_positions, raw_synced=raw_synced,
                         cadu_wrapper_min_bytes=cadu_wrapper_min_bytes,
                         cadu_wrapper_max_bytes=cadu_wrapper_max_bytes)


def _should_corrupt(p: ChainParams, prep: _PayloadPrep, i: int, vcid: Optional[int]) -> bool:
    """Whether CADU `i` (assigned to Virtual Channel `vcid`, or None outside
    "transfer_frame" + vcid_list) is one of the CADUs ChainParams.
    corrupt_rs_symbols targets -- see corrupt_cadu_indices/corrupt_vc."""
    if p.corrupt_rs_symbols <= 0:
        return False
    if prep.corrupt_cadu_index_set is not None and i in prep.corrupt_cadu_index_set:
        return True
    return p.corrupt_vc is not None and vcid == p.corrupt_vc


def _corrupt_rs_region(rs_region: bytes, p: ChainParams, i: int) -> bytes:
    """Flip exactly `p.corrupt_rs_symbols` distinct symbols of interleaved
    codeword `p.corrupt_codeword_index` within `rs_region` (the RS-coded
    byte block: rs_n*interleave_depth bytes, substream j = rs_region[j::
    interleave_depth], same layout reed_solomon.rs_encode_interleaved()
    produces/consumes) -- an exact, reproducible per-codeword error count,
    for boundary-testing a receiver's RS decoder (see ChainParams.
    corrupt_rs_symbols).

    Deterministic per CADU index `i` and ChainParams.corrupt_seed alone (no
    state carried between calls), so run_chain() and export_chain()
    (batched/streaming) always compute the same corruption for the same
    CADU regardless of batching.

    Applied here, before scrambling/ASM, rather than on the final
    transmitted bits: XOR corruption commutes with the pseudo-randomizer's
    own XOR scrambling (each is just XOR-ing a fixed mask over the same
    bytes), so the two orderings produce byte-for-byte identical output --
    this is simply the more convenient place to work in exact RS symbol
    positions.
    """
    depth = p.interleave_depth
    rng = np.random.default_rng((p.corrupt_seed, i))
    codeword = np.frombuffer(rs_region, dtype=np.uint8)[p.corrupt_codeword_index::depth].copy()
    positions = rng.choice(len(codeword), size=p.corrupt_rs_symbols, replace=False)
    masks = rng.integers(1, 256, size=p.corrupt_rs_symbols)  # never 0: guarantees each flipped symbol actually changes
    codeword[positions] ^= masks.astype(np.uint8)
    out = bytearray(rs_region)
    out[p.corrupt_codeword_index::depth] = codeword.tobytes()
    return bytes(out)


def _cadu_bits(p: ChainParams, prep: _PayloadPrep, i: int) -> np.ndarray:
    """Bits for CADU index `i`: RS-encode + ASM-prepend for a Transfer Frame
    input, or straight pass-through + sync validation for a CADU input (see
    `_prepare_payload()`)."""
    if prep.is_cadu_input:
        if prep.has_real_source:
            # Real CADU start straight from the data (see
            # `_prepare_payload()`): always genuinely starts with the ASM
            # by construction, no separate validation needed. Always
            # exactly `unit_bytes` long -- the configured RS/interleave
            # length -- regardless of whatever per-record framing the
            # source wraps it in; that framing is simply skipped.
            if i < prep.n_synced_cadu:
                start = prep.cadu_positions[i]
                cadu = prep.raw_synced[start:start + prep.unit_bytes]
            else:
                cadu = bytes(prep.unit_bytes)  # zero-padded tail, beyond what the source actually has
        else:
            # Pseudo-random test payload (no real source): fixed-stride,
            # exactly as generated.
            cadu = prep.payload[i * prep.unit_bytes:(i + 1) * prep.unit_bytes]
        if _should_corrupt(p, prep, i, vcid=None):
            asm_len = len(p.asm)
            cadu = cadu[:asm_len] + _corrupt_rs_region(cadu[asm_len:], p, i)
        bits = bytes_to_bits(cadu)
        if prep.pn is not None:
            # Scramble only the RS-coded region, never the ASM -- see
            # `_prepare_payload()`'s note on why CADU input isn't assumed
            # to already be scrambled.
            asm_bit_len = len(p.asm) * 8
            bits[asm_bit_len:] = np.bitwise_xor(bits[asm_bit_len:], prep.pn)
        return bits
    frame = prep.payload[i * prep.frame_bytes:(i + 1) * prep.frame_bytes]
    vcid = None
    if p.vcid_list is not None:
        # Overwrite the first PRIMARY_HEADER_BYTES of the frame with a real
        # CCSDS TM primary header carrying this frame's assigned VCID (see
        # ChainParams.vcid_list) -- the rest of the synthetic payload is
        # left as-is (there's no real Space Packet structure to preserve
        # inside it; see transfer_frame.py's module docstring). Guaranteed
        # not to run on an uploaded real Transfer Frame file/bytes --
        # _validate_chain_params() rejects that combination up front.
        vcid, vc_frame_count = vcid_schedule_counts(p.vcid_list, i)
        header = build_primary_header(
            scid=p.spacecraft_id, vcid=vcid,
            mc_frame_count=i % 256, vc_frame_count=vc_frame_count % 256,
        )
        frame = header + frame[PRIMARY_HEADER_BYTES:]
    rs_block = (rs_encode_interleaved(frame, p.rs_k, p.rs_n, p.interleave_depth)
                if p.fec_rs else frame)
    if _should_corrupt(p, prep, i, vcid):
        rs_block = _corrupt_rs_region(rs_block, p, i)
    rs_bits = bytes_to_bits(rs_block)
    if prep.pn is not None:
        rs_bits = np.bitwise_xor(rs_bits, prep.pn)
    return np.concatenate([prep.asm_bits, rs_bits])


def _chain_meta(p: ChainParams, prep: _PayloadPrep, sample_rate: float) -> dict:
    return {
        "modulation": p.modulation,
        "encoding": p.encoding,
        "bit_rate": p.bit_rate,
        "symbol_rate": p.symbol_rate,
        "sample_rate": sample_rate,
        "samples_per_symbol": p.sps,
        "rrc_alpha": p.rrc_alpha,
        "rrc_span": p.rrc_span,
        "input_format": p.input_format,
        "cadu_sync_skipped_bytes": prep.sync_skipped_bytes if prep.is_cadu_input else None,
        "cadu_wrapper_bytes_min": prep.cadu_wrapper_min_bytes,
        "cadu_wrapper_bytes_max": prep.cadu_wrapper_max_bytes,
        "fec_rs": p.fec_rs and not prep.is_cadu_input,
        "fec_conv": p.fec_conv,
        "conv_rate": p.conv_rate,
        "conv_invert_g2": p.conv_invert_g2 and p.conv_rate == "1/2",
        "randomizer": p.randomizer,
        "rs_e": p.rs_e,
        "rs_k": p.rs_k,
        "rs_n": p.rs_n,
        "interleave_depth": p.interleave_depth,
        "n_cadu": p.n_cadu,
        "vcid_list": p.vcid_list,
        "spacecraft_id": p.spacecraft_id if p.vcid_list is not None else None,
        "corrupt_rs_symbols": p.corrupt_rs_symbols if p.corrupt_rs_symbols > 0 else None,
        "corrupt_codeword_index": p.corrupt_codeword_index if p.corrupt_rs_symbols > 0 else None,
        "corrupt_cadu_indices": p.corrupt_cadu_indices if p.corrupt_rs_symbols > 0 else None,
        "corrupt_vc": p.corrupt_vc if p.corrupt_rs_symbols > 0 else None,
        "freq_offset_hz": p.freq_offset_hz if p.freq_offset_hz != 0 else None,
        "iq_gain_imbalance_db": p.iq_gain_imbalance_db if p.iq_gain_imbalance_db != 0 else None,
        "iq_phase_imbalance_deg": p.iq_phase_imbalance_deg if p.iq_phase_imbalance_deg != 0 else None,
        "phase_noise_linewidth_hz": p.phase_noise_linewidth_hz if p.phase_noise_linewidth_hz > 0 else None,
    }


def _apply_impairments(
    iq: np.ndarray, p: ChainParams, sample_rate: float, start_sample: int,
    phase_noise_gen: Optional[PhaseNoiseGenerator],
) -> np.ndarray:
    """Applies ChainParams' transmitter impairments to a chunk of
    pulse-shaped IQ: LO frequency offset and phase noise first, then IQ
    modulator gain/phase imbalance last. Skips a stage entirely when its
    parameter is at the "disabled" value, so a run with none of them
    configured is bit-for-bit identical to before this feature existed.
    `phase_noise_gen` is created once by the caller (None when disabled)
    and shared across every chunk of one export, so its random walk and
    PRNG state carry correctly across export_chain()'s batches;
    `start_sample` is this chunk's absolute sample offset in the whole
    signal, for an exact (state-free) frequency-offset phase ramp
    regardless of batching.

    This chain's 0 Hz *is* the transmitter's intended RF center frequency
    (there's no separate physical upconversion stage in software -- see
    impairments.py's module docstring): apply_iq_imbalance()'s mirror-
    image term reflects about that 0 Hz, so it only appears as a visible,
    separate image in the spectrum when the wanted signal itself is
    *not* centered at 0 Hz. Doing frequency offset (which displaces the
    wanted signal away from 0 Hz) before imbalance (which then mirrors
    that displaced signal to the opposite side of 0 Hz) reproduces the
    classic, visible image-frequency picture; the reverse order mirrors
    a still-symmetric-about-0-Hz signal onto itself, which for this
    project's random-data QPSK (whose own spectrum is already symmetric
    about 0 Hz when freq_offset_hz is 0) folds invisibly back into the
    same band -- confirmed empirically: swapping this order took a
    3 dB/15 deg imbalance's image from indistinguishable from the noise
    floor to landing within 0.1 dB of the closed-form image rejection
    ratio predicted for a pure tone."""
    if p.freq_offset_hz != 0:
        iq = apply_frequency_offset(iq, p.freq_offset_hz, sample_rate, start_sample=start_sample)
    if phase_noise_gen is not None:
        iq = phase_noise_gen.apply(iq)
    if p.iq_gain_imbalance_db != 0 or p.iq_phase_imbalance_deg != 0:
        iq = apply_iq_imbalance(iq, p.iq_gain_imbalance_db, p.iq_phase_imbalance_deg)
    return iq


def run_chain(
    p: ChainParams,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ChainResult:
    """Run the full chain in memory and return the whole result. Used for
    the interactive live preview (deliberately kept to a small n_cadu) --
    for a large/whole-file export, use `export_chain()` instead, which
    processes CADUs in bounded-memory batches and streams straight to disk
    regardless of `n_cadu`.

    `progress_callback(fraction, message)`, if given, is called
    periodically with fraction in [0, 1]; it is not called at all for a
    cheap/small run, so callers can pass it unconditionally."""
    t0 = time.time()
    prep = _prepare_payload(p)

    # Throttled so a large n_cadu doesn't spend more time updating a UI
    # widget than actually encoding (a plain Streamlit progress bar update
    # is not free at tens/hundreds of thousands of calls).
    report_every = max(1, p.n_cadu // 200)
    cadu_bit_chunks = []
    for i in range(p.n_cadu):
        cadu_bit_chunks.append(_cadu_bits(p, prep, i))
        if progress_callback is not None and ((i + 1) % report_every == 0 or i + 1 == p.n_cadu):
            progress_callback(
                _PROGRESS_RS_ASM_WEIGHT * (i + 1) / p.n_cadu,
                f"{'Loading' if prep.is_cadu_input else 'Encoding'} CADU {i + 1:,}/{p.n_cadu:,}...".replace(",", " "),
            )
    cadu_bytes = len(p.asm) + prep.rs_block_bytes

    bits = np.concatenate(cadu_bit_chunks)
    # Convolutional coding stays continuous across the whole burst (the
    # register is never reset per CADU), matching a physical coder running
    # continuously across the channel.
    coded_bits = (conv_encode(bits, invert_g2=p.conv_invert_g2, rate=p.conv_rate)
                  if p.fec_conv else bits)

    bipolar = bits_to_nrzl(coded_bits)
    symbols = map_symbols(bipolar, p.modulation)

    if progress_callback is not None:
        progress_callback(_PROGRESS_RS_ASM_WEIGHT + _PROGRESS_CONV_MAP_WEIGHT, "Pulse shaping (RRC filter)...")

    taps = rrc_taps(p.rrc_alpha, p.rrc_span, p.sps)
    pulse_progress = (
        (lambda frac: progress_callback(
            _PROGRESS_RS_ASM_WEIGHT + _PROGRESS_CONV_MAP_WEIGHT + _PROGRESS_PULSE_WEIGHT * frac,
            "Pulse shaping (RRC filter)...",
        ))
        if progress_callback is not None else None
    )
    iq = pulse_shape(symbols, p.sps, taps, progress_callback=pulse_progress)

    sample_rate = p.symbol_rate * p.sps
    phase_noise_gen = (
        PhaseNoiseGenerator(p.phase_noise_linewidth_hz, sample_rate, p.impairment_seed)
        if p.phase_noise_linewidth_hz > 0 else None
    )
    iq = _apply_impairments(iq, p, sample_rate, start_sample=0, phase_noise_gen=phase_noise_gen)

    if progress_callback is not None:
        progress_callback(1.0, "Done")

    elapsed = time.time() - t0

    meta = _chain_meta(p, prep, sample_rate)
    meta["n_symbols"] = len(symbols)
    meta["n_iq_samples"] = len(iq)

    return ChainResult(iq=iq, symbols=symbols, frame_bytes=prep.frame_bytes, cadu_bytes=cadu_bytes,
                        sample_rate=sample_rate, elapsed=elapsed, meta=meta)


# Target size, in bytes, of one export_chain() batch's native-rate complex128
# IQ -- chosen so peak memory during export stays at this rough order of
# magnitude regardless of n_cadu (i.e. export duration), instead of growing
# with it as the old whole-array run_chain()-based export did.
_EXPORT_BATCH_TARGET_BYTES = 64 * 1024 * 1024

# Above this much native-rate IQ, resampling (scipy.signal.resample_poly,
# which needs its whole input -- and a comparably sized output -- in memory
# at once, unlike every other stage of export_chain()) is refused upfront
# rather than attempted, so a combination that won't fit fails fast with an
# actionable message instead of after a long, otherwise-bounded-memory run.
_RESAMPLE_MEMORY_GUARD_BYTES = 2 * 1024 ** 3

# Samples read back per iteration when re-streaming the native-rate scratch
# file for normalization/packing (not a memory-safety bound by itself --
# _EXPORT_BATCH_TARGET_BYTES already sizes generation batches -- just a
# reasonable I/O granularity for the second pass).
_SCRATCH_READ_CHUNK_SAMPLES = 4_000_000


def _native_bytes_per_cadu(p: ChainParams, unit_bytes: int) -> float:
    """Rough estimate of native-rate complex128 IQ bytes produced per CADU,
    for sizing export_chain()'s batches and its resample memory guard. Not
    exact for punctured convolutional rates (puncture-pattern boundary
    effects aren't accounted for) -- only needs to be close enough to size
    a bounded-memory batch or an upfront feasibility check, not to predict
    an exact byte count."""
    bits_per_cadu = unit_bytes * 8
    conv_expansion = 1.0
    if p.fec_conv:
        if p.conv_rate == "1/2":
            conv_expansion = 2.0
        else:
            num, den = p.conv_rate.split("/")
            conv_expansion = int(den) / int(num)
    symbols_per_cadu = bits_per_cadu * conv_expansion / BITS_PER_SYMBOL[p.modulation]
    return symbols_per_cadu * p.sps * 16  # complex128, 16 bytes/sample


def export_chain(
    p: ChainParams,
    output_path: str,
    output_dtype: str = "float32",
    peak: float = 0.9,
    target_fs: Optional[float] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Generate the full chain's output and write it directly to
    `output_path`, in memory bounded by a batch of CADUs rather than by
    `p.n_cadu` (i.e. export duration) -- unlike `run_chain()`, which
    materializes the whole bits/coded-bits/symbols/IQ arrays for the entire
    export at once and is only meant for the small, interactive live
    preview. Use this for the actual export (CLI output file, or a GUI
    "Generate export file" click), of any size.

    CADUs are processed in batches (RS/ASM or CADU pass-through, exactly as
    in `run_chain()`/`_cadu_bits()`); the convolutional encoder
    (`ConvEncoder`) and RRC pulse shaper (`RRCPulseShaper`) carry their
    state across batches so the result is bit-for-bit identical to a single
    non-batched run. Each batch's native-rate IQ is appended, as raw
    complex128, to a temporary scratch file, while tracking the running peak
    |I|/|Q| across all of it -- so peak normalization still uses the true,
    exact global peak (matching `normalize_peak()`) even though the signal
    is never held in memory all at once. Only once that true peak is known
    is the scratch file re-read in chunks, scaled, dtype-converted and
    packed straight into `output_path` -- equivalent to
    `pack_iq_interleaved(normalize_peak(run_chain(p).iq, peak), output_dtype)`
    without ever holding the whole thing in memory. For `int16` output this
    is exactly byte-for-byte identical; for `float32`, batching changes the
    order convolution terms are summed in, which can (rarely) flip the
    last bit of a float32 value by float64 rounding noise around 1e-16 --
    around nine orders of magnitude below int16's own quantization step,
    so it never affects the signal in any way that matters.

    Optional resampling (`target_fs`) is the one stage that is *not*
    streamed -- `scipy.signal.resample_poly` needs its whole input (and
    produces a comparably sized whole output) in memory at once -- so it is
    checked upfront against `_RESAMPLE_MEMORY_GUARD_BYTES` and refused
    with a clear message rather than attempted and left to fail deep into
    a long run.

    Returns a dict with `cadu_bytes`, `elapsed` and `meta` (same shape as
    `ChainResult.meta`, plus the output-file fields `generate_signal.py`/
    `app.py` used to add by hand after calling `run_chain()`).
    """
    if output_dtype not in ("float32", "int16"):
        raise ValueError(f"unsupported output dtype {output_dtype!r} (expected 'float32' or 'int16')")

    t0 = time.time()
    _validate_chain_params(p)
    sample_rate = p.symbol_rate * p.sps
    taps = rrc_taps(p.rrc_alpha, p.rrc_span, p.sps)
    resampling = target_fs is not None and target_fs != sample_rate

    # Sized from p.n_cadu and the cheap, data-independent unit_bytes alone
    # (no payload generated/read yet): a pathologically large n_cadu must
    # be refused *before* _prepare_payload() below, which otherwise would
    # itself attempt a multi-GB allocation building that large a payload.
    _, guard_unit_bytes, _, _ = _unit_bytes(p)
    if resampling:
        est_native_bytes = p.n_cadu * _native_bytes_per_cadu(p, guard_unit_bytes)
        if est_native_bytes > _RESAMPLE_MEMORY_GUARD_BYTES:
            raise MemoryError(
                f"Resampling this export (~{est_native_bytes / 1e9:.1f} GB of native-rate "
                "IQ) needs the whole signal in memory at once and isn't streamed, unlike "
                "the rest of generation. Export without resampling (disable it) and "
                "resample the resulting file separately with a tool that streams it, or "
                "reduce the export size (fewer CADUs)."
            )

    prep = _prepare_payload(p)
    batch_n_cadu = max(1, int(_EXPORT_BATCH_TARGET_BYTES / max(_native_bytes_per_cadu(p, prep.unit_bytes), 1)))

    scratch_fd, scratch_path = tempfile.mkstemp(prefix="hktm_export_native_", suffix=".raw")
    os.close(scratch_fd)
    try:
        conv_encoder = ConvEncoder(invert_g2=p.conv_invert_g2, rate=p.conv_rate) if p.fec_conv else None
        shaper = RRCPulseShaper(p.sps, taps)
        phase_noise_gen = (
            PhaseNoiseGenerator(p.phase_noise_linewidth_hz, sample_rate, p.impairment_seed)
            if p.phase_noise_linewidth_hz > 0 else None
        )
        bits_per_symbol = BITS_PER_SYMBOL[p.modulation]
        # Leftover coded bits, when a batch's coded-bit count isn't a whole
        # number of symbols (possible with a punctured rate), carried into
        # the next batch so symbol mapping (QPSK's I/Q pairing, or BPSK's
        # 1:1 mapping) never splits a symbol across a batch boundary -- see
        # mapping.map_symbols().
        pending_bit = np.empty(0, dtype=np.uint8)
        global_peak = 0.0
        n_symbols = 0
        n_native_samples = 0

        with open(scratch_path, "wb") as scratch:
            for batch_start in range(0, p.n_cadu, batch_n_cadu):
                batch_end = min(batch_start + batch_n_cadu, p.n_cadu)
                batch_bits = np.concatenate([_cadu_bits(p, prep, i) for i in range(batch_start, batch_end)])
                coded = conv_encoder.encode(batch_bits) if conv_encoder is not None else batch_bits
                if len(pending_bit):
                    coded = np.concatenate([pending_bit, coded])
                    pending_bit = np.empty(0, dtype=np.uint8)
                carry = len(coded) % bits_per_symbol
                if carry:
                    pending_bit = coded[-carry:]
                    coded = coded[:-carry]

                symbols = map_symbols(bits_to_nrzl(coded), p.modulation)
                n_symbols += len(symbols)
                iq_batch = shaper.shape(symbols)
                iq_batch = _apply_impairments(iq_batch, p, sample_rate, n_native_samples, phase_noise_gen)
                if len(iq_batch):
                    global_peak = max(global_peak, np.abs(iq_batch.real).max(), np.abs(iq_batch.imag).max())
                    n_native_samples += len(iq_batch)
                    scratch.write(iq_batch.tobytes())

                if progress_callback is not None:
                    progress_callback(
                        0.7 * batch_end / p.n_cadu,
                        f"Generating CADU {batch_end:,}/{p.n_cadu:,}...".replace(",", " "),
                    )

            tail = shaper.flush()
            tail = _apply_impairments(tail, p, sample_rate, n_native_samples, phase_noise_gen)
            if len(tail):
                global_peak = max(global_peak, np.abs(tail.real).max(), np.abs(tail.imag).max())
                n_native_samples += len(tail)
                scratch.write(tail.tobytes())

        if len(pending_bit):
            raise ValueError(
                f"coded bitstream length isn't a whole number of {p.modulation} symbols "
                f"({bits_per_symbol} bit(s) each)"
            )

        scale = (peak / global_peak) if global_peak > 0 else 1.0

        if progress_callback is not None:
            progress_callback(0.7, "Normalizing..." if resampling else "Normalizing and packing...")

        if resampling:
            # Not streamed (see the upfront guard above): read the whole
            # scaled native-rate signal back, resample, then pack+write.
            native = np.fromfile(scratch_path, dtype=np.complex128) * scale
            resampled = resample_iq(native, sample_rate, target_fs)
            del native
            if progress_callback is not None:
                progress_callback(0.9, "Packing...")
            with open(output_path, "wb") as out:
                out.write(pack_iq_interleaved(resampled, output_dtype))
            output_fs = target_fs
            n_output_samples = len(resampled)
            del resampled
        else:
            output_fs = sample_rate
            n_output_samples = 0
            with open(scratch_path, "rb") as scratch, open(output_path, "wb") as out:
                while True:
                    raw = np.fromfile(scratch, dtype=np.complex128, count=_SCRATCH_READ_CHUNK_SAMPLES)
                    if len(raw) == 0:
                        break
                    out.write(pack_iq_interleaved(raw * scale, output_dtype))
                    n_output_samples += len(raw)
                    if progress_callback is not None:
                        progress_callback(0.7 + 0.3 * n_output_samples / max(n_native_samples, 1), "Packing...")
    finally:
        try:
            os.remove(scratch_path)
        except OSError:
            pass

    if progress_callback is not None:
        progress_callback(1.0, "Done")

    elapsed = time.time() - t0
    meta = _chain_meta(p, prep, sample_rate)
    meta.update({
        "n_symbols": n_symbols,
        "n_iq_samples": n_native_samples,
        "format": "raw interleaved, no header (I0,Q0,I1,Q1,...)",
        "output_dtype": output_dtype,
        "output_peak": peak,
        "output_sample_rate": output_fs,
        "output_n_samples": n_output_samples,
        "output_duration_s": n_output_samples / output_fs if output_fs else None,
        "carrier_note": "file is baseband IQ (no carrier/frequency information); "
                         "set the intended RF center frequency manually on the playback instrument",
    })
    cadu_bytes = len(p.asm) + prep.rs_block_bytes
    return {"cadu_bytes": cadu_bytes, "elapsed": elapsed, "meta": meta}
