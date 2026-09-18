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
from .mapping import bits_to_nrzl, qpsk_gray_map
from .pulse_shaping import rrc_taps, pulse_shape, RRCPulseShaper
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
    asm: bytes = ASM
    input_format: str = "transfer_frame"  # "transfer_frame" (raw, gets RS/randomizer/ASM applied here)
                                           # or "cadu" (already ASM+RS[+randomized]: used as-is, no re-encoding)
    n_cadu: int = 100
    payload_source: str | None = None
    payload_bytes: bytes | None = None
    seed: int = 42

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
    # Set only for CADU input from a real source with at least two ASMs
    # found: CADUs are delimited directly by consecutive ASM positions in
    # `raw_synced` (`cadu_positions[i]` to `cadu_positions[i+1]`), so their
    # length can vary from one CADU to the next -- confirmed against a real
    # captured pass. None when there's only one real CADU to work with (no
    # second ASM found), in which case the single CADU falls back to
    # `unit_bytes` (the configured RS/interleave length, best effort).
    cadu_positions: Optional[list] = None
    raw_synced: Optional[bytes] = None
    # The measured length is no longer a single global constant once CADUs
    # can vary in length -- these describe that variability for meta/UI
    # reporting. `detected_unit_bytes` is the first CADU's measured length
    # (kept for backward-compatible simple display); vs.
    # `configured_unit_bytes`, what the configured RS-E/interleave-depth
    # predicted. They differ whenever the real CADUs weren't built with
    # this tool's exact RS(255,*) interleaved framing.
    detected_unit_bytes: Optional[int] = None
    configured_unit_bytes: Optional[int] = None
    cadu_length_varies: bool = False
    cadu_length_min_bytes: Optional[int] = None
    cadu_length_max_bytes: Optional[int] = None


def _validate_chain_params(p: ChainParams) -> None:
    if p.modulation != "QPSK":
        raise NotImplementedError(f"modulation {p.modulation!r} not implemented (baseline: QPSK)")
    if p.encoding != "NRZ-L":
        raise NotImplementedError(f"encoding {p.encoding!r} not implemented (baseline: NRZ-L)")
    if p.input_format not in ("transfer_frame", "cadu"):
        raise NotImplementedError(f"input_format {p.input_format!r} not implemented (expected 'transfer_frame' or 'cadu')")


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
    *already* a stream of complete CADUs (ASM + RS-encoded, and already
    pseudo-randomized if that's how they were built) -- e.g. captured or
    previously generated CADUs, not raw Transfer Frames. In that case RS
    encoding, the pseudo-randomizer and the ASM prepend must all be
    skipped: applying them again would double-encode and prefix a second
    ASM in front of data that already has one. A real/captured CADU file
    is also not guaranteed to start exactly on a CADU boundary (leading
    idle line-fill, or an excerpt starting mid-stream), unlike a stream
    this tool built itself, so the raw source is searched for the first
    real ASM before slicing into CADUs. Those CADUs are then delimited
    directly by consecutive ASM positions in the data (`find_all_cadu_
    positions()`), not by a single length measured once and trusted for
    the rest of the stream: a real downlink's CADUs aren't guaranteed to
    all be the same length -- confirmed against a real captured pass,
    where 5 consecutive CADUs were 1348 bytes and a 6th was 1344 (most
    likely because they carry a variable number of packed Transfer
    Frames/Space Packets, not a fixed-size RS-interleaved codeblock at
    all) -- and trusting either the configured RS-E/interleave-depth or a
    single once-measured length for the *byte length* (as opposed to for
    actually RS-decoding, which CADU input skips entirely) would silently
    misalign some later CADU whenever the real lengths vary.
    """
    _validate_chain_params(p)
    frame_bytes, unit_bytes, rs_block_bytes, is_cadu_input = _unit_bytes(p)
    asm_bits = bytes_to_bits(p.asm)

    has_real_source = p.payload_source is not None or p.payload_bytes is not None
    sync_skipped_bytes = 0
    n_synced_cadu = 0
    cadu_positions = None
    raw_synced = None
    detected_unit_bytes = None
    configured_unit_bytes = None
    cadu_length_varies = False
    cadu_length_min_bytes = None
    cadu_length_max_bytes = None
    if is_cadu_input and has_real_source:
        # Read the raw source in full (not just the n_cadu*unit_bytes we
        # ultimately need) so there's enough of it to search past any
        # leading junk, locate every real ASM, and still have n_cadu whole
        # CADUs after the first one.
        raw = p.payload_bytes if p.payload_bytes is not None else open(p.payload_source, "rb").read()
        sync_skipped_bytes = find_cadu_sync(raw, p.asm)
        raw_synced = raw[sync_skipped_bytes:]
        configured_unit_bytes = unit_bytes

        positions = find_all_cadu_positions(raw_synced, p.asm)
        if len(positions) >= 2:
            # Real CADU boundaries, straight from the data: CADU i is
            # raw_synced[positions[i]:positions[i+1]]. The last found ASM's
            # own CADU isn't included (its end isn't known without a
            # following ASM), matching the existing "never guess, zero-pad
            # instead" convention for anything beyond what's really there.
            cadu_positions = positions
            n_synced_cadu = len(positions) - 1
            gaps = [positions[j + 1] - positions[j] for j in range(n_synced_cadu)]
            detected_unit_bytes = gaps[0]
            cadu_length_varies = len(set(gaps)) > 1
            cadu_length_min_bytes = min(gaps)
            cadu_length_max_bytes = max(gaps)
            unit_bytes = detected_unit_bytes  # representative length: padding fallback, batch sizing
            rs_block_bytes = unit_bytes - len(p.asm)
        else:
            # Only one real ASM in the whole source: there's nothing to
            # measure a real length from, so fall back to the configured
            # RS/interleave-derived length for that lone CADU (best effort).
            n_synced_cadu = 1

        payload = b""  # unused in this branch -- _cadu_bits() reads cadu_positions/raw_synced instead
    else:
        payload = generate_payload(p.n_cadu, unit_bytes, p.payload_source, p.payload_bytes, p.seed)

    pn = (pn_sequence(rs_block_bytes * 8, p.randomizer)
          if (p.randomizer != "none" and not is_cadu_input) else None)

    return _PayloadPrep(payload=payload, unit_bytes=unit_bytes, frame_bytes=frame_bytes,
                         rs_block_bytes=rs_block_bytes, is_cadu_input=is_cadu_input,
                         has_real_source=has_real_source, n_synced_cadu=n_synced_cadu,
                         sync_skipped_bytes=sync_skipped_bytes, asm_bits=asm_bits, pn=pn,
                         cadu_positions=cadu_positions, raw_synced=raw_synced,
                         detected_unit_bytes=detected_unit_bytes, configured_unit_bytes=configured_unit_bytes,
                         cadu_length_varies=cadu_length_varies, cadu_length_min_bytes=cadu_length_min_bytes,
                         cadu_length_max_bytes=cadu_length_max_bytes)


def _cadu_bits(p: ChainParams, prep: _PayloadPrep, i: int) -> np.ndarray:
    """Bits for CADU index `i`: RS-encode + ASM-prepend for a Transfer Frame
    input, or straight pass-through + sync validation for a CADU input (see
    `_prepare_payload()`)."""
    if prep.is_cadu_input:
        if prep.cadu_positions is not None:
            # Real CADU boundaries straight from the data (see
            # `_prepare_payload()`): always genuinely starts with the ASM
            # by construction, no separate validation needed.
            if i < prep.n_synced_cadu:
                cadu = prep.raw_synced[prep.cadu_positions[i]:prep.cadu_positions[i + 1]]
            else:
                cadu = bytes(prep.unit_bytes)  # zero-padded tail, beyond what the source actually has
        elif prep.has_real_source:
            # Only one real ASM was found in the whole source (see
            # `_prepare_payload()`): that lone CADU falls back to the
            # configured RS/interleave length, since there's no second ASM
            # to measure a real one from.
            if i < prep.n_synced_cadu:
                cadu = prep.raw_synced[i * prep.unit_bytes:(i + 1) * prep.unit_bytes]
                if cadu[:len(p.asm)] != p.asm:
                    raise ValueError(
                        f"Lost CADU sync at CADU {i} (byte offset "
                        f"{prep.sync_skipped_bytes + i * prep.unit_bytes} in the source): "
                        f"expected ASM {p.asm.hex()}, got {cadu[:len(p.asm)].hex()}. CADU length "
                        "came from the configured RS error correction E and interleave depth "
                        "(no second ASM was found in the source to measure the real length "
                        "from) -- check that they match how this CADU was actually built."
                    )
            else:
                cadu = bytes(prep.unit_bytes)
        else:
            # Pseudo-random test payload (no real source): fixed-stride,
            # exactly as generated.
            cadu = prep.payload[i * prep.unit_bytes:(i + 1) * prep.unit_bytes]
        return bytes_to_bits(cadu)
    frame = prep.payload[i * prep.frame_bytes:(i + 1) * prep.frame_bytes]
    rs_block = (rs_encode_interleaved(frame, p.rs_k, p.rs_n, p.interleave_depth)
                if p.fec_rs else frame)
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
        "cadu_length_detected_bytes": prep.detected_unit_bytes,
        "cadu_length_configured_bytes": prep.configured_unit_bytes,
        "cadu_length_mismatch": (
            prep.detected_unit_bytes is not None
            and prep.configured_unit_bytes is not None
            and prep.detected_unit_bytes != prep.configured_unit_bytes
        ),
        "cadu_length_varies": prep.cadu_length_varies,
        "cadu_length_min_bytes": prep.cadu_length_min_bytes,
        "cadu_length_max_bytes": prep.cadu_length_max_bytes,
        "fec_rs": p.fec_rs and not prep.is_cadu_input,
        "fec_conv": p.fec_conv,
        "conv_rate": p.conv_rate,
        "conv_invert_g2": p.conv_invert_g2 and p.conv_rate == "1/2",
        "randomizer": "none" if prep.is_cadu_input else p.randomizer,
        "rs_e": p.rs_e,
        "rs_k": p.rs_k,
        "rs_n": p.rs_n,
        "interleave_depth": p.interleave_depth,
        "n_cadu": p.n_cadu,
    }


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
    symbols = qpsk_gray_map(bipolar)

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

    if progress_callback is not None:
        progress_callback(1.0, "Done")

    sample_rate = p.symbol_rate * p.sps
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
    symbols_per_cadu = bits_per_cadu * conv_expansion / 2.0  # QPSK: 2 bits/symbol
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
        # A single leftover coded bit, when a batch's coded-bit count is
        # odd (possible with a punctured rate), carried into the next
        # batch so QPSK's I/Q pairing never splits across a batch boundary
        # -- see mapping.qpsk_gray_map().
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
                if len(coded) % 2:
                    pending_bit = coded[-1:]
                    coded = coded[:-1]

                symbols = qpsk_gray_map(bits_to_nrzl(coded))
                n_symbols += len(symbols)
                iq_batch = shaper.shape(symbols)
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
            if len(tail):
                global_peak = max(global_peak, np.abs(tail.real).max(), np.abs(tail.imag).max())
                n_native_samples += len(tail)
                scratch.write(tail.tobytes())

        if len(pending_bit):
            raise ValueError("bipolar stream length must be even for QPSK pairing")

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
