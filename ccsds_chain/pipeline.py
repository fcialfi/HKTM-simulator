"""Shared chain orchestration, used by both the CLI (generate_signal.py) and
the GUI (app.py) so the two never drift apart."""

from dataclasses import dataclass, field
from typing import Callable, Optional
import time

import numpy as np

from .reed_solomon import rs_encode_interleaved
from .convolutional import conv_encode
from .scrambler import pn_sequence
from .mapping import bits_to_nrzl, qpsk_gray_map
from .pulse_shaping import rrc_taps, pulse_shape
from .utils import bytes_to_bits, generate_payload

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


def run_chain(
    p: ChainParams,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ChainResult:
    """Run the full chain. `progress_callback(fraction, message)`, if given,
    is called periodically with fraction in [0, 1] -- useful for a UI
    progress bar on a large (e.g. whole-file) export; it is not called at
    all for a cheap/small run, so callers can pass it unconditionally."""
    if p.modulation != "QPSK":
        raise NotImplementedError(f"modulation {p.modulation!r} not implemented (baseline: QPSK)")
    if p.encoding != "NRZ-L":
        raise NotImplementedError(f"encoding {p.encoding!r} not implemented (baseline: NRZ-L)")

    t0 = time.time()
    frame_bytes = p.rs_k * p.interleave_depth

    payload = generate_payload(p.n_cadu, frame_bytes, p.payload_source, p.payload_bytes, p.seed)

    # Build each CADU = ASM + RS-encoded codeblock, per CCSDS 131.0-B-3's
    # "Overall Structure of Channel Coding": RS encode -> pseudo-randomize
    # (the codeblock only, never the ASM) -> attach ASM (this forms the
    # CADU) -> convolutionally encode the *stream of CADUs*. So the ASM IS
    # convolutionally coded along with the rest: a convolutionally-coded
    # stream is continuously Viterbi-decoded at the receiver (no framing
    # needed to decode it), and frame sync is recovered by correlating for
    # the literal ASM pattern in the *decoded* bitstream, not before
    # decoding. The pseudo-randomizer is reinitialized for each CADU's
    # codeblock (not carried over between CADUs) -- since every CADU's
    # codeblock is the same length, it is always reinitialized to the same
    # state, so the PN sequence is generated once and reused rather than
    # regenerated (an O(bits) pure-Python LFSR) on every iteration.
    asm_bits = bytes_to_bits(p.asm)
    rs_block_bytes = p.rs_n * p.interleave_depth if p.fec_rs else frame_bytes
    pn = pn_sequence(rs_block_bytes * 8, p.randomizer) if p.randomizer != "none" else None
    # Throttled so a large n_cadu doesn't spend more time updating a UI
    # widget than actually encoding (a plain Streamlit progress bar update
    # is not free at tens/hundreds of thousands of calls).
    report_every = max(1, p.n_cadu // 200)
    cadu_bit_chunks = []
    for i in range(p.n_cadu):
        frame = payload[i * frame_bytes:(i + 1) * frame_bytes]
        rs_block = (rs_encode_interleaved(frame, p.rs_k, p.rs_n, p.interleave_depth)
                    if p.fec_rs else frame)
        rs_bits = bytes_to_bits(rs_block)
        if pn is not None:
            rs_bits = np.bitwise_xor(rs_bits, pn)
        cadu_bit_chunks.append(asm_bits)
        cadu_bit_chunks.append(rs_bits)
        if progress_callback is not None and ((i + 1) % report_every == 0 or i + 1 == p.n_cadu):
            progress_callback(
                _PROGRESS_RS_ASM_WEIGHT * (i + 1) / p.n_cadu,
                f"Encoding CADU {i + 1:,}/{p.n_cadu:,}...".replace(",", " "),
            )
    cadu_bytes = len(p.asm) + rs_block_bytes

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

    meta = {
        "modulation": p.modulation,
        "encoding": p.encoding,
        "bit_rate": p.bit_rate,
        "symbol_rate": p.symbol_rate,
        "sample_rate": sample_rate,
        "samples_per_symbol": p.sps,
        "rrc_alpha": p.rrc_alpha,
        "rrc_span": p.rrc_span,
        "fec_rs": p.fec_rs,
        "fec_conv": p.fec_conv,
        "conv_rate": p.conv_rate,
        "conv_invert_g2": p.conv_invert_g2 and p.conv_rate == "1/2",
        "randomizer": p.randomizer,
        "rs_e": p.rs_e,
        "rs_k": p.rs_k,
        "rs_n": p.rs_n,
        "interleave_depth": p.interleave_depth,
        "n_cadu": p.n_cadu,
        "n_symbols": len(symbols),
        "n_iq_samples": len(iq),
    }

    return ChainResult(iq=iq, symbols=symbols, frame_bytes=frame_bytes, cadu_bytes=cadu_bytes,
                        sample_rate=sample_rate, elapsed=elapsed, meta=meta)
