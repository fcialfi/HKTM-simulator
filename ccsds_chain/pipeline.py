"""Shared chain orchestration, used by both the CLI (generate_signal.py) and
the GUI (app.py) so the two never drift apart."""

from dataclasses import dataclass, field
import time

import numpy as np

from .reed_solomon import rs_encode_interleaved
from .convolutional import conv_encode
from .scrambler import apply_scrambler
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
    rs_k: int = 223
    rs_n: int = 255
    interleave_depth: int = 5
    fec_rs: bool = True
    fec_conv: bool = True
    conv_invert_g2: bool = True
    scrambling: bool = False
    asm: bytes = ASM
    n_cadu: int = 100
    payload_source: str | None = None
    payload_bytes: bytes | None = None
    seed: int = 42


@dataclass
class ChainResult:
    iq: np.ndarray
    symbols: np.ndarray
    frame_bytes: int
    cadu_bytes: int
    sample_rate: float
    elapsed: float
    meta: dict = field(default_factory=dict)


def run_chain(p: ChainParams) -> ChainResult:
    if p.modulation != "QPSK":
        raise NotImplementedError(f"modulation {p.modulation!r} not implemented (baseline: QPSK)")
    if p.encoding != "NRZ-L":
        raise NotImplementedError(f"encoding {p.encoding!r} not implemented (baseline: NRZ-L)")

    t0 = time.time()
    frame_bytes = p.rs_k * p.interleave_depth

    payload = generate_payload(p.n_cadu, frame_bytes, p.payload_source, p.payload_bytes, p.seed)

    cadus = bytearray()
    for i in range(p.n_cadu):
        frame = payload[i * frame_bytes:(i + 1) * frame_bytes]
        rs_block = (rs_encode_interleaved(frame, p.rs_k, p.rs_n, p.interleave_depth)
                    if p.fec_rs else frame)
        cadus += p.asm
        cadus += rs_block
    cadu_bytes = len(p.asm) + (p.rs_n * p.interleave_depth if p.fec_rs else frame_bytes)

    bits = bytes_to_bits(bytes(cadus))
    coded_bits = conv_encode(bits, invert_g2=p.conv_invert_g2) if p.fec_conv else bits

    bipolar = bits_to_nrzl(coded_bits)
    if p.scrambling:
        bipolar = apply_scrambler(bipolar)

    symbols = qpsk_gray_map(bipolar)

    taps = rrc_taps(p.rrc_alpha, p.rrc_span, p.sps)
    iq = pulse_shape(symbols, p.sps, taps)

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
        "conv_invert_g2": p.conv_invert_g2,
        "scrambling": p.scrambling,
        "rs_k": p.rs_k,
        "rs_n": p.rs_n,
        "interleave_depth": p.interleave_depth,
        "n_cadu": p.n_cadu,
        "n_symbols": len(symbols),
        "n_iq_samples": len(iq),
        "format": "raw interleaved int16 LE, 12-bit two's complement [-2048,2047] (I0,Q0,I1,Q1,...)",
    }

    return ChainResult(iq=iq, symbols=symbols, frame_bytes=frame_bytes, cadu_bytes=cadu_bytes,
                        sample_rate=sample_rate, elapsed=elapsed, meta=meta)
