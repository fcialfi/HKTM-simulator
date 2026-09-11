#!/usr/bin/env python3
"""Generate a CCSDS-chain test signal (baseline: QPSK) as raw IQ, for
injection via RF-Catcher (TestTree) Capture & Playback.

Chain: payload -> RS(255,223) interleaved x5 -> ASM prepend -> convolutional
K=7 rate 1/2 -> NRZ-L -> optional CCSDS scrambler -> QPSK (Gray) -> RRC ->
raw interleaved float32 IQ.

See README.md for architecture assumptions, limitations, and open TODOs
before using the output against real ground equipment.
"""

import argparse
import json
import time

import numpy as np

from ccsds_chain.reed_solomon import rs_encode_interleaved
from ccsds_chain.convolutional import conv_encode
from ccsds_chain.scrambler import apply_scrambler
from ccsds_chain.mapping import bits_to_nrzl, qpsk_gray_map
from ccsds_chain.pulse_shaping import rrc_taps, pulse_shape
from ccsds_chain.utils import bytes_to_bits, generate_payload, write_iq_interleaved_float32

# --------------------------------------------------------------------------
# PARAMETRI CONFIGURABILI (rif. AWS-OSE-ICD-0063, baseline CCSDS 131.0-B-2)
# --------------------------------------------------------------------------
MODULATION = "QPSK"            # selezionabile: solo QPSK implementata (baseline)
ENCODING = "NRZ-L"             # selezionabile: solo NRZ-L implementata (baseline)
BIT_RATE = 3_570_000           # bps, post-codifica, header incluso (informativo)
SYMBOL_RATE = 1_785_000        # symbol/s (baseline QPSK: bit_rate/2)
SAMPLES_PER_SYM = 4            # oversampling factor
RRC_ALPHA = 0.35
RRC_SPAN = 8                   # taps RRC = RRC_SPAN * SAMPLES_PER_SYM + 1

RS_K = 223
RS_N = 255
INTERLEAVE_DEPTH = 5

FEC_RS = True                  # selezionabile
FEC_CONV = True                # selezionabile
CONV_INVERT_G2 = True          # bool, convenzione CCSDS standard (Viterbi Inverted)

SCRAMBLING = False             # bool, scrambler CCSDS opzionale

ASM = bytes.fromhex("1ACFFC1D")  # 4 byte, non codificato

N_CADU = 100                   # numero di CADU da generare (durata segnale)
PAYLOAD_SOURCE = None          # path a file con Transfer Frame reali, o None = pseudo-random
PAYLOAD_SEED = 42              # seed per riproducibilita' del payload pseudo-random

OUTPUT_PATH = "output_iq.raw"
# --------------------------------------------------------------------------


def build_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cadu", type=int, default=N_CADU)
    p.add_argument("--alpha", type=float, default=RRC_ALPHA)
    p.add_argument("--sps", type=int, default=SAMPLES_PER_SYM)
    p.add_argument("--span", type=int, default=RRC_SPAN)
    p.add_argument("--no-rs", action="store_true", help="disable RS(255,223) FEC")
    p.add_argument("--no-conv", action="store_true", help="disable convolutional FEC")
    p.add_argument("--no-invert-g2", action="store_true")
    p.add_argument("--scramble", action="store_true", help="enable CCSDS scrambler")
    p.add_argument("--payload-source", type=str, default=PAYLOAD_SOURCE)
    p.add_argument("--seed", type=int, default=PAYLOAD_SEED)
    p.add_argument("-o", "--output", type=str, default=OUTPUT_PATH)
    return p.parse_args()


def main():
    args = build_cli()
    if MODULATION != "QPSK":
        raise NotImplementedError(f"modulation {MODULATION!r} not implemented (baseline: QPSK)")
    if ENCODING != "NRZ-L":
        raise NotImplementedError(f"encoding {ENCODING!r} not implemented (baseline: NRZ-L)")

    fec_rs = FEC_RS and not args.no_rs
    fec_conv = FEC_CONV and not args.no_conv
    invert_g2 = CONV_INVERT_G2 and not args.no_invert_g2
    scrambling = SCRAMBLING or args.scramble

    frame_bytes = RS_K * INTERLEAVE_DEPTH
    t0 = time.time()

    print(f"[1/8] Generazione payload: {args.n_cadu} CADU x {frame_bytes} byte "
          f"({'file: ' + args.payload_source if args.payload_source else 'pseudo-random, seed=' + str(args.seed)})")
    payload = generate_payload(args.n_cadu, frame_bytes, args.payload_source, args.seed)

    print(f"[2/8] RS({RS_N},{RS_K}) encode, interleave depth {INTERLEAVE_DEPTH}"
          f"{' (SKIPPED)' if not fec_rs else ''}")
    cadus = bytearray()
    for i in range(args.n_cadu):
        frame = payload[i * frame_bytes:(i + 1) * frame_bytes]
        if fec_rs:
            rs_block = rs_encode_interleaved(frame, RS_K, RS_N, INTERLEAVE_DEPTH)
        else:
            # without RS, the "codeblock" is just the raw frame data (test-only mode)
            rs_block = frame
        cadus += ASM
        cadus += rs_block
    cadu_bytes = len(ASM) + (RS_N * INTERLEAVE_DEPTH if fec_rs else frame_bytes)
    print(f"       -> {args.n_cadu} CADU x {cadu_bytes} byte = {len(cadus)} byte totali")

    print("[3/8] ASM (0x1ACFFC1D) gia' inserito per CADU")

    bits = bytes_to_bits(bytes(cadus))
    print(f"[4/8] Convolutional K=7 rate 1/2 (G1=171o, G2=133o, invert_g2={invert_g2})"
          f"{' (SKIPPED)' if not fec_conv else ''}")
    coded_bits = conv_encode(bits, invert_g2=invert_g2) if fec_conv else bits

    print("[5/8] NRZ-L mapping (bipolare diretto)")
    bipolar = bits_to_nrzl(coded_bits)

    print(f"[6/8] Scrambler CCSDS (seed 0xFF){'' if scrambling else ' (SKIPPED)'}")
    if scrambling:
        bipolar = apply_scrambler(bipolar)

    print("[7/8] QPSK mapping (Gray, energia unitaria)")
    symbols = qpsk_gray_map(bipolar)

    print(f"[8/8] Pulse shaping RRC (alpha={args.alpha}, span={args.span}, sps={args.sps})")
    taps = rrc_taps(args.alpha, args.span, args.sps)
    iq = pulse_shape(symbols, args.sps, taps)

    write_iq_interleaved_float32(args.output, iq)

    sample_rate = SYMBOL_RATE * args.sps
    meta = {
        "modulation": MODULATION,
        "encoding": ENCODING,
        "bit_rate": BIT_RATE,
        "symbol_rate": SYMBOL_RATE,
        "sample_rate": sample_rate,
        "samples_per_symbol": args.sps,
        "rrc_alpha": args.alpha,
        "rrc_span": args.span,
        "fec_rs": fec_rs,
        "fec_conv": fec_conv,
        "conv_invert_g2": invert_g2,
        "scrambling": scrambling,
        "rs_k": RS_K,
        "rs_n": RS_N,
        "interleave_depth": INTERLEAVE_DEPTH,
        "n_cadu": args.n_cadu,
        "n_symbols": len(symbols),
        "n_iq_samples": len(iq),
        "format": "raw interleaved float32 (I0,Q0,I1,Q1,...)",
    }
    meta_path = args.output.rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    duration_s = len(symbols) / SYMBOL_RATE
    print(f"\nCompletato in {dt:.2f}s")
    print(f"  Simboli QPSK: {len(symbols)}  (durata segnale: {duration_s * 1000:.2f} ms)")
    print(f"  Campioni IQ:  {len(iq)}  @ {sample_rate / 1e6:.3f} MS/s")
    print(f"  Output:       {args.output} ({len(iq) * 8 / 1e6:.2f} MB)")
    print(f"  Metadata:     {meta_path}")


if __name__ == "__main__":
    main()
