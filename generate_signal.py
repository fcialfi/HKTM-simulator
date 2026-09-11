#!/usr/bin/env python3
"""Generate a CCSDS-chain test signal (baseline: QPSK) as raw IQ, for
injection via RF-Catcher (TestTree) Capture & Playback.

Chain: payload -> RS(255,223) interleaved x5 -> optional CCSDS scrambler
(excludes ASM) -> ASM prepend per CADU -> convolutional K=7 rate 1/2 (over
the CADU stream, ASM included) -> NRZ-L -> QPSK (Gray) -> RRC -> raw
interleaved float32 IQ.

See README.md for architecture assumptions, limitations, and open TODOs
before using the output against real ground equipment. For an interactive
GUI with a real-time spectrum view, see `streamlit run app.py`.
"""

import argparse
import json

from ccsds_chain.pipeline import ChainParams, run_chain
from ccsds_chain.utils import write_iq_interleaved_float32

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

    params = ChainParams(
        modulation=MODULATION,
        encoding=ENCODING,
        bit_rate=BIT_RATE,
        symbol_rate=SYMBOL_RATE,
        sps=args.sps,
        rrc_alpha=args.alpha,
        rrc_span=args.span,
        rs_k=RS_K,
        rs_n=RS_N,
        interleave_depth=INTERLEAVE_DEPTH,
        fec_rs=FEC_RS and not args.no_rs,
        fec_conv=FEC_CONV and not args.no_conv,
        conv_invert_g2=CONV_INVERT_G2 and not args.no_invert_g2,
        scrambling=SCRAMBLING or args.scramble,
        asm=ASM,
        n_cadu=args.n_cadu,
        payload_source=args.payload_source,
        seed=args.seed,
    )

    print(f"[1/8] Generazione payload: {params.n_cadu} CADU x {params.rs_k * params.interleave_depth} byte "
          f"({'file: ' + params.payload_source if params.payload_source else 'pseudo-random, seed=' + str(params.seed)})")
    print(f"[2/8] RS({params.rs_n},{params.rs_k}) encode, interleave depth {params.interleave_depth}"
          f"{' (SKIPPED)' if not params.fec_rs else ''}")
    print(f"[3/8] Scrambler CCSDS (seed 0xFF, excludes ASM){'' if params.scrambling else ' (SKIPPED)'}")
    print("[4/8] ASM (0x1ACFFC1D) attached per CADU (forms the CADU)")
    print(f"[5/8] Convolutional K=7 rate 1/2 over the CADU stream, ASM included "
          f"(G1=171o, G2=133o, invert_g2={params.conv_invert_g2})"
          f"{' (SKIPPED)' if not params.fec_conv else ''}")
    print("[6/8] NRZ-L mapping (direct bipolar)")
    print("[7/8] QPSK mapping (Gray, unit energy)")
    print(f"[8/8] Pulse shaping RRC (alpha={params.rrc_alpha}, span={params.rrc_span}, sps={params.sps})")

    result = run_chain(params)

    print(f"       -> {params.n_cadu} CADU x {result.cadu_bytes} byte")

    write_iq_interleaved_float32(args.output, result.iq)

    meta_path = args.output.rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(result.meta, f, indent=2)

    duration_s = len(result.symbols) / SYMBOL_RATE
    print(f"\nCompletato in {result.elapsed:.2f}s")
    print(f"  Simboli QPSK: {len(result.symbols)}  (durata segnale: {duration_s * 1000:.2f} ms)")
    print(f"  Campioni IQ:  {len(result.iq)}  @ {result.sample_rate / 1e6:.3f} MS/s")
    print(f"  Output:       {args.output} ({len(result.iq) * 8 / 1e6:.2f} MB)")
    print(f"  Metadata:     {meta_path}")


if __name__ == "__main__":
    main()
