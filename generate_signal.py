#!/usr/bin/env python3
"""Generate a CCSDS-chain test signal (baseline: QPSK) as raw IQ, for
injection via RF-Catcher (TestTree) Capture & Playback or any other IQ
recorder/replayer.

Chain: payload -> RS(255,223 or 255,239) interleaved -> optional CCSDS
pseudo-randomizer (excludes ASM) -> ASM prepend per CADU -> convolutional
K=7, rate 1/2 (or punctured to 2/3, 3/4, 5/6, 7/8) over the CADU stream
(ASM included) -> NRZ-L -> QPSK (Gray) -> RRC -> peak-normalize -> optional
resample -> raw interleaved IQ (float32 or int16), no header.

See README.md for architecture assumptions, limitations, and open TODOs
before using the output against real ground equipment. For an interactive
GUI with a real-time spectrum view, see `streamlit run app.py`.
"""

import argparse
import datetime
import json
import os

from ccsds_chain.pipeline import ChainParams, run_chain
from ccsds_chain.utils import normalize_peak, resample_iq, pack_iq_interleaved, resample_ratio

# --------------------------------------------------------------------------
# PARAMETRI CONFIGURABILI (rif. AWS-OSE-ICD-0063, baseline CCSDS 131.0-B-2)
# --------------------------------------------------------------------------
MODULATION = "QPSK"            # selezionabile: solo QPSK implementata (baseline)
ENCODING = "NRZ-L"             # selezionabile: solo NRZ-L implementata (baseline)
BIT_RATE = 3_570_000           # bps, post-codifica, header incluso (informativo)
SYMBOL_RATE = 1_785_000        # symbol/s (baseline QPSK: bit_rate/2)
SAMPLES_PER_SYM = 4            # oversampling factor "nativo" (interno, pre-resample)
RRC_ALPHA = 0.35
RRC_SPAN = 8                   # taps RRC = RRC_SPAN * SAMPLES_PER_SYM + 1

RS_E = 16                       # error correction capability, in symbols: 8 or 16
RS_N = 255
INTERLEAVE_DEPTH = 5           # 1, 2, 3, 4, 5, or 8

FEC_RS = True                  # selezionabile
FEC_CONV = True                # selezionabile
CONV_RATE = "1/2"              # 1/2, 2/3, 3/4, 5/6, 7/8 (puntura)
CONV_INVERT_G2 = True          # bool, convenzione CCSDS standard (solo rate 1/2)

RANDOMIZER = "none"            # "none", "short" (255-bit, legacy), "long" (131071-bit)

ASM = bytes.fromhex("1ACFFC1D")  # 4 byte, non codificato

N_CADU = 100                   # numero di CADU da generare (durata segnale)
PAYLOAD_SOURCE = None          # path a file con Transfer Frame reali, o None = pseudo-random
PAYLOAD_SEED = 42              # seed per riproducibilita' del payload pseudo-random

OUTPUT_DTYPE = "float32"       # "float32" ([-1,+1]) o "int16" (12 bit, formato RF-Catcher, [-2048,2047])
OUTPUT_PEAK = 0.9              # ampiezza di picco normalizzata (margine anti-clipping)
TARGET_FS = None               # Hz, o None per usare simbol_rate*sps nativo (nessun resample)
OUTPUT_DIR = "output"
# --------------------------------------------------------------------------


def build_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cadu", type=int, default=N_CADU)
    p.add_argument("--alpha", type=float, default=RRC_ALPHA)
    p.add_argument("--sps", type=int, default=SAMPLES_PER_SYM)
    p.add_argument("--span", type=int, default=RRC_SPAN)
    p.add_argument("--no-rs", action="store_true", help="disable Reed-Solomon FEC")
    p.add_argument("--rs-e", type=int, choices=[8, 16], default=RS_E, help="RS error correction capability")
    p.add_argument("--interleave-depth", type=int, choices=[1, 2, 3, 4, 5, 8], default=INTERLEAVE_DEPTH)
    p.add_argument("--no-conv", action="store_true", help="disable convolutional FEC")
    p.add_argument("--conv-rate", choices=["1/2", "2/3", "3/4", "5/6", "7/8"], default=CONV_RATE)
    p.add_argument("--no-invert-g2", action="store_true")
    p.add_argument("--randomizer", choices=["none", "short", "long"], default=RANDOMIZER,
                    help="CCSDS pseudo-randomizer: 'long' (131071-bit, current standard default), "
                         "'short' (255-bit, legacy), or 'none'")
    p.add_argument("--payload-source", type=str, default=PAYLOAD_SOURCE)
    p.add_argument("--seed", type=int, default=PAYLOAD_SEED)
    p.add_argument("--dtype", choices=["float32", "int16"], default=OUTPUT_DTYPE,
                    help="IQ sample format for the output file; int16 is the RF-Catcher "
                         "format (little-endian, 12 significant bits, range [-2048, 2047])")
    p.add_argument("--peak", type=float, default=OUTPUT_PEAK,
                    help="normalized peak amplitude before dtype conversion (headroom against clipping)")
    p.add_argument("--target-fs", type=float, default=TARGET_FS,
                    help="resample the output to this exact sample rate in Hz (e.g. 10e6); "
                         "default: no resampling, native symbol_rate*sps")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="output path; default: output/qpsk_ccsds_<fs>Msps_<n_cadu>cadu_<timestamp>.iq")
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
        rs_e=args.rs_e,
        rs_n=RS_N,
        interleave_depth=args.interleave_depth,
        fec_rs=FEC_RS and not args.no_rs,
        fec_conv=FEC_CONV and not args.no_conv,
        conv_rate=args.conv_rate,
        conv_invert_g2=CONV_INVERT_G2 and not args.no_invert_g2,
        randomizer=args.randomizer,
        asm=ASM,
        n_cadu=args.n_cadu,
        payload_source=args.payload_source,
        seed=args.seed,
    )

    print(f"[1/9] Generazione payload: {params.n_cadu} CADU x {params.rs_k * params.interleave_depth} byte "
          f"({'file: ' + params.payload_source if params.payload_source else 'pseudo-random, seed=' + str(params.seed)})")
    print(f"[2/9] RS({params.rs_n},{params.rs_k}) E={params.rs_e} encode, interleave depth {params.interleave_depth}"
          f"{' (SKIPPED)' if not params.fec_rs else ''}")
    print(f"[3/9] Pseudo-randomizer CCSDS ({params.randomizer}, excludes ASM)"
          f"{'' if params.randomizer != 'none' else ' (SKIPPED)'}")
    print("[4/9] ASM (0x1ACFFC1D) attached per CADU (forms the CADU)")
    print(f"[5/9] Convolutional K=7 rate {params.conv_rate} over the CADU stream, ASM included "
          f"(G1=171o, G2=133o, invert_g2={params.conv_invert_g2 and params.conv_rate == '1/2'})"
          f"{' (SKIPPED)' if not params.fec_conv else ''}")
    print("[6/9] NRZ-L mapping (direct bipolar)")
    print("[7/9] QPSK mapping (Gray, unit energy)")
    print(f"[8/9] Pulse shaping RRC (alpha={params.rrc_alpha}, span={params.rrc_span}, sps={params.sps})")

    result = run_chain(params)

    print(f"       -> {params.n_cadu} CADU x {result.cadu_bytes} byte")

    iq = normalize_peak(result.iq, args.peak)
    output_fs = result.sample_rate
    if args.target_fs is not None and args.target_fs != result.sample_rate:
        up, down = resample_ratio(result.sample_rate, args.target_fs)
        print(f"[9/9] Resampling {result.sample_rate/1e6:.3f} MS/s -> {args.target_fs/1e6:.3f} MS/s "
              f"(ratio {up}/{down}, scipy.signal.resample_poly), format={args.dtype}, peak={args.peak}")
        iq = resample_iq(iq, result.sample_rate, args.target_fs)
        output_fs = args.target_fs
    else:
        print(f"[9/9] No resampling (native {output_fs/1e6:.3f} MS/s), format={args.dtype}, peak={args.peak}")

    output_path = args.output
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            OUTPUT_DIR, f"qpsk_ccsds_{output_fs/1e6:.1f}Msps_{params.n_cadu}cadu_{timestamp}.iq")

    iq_bytes = pack_iq_interleaved(iq, args.dtype)
    with open(output_path, "wb") as f:
        f.write(iq_bytes)

    duration_s = len(iq) / output_fs
    meta = dict(result.meta)
    meta.update({
        "format": "raw interleaved, no header (I0,Q0,I1,Q1,...)",
        "output_dtype": args.dtype,
        "output_peak": args.peak,
        "output_sample_rate": output_fs,
        "output_n_samples": len(iq),
        "output_duration_s": duration_s,
        "carrier_note": "file is baseband IQ (no carrier/frequency information); "
                         "set the intended RF center frequency manually on the playback instrument",
    })
    meta_path = output_path.rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCompletato in {result.elapsed:.2f}s")
    print(f"  Simboli QPSK:  {len(result.symbols)}")
    print(f"  Campioni IQ:   {len(iq)}  @ {output_fs / 1e6:.3f} MS/s (formato {args.dtype})")
    print(f"  Durata segnale: {duration_s * 1000:.2f} ms")
    print(f"  Output:        {output_path} ({len(iq_bytes) / 1e6:.2f} MB)")
    print(f"  Metadata:      {meta_path}")


if __name__ == "__main__":
    main()
