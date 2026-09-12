#!/usr/bin/env python3
"""Verify the occupied bandwidth of a generated IQ file via FFT/Welch PSD.

Reads <file>.meta.json (written by generate_signal.py) alongside the IQ
file for sample rate etc., unless overridden on the command line.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ccsds_chain.utils import unpack_iq_interleaved
from ccsds_chain.spectrum import welch_psd, contiguous_bandwidth, null_to_null_bandwidth


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("iq_file")
    ap.add_argument("--sample-rate", type=float, default=None,
                    help="override meta.json output_sample_rate (Hz)")
    ap.add_argument("--dtype", choices=["float32", "int16"], default=None,
                    help="override meta.json output_dtype")
    ap.add_argument("--nperseg", type=int, default=4096)
    ap.add_argument("--plot", type=str, default=None, help="save spectrum plot PNG to this path")
    args = ap.parse_args()

    meta_path = args.iq_file.rsplit(".", 1)[0] + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    fs = args.sample_rate or meta.get("output_sample_rate") or meta.get("sample_rate")
    if fs is None:
        raise SystemExit("sample rate unknown: pass --sample-rate or generate the matching .meta.json")
    dtype = args.dtype or meta.get("output_dtype", "float32")

    with open(args.iq_file, "rb") as f:
        iq = unpack_iq_interleaved(f.read(), dtype)
    print(f"File: {args.iq_file}  ({len(iq)} campioni IQ @ {fs / 1e6:.3f} MS/s, formato {dtype})")

    psd = welch_psd(iq, args.nperseg)
    freqs = np.fft.fftshift(np.fft.fftfreq(args.nperseg, d=1 / fs))

    peak = np.max(psd)
    db = 10 * np.log10(psd / peak)

    bw_3db = contiguous_bandwidth(freqs, db, -3.0)
    bw_null = null_to_null_bandwidth(freqs, db)

    symbol_rate = meta.get("symbol_rate")
    alpha = meta.get("rrc_alpha")
    print(f"Banda occupata -3dB:      {bw_3db / 1e6:.4f} MHz")
    print(f"Banda null-nullo:         {bw_null / 1e6:.4f} MHz")
    if symbol_rate and alpha is not None:
        print(f"Attese (Rs={symbol_rate / 1e6:.3f} MHz, alpha={alpha}): "
              f"~{symbol_rate / 1e6:.2f}-{symbol_rate * 1.05 / 1e6:.2f} MHz (-3dB), "
              f"~{symbol_rate * (1 + alpha) / 1e6:.2f} MHz (null-nullo, teorico Rs*(1+alpha))")

    if args.plot:
        plt.figure(figsize=(9, 5))
        plt.plot(freqs / 1e6, db)
        plt.axhline(-3, color="orange", linestyle="--", linewidth=0.8, label="-3 dB")
        plt.axvline(-bw_null / 2e6, color="red", linestyle="--", linewidth=0.8, label="null")
        plt.axvline(bw_null / 2e6, color="red", linestyle="--", linewidth=0.8)
        plt.xlabel("Frequenza (MHz)")
        plt.ylabel("PSD relativa (dB)")
        plt.title(f"Spettro segnale generato ({os.path.basename(args.iq_file)})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"Grafico salvato: {args.plot}")


if __name__ == "__main__":
    main()
