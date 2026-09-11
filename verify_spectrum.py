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

from ccsds_chain.utils import read_iq_interleaved_float32


def welch_psd(x: np.ndarray, nperseg: int = 4096):
    window = np.hanning(nperseg)
    win_energy = np.sum(window ** 2)
    n_segs = len(x) // nperseg
    if n_segs < 1:
        raise ValueError("signal shorter than one FFT segment; reduce --nperseg or generate more CADU")
    acc = np.zeros(nperseg)
    for i in range(n_segs):
        seg = x[i * nperseg:(i + 1) * nperseg] * window
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc += np.abs(spec) ** 2
    acc /= (n_segs * win_energy)
    return acc


def contiguous_bandwidth(freqs: np.ndarray, db: np.ndarray, threshold_db: float) -> float:
    center_idx = int(np.argmin(np.abs(freqs)))
    left, right = center_idx, center_idx
    while left > 0 and db[left - 1] >= threshold_db:
        left -= 1
    while right < len(db) - 1 and db[right + 1] >= threshold_db:
        right += 1
    return freqs[right] - freqs[left]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("iq_file")
    ap.add_argument("--sample-rate", type=float, default=None, help="override meta.json sample_rate (Hz)")
    ap.add_argument("--nperseg", type=int, default=4096)
    ap.add_argument("--plot", type=str, default=None, help="save spectrum plot PNG to this path")
    args = ap.parse_args()

    meta_path = args.iq_file.rsplit(".", 1)[0] + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    fs = args.sample_rate or meta.get("sample_rate")
    if fs is None:
        raise SystemExit("sample rate unknown: pass --sample-rate or generate the matching .meta.json")

    iq = read_iq_interleaved_float32(args.iq_file)
    print(f"File: {args.iq_file}  ({len(iq)} campioni IQ @ {fs / 1e6:.3f} MS/s)")

    psd = welch_psd(iq, args.nperseg)
    freqs = np.fft.fftshift(np.fft.fftfreq(args.nperseg, d=1 / fs))

    peak = np.max(psd)
    db = 10 * np.log10(psd / peak)

    bw_3db = contiguous_bandwidth(freqs, db, -3.0)
    bw_20db = contiguous_bandwidth(freqs, db, -20.0)

    symbol_rate = meta.get("symbol_rate")
    alpha = meta.get("rrc_alpha")
    print(f"Banda occupata -3dB:  {bw_3db / 1e6:.4f} MHz")
    print(f"Banda occupata -20dB: {bw_20db / 1e6:.4f} MHz")
    if symbol_rate and alpha is not None:
        print(f"Attese (Rs={symbol_rate / 1e6:.3f} MHz, alpha={alpha}): "
              f"~{symbol_rate / 1e6:.2f}-{symbol_rate * 1.05 / 1e6:.2f} MHz (-3dB), "
              f"~{symbol_rate * (1 + alpha) / 1e6:.2f} MHz (-20dB, teorico Rs*(1+alpha))")

    if args.plot:
        plt.figure(figsize=(9, 5))
        plt.plot(freqs / 1e6, db)
        plt.axhline(-3, color="orange", linestyle="--", linewidth=0.8, label="-3 dB")
        plt.axhline(-20, color="red", linestyle="--", linewidth=0.8, label="-20 dB")
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
