#!/usr/bin/env python3
"""Analyze a real recorded CCSDS downlink (RF-Catcher .rfcatcher or raw IQ).

Reads a slice of the recording and reports, step by step:

  1. file format (RF-Catcher .rfcatcher files are tar archives around a raw
     int16 IQ member -- detected and unwrapped automatically)
  2. symbol rate (cyclostationary |x|^2 line) and its offset from nominal
  3. modulation, carrier offset and drift (4th/2nd-power line), i.e.
     residual Doppler
  4. spectrum: how many discrete spectral lines sit on top of the
     occupied band (a clean, random-data signal has almost none)
  5. demodulation: RRC matched filter, timing, carrier phase -> Es/N0,
     IQ gain/phase imbalance, residual phase noise
  6. decoding: Viterbi (K=7 r=1/2) + ASM search, CADU length, which
     pseudo-randomizer the link uses (none/short/long), TM primary header
     fields (SCID, VCIDs, idle frames) and how repetitive the on-air
     content is (idle frames + a short PN make a nearly periodic signal)

Example (the AWS_2 pass):
    python analyze_recording.py C:\\RF-Catcher\\AWS_2_split.rfcatcher --fs 10e6 \\
        --duration 1 --plot aws2.png
"""

import argparse
import os
import sys
from fractions import Fraction
from typing import Callable, Optional

import numpy as np
import scipy.fft as sfft
import scipy.signal as ss

from ccsds_chain.pulse_shaping import rrc_taps
from ccsds_chain.scrambler import pn_sequence
from ccsds_chain.utils import bits_to_bytes, bytes_to_bits
from ccsds_chain.viterbi import viterbi_decode

ASM = bytes.fromhex("1ACFFC1D")
ASM_BITS = bytes_to_bits(ASM).astype(np.int8)


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------

def locate_iq(path: str, header_bytes: int = 0) -> tuple[int, int, str]:
    """(data offset, data length in bytes, description) of the IQ samples
    in `path`. An RF-Catcher .rfcatcher file is a POSIX tar archive whose
    first member is the raw IQ file; anything else is taken as raw IQ after
    `header_bytes`."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(512)
    if len(head) == 512 and head[257:262] == b"ustar":
        name = head[:100].split(b"\0")[0].decode(errors="replace")
        size = int(head[124:136].split(b"\0")[0].strip() or b"0", 8)
        available = min(size, file_size - 512)
        desc = f"tar archive, member '{name}' ({size / 1e9:.2f} GB"
        desc += ")" if available == size else f", only {available / 1e6:.1f} MB present in this file)"
        return 512, available, desc
    return header_bytes, file_size - header_bytes, "raw IQ"


def load_iq(path: str, fs: float, offset_s: float, duration_s: float,
            dtype: str = "int16", header_bytes: int = 0) -> tuple[np.ndarray, str]:
    """Complex IQ slice [offset_s, offset_s + duration_s) of the recording,
    read through a memory map so multi-GB files are fine."""
    data_offset, data_bytes, desc = locate_iq(path, header_bytes)
    np_dtype = np.dtype("<i2") if dtype == "int16" else np.dtype("<f4")
    n_total = data_bytes // (2 * np_dtype.itemsize)
    start = int(offset_s * fs)
    stop = min(n_total, start + int(duration_s * fs))
    if start >= n_total:
        raise SystemExit(f"--offset {offset_s} s is past the end of the data ({n_total / fs:.2f} s)")
    raw = np.memmap(path, dtype=np_dtype, mode="r", offset=data_offset, shape=(2 * n_total,))
    seg = np.asarray(raw[2 * start:2 * stop], dtype=np.float32)
    desc += f"; {n_total / fs:.2f} s of IQ at {fs / 1e6:g} Msps, analyzing {start / fs:.2f}-{stop / fs:.2f} s"
    return (seg[0::2] + 1j * seg[1::2]).astype(np.complex64), desc


# --------------------------------------------------------------------------
# 2-4. Spectrum, symbol rate, carrier
# --------------------------------------------------------------------------

def _peak_freq(z: np.ndarray, fs: float, lo: float, hi: float, pad: int = 2) -> tuple[float, float]:
    """Frequency of the strongest FFT line of `z` within [lo, hi] (parabolic
    interpolation between bins), and its height over the band median in dB."""
    n_fft = sfft.next_fast_len(len(z) * pad)
    spec = np.abs(sfft.fft(z, n_fft))
    freqs = np.fft.fftfreq(n_fft, 1 / fs)
    band = np.where((freqs >= lo) & (freqs <= hi))[0]
    k = band[np.argmax(spec[band])]
    a, b, c = spec[(k - 1) % n_fft], spec[k], spec[(k + 1) % n_fft]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return freqs[k] + delta * fs / n_fft, 20 * np.log10(b / np.median(spec[band]))


def estimate_symbol_rate(x: np.ndarray, fs: float, rs_nominal: float) -> float:
    """Symbol rate from the cyclostationary line of |x|^2."""
    n = min(len(x), 2 ** 23)
    env = np.abs(x[:n]).astype(np.float64) ** 2
    rs, _ = _peak_freq(env - env.mean(), fs, 0.9 * rs_nominal, 1.1 * rs_nominal)
    return rs


def estimate_carrier(x: np.ndarray, fs: float, max_offset: float, power: int,
                     n_chunks: int = 8) -> tuple[float, float, list]:
    """Carrier offset at the slice midpoint and its drift rate (Hz/s), from
    the `power`-th power line (4 for QPSK, 2 for BPSK) in `n_chunks` pieces."""
    L = len(x) // n_chunks
    times, freqs = [], []
    for i in range(n_chunks):
        chunk = x[i * L:(i + 1) * L].astype(np.complex128) ** power
        f, _ = _peak_freq(chunk, fs, -power * max_offset, power * max_offset, pad=4)
        times.append((i + 0.5) * L / fs)
        freqs.append(f / power)
    rate, f_mid = np.polyfit(np.array(times) - len(x) / fs / 2, freqs, 1)
    return f_mid, rate, freqs


def detect_modulation(x: np.ndarray, fs: float, max_offset: float) -> str:
    """BPSK if the squared signal already shows a strong carrier line."""
    n = min(len(x), 2 ** 21)
    z = x[:n].astype(np.complex128)
    _, snr2 = _peak_freq(z ** 2, fs, -2 * max_offset, 2 * max_offset)
    _, snr4 = _peak_freq(z ** 4, fs, -4 * max_offset, 4 * max_offset)
    return "BPSK" if snr2 > snr4 - 3 else "QPSK"


def count_spectral_lines(x: np.ndarray, fs: float, fc: float, rs: float, alpha: float,
                         nperseg: int = 65536) -> tuple[np.ndarray, np.ndarray, int]:
    """PSD (dB, centered) and the number of discrete lines standing >6 dB
    above the local spectrum inside the signal's occupied band."""
    f, p = ss.welch(x[:min(len(x), 8_000_000)], fs=fs, nperseg=nperseg,
                    return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    pdb = 10 * np.log10(p + 1e-30)
    excess = pdb - ss.medfilt(pdb, 301)
    in_band = np.abs(f - fc) < (1 + alpha) * rs / 2
    hits = np.where(in_band & (excess > 6))[0]
    n_lines = 0 if len(hits) == 0 else 1 + int(np.sum(np.diff(hits) > 3))
    return f, pdb, n_lines


# --------------------------------------------------------------------------
# 5. Demodulation
# --------------------------------------------------------------------------

def demodulate(x: np.ndarray, fs: float, rs: float, fc: float, fdot: float, alpha: float,
               sps: int = 8, block: int = 20000) -> np.ndarray:
    """Matched-filtered symbols (carrier phase not yet removed): mix down,
    resample to ~`sps` samples/symbol, RRC filter, then pick the timing
    phase maximizing symbol energy, tracked block by block (+-1 sample,
    never wrapped, so no symbol is ever dropped or repeated)."""
    t = np.arange(len(x)) / fs
    t -= t[len(t) // 2]
    y = x * np.exp(-2j * np.pi * (fc * t + 0.5 * fdot * t ** 2)).astype(np.complex64)
    r = Fraction(sps * rs / fs).limit_denominator(1000)
    y = ss.resample_poly(y, r.numerator, r.denominator)
    sps_eff = fs * r.numerator / r.denominator / rs
    y = ss.fftconvolve(y, rrc_taps(alpha, 16, sps), mode="same")

    n_sym = int((len(y) - 2 * sps_eff) / sps_eff)
    out = np.empty(n_sym, dtype=np.complex64)

    def sample(idx):
        # Linear interpolation, done directly: np.interp on y.real/y.imag
        # would copy the whole (non-contiguous) signal on every call.
        i0 = np.clip(np.floor(idx).astype(np.int64), 0, len(y) - 2)
        frac = idx - i0
        return y[i0] * (1 - frac) + y[i0 + 1] * frac

    t0 = None
    for b in range(0, n_sym, block):
        idx = np.arange(b, min(b + block, n_sym)) * sps_eff
        candidates = np.arange(0, sps_eff, 0.25) if t0 is None else t0 + np.arange(-1, 1.01, 0.125)
        energies = [np.mean(np.abs(sample(idx + c)) ** 2) for c in candidates]
        t0 = candidates[int(np.argmax(energies))]
        out[b:b + len(idx)] = sample(idx + t0)
    return out


def remove_carrier_phase(s: np.ndarray, power: int, block: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise `power`-th power phase estimate, unwrapped and removed.
    Returns gain-normalized symbols on the ideal constellation and the
    per-block phase track (radians)."""
    nb = len(s) // block
    s = s[:nb * block].reshape(nb, block)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    ph = np.unwrap(np.angle(np.sum(s ** power, axis=1))) / power
    rot = np.pi / 4 if power == 4 else 0.0
    d = (s * np.exp(-1j * (ph[:, None] + rot))).ravel()
    # Scale so the ideal points sit at +-1/sqrt(2) (QPSK) or +-1 (BPSK).
    scale = np.mean(np.abs(d.real)) * (np.sqrt(2) if power == 4 else 1.0)
    return d / scale, ph


def symbol_metrics(d: np.ndarray, modulation: str) -> dict:
    """Es/N0 (decision-aided EVM and blind M2M4), IQ imbalance."""
    if modulation == "QPSK":
        ref = (np.sign(d.real) + 1j * np.sign(d.imag)) / np.sqrt(2)
    else:
        ref = np.sign(d.real).astype(np.complex64)
    g = np.vdot(ref, d) / np.vdot(ref, ref)
    dn = d / g
    evm = np.sqrt(np.mean(np.abs(dn - ref) ** 2) / np.mean(np.abs(ref) ** 2))
    m2, m4 = np.mean(np.abs(dn) ** 2), np.mean(np.abs(dn) ** 4)
    s_hat = np.sqrt(max(2 * m2 ** 2 - m4, 1e-12))
    out = {"esn0_evm_db": -20 * np.log10(evm), "esn0_m2m4_db": 10 * np.log10(s_hat / max(m2 - s_hat, 1e-12)),
           "evm_pct": 100 * evm}
    if modulation == "QPSK":
        i, q = dn.real, dn.imag
        out["iq_gain_db"] = 20 * np.log10(np.sqrt(np.mean(i ** 2)) / np.sqrt(np.mean(q ** 2)))
        out["iq_skew_deg"] = np.degrees(np.arcsin(np.clip(np.mean(i * q) / np.sqrt(np.mean(i ** 2) * np.mean(q ** 2)), -1, 1)))
    return out


def phase_noise_bands(ph: np.ndarray, rs: float, block: int = 128) -> dict:
    """RMS residual carrier phase (deg) per offset-frequency band, after
    removing a 2nd-order polynomial (residual frequency and Doppler rate)."""
    tb = block / rs
    t = np.arange(len(ph)) * tb
    resid = ph - np.polyval(np.polyfit(t, ph, 2), t)
    f, p = ss.welch(resid, fs=1 / tb, nperseg=min(len(resid), 4096))
    bands = {}
    for lo, hi in [(10, 100), (100, 1000), (1000, 0.5 / tb)]:
        m = (f >= lo) & (f < hi)
        bands[f"{lo:g}-{hi:.0f} Hz"] = np.degrees(np.sqrt(np.sum(p[m]) * (f[1] - f[0])))
    return bands


# --------------------------------------------------------------------------
# 6. Decoding
# --------------------------------------------------------------------------

def _hard_bits(d: np.ndarray, modulation: str, rot: int, conj: bool) -> np.ndarray:
    z = d * np.exp(1j * np.pi / 2 * rot)
    z = np.conj(z) if conj else z
    if modulation == "BPSK":
        return (z.real > 0).astype(np.uint8)
    bits = np.empty(2 * len(z), dtype=np.uint8)
    bits[0::2], bits[1::2] = z.real > 0, z.imag > 0
    return bits


def find_asm(bits: np.ndarray, min_match: int = 30) -> np.ndarray:
    corr = np.correlate(2 * bits.astype(np.int8) - 1, 2 * ASM_BITS - 1, "valid")
    return np.where(corr >= 2 * min_match - 32)[0]


def decode(d: np.ndarray, modulation: str, max_symbols: int) -> tuple[Optional[np.ndarray], str]:
    """Info bits of the best of 8 (phase, I/Q swap) hypotheses x {K=7 r=1/2
    with/without G2 inversion, uncoded}: the one showing the most ASMs."""
    probe = d[:20000]
    rots = range(4) if modulation == "QPSK" else range(2)
    hits = []  # (ASM count, deviations from the standard convention, hypothesis)
    for conj in (False, True):
        for rot in rots:
            coded = _hard_bits(probe, modulation, rot, conj)
            even = coded[:len(coded) // 2 * 2]
            candidates = [(None, coded)] + [
                (inv, viterbi_decode(even, len(even) // 2, "1/2", invert_g2=inv)) for inv in (True, False)]
            for inv, bits in candidates:
                for polarity in (0, 1):
                    n = len(find_asm(bits ^ polarity))
                    hits.append((n, 2 * int(inv is False) + polarity + int(conj), (rot, conj, inv, polarity)))
    max_count = max(h[0] for h in hits)
    if max_count < 2:
        return None, "no ASM found under any hypothesis (not K=7 r=1/2 or uncoded CCSDS?)"
    # Several hypotheses are equivalent for QPSK (inverting G2 flips every Q
    # bit, which an I/Q swap/conjugation mimics); among those finding
    # (nearly) every ASM, report the one closest to the standard convention
    # (G2 inverted, no swap, no polarity flip).
    best = min((h for h in hits if h[0] >= max_count - 1), key=lambda h: h[1])[2]
    rot, conj, inv, polarity = best
    label = "uncoded" if inv is None else f"conv K=7 r1/2 (G2 {'inverted' if inv else 'not inverted'})"
    coded = _hard_bits(d[:max_symbols], modulation, rot, conj)
    if inv is None:
        bits = coded
    else:
        coded = coded[:len(coded) // 2 * 2]
        bits = viterbi_decode(coded, len(coded) // 2, "1/2", invert_g2=inv)
    desc = f"{label}, phase rotation {90 * rot} deg{', I/Q swapped' if conj else ''}" + \
           (", inverted polarity" if polarity else "")
    if modulation == "QPSK" and inv is not None:
        desc += " -- with QPSK, G2 inversion and an I/Q swap can't be told apart"
    return bits ^ polarity, desc


def analyze_frames(bits: np.ndarray) -> dict:
    """CADU length, randomizer and TM primary header statistics."""
    pos = find_asm(bits)
    if len(pos) < 2:
        return {"error": "fewer than 2 ASMs in the decoded stream"}
    spacing = np.diff(pos)
    cadu_bits = int(np.bincount(spacing).argmax())
    cadus = [bits_to_bytes(bits[p:p + cadu_bits]) for p in pos if p + cadu_bits <= len(bits)]
    body_len = cadu_bits // 8 - len(ASM)
    bodies = np.array([np.frombuffer(c[len(ASM):], np.uint8) for c in cadus])

    def header_score(frames):
        mcfc = frames[:, 2].astype(int)
        version_ok = np.mean((frames[:, 0] >> 6) == 0)
        counter_ok = np.mean((np.diff(mcfc) % 256) == 1) if len(frames) > 1 else 0
        return version_ok + counter_ok

    scored = {}
    for mode in ("none", "short", "long"):
        pn = 0 if mode == "none" else np.packbits(pn_sequence(body_len * 8, mode))
        scored[mode] = (header_score(bodies ^ pn), bodies ^ pn)
    randomizer = max(scored, key=lambda m: scored[m][0])
    frames = scored[randomizer][1]

    words = (frames[:, 0].astype(int) << 8) | frames[:, 1]
    fhp = ((frames[:, 4].astype(int) << 8) | frames[:, 5]) & 0x7FF
    vcids, counts = np.unique((words >> 1) & 7, return_counts=True)
    fill_vals, fill_counts = np.unique(frames[:, 6:], return_counts=True)
    changed = (bodies[1:] != bodies[:-1]).sum(axis=1)
    return {
        "n_cadu": len(cadus),
        "cadu_bytes": cadu_bits // 8,
        "asm_spacing_consistent": bool(np.all(spacing % cadu_bits == 0)),
        "randomizer": randomizer,
        "randomizer_confidence": scored[randomizer][0] / 2,
        "scid": sorted(set(((words >> 4) & 0x3FF).tolist())),
        "vcid_counts": dict(zip(vcids.tolist(), counts.tolist())),
        "idle_frames_pct": 100 * np.mean(fhp == 0x7FE),
        "dominant_data_byte": (int(fill_vals[fill_counts.argmax()]), 100 * fill_counts.max() / frames[:, 6:].size),
        "bytes_changed_between_cadus": float(np.median(changed)) if len(changed) else None,
        "cadu_bits": cadu_bits,
        # ASM + de-randomized frame per CADU: the "asm_frame" input format,
        # ready to feed back into the generator.
        "records": b"".join(ASM + row.tobytes() for row in frames),
    }


def periodicity(d: np.ndarray, lag: int, modulation: str) -> float:
    """Normalized correlation of the hard symbol decisions with themselves
    `lag` symbols later (~0 for random data, ->1 for a repeating pattern)."""
    if lag <= 0 or lag >= len(d):
        return float("nan")
    h = np.sign(d.real) + (1j * np.sign(d.imag) if modulation == "QPSK" else 0)
    return float(np.abs(np.vdot(h[:-lag], h[lag:])) / np.vdot(h[:-lag], h[:-lag]).real)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def plot_report(path: str, r: dict, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, series = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=grid, linewidth=0.8)
        ax.tick_params(colors=ink2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)

    axes[0].plot(r["psd_freqs"] / 1e6, r["psd_db"], color=series, linewidth=1)
    axes[0].set(xlabel="Frequency (MHz)", ylabel="PSD (dB rel. peak)", title="Spectrum")

    pts = r["constellation"]
    axes[1].plot(pts.real, pts.imag, ".", color=series, markersize=1.5, alpha=0.35)
    axes[1].set(xlabel="I", ylabel="Q", title="Constellation (after carrier recovery)", aspect="equal")

    axes[2].plot(r["phase_t"] * 1e3, r["phase_resid_deg"], color=series, linewidth=0.8)
    axes[2].set(xlabel="Time (ms)", ylabel="Residual phase (deg)", title="Carrier phase residual")

    for ax in axes:
        ax.title.set_color(ink)
        ax.xaxis.label.set_color(ink2)
        ax.yaxis.label.set_color(ink2)
    fig.suptitle(title, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def analyze(path: str, fs: float, offset_s: float = 0.0, duration_s: float = 1.0,
            dtype: str = "int16", header_bytes: int = 0, rs_nominal: float = 1.785e6,
            alpha: float = 0.35, max_carrier_offset: float = 200e3, modulation: str = "auto",
            do_decode: bool = True, decode_symbols: int = 400_000,
            progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Run every step on a slice of the recording; returns the measurements
    plus light-weight arrays for plotting (shared by the CLI and the GUI)."""
    def step(frac, msg):
        if progress is not None:
            progress(frac, msg)

    step(0.0, "Reading the recording...")
    x, desc = load_iq(path, fs, offset_s, duration_s, dtype, header_bytes)
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    full_scale = 2048 if dtype == "int16" else 1.0
    r = {"file": desc, "dtype": dtype, "rms": rms, "peak": float(np.abs(x).max()),
         "dbfs": 20 * np.log10(max(rms, 1e-12) / full_scale), "rs_nominal": rs_nominal}

    step(0.05, "Estimating symbol rate and carrier...")
    r["modulation_auto"] = modulation == "auto"
    r["modulation"] = detect_modulation(x, fs, max_carrier_offset) if modulation == "auto" else modulation
    power = 4 if r["modulation"] == "QPSK" else 2
    r["rs"] = estimate_symbol_rate(x, fs, rs_nominal)
    r["rs_ppm"] = (r["rs"] / rs_nominal - 1) * 1e6
    r["fc"], r["fdot"], _ = estimate_carrier(x, fs, max_carrier_offset, power)

    step(0.15, "Computing the spectrum...")
    f, pdb, r["n_lines"] = count_spectral_lines(x, fs, r["fc"], r["rs"], alpha)
    r["psd_freqs"], r["psd_db"] = f, pdb - pdb.max()

    step(0.25, "Demodulating...")
    d, ph = remove_carrier_phase(demodulate(x, fs, r["rs"], r["fc"], r["fdot"], alpha), power)
    del x
    r["n_symbols"] = len(d)
    r.update(symbol_metrics(d, r["modulation"]))
    r["phase_noise"] = phase_noise_bands(ph, r["rs"])
    r["constellation"] = d[:: max(1, len(d) // 20000)]
    t = np.arange(len(ph)) * 128 / r["rs"]
    r["phase_t"], r["phase_resid_deg"] = t, np.degrees(ph - np.polyval(np.polyfit(t, ph, 2), t))

    r["decoding"], r["frames"] = None, None
    if do_decode:
        step(0.4, "Viterbi decoding (roughly 20-30 s per million symbols)...")
        bits, r["decoding"] = decode(d, r["modulation"], decode_symbols)
        if bits is not None:
            step(0.95, "Analyzing frames...")
            fr = analyze_frames(bits)
            if "error" not in fr:
                sym_per_cadu = fr["cadu_bits"] * (2 if "conv" in r["decoding"] else 1) // (
                    2 if r["modulation"] == "QPSK" else 1)
                fr["sym_per_cadu"] = sym_per_cadu
                fr["periodicity"] = periodicity(d, sym_per_cadu, r["modulation"])
            r["frames"] = fr
    step(1.0, "Done")
    return r


def report_lines(r: dict) -> list:
    """Human-readable report of an `analyze()` result."""
    lines = [f"[1] File: {r['file']}",
             f"    Level: rms {r['rms']:.1f}, peak {r['peak']:.1f} ({r['dbfs']:.1f} dBFS rms"
             f"{' vs 12-bit full scale' if r['dtype'] == 'int16' else ''})",
             f"[2] Symbol rate: {r['rs']:,.1f} sps  ({r['rs'] - r['rs_nominal']:+,.1f} Hz, {r['rs_ppm']:+.1f} ppm "
             f"vs nominal {r['rs_nominal']:,.0f})".replace(",", " "),
             f"[3] Modulation: {r['modulation']}{' (auto-detected)' if r['modulation_auto'] else ''}; "
             f"carrier offset {r['fc'] / 1e3:+.3f} kHz, drift {r['fdot']:+.1f} Hz/s over the slice",
             f"[4] Spectrum: {r['n_lines']} discrete lines >6 dB above the local spectrum inside the occupied "
             f"band ({'random-like data' if r['n_lines'] < 20 else 'strongly periodic content'})",
             f"[5] Demodulated {r['n_symbols']:,} symbols".replace(",", " "),
             f"    Es/N0: {r['esn0_evm_db']:.1f} dB (EVM {r['evm_pct']:.1f}%), blind M2M4 {r['esn0_m2m4_db']:.1f} dB"]
    if r["modulation"] == "QPSK":
        lines.append(f"    IQ imbalance: gain {r['iq_gain_db']:+.2f} dB, quadrature skew {r['iq_skew_deg']:+.2f} deg")
    lines.append("    Residual phase rms: " + ", ".join(f"{k}: {v:.2f} deg" for k, v in r["phase_noise"].items()))
    if r["decoding"] is not None:
        lines.append(f"[6] Decoding: {r['decoding']}")
        fr = r["frames"]
        if fr is not None and "error" in fr:
            lines.append(f"    {fr['error']}")
        elif fr is not None:
            byte, share = fr["dominant_data_byte"]
            lines += [
                f"    {fr['n_cadu']} CADUs of {fr['cadu_bytes']} bytes (ASM every {fr['cadu_bits']} bits"
                f"{'' if fr['asm_spacing_consistent'] else ', with gaps'})",
                f"    Pseudo-randomizer: {fr['randomizer']} "
                f"(header check passes on {100 * fr['randomizer_confidence']:.0f}% of frames)",
                f"    TM header: SCID {fr['scid']}, frames per VCID {fr['vcid_counts']}, "
                f"idle frames (FHP=0x7FE) {fr['idle_frames_pct']:.0f}%",
                f"    Data field: most common byte 0x{byte:02X} ({share:.0f}% of it); median "
                f"{fr['bytes_changed_between_cadus']:.0f} of {fr['cadu_bytes'] - 4} bytes change from one "
                "CADU to the next on air",
                f"    Symbol correlation one CADU apart ({fr['sym_per_cadu']} symbols): {fr['periodicity']:.2f} "
                f"({'nearly periodic signal' if fr['periodicity'] > 0.5 else 'random-like'})",
            ]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help=".rfcatcher (tar) or raw interleaved IQ file")
    ap.add_argument("--fs", type=float, required=True, help="recording sample rate, Hz (e.g. 10e6)")
    ap.add_argument("--dtype", choices=["int16", "float32"], default="int16")
    ap.add_argument("--header-bytes", type=int, default=0, help="bytes to skip in a raw (non-tar) file")
    ap.add_argument("--offset", type=float, default=0.0, help="start of the analyzed slice, s")
    ap.add_argument("--duration", type=float, default=1.0, help="length of the analyzed slice, s")
    ap.add_argument("--rs-nominal", type=float, default=1.785e6, help="expected symbol rate, sps")
    ap.add_argument("--alpha", type=float, default=0.35, help="RRC roll-off of the matched filter")
    ap.add_argument("--max-carrier-offset", type=float, default=200e3, help="carrier search range, +-Hz")
    ap.add_argument("--modulation", choices=["auto", "QPSK", "BPSK"], default="auto")
    ap.add_argument("--decode-symbols", type=int, default=400_000,
                    help="max symbols to Viterbi-decode (default ~40 CADUs, ~15 s; roughly 20-30 s per million)")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--plot", type=str, default=None, help="save a spectrum/constellation/phase PNG here")
    ap.add_argument("--save-frames", type=str, default=None,
                    help="write the decoded frames here as ASM + de-randomized Transfer Frame records "
                         "(the generator's 'asm_frame' input format)")
    args = ap.parse_args()

    r = analyze(args.recording, args.fs, args.offset, args.duration, args.dtype, args.header_bytes,
                args.rs_nominal, args.alpha, args.max_carrier_offset, args.modulation,
                not args.no_decode, args.decode_symbols)
    print("\n".join(report_lines(r)))
    if args.save_frames and r["frames"] and "records" in r["frames"]:
        with open(args.save_frames, "wb") as fh:
            fh.write(r["frames"]["records"])
        print(f"Decoded frames saved to {args.save_frames}")
    if args.plot:
        plot_report(args.plot, r, os.path.basename(args.recording))
        print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    sys.exit(main())
