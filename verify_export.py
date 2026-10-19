#!/usr/bin/env python3
"""Verify a GENERATED export end to end, before blaming the playback chain.

Scans the whole file (an RF-Catcher .rfcatcher or raw int16/float32 IQ, any
size, streamed in chunks) with the transmitter's own parameters and reports,
with the time in the file where it happens:

  * level: RMS and peak per window, clipping, runs of zero samples (a gap)
  * quality: decision-directed MER per window -- constant along a clean
    export (about 35 dB at roll-off 0.35, 55 dB at roll-off 1, limited by
    the RRC filter's own residual ISI); a drop means distortion or a timing
    break there
  * frame stream: every CADU's ASM is found directly in the demodulated,
    still convolutionally coded bits (the last 26 ASM bits always produce
    the same coded pattern, whatever precedes them -- no Viterbi needed),
    and the spacing between consecutive ASMs must be exactly one CADU. Any
    other spacing is a slip: samples missing or repeated in the file.

The generator's timing is known exactly (integer samples per symbol, no
carrier offset unless impairments were enabled), so no synchronization
loop is involved: anything reported is in the FILE. If this reports a clean
file but the demodulator still loses frames, the problem is in playback or
in the receiving chain.

Example (QPSK 1.785 Msps, roll-off 1, I=5, rate 1/2):
    python verify_export.py C:\\data\\test.rfcatcher --symbol-rate 1.785e6 --alpha 1
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

from ccsds_chain import rfcatcher
from ccsds_chain.convolutional import K, PUNCTURE_PATTERNS, _puncture, _rate_half_encode_from_history
from ccsds_chain.mapping import BITS_PER_SYMBOL
from ccsds_chain.pulse_shaping import pulse_taps
from ccsds_chain.utils import INT16_FULL_SCALE, bytes_to_bits

ASM = bytes.fromhex("1ACFFC1D")
CHUNK_SYMBOLS = 1_000_000


def locate_iq(path: str) -> tuple[int, int]:
    """(offset, length in bytes) of the IQ samples: the first tar member of a
    .rfcatcher, else the whole file."""
    import tarfile
    with open(path, "rb") as f:
        head = f.read(512)
    if len(head) == 512 and head[257:262] == b"ustar":
        with open(path, "rb") as f:
            member = tarfile.open(fileobj=f, mode="r:").next()
        return member.offset_data, min(member.size, os.path.getsize(path) - member.offset_data)
    return 0, os.path.getsize(path)


def coded_asm_patterns(rate: str, invert_g2: bool) -> list[tuple[np.ndarray, int]]:
    """For each puncture phase at the ASM's first bit: (coded bits produced by
    ASM bits K-1..31, which don't depend on what came before the ASM, and the
    number of coded bits the first K-1 ASM bits produce before them)."""
    asm_bits = bytes_to_bits(ASM)
    g1, g2, _ = _rate_half_encode_from_history(asm_bits, invert_g2 and rate == "1/2",
                                               np.zeros(K - 1, dtype=np.uint8))
    pattern = PUNCTURE_PATTERNS[rate]
    period = 1 if pattern is None else len(pattern[0])
    out = []
    for phase in range(period):
        head, _ = _puncture(g1[:K - 1], g2[:K - 1], rate, phase)
        tail, _ = _puncture(g1[K - 1:], g2[K - 1:], rate, (phase + K - 1) % period)
        out.append((tail, len(head)))
    return out


def coded_length(n_bits: int, rate: str, phase: int) -> int:
    """Coded bits produced by `n_bits` input bits starting at puncture `phase`."""
    pattern = PUNCTURE_PATTERNS[rate]
    if pattern is None:
        return 2 * n_bits
    period = len(pattern[0])
    keep = np.array(pattern[0], dtype=int) + np.array(pattern[1], dtype=int)
    idx = (np.arange(n_bits) + phase) % period
    return int(keep[idx].sum())


def verify(path: str, symbol_rate: float, fs: Optional[float] = None, dtype: str = "int16",
           modulation: str = "QPSK", pulse: str = "rrc", alpha: float = 0.35, span: int = 8,
           conv_rate: str = "1/2", invert_g2: bool = True, interleave_depth: int = 5,
           rs_n: int = 255, window_s: float = 1.0, max_asm_errors: int = 2,
           progress=None) -> dict:
    if fs is None:
        meta = rfcatcher.read_metadata(path) or {}
        fs = rfcatcher.parse_quantity(meta.get("rate"))
        if not fs:
            raise ValueError("no sample rate in the file's metadata: pass --fs")
    sps_f = fs / symbol_rate
    sps = int(round(sps_f))
    if abs(sps_f - sps) > 1e-6:
        raise ValueError(f"sample rate {fs:.0f} is not an integer multiple of the symbol rate "
                         f"{symbol_rate:.0f} ({sps_f:.4f} samples/symbol): resampled exports "
                         "can't be checked symbol-exactly by this tool")
    taps = pulse_taps(pulse, alpha, span, sps)
    d = len(taps) - 1                     # TX + RX filter delay, samples
    bps = BITS_PER_SYMBOL[modulation]
    sample_bytes = 4 if dtype == "int16" else 8
    offset, nbytes = locate_iq(path)
    n_samples = nbytes // sample_bytes
    n_sym = (n_samples - d) // sps
    cadu_bits = (len(ASM) + rs_n * interleave_depth) * 8

    patterns = coded_asm_patterns(conv_rate, invert_g2)
    plen = len(patterns[0][0])
    pm = [(2.0 * p.astype(np.float32) - 1.0)[::-1] for p, _ in patterns]
    threshold = plen - 2 * max_asm_errors

    mm = np.memmap(path, dtype="<i2" if dtype == "int16" else "<f4", mode="r",
                   offset=offset, shape=(2 * n_samples,))
    scale = 1.0 / INT16_FULL_SCALE if dtype == "int16" else 1.0

    win = max(1, int(round(window_s * fs)))
    windows = {}
    asm_hits = []                          # (coded bit index of ASM bit 0, phase)
    carry = np.empty(0, dtype=np.float32)  # last coded bits of the previous chunk (as +-1)
    carry_start = 0
    zero_runs = []
    prev_zero_tail = 0

    for k0 in range(0, n_sym, CHUNK_SYMBOLS):
        k1 = min(k0 + CHUNK_SYMBOLS, n_sym)
        s_lo, s_hi = k0 * sps, (k1 - 1) * sps + d + 1      # samples feeding these symbols
        raw = np.asarray(mm[2 * s_lo:2 * s_hi], dtype=np.float32) * scale
        x = raw[0::2] + 1j * raw[1::2]
        # matched filter evaluated only at the symbol instants (polyphase)
        idx = d + np.arange(k1 - k0) * sps                # inside x
        s = np.zeros(k1 - k0, dtype=np.complex64)
        for j, h in enumerate(taps):
            s += h * x[idx - j]

        # level, clipping, gaps -- per sample, on this chunk's new samples
        new = x[:(k1 - k0) * sps]
        a = np.abs(new)
        axis_peak = np.maximum(np.abs(new.real), np.abs(new.imag))
        clip = (np.abs(raw[:2 * len(new)]) >= 2047 * scale) if dtype == "int16" else np.zeros(1, bool)
        zero = (new == 0)
        if zero.any():
            # runs of >= 2 symbols of exact zeros: a hole in the file
            edges = np.flatnonzero(np.diff(np.concatenate([[prev_zero_tail > 0], zero, [False]]).astype(np.int8)))
            for st, en in zip(edges[0::2], edges[1::2]):
                if en - st >= 2 * sps:
                    zero_runs.append(((s_lo + st) / fs, (en - st) / fs))
        prev_zero_tail = 0

        # decision-directed MER per window
        if modulation == "QPSK":
            ref = (np.sign(s.real) + 1j * np.sign(s.imag))
        else:
            ref = np.sign(s.real) + 0j
        g = np.vdot(ref, s) / np.vdot(ref, ref)
        err = np.abs(s - g * ref) ** 2
        sig = np.abs(g * ref) ** 2
        w_idx = (s_lo + np.arange(len(s)) * sps) // win
        for w in np.unique(w_idx):
            m = w_idx == w
            rec = windows.setdefault(int(w), [0.0, 0.0, 0.0, 0.0, 0, 0])
            rec[0] += float(sig[m].sum()); rec[1] += float(err[m].sum())
            sm = (np.arange(len(new)) + s_lo) // win == w
            rec[2] += float((a[sm] ** 2).sum()); rec[3] = max(rec[3], float(axis_peak[sm].max()) if sm.any() else 0)
            rec[4] += int(sm.sum()); rec[5] += int(clip[np.repeat(sm, 2)].sum()) if dtype == "int16" else 0

        # hard coded bits, then the ASM pattern search across the chunk boundary
        if modulation == "QPSK":
            bits = np.empty(2 * len(s), dtype=np.float32)
            bits[0::2] = np.where(s.real > 0, 1.0, -1.0); bits[1::2] = np.where(s.imag > 0, 1.0, -1.0)
        else:
            bits = np.where(s.real > 0, 1.0, -1.0).astype(np.float32)
        stream = np.concatenate([carry, bits])
        base = carry_start
        for phase, (pat, (_, head_len)) in enumerate(zip(pm, patterns)):
            corr = np.convolve(stream, pat, mode="valid")
            for pos in np.flatnonzero(corr >= threshold):
                asm_hits.append((base + int(pos) - head_len, phase))
        keep = plen - 1
        carry = stream[-keep:]
        carry_start = base + len(stream) - keep
        if progress:
            progress(k1 / n_sym)

    # ASM spacing: exactly one CADU's coded length between consecutive ASMs
    asm_hits = sorted(set(asm_hits))
    # A genuine ASM has a neighbour exactly one CADU before or after it; a
    # chance match in the data (possible with the short coded patterns of the
    # high punctured rates) has neither.
    positions = {pos for pos, _ in asm_hits}
    nxt = {pos: pos + coded_length(cadu_bits, conv_rate, ph) for pos, ph in asm_hits}
    has_prev = set(nxt.values())
    asm_hits = [(pos, ph) for pos, ph in asm_hits if nxt[pos] in positions or pos in has_prev]
    slips = []
    for (p0, ph0), (p1, _) in zip(asm_hits, asm_hits[1:]):
        expected = coded_length(cadu_bits, conv_rate, ph0)
        if p1 - p0 != expected:
            t = (d + (p1 // bps) * sps) / fs
            slips.append({"time_s": t, "spacing_bits": p1 - p0, "expected_bits": expected})
    num, den = (1, 2) if conv_rate == "1/2" else map(int, conv_rate.split("/"))
    expected_asms = n_sym * bps / (cadu_bits * den / num)     # coded bits / coded bits per CADU

    rows = []
    for w in sorted(windows):
        sig, err, pw, pk, n, clipped = windows[w]
        rows.append({"t_s": w * win / fs, "mer_db": 10 * np.log10(sig / err) if err > 0 else float("inf"),
                     "rms": np.sqrt(pw / max(n, 1)) * (INT16_FULL_SCALE if dtype == "int16" else 1),
                     "peak": pk * (INT16_FULL_SCALE if dtype == "int16" else 1), "clipped": clipped})
    return {"fs": fs, "sps": sps, "duration_s": n_samples / fs, "n_symbols": n_sym,
            "asms": len(asm_hits), "expected_asms": int(expected_asms), "slips": slips,
            "zero_runs": zero_runs, "windows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help=".rfcatcher, or raw interleaved IQ")
    ap.add_argument("--symbol-rate", type=float, required=True, help="symbols/s, e.g. 1.785e6")
    ap.add_argument("--fs", type=float, default=None, help="sample rate (default: from the .rfcatcher metadata)")
    ap.add_argument("--dtype", choices=["int16", "float32"], default="int16")
    ap.add_argument("--modulation", choices=["QPSK", "BPSK"], default="QPSK")
    ap.add_argument("--pulse", choices=["rrc", "nrz"], default="rrc")
    ap.add_argument("--alpha", type=float, default=0.35, help="RRC roll-off used at generation")
    ap.add_argument("--span", type=int, default=8, help="RRC span (symbols) used at generation")
    ap.add_argument("--conv-rate", choices=list(PUNCTURE_PATTERNS), default="1/2")
    ap.add_argument("--no-invert-g2", action="store_true", help="rate 1/2 generated without G2 inversion")
    ap.add_argument("--interleave", type=int, default=5, help="RS interleave depth I (CADU = 4 + 255*I bytes)")
    ap.add_argument("--window", type=float, default=1.0, help="statistics window, s")
    args = ap.parse_args()

    def progress(frac):
        print(f"\r  scanning... {frac * 100:5.1f}%", end="", file=sys.stderr, flush=True)

    r = verify(args.file, args.symbol_rate, args.fs, args.dtype, args.modulation, args.pulse, args.alpha,
               args.span, args.conv_rate, not args.no_invert_g2, args.interleave, window_s=args.window,
               progress=progress)
    print(file=sys.stderr)
    w = r["windows"]
    mer = np.array([x["mer_db"] for x in w]); rms = np.array([x["rms"] for x in w])
    print(f"File: {r['duration_s']:.1f} s at {r['fs'] / 1e6:.3f} Msps ({r['sps']} samples/symbol)")
    print(f"Level: RMS {rms.min():.1f} .. {rms.max():.1f}, peak {max(x['peak'] for x in w):.0f}, "
          f"clipped samples {sum(x['clipped'] for x in w)}")
    print(f"MER per {args.window:g} s window: min {mer.min():.1f} dB, median {np.median(mer):.1f} dB "
          f"(a clean export: ~35 dB at roll-off 0.35, ~55 dB at 1)")
    # A window is flagged if its MER is low in absolute terms or against the rest of
    # the file, or its level differs from the median (a clean export is flat).
    low = [x for x in w if x["mer_db"] < max(25.0, np.median(mer) - 3)
           or abs(20 * np.log10(x["rms"] / np.median(rms))) > 0.5]
    for x in low[:20]:
        print(f"  anomaly at {x['t_s']:8.1f} s: MER {x['mer_db']:.1f} dB, RMS {x['rms']:.1f}")
    if len(low) > 20:
        print(f"  ... {len(low) - 20} more windows")
    print(f"Gaps (runs of zero samples): {len(r['zero_runs'])}")
    for t, dur in r["zero_runs"][:20]:
        print(f"  gap at {t:8.3f} s, {dur * 1e6:.1f} us")
    print(f"ASMs found: {r['asms']} (about {r['expected_asms']} expected); slips: {len(r['slips'])}")
    if r["slips"]:
        print("  (after a slip the timing grid is off: later MER figures are not meaningful)")
    for s in r["slips"][:20]:
        print(f"  slip at {s['time_s']:8.3f} s: ASM spacing {s['spacing_bits']} coded bits "
              f"instead of {s['expected_bits']}")
    clean = not low and not r["zero_runs"] and not r["slips"]
    print("RESULT:", "file is CLEAN -- look at playback / the receiving chain" if clean else
          "the FILE has the anomalies listed above")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
