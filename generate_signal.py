#!/usr/bin/env python3
"""Generate a CCSDS-chain test signal (baseline: QPSK, BPSK also available)
as raw IQ, for injection via RF-Catcher (TestTree) Capture & Playback or any
other IQ recorder/replayer.

Chain: payload -> RS(255,223 or 255,239) interleaved -> optional CCSDS
pseudo-randomizer (excludes ASM) -> ASM prepend per CADU -> convolutional
K=7, rate 1/2 (or punctured to 2/3, 3/4, 5/6, 7/8) over the CADU stream
(ASM included) -> NRZ-L -> QPSK (Gray) or BPSK -> RRC -> peak-normalize ->
optional resample -> raw interleaved IQ (float32 or int16), no header.

With an ASM-framed --input-format, the payload already starts each record
with the ASM, and RS and the ASM prepend are skipped to avoid double-
encoding. Each record is exactly 4 + 255*I bytes (1279 at I=5); anything
past that before the next ASM is receiver-added overhead and is skipped:
  asm_frame  ASM + Transfer Frame, NOT scrambled (e.g. a ground station's
             decoded output): the frame is scrambled here (never the ASM),
             --randomizer long by default ('none' is rejected).
  cadu       CADU exactly as on the air (already scrambled): used verbatim,
             --randomizer must be 'none'.

See README.md for architecture assumptions, limitations, and open TODOs
before using the output against real ground equipment. For an interactive
GUI with a real-time spectrum view, see `streamlit run app.py`.
"""

import argparse
import datetime
import json
import os

from ccsds_chain.mapping import BITS_PER_SYMBOL
from ccsds_chain.pipeline import ASM_FRAMED_INPUT_FORMATS, INPUT_FORMATS, ChainParams, export_chain
from ccsds_chain.utils import resample_ratio

# --------------------------------------------------------------------------
# CONFIGURABLE PARAMETERS (baseline CCSDS 131.0-B-5)
# --------------------------------------------------------------------------
MODULATION = "QPSK"            # selectable: "QPSK" (baseline, 2 bits/symbol) or "BPSK" (1 bit/symbol)
ENCODING = "NRZ-L"             # selectable: only NRZ-L implemented (baseline)
SYMBOL_RATE = 1_785_000        # symbol/s (bit_rate = symbol_rate * bits/symbol, derived below from MODULATION)
SAMPLES_PER_SYM = 4            # "native" oversampling factor (internal, pre-resample)
RRC_ALPHA = 0.35
RRC_SPAN = 8                   # RRC taps = RRC_SPAN * SAMPLES_PER_SYM + 1

RS_E = 16                       # error correction capability, in symbols: 8 or 16
RS_N = 255
INTERLEAVE_DEPTH = 5           # 1, 2, 3, 4, 5, or 8

FEC_RS = True                  # selectable
FEC_CONV = True                # selectable
CONV_RATE = "1/2"              # 1/2, 2/3, 3/4, 5/6, 7/8 (puncturing)
CONV_INVERT_G2 = True          # bool, CCSDS standard convention (rate 1/2 only)

RANDOMIZER = "none"            # "none", "short" (255-bit, legacy), "long" (131071-bit)

ASM = bytes.fromhex("1ACFFC1D")  # 4 bytes, uncoded

INPUT_FORMAT = "transfer_frame"  # "transfer_frame" (default), "asm_frame" or "cadu" (see docstring)
N_CADU = 100                   # number of CADUs to generate (signal duration)
PAYLOAD_SOURCE = None          # path to a file with real Transfer Frames (or CADUs, if --input-format cadu), or None = pseudo-random
PAYLOAD_SEED = 42              # seed for reproducibility of the pseudo-random payload

OUTPUT_DTYPE = "float32"       # "float32" ([-1,+1]) or "int16" (12-bit, RF-Catcher format, [-2048,2047])
OUTPUT_PEAK = 0.9              # normalized peak amplitude (anti-clipping headroom)
TARGET_FS = None               # Hz, or None to use native symbol_rate*sps (no resampling)
OUTPUT_DIR = "output"
# --------------------------------------------------------------------------


def build_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cadu", type=int, default=N_CADU)
    p.add_argument("--modulation", choices=["QPSK", "BPSK"], default=MODULATION,
                    help="QPSK (baseline, 2 bits/symbol) or BPSK (1 bit/symbol, half the bit "
                         "rate at the same symbol rate -- the modulation the base rate-1/2 "
                         "convolutional code's G2 inversion, CCSDS 3.3.1(5), is specified against)")
    p.add_argument("--alpha", type=float, default=RRC_ALPHA)
    p.add_argument("--sps", type=int, default=SAMPLES_PER_SYM)
    p.add_argument("--span", type=int, default=RRC_SPAN)
    p.add_argument("--no-rs", action="store_true", help="disable Reed-Solomon FEC")
    p.add_argument("--rs-e", type=int, choices=[8, 16], default=RS_E, help="RS error correction capability")
    p.add_argument("--interleave-depth", type=int, choices=[1, 2, 3, 4, 5, 8], default=INTERLEAVE_DEPTH)
    p.add_argument("--no-conv", action="store_true", help="disable convolutional FEC")
    p.add_argument("--conv-rate", choices=["1/2", "2/3", "3/4", "5/6", "7/8"], default=CONV_RATE)
    p.add_argument("--no-invert-g2", action="store_true")
    p.add_argument("--randomizer", choices=["none", "short", "long"], default=None,
                    help="CCSDS pseudo-randomizer: 'long' (131071-bit, current standard default), "
                         "'short' (255-bit, legacy), or 'none'. Default: 'long' with --input-format "
                         f"asm_frame (where 'none' is rejected), otherwise '{RANDOMIZER}'. Must be "
                         "'none' with --input-format cadu")
    p.add_argument("--input-format", choices=list(INPUT_FORMATS), default=INPUT_FORMAT,
                    help="what --payload-source/the generated payload represents: 'transfer_frame' "
                         "(raw, uncoded data -- RS/randomizer/ASM applied here, default), "
                         "'asm_frame' (ASM + Transfer Frame, NOT scrambled -- scrambled here, "
                         "never the ASM) or 'cadu' (CADU exactly as on the air, already "
                         "scrambled -- used verbatim). Neither ASM-framed format gets RS or a new "
                         "ASM; each record is 4 + 255*I bytes, extra bytes before the next ASM "
                         "are skipped. The convolutional stage still runs on all formats")
    p.add_argument("--payload-source", type=str, default=PAYLOAD_SOURCE)
    p.add_argument("--seed", type=int, default=PAYLOAD_SEED)
    p.add_argument("--vcid-list", type=str, default=None,
                    help="comma-separated Virtual Channel IDs (0-7), e.g. '0,1,2', round-"
                         "robined across generated Transfer Frames: builds a real CCSDS TM "
                         "primary header (132.0-B-3 4.1.2) carrying each frame's assigned "
                         "VCID, for validating a receiver correctly identifies/routes frames "
                         "by Virtual Channel. Repeat a value to weight it more heavily (e.g. "
                         "'0,0,1' gives VC 0 twice VC 1's share). Only valid with "
                         "--input-format transfer_frame and no --payload-source (it would "
                         "overwrite the first 6 bytes of real data); default: no synthetic "
                         "header, unchanged payload")
    p.add_argument("--spacecraft-id", type=int, default=0x123,
                    help="10-bit Spacecraft ID (0-1023) for the synthetic primary header "
                         "when --vcid-list is set; otherwise unused")
    p.add_argument("--corrupt-rs-symbols", type=int, default=0,
                    help="deterministically flip exactly this many RS symbols within one "
                         "interleaved codeword (--corrupt-codeword-index) of each targeted "
                         "CADU (--corrupt-cadu-indices and/or --corrupt-vc), to boundary-test "
                         "a receiver's RS decoder against its declared correction capability "
                         "E (--rs-e): E errors must still decode perfectly, E+1 must fail -- "
                         "an exact, reproducible per-codeword error count a replayer's AWGN "
                         "can't guarantee hitting. Needs --no-rs NOT set (or --input-format "
                         "cadu, already RS-coded); default: 0 (disabled)")
    p.add_argument("--corrupt-codeword-index", type=int, default=0,
                    help="which of the --interleave-depth interleaved RS codewords to target "
                         "(0-based); only used with --corrupt-rs-symbols")
    p.add_argument("--corrupt-cadu-indices", type=str, default=None,
                    help="comma-separated 0-based CADU indices to corrupt, e.g. '0,5,10'; "
                         "combines with --corrupt-vc (either match corrupts that CADU) -- at "
                         "least one of the two is required with --corrupt-rs-symbols")
    p.add_argument("--corrupt-vc", type=int, default=None,
                    help="corrupt every CADU assigned to this Virtual Channel ID (requires "
                         "--vcid-list) -- e.g. to verify a receiver's per-VC FER accounting "
                         "attributes errors to the right channel only")
    p.add_argument("--corrupt-seed", type=int, default=777,
                    help="seed for which symbol positions/values get flipped -- deterministic "
                         "and reproducible across runs")
    p.add_argument("--freq-offset-hz", type=float, default=0.0,
                    help="residual LO frequency offset applied to the pulse-shaped IQ, in Hz "
                         "(positive or negative) -- for validating a receiver's carrier-"
                         "recovery loop against a real (imperfect) transmitter LO, not just a "
                         "perfectly on-frequency signal; default: 0 (disabled)")
    p.add_argument("--iq-gain-imbalance-db", type=float, default=0.0,
                    help="IQ modulator I/Q branch gain mismatch, in dB -- produces a mirror-"
                         "image tone in the spectrum; default: 0 (disabled)")
    p.add_argument("--iq-phase-imbalance-deg", type=float, default=0.0,
                    help="IQ modulator phase deviation from ideal 90-degree I/Q separation, in "
                         "degrees -- combines with --iq-gain-imbalance-db in the same mirror-"
                         "image model; default: 0 (disabled)")
    p.add_argument("--phase-noise-linewidth-hz", type=float, default=0.0,
                    help="free-running-oscillator single-sideband 3 dB linewidth, in Hz -- "
                         "generates Wiener (random-walk) phase noise on the IQ, for validating "
                         "a receiver's tolerance to constellation smearing from a real "
                         "transmitter LO; default: 0 (disabled)")
    p.add_argument("--impairment-seed", type=int, default=2718,
                    help="seed for the phase noise random walk -- deterministic and "
                         "reproducible across runs")
    p.add_argument("--pa-backoff-db", type=float, default=None,
                    help="power amplifier saturation point (Rapp AM-AM model), in dB above "
                         "this project's reference unit amplitude (1.0, an ideal symbol's own "
                         "magnitude) -- models PA compression/saturation, the last physical "
                         "stage before the antenna: negative values saturate more aggressively "
                         "(visible spectral regrowth just outside the occupied bandwidth); "
                         "default: not set (disabled)")
    p.add_argument("--pa-smoothness", type=float, default=3.0,
                    help="Rapp model knee sharpness (higher = sharper transition from linear "
                         "to saturated); only used with --pa-backoff-db")
    p.add_argument("--pa-am-pm-deg-per-db", type=float, default=0.0,
                    help="AM-PM conversion, in degrees of phase shift per dB of AM-AM "
                         "compression (the same figure real TWTA/SSPA datasheets quote); "
                         "0 (default) disables it; only used with --pa-backoff-db")
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

    vcid_list = None
    if args.vcid_list is not None:
        try:
            vcid_list = [int(v.strip()) for v in args.vcid_list.split(",") if v.strip() != ""]
            if not vcid_list:
                raise ValueError("pattern is empty")
        except ValueError as e:
            raise SystemExit(f"--vcid-list: invalid pattern {args.vcid_list!r} ({e}); "
                              "expected comma-separated integers 0-7, e.g. '0,1,2'")

    corrupt_cadu_indices = None
    if args.corrupt_cadu_indices is not None:
        try:
            corrupt_cadu_indices = [int(v.strip()) for v in args.corrupt_cadu_indices.split(",") if v.strip() != ""]
            if not corrupt_cadu_indices:
                raise ValueError("list is empty")
        except ValueError as e:
            raise SystemExit(f"--corrupt-cadu-indices: invalid list {args.corrupt_cadu_indices!r} ({e}); "
                              "expected comma-separated integers, e.g. '0,5,10'")

    params = ChainParams(
        modulation=args.modulation,
        encoding=ENCODING,
        bit_rate=SYMBOL_RATE * BITS_PER_SYMBOL[args.modulation],
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
        randomizer=args.randomizer if args.randomizer is not None
                   else ("long" if args.input_format == "asm_frame" else RANDOMIZER),
        asm=ASM,
        input_format=args.input_format,
        n_cadu=args.n_cadu,
        payload_source=args.payload_source,
        seed=args.seed,
        vcid_list=vcid_list,
        spacecraft_id=args.spacecraft_id,
        corrupt_rs_symbols=args.corrupt_rs_symbols,
        corrupt_codeword_index=args.corrupt_codeword_index,
        corrupt_cadu_indices=corrupt_cadu_indices,
        corrupt_vc=args.corrupt_vc,
        corrupt_seed=args.corrupt_seed,
        freq_offset_hz=args.freq_offset_hz,
        iq_gain_imbalance_db=args.iq_gain_imbalance_db,
        iq_phase_imbalance_deg=args.iq_phase_imbalance_deg,
        phase_noise_linewidth_hz=args.phase_noise_linewidth_hz,
        impairment_seed=args.impairment_seed,
        pa_backoff_db=args.pa_backoff_db,
        pa_smoothness=args.pa_smoothness,
        pa_am_pm_deg_per_db=args.pa_am_pm_deg_per_db,
    )
    is_cadu_input = params.input_format in ASM_FRAMED_INPUT_FORMATS
    unit_bytes = len(params.asm) + params.rs_n * params.interleave_depth if is_cadu_input else params.rs_k * params.interleave_depth

    print(f"[1/9] Payload generation: {params.n_cadu} CADU x {unit_bytes} bytes "
          f"({'file: ' + params.payload_source if params.payload_source else 'pseudo-random, seed=' + str(params.seed)}), "
          f"input-format={params.input_format}")
    if params.vcid_list is not None:
        print(f"       -> synthetic Transfer Frame headers: spacecraft_id=0x{params.spacecraft_id:03X}, "
              f"VCID pattern {params.vcid_list} (round-robined)")
    if params.corrupt_rs_symbols > 0:
        target = []
        if params.corrupt_cadu_indices is not None:
            target.append(f"CADU indices {params.corrupt_cadu_indices}")
        if params.corrupt_vc is not None:
            target.append(f"VC{params.corrupt_vc}")
        print(f"       -> deterministic corruption: {params.corrupt_rs_symbols} symbol(s) flipped in "
              f"codeword {params.corrupt_codeword_index} of {' and '.join(target)}")
    if is_cadu_input:
        print(f"[2/9] RS SKIPPED -- input already carries the {unit_bytes - len(params.asm)}-byte "
              "block after each ASM")
        if params.input_format == "asm_frame":
            print(f"[3/9] Pseudo-randomizer CCSDS ({params.randomizer}, excludes ASM) -- applied to the "
                  "input's not-yet-scrambled frames")
        else:
            print("[3/9] Pseudo-randomizer SKIPPED -- input CADUs are already as on the air")
        print("[4/9] ASM SKIPPED -- input already carries it per record")
    else:
        print(f"[2/9] RS({params.rs_n},{params.rs_k}) E={params.rs_e} encode, interleave depth {params.interleave_depth}"
              f"{' (SKIPPED)' if not params.fec_rs else ''}")
        print(f"[3/9] Pseudo-randomizer CCSDS ({params.randomizer}, excludes ASM)"
              f"{'' if params.randomizer != 'none' else ' (SKIPPED)'}")
        print("[4/9] ASM (0x1ACFFC1D) attached per CADU (forms the CADU)")
    print(f"[5/9] Convolutional K=7 rate {params.conv_rate} over the CADU stream, ASM included "
          f"(G1=171o, G2=133o, invert_g2={params.conv_invert_g2 and params.conv_rate == '1/2'})"
          f"{' (SKIPPED)' if not params.fec_conv else ''}")
    print("[6/9] NRZ-L mapping (direct bipolar)")
    print(f"[7/9] {params.modulation} mapping"
          + (" (Gray, unit energy)" if params.modulation == "QPSK" else " (unit energy, I only)"))
    print(f"[8/9] Pulse shaping RRC (alpha={params.rrc_alpha}, span={params.rrc_span}, sps={params.sps})")
    impairments = []
    if params.freq_offset_hz != 0:
        impairments.append(f"freq offset {params.freq_offset_hz:+.1f} Hz")
    if params.iq_gain_imbalance_db != 0 or params.iq_phase_imbalance_deg != 0:
        impairments.append(f"IQ imbalance {params.iq_gain_imbalance_db:+.2f} dB / {params.iq_phase_imbalance_deg:+.2f} deg")
    if params.phase_noise_linewidth_hz > 0:
        impairments.append(f"phase noise {params.phase_noise_linewidth_hz:.1f} Hz linewidth")
    if params.pa_backoff_db is not None:
        pa_desc = f"PA backoff {params.pa_backoff_db:+.1f} dB, smoothness {params.pa_smoothness:.1f}"
        if params.pa_am_pm_deg_per_db != 0:
            pa_desc += f", AM-PM {params.pa_am_pm_deg_per_db:.1f} deg/dB"
        impairments.append(pa_desc)
    if impairments:
        print(f"       -> transmitter impairments: {', '.join(impairments)}")

    native_fs = params.symbol_rate * params.sps
    output_fs = args.target_fs if (args.target_fs is not None and args.target_fs != native_fs) else native_fs
    if output_fs != native_fs:
        up, down = resample_ratio(native_fs, output_fs)
        print(f"[9/9] Resampling {native_fs/1e6:.3f} MS/s -> {output_fs/1e6:.3f} MS/s "
              f"(ratio {up}/{down}, scipy.signal.resample_poly), format={args.dtype}, peak={args.peak}")
    else:
        print(f"[9/9] No resampling (native {output_fs/1e6:.3f} MS/s), format={args.dtype}, peak={args.peak}")

    output_path = args.output
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            OUTPUT_DIR, f"qpsk_ccsds_{output_fs/1e6:.1f}Msps_{params.n_cadu}cadu_{timestamp}.iq")

    # export_chain() processes CADUs in bounded-memory batches and streams
    # straight to output_path, so peak memory stays roughly constant
    # regardless of n_cadu (export duration) -- unlike building the whole
    # bits/coded-bits/symbols/IQ arrays in memory at once, which is what
    # run_chain() does (fine for the GUI's small live preview, not for a
    # potentially large export like this one).
    export_result = export_chain(
        params, output_path, output_dtype=args.dtype, peak=args.peak,
        target_fs=args.target_fs,
    )

    skipped = export_result["meta"].get("cadu_sync_skipped_bytes")
    if skipped:
        print(f"       -> synced to first CADU boundary, skipped {skipped} leading byte(s) of unframed data")
    wrapper_min = export_result["meta"].get("cadu_wrapper_bytes_min")
    wrapper_max = export_result["meta"].get("cadu_wrapper_bytes_max")
    if wrapper_min is not None:
        wrapper_note = (f"{wrapper_min} bytes" if wrapper_min == wrapper_max
                         else f"{wrapper_min}-{wrapper_max} bytes")
        print(f"       -> {wrapper_note} of per-record framing stripped between CADUs (not part of the CADU)")
    print(f"       -> {params.n_cadu} CADU x {export_result['cadu_bytes']} byte")

    meta_path = output_path.rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(export_result["meta"], f, indent=2)

    output_n_samples = export_result["meta"]["output_n_samples"]
    output_bytes = os.path.getsize(output_path)
    print(f"\nDone in {export_result['elapsed']:.2f}s")
    print(f"  {params.modulation} symbols:  {export_result['meta']['n_symbols']}")
    print(f"  IQ samples:    {output_n_samples}  @ {output_fs / 1e6:.3f} MS/s (format {args.dtype})")
    print(f"  Signal duration: {export_result['meta']['output_duration_s'] * 1000:.2f} ms")
    print(f"  Output:        {output_path} ({output_bytes / 1e6:.2f} MB)")
    print(f"  Metadata:      {meta_path}")


if __name__ == "__main__":
    main()
