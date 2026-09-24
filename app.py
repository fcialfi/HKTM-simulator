#!/usr/bin/env python3
"""HKTM CCSDS Signal Generator -- GUI (Streamlit).

Graphical interface for generating the CCSDS test signal (baseline QPSK)
for injection via RF-Catcher (TestTree) Capture & Playback. Every parameter
change recomputes the chain and updates the spectrum, constellation and
metrics in real time.

Usage: streamlit run app.py
"""

import dataclasses
import datetime
import json
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ccsds_chain.mapping import BITS_PER_SYMBOL
from ccsds_chain.pipeline import ASM, ASM_FRAMED_INPUT_FORMATS, ChainParams, export_chain, run_chain
from ccsds_chain.pulse_shaping import matched_filter_sample, rrc_taps
from ccsds_chain.spectrum import welch_psd
from ccsds_chain.utils import find_all_cadu_positions, find_cadu_sync, resample_ratio

# Exported IQ files are written here (same convention as generate_signal.py's
# CLI default) rather than held fully in memory: export_chain() streams
# straight to disk in bounded-memory batches regardless of export size, so
# large exports no longer need to fit in RAM to be generated.
EXPORT_DIR = "output"

# Above this size, the generated file is left on disk (its path is always
# shown) but not also offered through st.download_button: Streamlit reads
# a download's data fully into memory to serve it, so a very large file
# would reintroduce the same peak-memory problem export_chain() avoids for
# generation, just at download time instead.
_DOWNLOAD_BUTTON_SIZE_LIMIT = 1 * 1024 ** 3

st.set_page_config(
    page_title="HKTM CCSDS Signal Generator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(ellipse 120% 80% at 50% -10%, #142033 0%, #0b0f19 55%); }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"] [data-testid="stMainMenu"],
header[data-testid="stHeader"] [data-testid="stAppDeployButton"] { visibility: hidden; }
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    background: rgba(0,212,255,0.14) !important;
    border: 1px solid rgba(0,212,255,0.4) !important;
    border-radius: 6px !important;
}
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] svg,
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] span { color: #00d4ff !important; }

.hktm-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.6rem; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, rgba(0,212,255,0.10), rgba(0,212,255,0.02));
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 14px;
}
.hktm-title { font-size: 1.5rem; font-weight: 700; color: #eaf6ff; letter-spacing: 0.3px; }
.hktm-title span { color: #00d4ff; }
.hktm-subtitle { font-size: 0.85rem; color: #8aa0bd; margin-top: 2px; }
.hktm-badges { display: flex; gap: 8px; }
.hktm-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600;
    padding: 4px 10px; border-radius: 999px;
    background: rgba(90,110,140,0.16); border: 1px solid rgba(120,140,170,0.30); color: #9fb0c9;
}

.hktm-panel {
    border: 1px solid rgba(255,255,255,0.07); border-radius: 16px;
    padding: 1.2rem 1.3rem 1.4rem; margin-bottom: 1rem;
    background: rgba(255,255,255,0.012);
}
.panel-head { display: flex; align-items: center; gap: 9px; margin-bottom: 1rem; }
.panel-head .dot { width: 7px; height: 7px; border-radius: 50%; background: #00d4ff; box-shadow: 0 0 8px #00d4ff; }
.panel-head.neutral .dot { background: #7d90ac; box-shadow: none; }
.panel-head h2 { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #8aa0bd; margin: 0; }

.metric-hero {
    background: linear-gradient(160deg, rgba(0,212,255,0.08), rgba(0,212,255,0.015));
    border: 1px solid rgba(0,212,255,0.35);
    border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
    cursor: help;
}
.metric-hero .label { font-size: 0.68rem; color: #8aa0bd; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
.metric-hero .value { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; color: #00d4ff; font-weight: 700; margin-top: 4px; line-height: 1; }
.metric-hero .unit { font-size: 0.78rem; color: #56698a; margin-left: 4px; }
.metric-hero .sub { font-size: 0.7rem; color: #56698a; margin-top: 5px; }

.secondary-strip {
    display: flex; flex-wrap: wrap; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; background: #0d1220; overflow: hidden;
}
.sec-item {
    flex: 1 1 140px; padding: 0.6rem 0.9rem; border-right: 1px solid rgba(255,255,255,0.07);
    display: flex; align-items: baseline; gap: 6px; justify-content: space-between;
}
.sec-item:last-child { border-right: none; }
.sec-item .label { font-size: 0.68rem; color: #56698a; text-transform: uppercase; letter-spacing: 0.4px; }
.sec-item .value { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #8aa0bd; font-weight: 600; }

.stage-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 0.4rem 0 1rem 0; }
.stage-chip {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600;
    padding: 5px 11px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03); color: #55698a;
}
.stage-chip.on { background: rgba(90,110,140,0.16); border-color: rgba(120,140,170,0.35); color: #9fb0c9; }
.stage-chip .arrow { color: #33455f; margin-left: 8px; }

section[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] h3 { color: #00d4ff !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 1.2rem; }

div[data-testid="stExpander"] { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Modulation & Encoding")
    modulation = st.selectbox(
        "Modulation", ["QPSK", "BPSK"],
        help=(
            "QPSK (baseline): 2 bits/symbol, 1 CCSDS-native RS/interleave-compatible "
            "chain. BPSK: 1 bit/symbol -- half the bit rate at the same symbol rate/"
            "occupied bandwidth (or double the occupied bandwidth for the same bit "
            "rate), but the convolutional code's rate-1/2 G2 inversion (3.3.1(5)) is "
            "specified against BPSK symbol synchronizers, so it's the modulation the "
            "base rate-1/2 code was originally designed for."
        ),
    )
    encoding = st.selectbox("Line encoding", ["NRZ-L"], help="Baseline. Other encodings: roadmap.")

    bits_per_symbol = BITS_PER_SYMBOL[modulation]
    rate_ref = st.radio(
        "Rate input", ["Symbol rate", "Bit rate"], horizontal=True,
        help=(
            "Symbol rate and bit rate aren't independent: for a given modulation "
            f"(here {modulation} = {bits_per_symbol} bits/symbol), bit rate = symbol rate x "
            "bits/symbol. Pick which one you want to drive directly -- the other is "
            "derived automatically so they can never go out of sync."
        ),
    )
    if rate_ref == "Symbol rate":
        symbol_rate = st.number_input("Symbol rate (S/s)", value=1_785_000, step=1_000, format="%d")
        bit_rate = symbol_rate * bits_per_symbol
        st.caption(f"Bit rate (derived): {bit_rate:,.0f} bps".replace(",", " "))
    else:
        bit_rate = st.number_input("Bit rate (bps)", value=3_570_000, step=1_000, format="%d")
        symbol_rate = bit_rate // bits_per_symbol
        st.caption(f"Symbol rate (derived): {symbol_rate:,.0f} S/s".replace(",", " "))

    st.markdown("### Input")
    input_format_options = {
        "Transfer Frame (unencoded)": "transfer_frame",
        "ASM + Transfer Frame (not scrambled)": "asm_frame",
        "CADU as on air (already scrambled)": "cadu",
    }
    input_format_label = st.radio(
        "Payload contains", list(input_format_options),
        help=(
            "What the payload below actually is. 'Transfer Frame' is raw, "
            "uncoded data: RS, the pseudo-randomizer and the ASM are all "
            "applied further down to build the CADUs from it.\n\n"
            "'ASM + Transfer Frame (not scrambled)' is what a ground station "
            "that does its own frame sync hands back (e.g. Cortex, once its "
            "per-record header/trailer is stripped): each record starts with "
            "the ASM, followed by a frame that is NOT scrambled. The frame is "
            "scrambled here (never the ASM); RS and the ASM are not re-applied.\n\n"
            "'CADU as on air' means the bytes are exactly as transmitted "
            "right before convolutional coding (already scrambled, if the "
            "link scrambles): used verbatim, no RS/scrambling/ASM applied.\n\n"
            "In both ASM-framed modes each record is exactly 4 + 255*I bytes "
            "(1279 at I=5): anything after that, before the next ASM, is "
            "receiver-added overhead and is skipped. Only the convolutional "
            "stage below (if enabled) is still applied."
        ),
    )
    input_format = input_format_options[input_format_label]
    is_cadu_input = input_format in ASM_FRAMED_INPUT_FORMATS
    if input_format == "asm_frame":
        st.caption(
            "ASM + Transfer Frame input: RS encoding and ASM prepend below are "
            "skipped; the frame after each ASM is scrambled with the pseudo-"
            "randomizer below (the ASM never is). Interleave depth sets each "
            "record's length (4 + 255*I bytes)."
        )
    elif input_format == "cadu":
        st.caption(
            "On-air CADU input: RS encoding, pseudo-randomizer and ASM prepend "
            "below are all skipped -- the CADUs are used verbatim. Interleave "
            "depth sets each CADU's length (4 + 255*I bytes)."
        )

    st.markdown("### FEC")
    fec_rs = st.checkbox(
        "Reed-Solomon", value=True, disabled=is_cadu_input,
        help="Not re-applied to ASM-framed input: the block after each ASM is used as-is." if is_cadu_input else None,
    )
    rs_col1, rs_col2 = st.columns(2)
    with rs_col1:
        rs_e = st.selectbox(
            "Error correction E", [16, 8], disabled=not fec_rs and not is_cadu_input,
            help=(
                "RS error-correction capability, in symbols (CCSDS 4.3.1, managed "
                "parameter 12.5). E=16 gives RS(255,223): more parity overhead, "
                "corrects up to 16 symbol errors per codeword. E=8 gives "
                "RS(255,239): less overhead, corrects up to 8."
                + (" Not used to size ASM-framed input (only interleave depth "
                   "is); it only matters there for error injection." if is_cadu_input else "")
            ),
        )
    with rs_col2:
        interleave_depth = st.selectbox(
            "Interleave depth I", [1, 2, 3, 4, 5, 8], index=4, disabled=not fec_rs and not is_cadu_input,
            help=(
                "Number of RS codewords interleaved together per CADU (CCSDS "
                "4.3.5, managed parameter 12.5). Higher I spreads a burst error "
                "across more codewords (each corrects a smaller share of it) at "
                "the cost of a larger CADU."
                + (" With ASM-framed input this sets each record's length "
                   "(4 + 255*I bytes, 1279 at I=5)." if is_cadu_input else "")
            ),
        )
    rs_k = 255 - 2 * rs_e
    # Mirrors ChainParams.frame_bytes / pipeline.py's frame_bytes computation:
    # the data-zone size of one CADU at the current RS/interleave settings,
    # needed below to tell how many complete CADUs a Transfer Frame file holds.
    frame_bytes = rs_k * interleave_depth
    # Full CADU length (ASM + RS-coded codeblock), needed below when the
    # uploaded file already contains complete CADUs rather than raw frames.
    cadu_unit_bytes = len(ASM) + 255 * interleave_depth
    unit_bytes = cadu_unit_bytes if is_cadu_input else frame_bytes

    fec_conv = st.checkbox("Convolutional K=7", value=True)
    conv_rate = st.selectbox(
        "Code rate", ["1/2", "2/3", "3/4", "5/6", "7/8"], disabled=not fec_conv,
        help=(
            "Convolutional code rate via puncturing (CCSDS 3.4, Table 3-1, "
            "managed parameter 12.4). Higher rate transmits fewer coded symbols "
            "per input bit (less bandwidth overhead) at the cost of weaker error "
            "correction. Rate 1/2 is the base, unpunctured code."
        ),
    )
    invert_g2 = st.checkbox(
        "Invert G2 (CCSDS convention)", value=True,
        disabled=not fec_conv or conv_rate != "1/2",
        help=(
            "G2 output symbol inversion (CCSDS 3.3.1(5)), needed at rate 1/2 to "
            "guarantee bit transitions for BPSK symbol synchronizers. Punctured "
            "codes (any rate other than 1/2) use no inversion (3.4.1(5))."
        ),
    )

    randomizer_options = {
        "Long (131071-bit)": "long", "Short (255-bit, legacy)": "short", "None": "none",
    }
    if input_format == "asm_frame":
        # The frames are known NOT to be scrambled yet: "None" would send
        # them unscrambled and a real receiver's descrambler would corrupt
        # every frame (the pipeline rejects it too).
        randomizer_choices = ["Short (255-bit, legacy)", "Long (131071-bit)"]
    elif input_format == "cadu":
        # Already scrambled as on air: scrambling again would undo it.
        randomizer_choices = ["None"]
    else:
        randomizer_choices = list(randomizer_options)
    randomizer_label = st.selectbox(
        "Pseudo-randomizer", randomizer_choices, disabled=input_format == "cadu",
        help=(
            "CCSDS section 10: scrambles the RS-coded data (never the ASM) to "
            "guarantee bit transitions, avoid spectral lines, and aid receiver "
            "acquisition. 'Long' is the current standard default (Issue 5, "
            "2023, managed parameter 12.3); 'Short' is kept only for backward "
            "compatibility with legacy systems."
            + (" With 'ASM + Transfer Frame (not scrambled)' input it is "
               "always applied: pick the one the real link uses."
               if input_format == "asm_frame" else "")
            + (" Not applied to 'CADU as on air' input, which is already "
               "scrambled." if input_format == "cadu" else "")
        ),
    )
    randomizer = randomizer_options[randomizer_label]

    st.markdown("### Payload & Duration")
    file_upload_label = {"transfer_frame": "Transfer Frame file", "asm_frame": "ASM + Transfer Frame file",
                         "cadu": "CADU file"}[input_format]
    payload_mode = st.radio("Payload source", ["Pseudo-random", file_upload_label], horizontal=True)
    payload_source_bytes = None
    seed = 42
    file_cadu_count = None
    vcid_list = None
    spacecraft_id = 0x123
    if payload_mode == "Pseudo-random":
        seed = st.number_input("Seed", value=42, step=1)
        n_cadu = st.slider(
            "Preview CADU count", 5, 300, 40, 5,
            help=(
                "Number of CADUs used for the live spectrum/constellation preview "
                "below. Kept small so every parameter change recomputes instantly. "
                "The exported file's length is set independently, further down, "
                "and is only (re)computed when you click 'Generate export file'."
            ),
        )

        if not is_cadu_input:
            enable_vc = st.checkbox(
                "Assign Virtual Channels",
                help=(
                    "Builds a real CCSDS TM primary header (132.0-B-3 4.1.2) into "
                    "each synthetic Transfer Frame, carrying a Virtual Channel ID "
                    "from the pattern below -- for validating that a receiver "
                    "correctly identifies and routes frames to each Virtual "
                    "Channel. Only overwrites the first 6 bytes of the synthetic "
                    "payload (the rest, and everything else in the chain, is "
                    "unaffected); not available with an uploaded real Transfer "
                    "Frame file, which already carries its own real header."
                ),
            )
            if enable_vc:
                vc_col1, vc_col2 = st.columns([2, 1])
                with vc_col1:
                    vcid_text = st.text_input(
                        "VCID pattern (comma-separated, 0-7)", value="0, 1, 2",
                        help=(
                            "Round-robined across generated frames: frame 0 gets "
                            "the first value, frame 1 the second, and so on, "
                            "repeating. Repeat a value to give it proportionally "
                            "more frames -- e.g. '0, 0, 1' gives Virtual Channel 0 "
                            "twice Virtual Channel 1's share."
                        ),
                    )
                with vc_col2:
                    spacecraft_id = st.number_input(
                        "Spacecraft ID", min_value=0, max_value=1023, value=0x123, step=1,
                        help="10-bit SCID (0-1023) carried in the synthetic primary header.",
                    )
                try:
                    vcid_list = [int(v.strip()) for v in vcid_text.split(",") if v.strip() != ""]
                    if not vcid_list:
                        raise ValueError("pattern is empty")
                    for v in vcid_list:
                        if not (0 <= v < 8):
                            raise ValueError(f"{v} is out of range (Virtual Channel IDs are 3-bit, 0-7)")
                except ValueError as e:
                    st.error(f"Invalid VCID pattern: {e}. Use comma-separated integers 0-7, e.g. '0, 1, 2'.")
                    st.stop()
                st.caption(
                    f"{len(vcid_list)}-frame pattern, cycling: " + ", ".join(f"VC{v}" for v in vcid_list)
                )
    else:
        uploaded = st.file_uploader(
            f"{file_upload_label} (binary)", type=None,
        )
        if uploaded is not None:
            payload_source_bytes = uploaded.read()
            sync_skipped_bytes = 0
            usable_bytes = payload_source_bytes
            file_unit_bytes = unit_bytes
            if is_cadu_input:
                # A real/captured CADU file isn't guaranteed to start exactly
                # on a CADU boundary (leading idle line-fill, or an excerpt
                # starting mid-stream): locate the first real ASM instead of
                # blindly slicing from byte 0, or CADU boundaries end up
                # misaligned with whatever junk precedes the real stream.
                try:
                    sync_skipped_bytes = find_cadu_sync(payload_source_bytes, ASM)
                except ValueError:
                    st.error(
                        f"No ASM (0x{ASM.hex()}) found anywhere in the uploaded file: "
                        "cannot locate a CADU boundary to synchronize to."
                    )
                    st.stop()
                usable_bytes = payload_source_bytes[sync_skipped_bytes:]
                # The CADU is always exactly `unit_bytes` long -- the length
                # the configured RS-E/interleave-depth predicts, never
                # measured from ASM spacing: a capture instrument commonly
                # wraps each CADU in its own record framing (e.g. RF-Catcher/
                # TestTree's CRT format: a fixed header + CADU + a short
                # postamble per record), and treating ASM-to-ASM distance as
                # the CADU length would fold that framing into what's fed to
                # convolutional coding as if it were coded CADU data.
                # Generation (export_chain/run_chain) does this same thing --
                # shown here too so the preview/count below match what
                # generation will actually do.
                positions = find_all_cadu_positions(usable_bytes, ASM)
                wrapper_gaps = []
                mismatch_at = None
                for j in range(len(positions) - 1):
                    gap = positions[j + 1] - positions[j]
                    if gap < unit_bytes:
                        mismatch_at = (positions[j], gap)
                        break
                    wrapper_gaps.append(gap - unit_bytes)
                if mismatch_at is not None:
                    pos, gap = mismatch_at
                    st.error(
                        f"Record length from interleave depth {interleave_depth} is 4 + 255 x "
                        f"{interleave_depth} = {unit_bytes} bytes, but the next ASM in the file "
                        f"is only {gap} bytes after the one at offset {sync_skipped_bytes + pos}. "
                        "Check that the interleave depth matches how this data was actually built."
                    )
                    st.stop()
                usable_positions = [pos for pos in positions if pos + unit_bytes <= len(usable_bytes)]
                file_cadu_count = len(usable_positions)
                leftover_bytes = len(usable_bytes) - (usable_positions[-1] + unit_bytes if usable_positions else 0)
            else:
                file_cadu_count = len(usable_bytes) // file_unit_bytes
                leftover_bytes = len(usable_bytes) - file_cadu_count * file_unit_bytes
            if file_cadu_count < 1:
                st.error(
                    f"File too short: {len(usable_bytes)} usable bytes after sync, but one "
                    f"CADU needs {file_unit_bytes} bytes."
                )
                st.stop()
            st.caption(
                (f"Synced to the first CADU after skipping {sync_skipped_bytes} leading "
                 f"byte{'s' if sync_skipped_bytes != 1 else ''} of unframed data -- " if sync_skipped_bytes else "")
                + f"File contains {file_cadu_count} complete CADU{'s' if file_cadu_count != 1 else ''} "
                f"of {file_unit_bytes} bytes each"
                + (f", {leftover_bytes} trailing bytes ignored (not a full CADU)" if leftover_bytes else "")
                + (f" ({min(wrapper_gaps)}-{max(wrapper_gaps)} bytes of per-record "
                   "framing stripped between CADUs)" if is_cadu_input and wrapper_gaps else "")
                + " -- generation uses exactly these, no padding."
            )
            n_cadu = min(file_cadu_count, 300)
            if file_cadu_count > n_cadu:
                st.caption(
                    f"Live preview uses the first {n_cadu} of {file_cadu_count} CADUs for "
                    "responsiveness; the export below always uses all of them."
                )
        else:
            st.caption("No file uploaded: falling back to pseudo-random.")
            n_cadu = st.slider(
                "Preview CADU count", 5, 300, 40, 5,
                help=(
                    "Number of CADUs used for the live spectrum/constellation preview "
                    "below. Kept small so every parameter change recomputes instantly."
                ),
            )

    corrupt_rs_symbols = 0
    corrupt_codeword_index = 0
    corrupt_cadu_indices = None
    corrupt_vc = None
    corrupt_seed = 777
    if fec_rs or is_cadu_input:
        st.markdown("### Deterministic Error Injection")
        enable_corruption = st.checkbox(
            "Inject controlled RS symbol errors",
            help=(
                "Deterministically flips an exact number of RS symbols within one "
                "interleaved codeword of chosen CADUs -- for boundary-testing a "
                "receiver's RS decoder against its declared correction capability E "
                "(E errors must still decode perfectly, E+1 must fail/be flagged), "
                "or for validating that a receiver's per-Virtual-Channel FER "
                "accounting attributes errors to the right channel and leaves "
                "others untouched. Complementary to a replayer's AWGN, which "
                "characterizes statistical BER/FER vs Eb/N0 well but can't "
                "guarantee hitting a precise per-codeword error count."
            ),
        )
        if enable_corruption:
            cc1, cc2 = st.columns(2)
            with cc1:
                corrupt_rs_symbols = st.number_input(
                    "Symbols to flip per codeword", min_value=1, max_value=255, value=int(rs_e), step=1,
                    help=f"Try {rs_e} (the declared E) to confirm it still decodes, and {rs_e + 1} to confirm it correctly fails.",
                )
            with cc2:
                corrupt_codeword_index = st.number_input(
                    "Codeword index", min_value=0, max_value=int(interleave_depth) - 1, value=0, step=1,
                    help="Which of the interleave_depth interleaved codewords to corrupt (0-based).",
                )
            target_mode = (
                st.radio("Target", ["Specific CADU indices", "Virtual Channel"], horizontal=True)
                if vcid_list is not None else "Specific CADU indices"
            )
            if target_mode == "Specific CADU indices":
                indices_text = st.text_input("CADU indices (comma-separated, 0-based)", value="0")
                try:
                    corrupt_cadu_indices = [int(v.strip()) for v in indices_text.split(",") if v.strip() != ""]
                    if not corrupt_cadu_indices:
                        raise ValueError("list is empty")
                except ValueError as e:
                    st.error(f"Invalid CADU index list: {e}. Use comma-separated integers, e.g. '0, 5, 10'.")
                    st.stop()
                st.caption(f"Corrupting {corrupt_rs_symbols} symbol(s) in codeword {corrupt_codeword_index} of CADU(s) {corrupt_cadu_indices}")
            else:
                corrupt_vc = st.number_input("Virtual Channel ID", min_value=0, max_value=7, value=vcid_list[0], step=1)
                st.caption(f"Corrupting {corrupt_rs_symbols} symbol(s) in codeword {corrupt_codeword_index} of every VC{corrupt_vc} CADU")

    st.markdown("### Pulse Shaping (RRC)")
    st.caption(f"Shapes the {modulation} symbols into a band-limited waveform (Root-Raised-Cosine filter) before D/A conversion.")
    rrc_alpha = st.slider(
        "Roll-off (alpha)", 0.05, 1.00, 0.35, 0.01,
        help=(
            "Excess-bandwidth factor of the RRC filter (0-1). Trades occupied "
            "bandwidth against tolerance to timing/synchronization error: lower "
            "alpha keeps the occupied bandwidth close to the symbol rate but makes "
            "pulses sharper and more sensitive to timing offsets; higher alpha "
            "widens the occupied bandwidth toward 2x the symbol rate but gives "
            "smoother, more timing-tolerant pulses. Theoretical occupied bandwidth "
            "= symbol rate x (1 + alpha). CCSDS commonly uses 0.35."
        ),
    )
    rrc_span = st.slider(
        "Span (symbols)", 4, 16, 8, 1,
        help=(
            "Length of the RRC filter's impulse response, in symbol periods. A "
            "longer span approximates the ideal (infinite) RRC filter more "
            "closely and reduces residual inter-symbol interference, at the cost "
            "of a longer start/end transient (group delay = span/2 symbols) and "
            "more computation."
        ),
    )
    sps = st.select_slider(
        "Samples/symbol", options=[2, 4, 8], value=4,
        help=(
            "Oversampling factor: number of I/Q samples generated per symbol. "
            "Sets the output sample rate = symbol rate x samples/symbol. Needs to "
            "be high enough to represent the occupied bandwidth (roughly "
            "symbol rate x (1 + alpha)) without aliasing."
        ),
    )

    st.markdown("### Transmitter Impairments")
    enable_impairments = st.checkbox(
        "Enable typical transmitter impairments",
        help=(
            "Applies real-transmitter non-idealities to the pulse-shaped IQ -- "
            "distinct from a replayer's AWGN (which characterizes receiver "
            "sensitivity vs Eb/N0): these validate a receiver's tolerance to an "
            "imperfect transmitter itself, grouped by which physical subsystem "
            "each comes from: the LO/synthesizer (frequency offset, phase "
            "noise), the IQ modulator (gain/phase imbalance), and the power "
            "amplifier (saturation/compression, AM-PM conversion) -- the last "
            "stage before the antenna, applied after the other two."
        ),
    )
    freq_offset_hz = 0.0
    iq_gain_imbalance_db = 0.0
    iq_phase_imbalance_deg = 0.0
    phase_noise_linewidth_hz = 0.0
    impairment_seed = 2718
    enable_pa = False
    pa_backoff_db = None
    pa_smoothness = 3.0
    pa_am_pm_deg_per_db = 0.0
    if enable_impairments:
        st.markdown("**LO / Synthesizer**")
        freq_offset_hz = st.number_input(
            "Frequency offset (Hz)", value=0.0, step=100.0,
            help="Constant residual LO frequency offset, positive or negative.",
        )
        phase_noise_linewidth_hz = st.number_input(
            "Phase noise linewidth (Hz)", min_value=0.0, value=0.0, step=10.0,
            help="Free-running-oscillator single-sideband 3 dB linewidth. 0 disables it.",
        )

        st.markdown("**IQ Modulator**")
        iq_col1, iq_col2 = st.columns(2)
        with iq_col1:
            iq_gain_imbalance_db = st.number_input(
                "IQ gain imbalance (dB)", value=0.0, step=0.1,
                help="I/Q branch gain mismatch in the IQ modulator.",
            )
        with iq_col2:
            iq_phase_imbalance_deg = st.number_input(
                "IQ phase imbalance (deg)", value=0.0, step=0.5,
                help="Deviation from ideal 90-degree I/Q separation.",
            )

        st.markdown("**Power Amplifier**")
        enable_pa = st.checkbox(
            "Saturation / compression",
            help=(
                "Rapp AM-AM model (soft saturation, standard for solid-state "
                "PAs) with optional AM-PM conversion -- the last physical stage "
                "before the antenna, applied after the LO and IQ modulator "
                "impairments above. Produces spectral regrowth just outside "
                "the occupied bandwidth (odd-order intermodulation), the same "
                "signature a spectrum analyzer's adjacent-channel power "
                "measurement is built to catch on a real PA."
            ),
        )
        if enable_pa:
            pa_col1, pa_col2 = st.columns(2)
            with pa_col1:
                pa_backoff_db = st.number_input(
                    "Input backoff (dB)", value=0.0, step=0.5,
                    help=(
                        "Saturation point, in dB above this project's reference "
                        "unit amplitude (an ideal symbol's own magnitude). More "
                        "negative = saturates more aggressively (more visible "
                        "spectral regrowth); 0 dB already produces noticeable "
                        "compression since the RRC-shaped envelope's peaks "
                        "routinely exceed that reference."
                    ),
                )
            with pa_col2:
                pa_smoothness = st.number_input(
                    "Knee smoothness", min_value=0.1, value=3.0, step=0.5,
                    help="Rapp model knee sharpness: higher = sharper transition from linear to saturated.",
                )
            pa_am_pm_deg_per_db = st.number_input(
                "AM-PM conversion (deg/dB)", value=0.0, step=0.5,
                help="Phase shift per dB of AM-AM compression -- the same figure real TWTA/SSPA datasheets quote. 0 disables it.",
            )

        active = []
        if freq_offset_hz != 0:
            active.append(f"{freq_offset_hz:+.0f} Hz offset")
        if phase_noise_linewidth_hz > 0:
            active.append(f"{phase_noise_linewidth_hz:.0f} Hz phase noise linewidth")
        if iq_gain_imbalance_db != 0 or iq_phase_imbalance_deg != 0:
            active.append(f"IQ imbalance {iq_gain_imbalance_db:+.2f} dB / {iq_phase_imbalance_deg:+.2f}°")
        if enable_pa:
            pa_desc = f"PA backoff {pa_backoff_db:+.1f} dB (smoothness {pa_smoothness:.1f})"
            if pa_am_pm_deg_per_db != 0:
                pa_desc += f", AM-PM {pa_am_pm_deg_per_db:.1f} deg/dB"
            active.append(pa_desc)
        st.caption("Active: " + ", ".join(active) if active else "Enabled, but all values are still 0 -- no effect yet.")

# --------------------------------------------------------------------------
# Build params & run chain
# --------------------------------------------------------------------------
params = ChainParams(
    modulation=modulation,
    encoding=encoding,
    bit_rate=int(bit_rate),
    symbol_rate=int(symbol_rate),
    sps=int(sps),
    rrc_alpha=float(rrc_alpha),
    rrc_span=int(rrc_span),
    rs_e=int(rs_e),
    interleave_depth=int(interleave_depth),
    fec_rs=fec_rs,
    fec_conv=fec_conv,
    conv_rate=conv_rate,
    conv_invert_g2=invert_g2,
    randomizer=randomizer,
    input_format=input_format,
    n_cadu=int(n_cadu),
    payload_bytes=payload_source_bytes,
    seed=int(seed),
    vcid_list=vcid_list,
    spacecraft_id=int(spacecraft_id),
    corrupt_rs_symbols=int(corrupt_rs_symbols),
    corrupt_codeword_index=int(corrupt_codeword_index),
    corrupt_cadu_indices=corrupt_cadu_indices,
    corrupt_vc=int(corrupt_vc) if corrupt_vc is not None else None,
    corrupt_seed=int(corrupt_seed),
    freq_offset_hz=float(freq_offset_hz),
    iq_gain_imbalance_db=float(iq_gain_imbalance_db),
    iq_phase_imbalance_deg=float(iq_phase_imbalance_deg),
    phase_noise_linewidth_hz=float(phase_noise_linewidth_hz),
    impairment_seed=int(impairment_seed),
    pa_backoff_db=float(pa_backoff_db) if enable_pa else None,
    pa_smoothness=float(pa_smoothness),
    pa_am_pm_deg_per_db=float(pa_am_pm_deg_per_db),
)

panel = st.container(border=True)
with panel:
    st.markdown(
        '<div class="panel-head"><span class="dot"></span><h2>Live Preview</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div class="hktm-header">
      <div>
        <div class="hktm-title">HKTM <span>CCSDS</span> Signal Generator</div>
        <div class="hktm-subtitle">CCSDS 131.0-B-5 &middot; RF-Catcher (TestTree) test signal injection</div>
      </div>
      <div class="hktm-badges">
        <div class="hktm-badge">{modulation}</div>
        <div class="hktm-badge">{symbol_rate/1e3:.0f} kS/s</div>
        <div class="hktm-badge">{"RS(255," + str(rs_k) + ")" if fec_rs or is_cadu_input else "RS off"}</div>
        <div class="hktm-badge">{"CONV K=7 r=" + conv_rate if fec_conv else "CONV off"}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline stage indicator
    def chip(label, active):
        cls = "stage-chip on" if active else "stage-chip"
        return f'<span class="{cls}">{label}</span>'

    stages_html = '<div class="stage-row">' + '<span class="arrow">&rarr;</span>'.join([
        chip({"asm_frame": "ASM+TF (in)", "cadu": "CADU (in)"}[input_format] if is_cadu_input else ("PAYLOAD +VC" if vcid_list is not None else "PAYLOAD"), True),
        chip(f"RS(255,{rs_k}) I={interleave_depth}" + (" [in input]" if is_cadu_input else ""), fec_rs and not is_cadu_input),
        chip(f"+{corrupt_rs_symbols} ERR", corrupt_rs_symbols > 0),
        chip(randomizer_label.split(" ")[0].upper(), randomizer != "none"),
        chip("+ASM" + (" [in input]" if is_cadu_input else ""), not is_cadu_input),
        chip(f"CONV K=7 r={conv_rate}", fec_conv),
        chip("NRZ-L", True),
        chip(f"{modulation} GRAY" if modulation == "QPSK" else modulation, True),
        chip(f"RRC &alpha;={rrc_alpha:.2f}", True),
        chip("TX IMPAIR", freq_offset_hz != 0 or iq_gain_imbalance_db != 0
             or iq_phase_imbalance_deg != 0 or phase_noise_linewidth_hz > 0 or enable_pa),
    ]) + '</div>'
    st.markdown(stages_html, unsafe_allow_html=True)

    with st.spinner("Recomputing CCSDS chain..."):
        try:
            result = run_chain(params)
            error = None
        except Exception as exc:  # surface pipeline errors in the UI instead of crashing
            result = None
            error = str(exc)

    if error:
        st.error(f"Error in the generation chain: {error}")
        st.stop()

    duration_ms = len(result.symbols) / params.symbol_rate * 1000

    # --------------------------------------------------------------------------
    # Spectrum (real-time)
    # --------------------------------------------------------------------------
    nperseg = 4096
    n_segs = len(result.iq) // nperseg
    if n_segs < 1:
        st.warning("Signal too short for spectral analysis with these parameters: increase the CADU count.")
        st.stop()

    psd = welch_psd(result.iq, nperseg)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1 / result.sample_rate))
    peak = np.max(psd)
    db = 10 * np.log10(psd / peak)

    bw_theoretical = params.symbol_rate * (1 + params.rrc_alpha)

    # --------------------------------------------------------------------------
    # Metric card: the theoretical occupied bandwidth is what RF-Catcher cares
    # about, shown as a single hero tile.
    # --------------------------------------------------------------------------
    hero_metrics = [
        ("Occupied BW (theoretical)", f"{bw_theoretical/1e6:.3f}", "MHz",
         "Rs x (1 + alpha), the reference for RF-Catcher",
         "Theoretical RRC spectral edge = symbol rate x (1 + alpha)."),
    ]
    hero_cols = st.columns(1)
    for col, (label, value, unit, sub, tooltip) in zip(hero_cols, hero_metrics):
        col.markdown(f"""
        <div class="metric-hero" title="{tooltip}">
          <div class="label">{label}</div>
          <div class="value">{value}<span class="unit">{unit}</span></div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    secondary_metrics = [
        ("Duration", f"{duration_ms:.1f} ms"),
        ("Sample rate", f"{result.sample_rate/1e6:.3f} MS/s"),
        (f"{modulation} symbols", f"{len(result.symbols):,}".replace(",", " ")),
        ("Compute time", f"{result.elapsed*1000:.0f} ms"),
    ]
    secondary_html = '<div class="secondary-strip">' + "".join(
        f'<div class="sec-item"><span class="label">{label}</span><span class="value">{value}</span></div>'
        for label, value in secondary_metrics
    ) + '</div>'
    st.markdown(secondary_html, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------------------------------
    # Main plots
    # --------------------------------------------------------------------------
    spec_col, side_col = st.columns([2.1, 1])

    with spec_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs / 1e6, y=db, mode="lines", name="PSD",
            line=dict(color="#00d4ff", width=1.6),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
        ))
        fig.add_vline(x=-bw_theoretical / 2e6, line=dict(color="#ff5c7a", width=1, dash="dash"))
        fig.add_vline(x=bw_theoretical / 2e6, line=dict(color="#ff5c7a", width=1, dash="dash"),
                      annotation_text="theoretical edge", annotation_position="top right",
                      annotation_font_color="#ff5c7a")
        fig.update_layout(
            title="Spectrum (PSD) -- real-time",
            xaxis_title="Frequency (MHz)", yaxis_title="Relative PSD (dB)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=460, margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(range=[-70, 5], gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Note: the dashed red lines mark the theoretical occupied bandwidth edge "
            "(+/- symbol_rate x (1 + alpha) / 2), matching the metric above."
        )

    with side_col:
        # Sampled from the actual (post-impairment) IQ through a matched RRC
        # filter, not from ChainResult.symbols -- symbols are fixed before
        # pulse shaping/impairments are ever applied, so plotting them would
        # always show a perfect constellation regardless of any Transmitter
        # Impairments setting. See pulse_shaping.matched_filter_sample().
        decision_points = matched_filter_sample(
            result.iq, len(result.symbols), int(sps), rrc_taps(rrc_alpha, rrc_span, int(sps)),
        )
        n_preview = min(3000, len(decision_points))
        idx = np.linspace(0, len(decision_points) - 1, n_preview).astype(int)
        sub = decision_points[idx]
        fig_const = go.Figure()
        fig_const.add_trace(go.Scattergl(
            x=sub.real, y=sub.imag, mode="markers",
            marker=dict(size=3, color="#00d4ff", opacity=0.35),
        ))
        fig_const.update_layout(
            title=f"{modulation} constellation", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=210, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(range=[-1.2, 1.2], gridcolor="rgba(255,255,255,0.06)", zeroline=True, zerolinecolor="rgba(255,255,255,0.15)"),
            yaxis=dict(range=[-1.2, 1.2], gridcolor="rgba(255,255,255,0.06)", zeroline=True, zerolinecolor="rgba(255,255,255,0.15)", scaleanchor="x"),
        )
        st.plotly_chart(fig_const, width='stretch')

        n_time = min(200, len(result.iq))
        t_axis = np.arange(n_time) / result.sample_rate * 1e6
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=t_axis, y=result.iq.real[:n_time], mode="lines",
                                       name="I", line=dict(color="#00d4ff", width=1.3)))
        fig_time.add_trace(go.Scatter(x=t_axis, y=result.iq.imag[:n_time], mode="lines",
                                       name="Q", line=dict(color="#ff5c7a", width=1.3)))
        fig_time.update_layout(
            title="I/Q over time (excerpt)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=210, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(title="us", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        st.plotly_chart(fig_time, width='stretch')


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
panel = st.container(border=True)
with panel:
    st.markdown(
        '<div class="panel-head neutral"><span class="dot"></span><h2>Export</h2></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Produces a raw baseband IQ file (no header, interleaved I0,Q0,I1,Q1,...) "
        "for an IQ recorder/replayer such as RF-Catcher's Capture & Playback. The "
        "file carries no carrier/frequency information -- set the intended RF "
        "center frequency manually on the playback instrument."
    )

    symbols_per_cadu = len(result.symbols) / params.n_cadu
    native_fs = params.symbol_rate * params.sps
    exp_col1, exp_col2 = st.columns([1, 2])
    with exp_col1:
        if payload_mode == file_upload_label and file_cadu_count is not None:
            export_n_cadu = st.number_input(
                "Export CADU count", min_value=1, max_value=int(file_cadu_count),
                value=int(file_cadu_count), step=1,
                help=(
                    f"Capped at the {file_cadu_count} complete CADUs available in "
                    f"the uploaded {file_upload_label.lower()}: generation never pads with "
                    "extra data, so this can't exceed what the file actually "
                    "contains. Lower it to export only a leading subset."
                ),
            )
        else:
            export_n_cadu = st.number_input(
                "Export CADU count", min_value=1, value=int(n_cadu), step=100,
                help=(
                    "Number of CADUs for the exported file -- independent of the "
                    "preview above, and only computed when you click 'Generate "
                    "export file' below, so a large value here doesn't slow down "
                    "live parameter tweaking."
                ),
            )
    with exp_col2:
        export_duration_s = export_n_cadu * symbols_per_cadu / params.symbol_rate

    fmt_col1, fmt_col2, fmt_col3 = st.columns(3)
    with fmt_col1:
        output_dtype_label = st.selectbox(
            "Output format", ["float32 ([-1, +1])", "int16 (12-bit, RF-Catcher)"],
            help=(
                "Sample data type for the raw IQ file. float32 is compatible "
                "with most modern SDR tooling; int16 matches the RF-Catcher "
                "(TestTree) format -- little-endian, 12 significant bits in "
                "two's complement, range [-2048, 2047] -- check what your "
                "IQ recorder/replayer requires."
            ),
        )
        output_dtype = "int16" if output_dtype_label.startswith("int16") else "float32"
    with fmt_col2:
        output_peak = st.number_input(
            "Peak amplitude", min_value=0.1, max_value=1.0, value=0.9, step=0.05,
            help="Normalized peak |I|/|Q| amplitude before dtype conversion, leaving headroom against clipping on playback.",
        )
    with fmt_col3:
        resample_enabled = st.checkbox(
            "Resample to fixed rate",
            help=(
                "Resample the exported file (via polyphase resampling) to an "
                "exact sample rate your instrument expects, independent of the "
                "chain's native symbol_rate x samples/symbol. The live preview "
                "and internal processing are unaffected -- this only reshapes "
                "the exported samples."
            ),
        )
        target_fs = st.number_input("Target sample rate (Hz)", min_value=1.0, value=10_000_000.0, step=1e6,
                                     disabled=not resample_enabled, format="%.0f")
        if resample_enabled:
            up, down = resample_ratio(native_fs, target_fs)
            st.caption(
                f"Native: {native_fs/1e6:.3f} MS/s -> resample {up}/{down} "
                f"({target_fs/native_fs:.3f}x) -> {target_fs/1e6:.3f} MS/s"
            )
        else:
            st.caption(f"Native sample rate: {native_fs/1e6:.3f} MS/s (no resampling)")

    export_output_fs = target_fs if resample_enabled else native_fs
    export_bytes_per_sample = 4 if output_dtype == "float32" else 2
    export_native_samples = export_n_cadu * symbols_per_cadu * params.sps
    export_samples = export_native_samples * (export_output_fs / native_fs)
    export_mb = export_samples * 2 * export_bytes_per_sample / 1e6
    # export_chain() generates in bounded-memory batches (~64 MB) regardless
    # of file size, so a large export only costs time/disk, not RAM -- the
    # one exception is resampling, which is not streamed (see export_chain()'s
    # docstring) and needs the whole native-rate signal in memory at once.
    if export_mb > 500:
        size_warning = (
            " -- large: generating this may take a while, and needs the whole native-rate "
            "signal in memory at once for resampling (refused upfront if it wouldn't fit)"
            if resample_enabled else
            " -- large: generating this may take a while (memory use stays bounded regardless of file size)"
        )
    else:
        size_warning = ""
    st.caption(f"~{export_duration_s:.2f} s of signal, ~{export_mb:,.0f} MB file{size_warning}")

    generate_clicked = st.button("Generate export file", width='stretch')

    export_key = (
        export_n_cadu, params.rs_e, params.interleave_depth, params.fec_rs, params.fec_conv,
        params.conv_rate, params.conv_invert_g2, params.randomizer, params.input_format, params.rrc_alpha,
        params.rrc_span, params.sps, params.symbol_rate, params.seed, payload_mode,
        output_dtype, output_peak, resample_enabled, target_fs if resample_enabled else None,
    )
    if generate_clicked:
        export_params = dataclasses.replace(params, n_cadu=int(export_n_cadu))
        progress_bar = st.progress(0, text=f"Generating {export_n_cadu:,} CADUs...".replace(",", " "))

        def _update_progress(frac: float, message: str) -> None:
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=message)

        # Clean up the previous export's file before starting a new one --
        # export_chain() writes straight to disk, so nothing here holds a
        # reference that would do this automatically.
        prev = st.session_state.get("export_data")
        if prev is not None and os.path.exists(prev["path"]):
            try:
                os.remove(prev["path"])
            except OSError:
                pass

        os.makedirs(EXPORT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fs_for_name = target_fs if resample_enabled else native_fs
        out_path = os.path.join(
            EXPORT_DIR, f"qpsk_ccsds_{out_fs_for_name/1e6:.1f}Msps_{export_n_cadu}cadu_{timestamp}.iq")

        try:
            # export_chain() processes CADUs in bounded-memory batches and
            # streams straight to out_path, so peak memory stays roughly
            # constant regardless of export_n_cadu -- unlike the old
            # run_chain()-based export, which built the whole bits/coded-
            # bits/symbols/IQ arrays in memory for the entire export at
            # once and could run out of memory on a large one.
            export_result = export_chain(
                export_params, out_path, output_dtype=output_dtype, peak=output_peak,
                target_fs=target_fs if resample_enabled else None,
                progress_callback=_update_progress,
            )
            progress_bar.empty()
            meta = export_result["meta"]
            file_size = os.path.getsize(out_path)
            st.session_state["export_data"] = {
                "path": out_path,
                "meta_bytes": json.dumps(meta, indent=2).encode(),
                "filename": os.path.basename(out_path),
                "file_size": file_size,
                "caption": (
                    f"Format: raw interleaved {output_dtype} (I0,Q0,I1,Q1,...) &middot; "
                    f"{file_size/1e6:.2f} MB &middot; {meta['output_n_samples']:,} samples @ "
                    f"{meta['output_sample_rate']/1e6:.3f} MS/s &middot; "
                    f"CADU: {export_n_cadu} x {export_result['cadu_bytes']} bytes"
                ),
            }
            st.session_state["export_key"] = export_key
        except MemoryError as e:
            progress_bar.empty()
            st.error(f"Export failed: {e}")

    if "export_data" in st.session_state:
        data = st.session_state["export_data"]
        dl_col1, dl_col2 = st.columns([1, 3])
        with dl_col1:
            if not os.path.exists(data["path"]):
                st.warning(f"Export file no longer found at `{data['path']}` (was it moved or deleted?).")
            elif data["file_size"] <= _DOWNLOAD_BUTTON_SIZE_LIMIT:
                with open(data["path"], "rb") as f:
                    iq_bytes = f.read()
                st.download_button("Download IQ file", data=iq_bytes,
                                    file_name=data["filename"], mime="application/octet-stream",
                                    width='stretch')
            else:
                st.info(
                    f"File too large ({data['file_size']/1e6:,.0f} MB) to also offer as a "
                    "browser download here -- it's already saved to disk at the path below; "
                    "retrieve it directly from there.".replace(",", " ")
                )
            st.download_button("Download metadata (.json)", data=data["meta_bytes"],
                                file_name=data["filename"].rsplit(".", 1)[0] + ".meta.json",
                                mime="application/json", width='stretch')
        with dl_col2:
            st.caption(data["caption"])
            st.caption(f"Saved to: `{data['path']}`")
            if st.session_state.get("export_key") != export_key:
                st.caption("Note: parameters changed since this file was generated -- click 'Generate export file' again to refresh.")
    else:
        st.caption("Click 'Generate export file' to produce a downloadable IQ file at the size set above.")

