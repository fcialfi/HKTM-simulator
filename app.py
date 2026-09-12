#!/usr/bin/env python3
"""HKTM CCSDS Signal Generator -- GUI (Streamlit).

Interfaccia grafica per la generazione del segnale di test CCSDS (baseline
QPSK) per iniezione via RF-Catcher (TestTree) Capture & Playback. Ogni
modifica ai parametri ricalcola la catena e aggiorna spettro, costellazione
e metriche in tempo reale.

Uso: streamlit run app.py
"""

import dataclasses
import datetime
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ccsds_chain.pipeline import ChainParams, run_chain
from ccsds_chain.spectrum import welch_psd, contiguous_bandwidth
from ccsds_chain.utils import normalize_peak, resample_iq, pack_iq_interleaved

BITS_PER_SYMBOL = {"QPSK": 2}

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
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.3); color: #00d4ff;
}

.metric-card {
    background: linear-gradient(160deg, #131a2a 0%, #0f1522 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.85rem 1rem; height: 100%;
    cursor: help;
}
.metric-card .label { font-size: 0.72rem; color: #7d90ac; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
.metric-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; color: #eaf6ff; font-weight: 600; margin-top: 2px; }
.metric-card .unit { font-size: 0.78rem; color: #56698a; margin-left: 4px; }
.metric-card.accent .value { color: #00d4ff; }
.metric-card.warn .value { color: #ffb454; }

.stage-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 0.4rem 0 1rem 0; }
.stage-chip {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600;
    padding: 5px 11px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03); color: #55698a;
}
.stage-chip.on { background: rgba(0,212,255,0.10); border-color: rgba(0,212,255,0.35); color: #00d4ff; }
.stage-chip .arrow { color: #33455f; margin-left: 8px; }

section[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] h3 { color: #00d4ff !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 1.2rem; }

div[data-testid="stExpander"] { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="hktm-header">
  <div>
    <div class="hktm-title">HKTM <span>CCSDS</span> Signal Generator</div>
    <div class="hktm-subtitle">AWS-OSE-ICD-0063 &middot; CCSDS 131.0-B-2 &middot; RF-Catcher (TestTree) test signal injection</div>
  </div>
  <div class="hktm-badges">
    <div class="hktm-badge">QPSK</div>
    <div class="hktm-badge">1785 kS/s</div>
    <div class="hktm-badge">RS(255,223)</div>
    <div class="hktm-badge">CONV K=7</div>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Modulation & Encoding")
    modulation = st.selectbox("Modulation", ["QPSK"], help="Baseline. Other modulations: roadmap.")
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

    st.markdown("### FEC")
    fec_rs = st.checkbox("Reed-Solomon", value=True)
    rs_col1, rs_col2 = st.columns(2)
    with rs_col1:
        rs_e = st.selectbox(
            "Error correction E", [16, 8], disabled=not fec_rs,
            help=(
                "RS error-correction capability, in symbols (CCSDS 4.3.1, managed "
                "parameter 12.5). E=16 gives RS(255,223): more parity overhead, "
                "corrects up to 16 symbol errors per codeword. E=8 gives "
                "RS(255,239): less overhead, corrects up to 8."
            ),
        )
    with rs_col2:
        interleave_depth = st.selectbox(
            "Interleave depth I", [1, 2, 3, 4, 5, 8], index=4, disabled=not fec_rs,
            help=(
                "Number of RS codewords interleaved together per CADU (CCSDS "
                "4.3.5, managed parameter 12.5). Higher I spreads a burst error "
                "across more codewords (each corrects a smaller share of it) at "
                "the cost of a larger CADU."
            ),
        )
    rs_k = 255 - 2 * rs_e

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

    randomizer_label = st.selectbox(
        "Pseudo-randomizer", ["Long (131071-bit)", "Short (255-bit, legacy)", "None"],
        help=(
            "CCSDS section 10: scrambles the RS-coded data (never the ASM) to "
            "guarantee bit transitions, avoid spectral lines, and aid receiver "
            "acquisition. 'Long' is the current standard default (Issue 5, "
            "2023, managed parameter 12.3); 'Short' is kept only for backward "
            "compatibility with legacy systems."
        ),
    )
    randomizer = {"Long (131071-bit)": "long", "Short (255-bit, legacy)": "short", "None": "none"}[randomizer_label]

    st.markdown("### Pulse Shaping (RRC)")
    st.caption("Shapes the QPSK symbols into a band-limited waveform (Root-Raised-Cosine filter) before D/A conversion.")
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

    st.markdown("### Payload & Duration")
    n_cadu = st.slider(
        "Preview CADU count", 5, 300, 40, 5,
        help=(
            "Number of CADUs used for the live spectrum/constellation preview "
            "below. Kept small so every parameter change recomputes instantly. "
            "The exported file's length is set independently, further down, "
            "and is only (re)computed when you click 'Generate export file'."
        ),
    )
    payload_mode = st.radio("Payload source", ["Pseudo-random", "Transfer Frame file"], horizontal=True)
    payload_source_bytes = None
    seed = 42
    if payload_mode == "Pseudo-random":
        seed = st.number_input("Seed", value=42, step=1)
    else:
        uploaded = st.file_uploader("Transfer Frame (binary)", type=None)
        if uploaded is not None:
            payload_source_bytes = uploaded.read()
        else:
            st.caption("No file uploaded: falling back to pseudo-random.")

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
    n_cadu=int(n_cadu),
    payload_bytes=payload_source_bytes,
    seed=int(seed),
)

# Pipeline stage indicator
def chip(label, active):
    cls = "stage-chip on" if active else "stage-chip"
    return f'<span class="{cls}">{label}</span>'

stages_html = '<div class="stage-row">' + '<span class="arrow">&rarr;</span>'.join([
    chip("PAYLOAD", True),
    chip(f"RS(255,{rs_k}) I={interleave_depth}", fec_rs),
    chip(randomizer_label.split(" ")[0].upper(), randomizer != "none"),
    chip("+ASM", True),
    chip(f"CONV K=7 r={conv_rate}", fec_conv),
    chip("NRZ-L", True),
    chip("QPSK GRAY", True),
    chip(f"RRC &alpha;={rrc_alpha:.2f}", True),
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

bw_3db = contiguous_bandwidth(freqs, db, -3.0)
bw_20db = contiguous_bandwidth(freqs, db, -20.0)
bw_theoretical = params.symbol_rate * (1 + params.rrc_alpha)

# --------------------------------------------------------------------------
# Metric cards
# --------------------------------------------------------------------------
cols = st.columns(7)
metrics = [
    ("-3dB Bandwidth", f"{bw_3db/1e6:.3f}", "MHz", "accent",
     "Width where the measured PSD stays within 3 dB of its peak. For an RRC-shaped "
     "signal this sits at +/-Rs/2 almost regardless of roll-off, so it tracks the "
     "symbol rate rather than the roll-off."),
    ("-20dB Bandwidth", f"{bw_20db/1e6:.3f}", "MHz", "accent",
     "Width where the measured PSD stays within 20 dB of its peak. Grows with the "
     "roll-off alpha, approaching the theoretical occupied bandwidth at high alpha."),
    ("Occupied BW (theoretical)", f"{bw_theoretical/1e6:.3f}", "MHz", "accent",
     "Theoretical RRC spectral edge = symbol rate x (1 + alpha). Not a live "
     "measurement -- compare it against the -20dB bandwidth above."),
    ("Signal duration", f"{duration_ms:.1f}", "ms", "", "Length of the generated burst."),
    ("Sample rate", f"{result.sample_rate/1e6:.3f}", "MS/s", "", "Symbol rate x samples/symbol."),
    ("QPSK symbols", f"{len(result.symbols):,}".replace(",", " "), "", "", "Total number of QPSK symbols generated."),
    ("Compute time", f"{result.elapsed*1000:.0f}", "ms", "warn" if result.elapsed > 1.0 else "",
     "Wall-clock time to run the full chain for this preview."),
]
for col, (label, value, unit, style, tooltip) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card {style}" title="{tooltip}">
      <div class="label">{label}</div>
      <div class="value">{value}<span class="unit">{unit}</span></div>
    </div>""", unsafe_allow_html=True)

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
    fig.add_hline(y=-3, line=dict(color="#ffb454", width=1, dash="dash"),
                  annotation_text="-3 dB", annotation_position="top left",
                  annotation_font_color="#ffb454")
    fig.add_hline(y=-20, line=dict(color="#ff5c7a", width=1, dash="dash"),
                  annotation_text="-20 dB", annotation_position="top left",
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
        "Note: the -3dB bandwidth (dashed orange) stays close to the symbol rate for "
        "any roll-off -- it's the -20dB line (dashed red) and the 'Occupied BW "
        "(theoretical)' metric above that grow with alpha."
    )

with side_col:
    n_preview = min(3000, len(result.symbols))
    idx = np.linspace(0, len(result.symbols) - 1, n_preview).astype(int)
    sub = result.symbols[idx]
    fig_const = go.Figure()
    fig_const.add_trace(go.Scattergl(
        x=sub.real, y=sub.imag, mode="markers",
        marker=dict(size=3, color="#00d4ff", opacity=0.35),
    ))
    fig_const.update_layout(
        title="QPSK constellation", template="plotly_dark",
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
st.markdown("### Export")
st.caption(
    "Produces a raw baseband IQ file (no header, interleaved I0,Q0,I1,Q1,...) "
    "for an IQ recorder/replayer such as RF-Catcher's Capture & Playback. The "
    "file carries no carrier/frequency information -- set the intended RF "
    "center frequency manually on the playback instrument."
)

symbols_per_cadu = len(result.symbols) / params.n_cadu
exp_col1, exp_col2 = st.columns([1, 2])
with exp_col1:
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

native_fs = params.symbol_rate * params.sps
export_output_fs = target_fs if resample_enabled else native_fs
export_bytes_per_sample = 4 if output_dtype == "float32" else 2
export_native_samples = export_n_cadu * symbols_per_cadu * params.sps
export_samples = export_native_samples * (export_output_fs / native_fs)
export_mb = export_samples * 2 * export_bytes_per_sample / 1e6
size_warning = " -- large: generating this may take a while and use significant RAM" if export_mb > 500 else ""
st.caption(f"~{export_duration_s:.2f} s of signal, ~{export_mb:,.0f} MB file{size_warning}")

generate_clicked = st.button("Generate export file", width='stretch')

export_key = (
    export_n_cadu, params.rs_e, params.interleave_depth, params.fec_rs, params.fec_conv,
    params.conv_rate, params.conv_invert_g2, params.randomizer, params.rrc_alpha,
    params.rrc_span, params.sps, params.symbol_rate, params.seed, payload_mode,
    output_dtype, output_peak, resample_enabled, target_fs if resample_enabled else None,
)
if generate_clicked:
    export_params = dataclasses.replace(params, n_cadu=int(export_n_cadu))
    with st.spinner(f"Generating {export_n_cadu} CADUs..."):
        export_result = run_chain(export_params)
        iq = normalize_peak(export_result.iq, output_peak)
        output_fs = export_result.sample_rate
        if resample_enabled and target_fs != export_result.sample_rate:
            iq = resample_iq(iq, export_result.sample_rate, target_fs)
            output_fs = target_fs
        meta = dict(export_result.meta)
        meta.update({
            "format": "raw interleaved, no header (I0,Q0,I1,Q1,...)",
            "output_dtype": output_dtype,
            "output_peak": output_peak,
            "output_sample_rate": output_fs,
            "output_n_samples": len(iq),
            "output_duration_s": len(iq) / output_fs,
            "carrier_note": "file is baseband IQ (no carrier/frequency information); "
                             "set the intended RF center frequency manually on the playback instrument",
        })
        iq_bytes = pack_iq_interleaved(iq, output_dtype)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["export_data"] = {
            "iq_bytes": iq_bytes,
            "meta_bytes": json.dumps(meta, indent=2).encode(),
            "filename": f"qpsk_ccsds_{output_fs/1e6:.1f}Msps_{export_n_cadu}cadu_{timestamp}.iq",
            "caption": (
                f"Format: raw interleaved {output_dtype} (I0,Q0,I1,Q1,...) &middot; "
                f"{len(iq_bytes)/1e6:.2f} MB &middot; {len(iq):,} samples @ "
                f"{output_fs/1e6:.3f} MS/s &middot; CADU: {export_n_cadu} x {export_result.cadu_bytes} bytes"
            ),
        }
        st.session_state["export_key"] = export_key

if "export_data" in st.session_state:
    data = st.session_state["export_data"]
    dl_col1, dl_col2 = st.columns([1, 3])
    with dl_col1:
        st.download_button("Download IQ file", data=data["iq_bytes"],
                            file_name=data["filename"], mime="application/octet-stream",
                            width='stretch')
        st.download_button("Download metadata (.json)", data=data["meta_bytes"],
                            file_name=data["filename"].rsplit(".", 1)[0] + ".meta.json",
                            mime="application/json", width='stretch')
    with dl_col2:
        st.caption(data["caption"])
        if st.session_state.get("export_key") != export_key:
            st.caption("Note: parameters changed since this file was generated -- click 'Generate export file' again to refresh.")
else:
    st.caption("Click 'Generate export file' to produce a downloadable IQ file at the size set above.")

with st.expander("See known limitations before use on real hardware"):
    st.markdown("""
- **Turbo coding and LDPC** (CCSDS 131.0-B-5 sections 6-8) are not implemented -- only Reed-Solomon, convolutional (with puncturing), and their concatenation.
- **Transfer Frame length constraints** (section 11) are not enforced: some combinations of E / interleave depth / convolutional rate / CADU count can produce an odd number of coded bits, which fails QPSK pairing (shown as an error, not a crash). This never affects the rate-1/2 baseline.
- **NRZ-L polarity**: bit 1 -> +1, bit 0 -> -1; not yet verified against the receiver/tool's expected polarity.
- **RF-Catcher's "IQ Converter" tool** may expect its own `.rfcatcher` container format rather than the raw binary produced here -- to be confirmed on real hardware.

Full details in `README.md`.
    """)
