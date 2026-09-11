#!/usr/bin/env python3
"""HKTM CCSDS Signal Generator -- GUI (Streamlit).

Interfaccia grafica per la generazione del segnale di test CCSDS (baseline
QPSK) per iniezione via RF-Catcher (TestTree) Capture & Playback. Ogni
modifica ai parametri ricalcola la catena e aggiorna spettro, costellazione
e metriche in tempo reale.

Uso: streamlit run app.py
"""

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ccsds_chain.pipeline import ChainParams, run_chain
from ccsds_chain.spectrum import welch_psd, contiguous_bandwidth

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
    fec_rs = st.checkbox("Reed-Solomon (255,223) interleave x5", value=True)
    fec_conv = st.checkbox("Convolutional K=7 rate 1/2", value=True)
    invert_g2 = st.checkbox("Invert G2 (CCSDS convention)", value=True, disabled=not fec_conv)
    scrambling = st.checkbox("CCSDS scrambler (LFSR seed 0xFF)", value=False)

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
    n_cadu = st.slider("CADU count", 5, 300, 40, 5,
                        help="Number of CADUs generated. Controls both the real-time preview and the exported file.")
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

    st.markdown("### Output")
    output_filename = st.text_input("IQ file name", value="output_iq.raw")

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
    fec_rs=fec_rs,
    fec_conv=fec_conv,
    conv_invert_g2=invert_g2,
    scrambling=scrambling,
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
    chip("RS(255,223)", fec_rs),
    chip("+ASM", True),
    chip("CONV K=7 r=1/2", fec_conv),
    chip("NRZ-L", True),
    chip("SCRAMBLER", scrambling),
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
exp_col1, exp_col2 = st.columns([1, 3])

interleaved = np.empty(2 * len(result.iq), dtype=np.float32)
interleaved[0::2] = result.iq.real.astype(np.float32)
interleaved[1::2] = result.iq.imag.astype(np.float32)
iq_bytes = interleaved.tobytes()
meta_bytes = json.dumps(result.meta, indent=2).encode()

with exp_col1:
    st.download_button("Download IQ (.raw)", data=iq_bytes,
                        file_name=output_filename, mime="application/octet-stream",
                        width='stretch')
    st.download_button("Download metadata (.json)", data=meta_bytes,
                        file_name=output_filename.rsplit(".", 1)[0] + ".meta.json",
                        mime="application/json", width='stretch')
with exp_col2:
    st.caption(
        f"Format: raw interleaved float32 (I0,Q0,I1,Q1,...) &middot; "
        f"{len(iq_bytes)/1e6:.2f} MB &middot; {len(result.iq):,} samples @ "
        f"{result.sample_rate/1e6:.3f} MS/s &middot; CADU: {params.n_cadu} x {result.cadu_bytes} bytes"
    )
    with st.expander("See known limitations before use on real hardware"):
        st.markdown("""
- **RS Alpha/Beta**: uses `reedsolo`'s default GF(256) parameters, not verified against the CCSDS representation expected by the receiver.
- **ASM** is included in the convolutional coding (and in scrambling, if enabled), following the literal order of the requested chain -- to be validated.
- **RF-Catcher IQ converter format** to be confirmed with TestTree.

Full details in `README.md`.
        """)
