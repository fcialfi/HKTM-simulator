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
from ccsds_chain.utils import iq_to_int16_interleaved

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

#MainMenu, footer, header { visibility: hidden; }

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
    st.markdown("### Modulazione & Encoding")
    modulation = st.selectbox("Modulazione", ["QPSK"], help="Baseline. Altre modulazioni: roadmap.")
    encoding = st.selectbox("Encoding di linea", ["NRZ-L"], help="Baseline. Altri encoding: roadmap.")
    symbol_rate = st.number_input("Symbol rate (S/s)", value=1_785_000, step=1_000, format="%d")
    bit_rate = st.number_input("Bit rate (bps, informativo)", value=3_570_000, step=1_000, format="%d")

    st.markdown("### FEC")
    fec_rs = st.checkbox("Reed-Solomon (255,223) interleave x5", value=True)
    fec_conv = st.checkbox("Convoluzionale K=7 rate 1/2", value=True)
    invert_g2 = st.checkbox("Invert G2 (convenzione CCSDS)", value=True, disabled=not fec_conv)
    scrambling = st.checkbox("Scrambler CCSDS (LFSR seed 0xFF)", value=False)

    st.markdown("### Pulse Shaping (RRC)")
    rrc_alpha = st.slider("Roll-off alpha", 0.05, 1.00, 0.35, 0.01)
    rrc_span = st.slider("Span (simboli)", 4, 16, 8, 1)
    sps = st.select_slider("Samples/simbolo", options=[2, 4, 8], value=4)

    st.markdown("### Payload & Durata")
    n_cadu = st.slider("N. CADU", 5, 300, 40, 5,
                        help="Numero di CADU generati. Controlla sia l'anteprima real-time sia il file esportato.")
    payload_mode = st.radio("Sorgente payload", ["Pseudo-random", "File Transfer Frame"], horizontal=True)
    payload_source_bytes = None
    seed = 42
    if payload_mode == "Pseudo-random":
        seed = st.number_input("Seed", value=42, step=1)
    else:
        uploaded = st.file_uploader("Transfer Frame (binario)", type=None)
        if uploaded is not None:
            payload_source_bytes = uploaded.read()
        else:
            st.caption("Nessun file caricato: uso pseudo-random come fallback.")

    st.markdown("### Output")
    output_filename = st.text_input("Nome file IQ", value="output_iq.raw")

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

with st.spinner("Ricalcolo catena CCSDS..."):
    try:
        result = run_chain(params)
        error = None
    except Exception as exc:  # surface pipeline errors in the UI instead of crashing
        result = None
        error = str(exc)

if error:
    st.error(f"Errore nella catena di generazione: {error}")
    st.stop()

duration_ms = len(result.symbols) / params.symbol_rate * 1000

# --------------------------------------------------------------------------
# Spectrum (real-time)
# --------------------------------------------------------------------------
nperseg = 4096
n_segs = len(result.iq) // nperseg
if n_segs < 1:
    st.warning("Segnale troppo corto per l'analisi spettrale con questi parametri: aumenta N. CADU.")
    st.stop()

psd = welch_psd(result.iq, nperseg)
freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1 / result.sample_rate))
peak = np.max(psd)
db = 10 * np.log10(psd / peak)

bw_3db = contiguous_bandwidth(freqs, db, -3.0)
bw_20db = contiguous_bandwidth(freqs, db, -20.0)

# --------------------------------------------------------------------------
# Metric cards
# --------------------------------------------------------------------------
cols = st.columns(6)
metrics = [
    ("Banda -3dB", f"{bw_3db/1e6:.3f}", "MHz", "accent"),
    ("Banda -20dB", f"{bw_20db/1e6:.3f}", "MHz", "accent"),
    ("Durata segnale", f"{duration_ms:.1f}", "ms", ""),
    ("Sample rate", f"{result.sample_rate/1e6:.3f}", "MS/s", ""),
    ("Simboli QPSK", f"{len(result.symbols):,}".replace(",", " "), "", ""),
    ("Tempo calcolo", f"{result.elapsed*1000:.0f}", "ms", "warn" if result.elapsed > 1.0 else ""),
]
for col, (label, value, unit, style) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card {style}">
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
        title="Spettro (PSD) -- real-time",
        xaxis_title="Frequenza (MHz)", yaxis_title="PSD relativa (dB)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=460, margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=[-70, 5], gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')

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
        title="Costellazione QPSK", template="plotly_dark",
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
        title="I/Q nel tempo (estratto)", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=210, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="µs", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    st.plotly_chart(fig_time, width='stretch')

# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
st.markdown("### Esporta")
exp_col1, exp_col2 = st.columns([1, 3])

iq_bytes = iq_to_int16_interleaved(result.iq).tobytes()
meta_bytes = json.dumps(result.meta, indent=2).encode()

with exp_col1:
    st.download_button("Scarica IQ (.raw)", data=iq_bytes,
                        file_name=output_filename, mime="application/octet-stream",
                        width='stretch')
    st.download_button("Scarica metadata (.json)", data=meta_bytes,
                        file_name=output_filename.rsplit(".", 1)[0] + ".meta.json",
                        mime="application/json", width='stretch')
with exp_col2:
    st.caption(
        f"Formato: raw interleaved int16 LE, 12 bit [-2048,2047] (I0,Q0,I1,Q1,...) &middot; "
        f"{len(iq_bytes)/1e6:.2f} MB &middot; {len(result.iq):,} campioni @ "
        f"{result.sample_rate/1e6:.3f} MS/s &middot; CADU: {params.n_cadu} x {result.cadu_bytes} byte"
    )
    with st.expander("Vedi limitazioni note prima dell'uso su hardware reale"):
        st.markdown("""
- **RS Alpha/Beta**: uso i parametri GF(256) di default di `reedsolo`, non verificati contro la rappresentazione CCSDS attesa dal ricevitore.
- **ASM** incluso nella codifica convoluzionale (e nello scrambling se attivo) seguendo l'ordine letterale della catena richiesta -- da validare.
- **Formato IQ Converter** RF-Catcher da confermare con TestTree.

Dettagli completi in `README.md`.
        """)
