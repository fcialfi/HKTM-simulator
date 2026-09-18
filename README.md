# HKTM-simulator

Test RF signal generator, modulated per the CCSDS chain (baseline QPSK), to
be injected via the **RF-Catcher (TestTree) Capture & Playback Application**.

References: AWS-OSE-ICD-0063 (Arctic Weather Satellite downlink),
CCSDS 131.0-B-5 (TM Synchronization and Channel Coding, Sept. 2023),
ECSS-E-ST-50-01C.

## Project structure

```
app.py                  GUI (Streamlit) with real-time spectrum/constellation
generate_signal.py      CLI, generates output_iq.raw + metadata
verify_spectrum.py      CLI, checks the occupied bandwidth of an existing IQ file
ccsds_chain/
  pipeline.py            chain orchestration (used by app.py and generate_signal.py)
  reed_solomon.py         RS(255,223)/(255,239), CCSDS-native GF(256) and dual-basis
  convolutional.py        convolutional K=7, rate 1/2 punctured to 2/3-7/8
  scrambler.py             CCSDS pseudo-randomizer (131071-bit and 255-bit legacy)
  mapping.py               NRZ-L + QPSK Gray
  pulse_shaping.py         RRC filter
  spectrum.py              PSD/occupied bandwidth (used by app.py and verify_spectrum.py)
  utils.py                 bit/byte helpers, IQ file I/O
```

## Baseline parameters

| Parameter       | Baseline value          | Selectable |
|-----------------|---------------------------|---------------|
| Modulation     | QPSK                       | yes (only QPSK implemented) |
| Bit rate        | 3,570 kbps (post-coding, header included) | yes (or symbol rate, kept in sync) |
| Symbol rate     | 1,785 kS/s                 | yes |
| Encoding        | NRZ-L                      | yes (only NRZ-L implemented) |
| RS FEC          | RS(255,223), interleave depth 5 | yes (on/off; E=8 or 16; I=1,2,3,4,5,8) |
| Convolutional FEC | rate 1/2, K=7, G1=171o, G2=133o | yes (on/off; rate 1/2, 2/3, 3/4, 5/6, 7/8; invert G2 only at rate 1/2) |
| Pseudo-randomizer | none                  | yes (none / 255-bit legacy / 131071-bit, standard default since 2023) |
| CADU            | 4-byte ASM (0x1ACFFC1D, uncoded) + 1275-byte RS-encoded = 1279 bytes | depends on I (CADU = 4 + 255*I bytes) |

## Implemented chain

Follows the official CCSDS 131.0-B-3 structure ("Overall Structure of Channel
Coding"): RS encode -> pseudo-random -> attach ASM (= CADU) -> convolutional
over the CADU stream:

```
payload -> RS(255,223) interleave x5 -> [scrambler, excludes ASM]
        -> + ASM (per CADU, forms the CADU) -> conv. K=7 r=1/2 (over the CADU stream)
        -> NRZ-L -> QPSK (Gray) -> RRC -> IQ int16 (RF-Catcher)
```

1. **Payload**: reproducible pseudo-random data (seed) per CADU, or real data
   from a file (`--payload-source`). The format of this data is selectable
   via `--input-format` (or, in the GUI, "Payload contains"):
   - `transfer_frame` (default): raw, uncoded Transfer Frames. RS, the
     pseudo-randomizer and the ASM are all applied here to build the CADUs.
   - `cadu`: already-complete CADUs (ASM + RS-encoded block, already
     pseudo-randomized if that's how they were built) -- e.g. captured or
     previously generated CADUs. In this case RS, the randomizer and the ASM
     are **not** re-applied (to avoid double-encoding and a second ASM in
     front of data that already has one); only the convolutional stage (if
     enabled) is still applied over the CADU stream, exactly as a physical
     coder downstream of an already-formed CADU stream would.

     A real/captured CADU file is not guaranteed to start exactly on a CADU
     boundary (unframed idle line-fill in front, or an excerpt starting
     mid-stream): with a real source, the tool searches the file for the
     first genuine ASM before slicing (see `find_cadu_sync`). Each CADU is
     always exactly `unit_bytes` long -- the length the configured RS error
     correction E and interleave depth predict (`4 + 255*I` bytes) -- never
     measured from the spacing between ASMs: a real capture commonly wraps
     each CADU in its own instrument-specific record framing (e.g.
     RF-Catcher/TestTree's own capture format: a fixed header, the CADU,
     then a short postamble, repeated per record), which is not part of the
     CADU and must never be fed into convolutional coding as if it were.
     The ASM is used only to find where each CADU **starts** (see
     `find_all_cadu_positions`); whatever lies between the end of one CADU
     and the next real ASM -- whatever it is, and however long -- is simply
     skipped. If the configured RS-E/interleave depth predicts a CADU
     *longer* than the real spacing to the next ASM found in the file, a
     clear, actionable error is raised (GUI and CLI): the configured
     settings don't match how these CADUs were actually built.
2. **Reed-Solomon**: RS(255,223) with E=16 (default) or RS(255,239) with
   E=8, interleaving at a selectable depth I=1,2,3,4,5,8 (byte `i` goes into
   sub-stream `i mod I`). CCSDS-native implementation (not a generic RS
   library): Galois field GF(256) with polynomial F(x)=x^8+x^7+x^2+x+1
   (0x187, not the "generic" 0x11D), generator polynomial g(x) with roots
   alpha^(11j) taken directly from the expanded coefficients in the
   standard's Annex G, and the mandatory **dual-basis (Berlekamp)**
   representation (4.3.9) applied only to the computed parity (the Transfer
   Frame bytes stay unchanged, being the "uncoded" part of the CADU -- see
   Annex F, "Transformational Equivalence"). GF(256), the g(x) coefficients
   and the dual-basis transform have all been checked against the reference
   tables and numeric examples the standard itself provides (Table F-1,
   Annex G, Examples 1-2 of Annex F), as well as for algebraic divisibility
   of the resulting codeword by all 2E required roots.
3. **CCSDS pseudo-randomizer** (optional, section 10): a long, 131071-bit
   sequence (the standard's default since 2023, polynomial x^17+x^14+1) or a
   short, 255-bit one (legacy, polynomial 1+x^3+x^5+x^7+x^8), via bit-wise
   XOR applied only to each CADU's RS-encoded block (never to the ASM),
   reinitialized at every CADU. Both sequences have been validated bit for
   bit against the reference sequences (first 40 bits) the standard
   provides.
4. **ASM** (4 bytes, `0x1ACFFC1D`) prepended to the (randomized or not)
   block of each CADU -- this is literally what CCSDS calls a CADU.
5. **Convolutional** K=7, CCSDS polynomials 171/133 octal, rate 1/2 (base)
   or punctured to 2/3, 3/4, 5/6, 7/8 (Table 3-1); G2 inversion applies only
   at rate 1/2 (punctured rates use no inversion, per 3.4.1). The encoder
   runs continuously over the whole stream of concatenated CADUs (ASM
   included), with the shift register cleared only once at the start of the
   entire signal (not per-CADU). The ASM is therefore convolutionally
   encoded along with the rest: for a convolutional stream, receiver-side
   frame sync is obtained by correlating the ASM in the **already-decoded**
   bitstream (continuous Viterbi decoding, no sync needed beforehand), not
   before decoding.
6. **NRZ-L**: direct bipolar mapping (bit 1 -> +1, bit 0 -> -1), no
   differential coding.
7. **QPSK Gray**: pairs of bipolar samples -> I/Q, normalized to unit
   average energy per symbol.
8. **RRC pulse shaping** (configurable alpha, default 0.35).
9. **Normalization**: the final signal is scaled to a configurable
   normalized peak amplitude (default 0.9 on a [-1,+1] scale), to leave
   headroom and prevent saturation/clipping during RF playback.
10. **Optional resampling**: if requested (CLI `--target-fs` parameter, or
    "Resample to fixed rate" in the GUI), the signal is resampled
    (`scipy.signal.resample_poly`, exact integer ratio) to a specific sample
    rate accepted by the playback instrument, independent of the native
    `symbol_rate x samples/symbol` frequency used internally by the chain.
11. **Output**: raw interleaved IQ file (`I0,Q0,I1,Q1,...`, no header), in
    `float32` (default, range [-1,+1]) or `int16` (selectable; RF-Catcher/
    TestTree format: little-endian, 12 significant bits in two's complement,
    LSB-aligned, range [-2048, 2047]), with a companion `.meta.json` file
    containing the parameters used, the exported file's actual sample
    rate/format, and a note that the file is baseband (no carrier frequency
    information: this must be set manually on the playback instrument, e.g.
    RF-Catcher's TX Freq field).

**Constant-memory generation (`export_chain`)**: both the CLI and the GUI's
"Generate export file" button use `ccsds_chain.pipeline.export_chain()`,
which processes CADUs in batches (sized to stay around ~64 MB of native IQ
per batch) instead of building the entire bit/coded-bit/symbol/IQ arrays in
RAM for the whole export, as `run_chain()` does (used only for the live
preview, deliberately capped to a small number of CADUs). The convolutional
encoder and the RRC filter carry their state across batches, so the result
is identical (byte-for-byte for `int16`; for `float32` it can differ in the
last mantissa bit due to float64 rounding noise, ~9 orders of magnitude
below the int16 quantization step) to a single non-batched run. Peak memory
therefore stays on the order of a single batch regardless of export
duration -- an export that used to exhaust RAM (killed by the OOM killer)
now uses a negligible amount. Optional resampling (`--target-fs`/"Resample
to fixed rate") remains the only non-batched stage (it needs the whole
signal in memory for `scipy.signal.resample_poly`): an export too large
combined with resampling is refused immediately with a clear message,
before generation starts, instead of failing partway through a long run.
In the GUI, the exported file is written to disk (`output/` folder, same as
the CLI) and also offered as a browser download only if it stays under
1 GB -- above that threshold it remains available at the path shown on the
page, because Streamlit's download button still has to load the entire file
into memory to serve it.

## Usage

```bash
pip install -r requirements.txt
```

### GUI (recommended)

```bash
streamlit run app.py
```

Opens a graphical interface (dark theme) with all chain parameters in the
sidebar. **Every parameter change recomputes the chain and updates in real
time**: spectrum (PSD with the -3dB/null-to-null bandwidth highlighted),
QPSK constellation, I/Q excerpt over time, and metrics (occupied bandwidth,
signal duration, compute time). Includes a visual indicator of active/
inactive pipeline stages and direct export of the IQ file + metadata from
the browser ("Download" buttons).

`app.py` and `generate_signal.py` share the same chain implementation
(`ccsds_chain/pipeline.py`): there is no risk of the GUI and CLI drifting
apart.

### CLI

```bash
python generate_signal.py                       # baseline parameters, saves to output/qpsk_ccsds_*.iq
python generate_signal.py --n-cadu 500 --randomizer long -o test.raw
python generate_signal.py --rs-e 8 --interleave-depth 2 --conv-rate 3/4 -o test2.raw
python generate_signal.py --dtype int16 --target-fs 10e6 --peak 0.9   # for an RF Recorder/Replayer

python verify_spectrum.py output/qpsk_ccsds_....iq --plot spectrum.png
```

Baseline parameters are constants at the top of `generate_signal.py`; the
most common options are also exposed via the CLI (`--help`), including
output format ones (`--dtype`, `--peak`, `--target-fs`). If `-o`/`--output`
is not given, the file is saved to `output/` as
`qpsk_ccsds_<fs>Msps_<n_cadu>cadu_<timestamp>.iq`.

## Limitations

- **Turbo coding and LDPC** (sections 6, 7, 8 of the standard) are not
  implemented: only Reed-Solomon, convolutional (with puncturing) and their
  concatenation.
- **Transfer Frame lengths** (section 11): the standard constrains the
  Transfer Frame lengths allowed for each coding scheme (to guarantee
  compatibility with the codeblock length, via "virtual fill" if needed).
  This is not implemented: combinations of E / interleave depth /
  convolutional rate / CADU count that produce an odd number of coded bits
  fail (a visible error in the GUI/CLI, not a crash) at the QPSK pairing
  step. In practice this only happens with punctured convolutional rates in
  specific combinations; the baseline (rate 1/2) is never affected.
- **NRZ-L convention**: bit 1 -> +1, bit 0 -> -1; to be checked against the
  polarity expected by the receiver/tool.
- **IQ file format**: raw interleaved (float32/int16, no header) follows
  the IQ format spec provided for the target Recorder/Replayer; it still
  needs to be confirmed with an end-to-end test on real hardware whether
  RF-Catcher (TestTree)'s "IQ Converter" tool requires a different
  `.rfcatcher` format (with its own header/metadata) instead of the raw
  binary produced here.
- **Payload**: currently pseudo-random test data (or raw Transfer Frames
  from a file); no real CCSDS Transfer Frame header is constructed (VCID,
  counters, CRC, etc.).
- **RRC filter transient**: since only a single RRC filter is applied (not
  a matched Tx/Rx pair), the output has a transient of `RRC_SPAN/2` symbols
  at the start and end.

## TODO

- [ ] Confirm the IQ Converter's expected input format (contact
      support@test-tree.com if needed)
- [ ] Verify the expected NRZ-L polarity
- [ ] Implement Turbo coding and LDPC (CCSDS 131.0-B-5 sections 6-8), if
      required by a specific receiver/test
- [ ] Implement Transfer Frame length constraints (section 11) and RS
      "virtual fill" (4.3.7-4.3.8), to avoid the QPSK parity error on
      non-standard parameter combinations
- [ ] End-to-end test: generation -> IQ Converter -> Capture & Playback ->
      RX loopback

## Spectrum verification

`verify_spectrum.py` computes the PSD (averaged Hanning-windowed
periodograms) and measures the occupied bandwidth at -3dB and the
null-to-null bandwidth (width of the main lobe between the first two nulls
measured on the spectrum) around the band center. With the baseline
parameters (Rs=1.785 MS/s, alpha=0.35), the expected -3dB occupied
bandwidth is close to Rs (~1.7-1.8 MHz), and the null-to-null bandwidth
close to Rs*(1+alpha) (~2.3-2.4 MHz, the exact theoretical value for an
ideal RRC filter).
