# HKTM-simulator

Test RF signal generator, modulated per the CCSDS chain (baseline QPSK), to
be injected via the **RF-Catcher (TestTree) Capture & Playback Application**.

References: CCSDS 131.0-B-5 (TM Synchronization and Channel Coding, Sept. 2023),
ECSS-E-ST-50-01C.

## Project structure

```
app.py                  GUI (Streamlit) with real-time spectrum/constellation
generate_signal.py      CLI, generates output_iq.raw + metadata
verify_spectrum.py      CLI, checks the occupied bandwidth of an existing IQ file
analyze_recording.py    CLI, analyzes a real recorded downlink (.rfcatcher or raw IQ)
ccsds_chain/
  pipeline.py            chain orchestration (used by app.py and generate_signal.py)
  reed_solomon.py         RS(255,223)/(255,239), CCSDS-native GF(256) and dual-basis
  convolutional.py        convolutional K=7, rate 1/2 punctured to 2/3-7/8
  scrambler.py             CCSDS pseudo-randomizer (131071-bit and 255-bit legacy)
  mapping.py               NRZ-L + QPSK Gray / BPSK
  pulse_shaping.py         RRC filter
  spectrum.py              PSD/occupied bandwidth (used by app.py and verify_spectrum.py)
  utils.py                 bit/byte helpers, IQ file I/O
  viterbi.py               Viterbi decoder for the K=7 convolutional code (self-verification)
  loopback.py              full digital-domain encode/decode loopback (self-verification)
```

## Baseline parameters

| Parameter       | Baseline value          | Selectable |
|-----------------|---------------------------|---------------|
| Modulation     | QPSK                       | yes (QPSK or BPSK) |
| Bit rate        | 3,570 kbps (post-coding, header included) | yes (or symbol rate, kept in sync; bit rate = symbol rate x bits/symbol, 2 for QPSK, 1 for BPSK) |
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
        -> NRZ-L -> QPSK (Gray) or BPSK -> RRC -> IQ int16 (RF-Catcher)
```

1. **Payload**: reproducible pseudo-random data (seed) per CADU, or real data
   from a file (`--payload-source`). The format of this data is selectable
   via `--input-format` (or, in the GUI, "Payload contains"):
   - `transfer_frame` (default): raw, uncoded Transfer Frames. RS, the
     pseudo-randomizer and the ASM are all applied here to build the CADUs.
   - `asm_frame`: ASM + Transfer Frame, **not scrambled** -- what a ground
     station that does its own frame sync hands back (e.g. Cortex, once its
     per-record header/trailer is stripped). Confirmed against a real
     capture, whose payload contained a plainly readable ASCII string (a
     firmware version tag) that genuinely scrambled bytes could never
     produce. RS and the ASM are **not** re-applied; the frame after each
     ASM is scrambled here with `--randomizer` (never the ASM), which
     defaults to `short` (legacy 255-bit) in this mode and cannot be
     `none` -- otherwise a real, CCSDS-conformant receiver's descrambler
     would corrupt every frame.
   - `cadu`: CADUs exactly as on the air right before convolutional coding
     (ASM + already-scrambled block, if the link scrambles). Used verbatim:
     no RS, no scrambling (`--randomizer` must be `none`), no new ASM.

     In both ASM-framed modes only the convolutional stage (if enabled) is
     still applied over the record stream, exactly as a physical coder
     downstream of an already-formed CADU stream would. Each record is
     always exactly `4 + 255*I` bytes -- **1279** at the default I=5 --
     and anything past that before the next ASM is receiver-added overhead
     and is skipped (see below).

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

     With `transfer_frame` input and synthetic (pseudo-random) payload,
     `--vcid-list` (or, in the GUI, "Assign Virtual Channels") builds a
     real 6-octet CCSDS TM primary header (132.0-B-3 "TM Space Data Link
     Protocol" 4.1.2 -- a different part of the standard from the channel
     coding the rest of this tool implements) into each generated frame,
     carrying a Virtual Channel ID from a comma-separated list, round-
     robined across frames (e.g. `0,0,1,2` gives Virtual Channel 0 twice
     any other VC's share). This is for validating that a receiver
     correctly *identifies and routes* frames to each Virtual Channel --
     something a single undifferentiated stream of payload bytes can't
     exercise. Only the primary header (Spacecraft ID, VCID, Master/
     Virtual Channel Frame Count, an "Idle Data" Data Field Status) is
     built; there's no real Space Packet structure or secondary header
     inside the frame, and it's rejected outright when combined with a
     real uploaded Transfer Frame file (it would overwrite the first 6
     bytes of real data with a synthetic header) -- see
     `ccsds_chain/transfer_frame.py`.

     `--corrupt-rs-symbols` (or, in the GUI, "Inject controlled RS symbol
     errors") deterministically flips an exact number of RS symbols within
     one interleaved codeword (`--corrupt-codeword-index`) of chosen CADUs
     (`--corrupt-cadu-indices` and/or `--corrupt-vc`, the latter requiring
     `--vcid-list`). This is complementary to a replayer's AWGN: a real
     noise sweep gives a proper statistical BER/FER-vs-Eb/N0 curve, but
     can't guarantee hitting a precise per-codeword error count, which
     makes boundary-testing a receiver's RS decoder against its declared
     correction capability E impractical that way -- exactly E symbol
     errors in one codeword must still decode perfectly, E+1 must fail (or
     be flagged), never silently miscorrect. Combined with `--vcid-list`
     and `--corrupt-vc`, it also validates that a receiver's per-Virtual-
     Channel FER accounting attributes injected errors to the right
     channel and leaves the others untouched. Requires an actual RS-coded
     region to corrupt (`--no-rs` not set, or `--input-format asm_frame`/`cadu`,
     already RS-coded by construction).
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
7. **Symbol mapping** (`--modulation`, default QPSK): **QPSK Gray** pairs
   consecutive bipolar samples into I/Q, normalized to unit average energy
   per symbol; **BPSK** instead carries one bipolar sample per symbol on I
   only (Q=0), at the same unit energy -- half the bit rate of QPSK at the
   same symbol rate, but the modulation the base rate-1/2 convolutional
   code's G2 inversion (3.3.1(5)) is specified against.
8. **RRC pulse shaping** (configurable alpha, default 0.35).
9. **Transmitter impairments** (optional, disabled by default -- "Transmitter
   Impairments" in the GUI, or the flags named below on the CLI), applied to
   the pulse-shaped IQ (`ccsds_chain/impairments.py`): a real transmitter's
   own non-idealities, distinct from a replayer's AWGN injection (which
   characterizes a receiver's sensitivity vs Eb/N0, not its tolerance to an
   imperfect transmitter). Grouped by which physical subsystem each comes
   from, applied in that order (frequency offset -> phase noise -> IQ
   imbalance -> PA nonlinearity), matching where each actually originates
   along a real signal path:
   - **LO/synthesizer** (`--freq-offset-hz` / `--phase-noise-linewidth-hz`):
     frequency offset is a constant residual LO error (Hz), for validating
     carrier-recovery acquisition/tracking against a real, not perfectly
     on-frequency, signal. Phase noise models a free-running oscillator's
     single-sideband linewidth (Hz) as a Wiener (random-walk) phase process,
     for validating tolerance to constellation smearing from a real
     transmitter LO.
   - **IQ modulator** (`--iq-gain-imbalance-db` / `--iq-phase-imbalance-deg`):
     models the modulator's I/Q branch mismatch as the standard
     `s' = A*s + B*conj(s)` mirror-image transform, producing a mirror tone
     at an image rejection ratio of `20*log10(|A|/|B|)` -- checked in
     `tests/test_impairments.py` against a synthetic tone's own FFT. Only
     becomes visible in the spectrum once a frequency offset has displaced
     the wanted signal away from 0 Hz first (this chain's 0 Hz *is* the
     transmitter's RF center frequency); with no offset, a symmetric random-
     data QPSK spectrum's own mirror folds invisibly back onto itself.
   - **Power amplifier** (`--pa-backoff-db` / `--pa-smoothness` /
     `--pa-am-pm-deg-per-db`, the last physical stage before the antenna):
     Rapp AM-AM saturation/compression model, with optional AM-PM
     conversion (degrees of phase shift per dB of compression, the same
     figure real TWTA/SSPA datasheets quote). `pa_backoff_db` is how far
     above this project's reference unit amplitude (1.0, an ideal symbol's
     own magnitude) the PA saturates -- unlike a pure phase rotation or a
     linear image term, this is a genuine nonlinearity, so it generates
     *spectral regrowth* (odd-order intermodulation) visible just outside
     the occupied bandwidth, the same adjacent-channel-power signature a
     real PA's compression produces.

   All are applied deterministically (LO impairments seeded via
   `--impairment-seed`) and carry their state correctly across
   `export_chain()`'s batches, exactly like every other stage in this chain.
   The GUI's live "QPSK constellation" plot reflects all of them: it samples
   the actual (possibly impaired) IQ through a matched RRC filter
   (`pulse_shaping.matched_filter_sample()`), not the ideal pre-pulse-shaping
   symbols -- e.g. a nonzero phase noise linewidth visibly spreads the 4
   QPSK points into a ring (constant-magnitude phase rotation), rather than
   always showing a perfect, unaffected constellation.
9b. **Channel Doppler** (optional, disabled by default -- `--doppler-hz` /
   `--doppler-rate-hz-s` on the CLI), applied last of all, after every
   transmitter impairment above: unlike those, this is not a transmitter
   non-ideality but a propagation effect, from relative motion between
   spacecraft and ground station during a real pass. `doppler_hz` is the
   instantaneous shift at the start of the exported signal; `doppler_rate_hz_s`
   is its constant rate of change (Hz/s) -- together a local-linear (chirp)
   approximation of a real LEO pass's Doppler curve, for validating a
   receiver's carrier-tracking loop against realistic pass dynamics rather
   than just a static frequency error. Implemented as a quadratic phase ramp
   (`impairments.apply_doppler()`), exact and reproducible across
   `export_chain()`'s batches like `--freq-offset-hz`.
10. **Normalization**: the final signal is scaled to a configurable
    normalized peak amplitude (default 0.9 on a [-1,+1] scale), to leave
    headroom and prevent saturation/clipping during RF playback.
11. **Optional resampling**: if requested (CLI `--target-fs` parameter, or
    "Resample to fixed rate" in the GUI), the signal is resampled
    (`scipy.signal.resample_poly`, exact integer ratio) to a specific sample
    rate accepted by the playback instrument, independent of the native
    `symbol_rate x samples/symbol` frequency used internally by the chain.
12. **Output**: raw interleaved IQ file (`I0,Q0,I1,Q1,...`, no header), in
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
constellation, I/Q excerpt over time, and metrics (occupied bandwidth,
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
python generate_signal.py --modulation BPSK -o test3.raw
python generate_signal.py --dtype int16 --target-fs 10e6 --peak 0.9   # for an RF Recorder/Replayer

# 3 Virtual Channels, VC 0 getting twice VC 1/2's share of frames -- for
# validating a receiver's VC identification/routing
python generate_signal.py --vcid-list 0,0,1,2 --spacecraft-id 291 -o test4.raw

# Exactly E=16 symbol errors in codeword 0 of CADU #10: must still decode
# perfectly. Bump to 17 to confirm the receiver correctly flags it instead.
python generate_signal.py --corrupt-rs-symbols 16 --corrupt-cadu-indices 10 -o test5.raw

# Corrupt every VC1 frame (17 errors, beyond E=16) to check the receiver's
# per-channel FER report blames VC1 alone, leaving VC0/VC2 clean
python generate_signal.py --vcid-list 0,1,2 --corrupt-rs-symbols 17 --corrupt-vc 1 -o test6.raw

# A real transmitter is never perfect: 5 kHz LO offset, 0.8 dB/3deg IQ
# imbalance, 200 Hz phase noise linewidth -- validates carrier recovery,
# image rejection and EVM tolerance against a real, not idealized, signal
python generate_signal.py --freq-offset-hz 5000 --iq-gain-imbalance-db 0.8 \
    --iq-phase-imbalance-deg 3 --phase-noise-linewidth-hz 200 -o test7.raw

# PA driven 6 dB into saturation, with 3 deg/dB AM-PM -- check the receiver
# tolerates the resulting spectral regrowth/EVM degradation
python generate_signal.py --pa-backoff-db -6 --pa-smoothness 3 --pa-am-pm-deg-per-db 3 -o test8.raw

# A real LEO pass: 30 kHz Doppler shift at export start, closing at -400 Hz/s
# -- validates carrier tracking against realistic pass dynamics, not just a
# static frequency error (this is a channel effect, not a transmitter one --
# combine freely with the transmitter impairments above)
python generate_signal.py --doppler-hz 30000 --doppler-rate-hz-s -400 -o test9.raw

python verify_spectrum.py output/qpsk_ccsds_....iq --plot spectrum.png
```

Baseline parameters are constants at the top of `generate_signal.py`; the
most common options are also exposed via the CLI (`--help`), including
output format ones (`--dtype`, `--peak`, `--target-fs`). If `-o`/`--output`
is not given, the file is saved to `output/` as
`qpsk_ccsds_<fs>Msps_<n_cadu>cadu_<timestamp>.iq`.

### Analyzing a real recording

`analyze_recording.py` characterizes a real recorded pass (RF-Catcher
`.rfcatcher` files are tar archives around an int16 IQ member and are
unwrapped automatically; raw IQ works too), so a simulated signal can be
matched to it or a receiver problem traced back to the signal:

```bash
python analyze_recording.py C:\RF-Catcher\AWS_2_split.rfcatcher --fs 10e6 \
    --offset 30 --duration 1 --plot aws2.png
```

It reports the symbol rate (and its ppm offset from `--rs-nominal`), the
carrier offset and drift, how many discrete spectral lines sit on the
signal, Es/N0, IQ imbalance and residual phase noise, then Viterbi-decodes
the symbols to find the CADU length, the pseudo-randomizer in use
(none/short/long, checked against the TM primary header), SCID/VCIDs, the
share of idle frames, and how repetitive the on-air content is. The PNG
shows the spectrum, the constellation after carrier recovery and the
residual carrier phase. Only `--duration` seconds from `--offset` are read
(memory-mapped), so multi-GB recordings are fine; decoding is pure Python
and takes about a minute per million symbols (`--no-decode` to skip it).

### Tests

```bash
pip install -r requirements.txt pytest
pytest
```

`tests/` covers the coding chain's own claimed invariants -- RS codeword
divisibility by every root required by CCSDS 131.0-B-5 4.3.4, dual-basis
transform invertibility, PN sequence periodicity, and bit-for-bit
equivalence between the batched/streaming `export_chain()` path and the
whole-array `run_chain()` reference for both plain and resampled output --
rather than hardcoded external test vectors, so a refactor that silently
breaks one of those properties fails CI instead of only showing up as a
subtly wrong spectrum. Runs in a few seconds; also run automatically on
every push/PR (`.github/workflows/tests.yml`).

### Self-verification decoder (loopback)

`ccsds_chain/viterbi.py` (Viterbi decoder for the K=7 convolutional code)
and `ccsds_chain/reed_solomon.py`'s `rs_decode_codeword`/
`rs_decode_interleaved` (Berlekamp-Massey/Chien-search/Forney) close the
loop on the two parts of the chain hand-derived from the CCSDS spec: given
what this project's own encoder produced -- including corrupted, so an
actual *correction* has to happen, not just a pass-through -- can this
project's own decoder recover the exact original data?
`ccsds_chain/loopback.py` ties both together (RS decode + ASM check +
descramble + Viterbi decode) into a single `decode_transfer_frame_stream()`
call, exercised end-to-end by `tests/test_loopback.py`.

This is scoped to the digital/bit domain only, not the actual generated IQ
waveform -- recovering symbols from real IQ needs a matched filter and
symbol-timing recovery, a separate, larger piece of receiver DSP this
project doesn't implement (it's a signal *generator*, per the top of this
file). What it validates instead is the part of the chain most likely to
hide a subtle spec-transcription bug: encoding alone can produce a
codeword divisible by the right roots (algebraically well-formed) without
ever proving that a *real* error in it is actually correctable, which only
running the decoder for real, against real injected errors, can show.

## Limitations

- **Turbo coding and LDPC** (sections 6, 7, 8 of the standard) are not
  implemented: only Reed-Solomon, convolutional (with puncturing) and their
  concatenation.
- **Transfer Frame lengths** (section 11): the standard constrains the
  Transfer Frame lengths allowed for each coding scheme (to guarantee
  compatibility with the codeblock length, via "virtual fill" if needed).
  This is not implemented: with QPSK, combinations of E / interleave depth /
  convolutional rate / CADU count that produce an odd number of coded bits
  fail (a visible error in the GUI/CLI, not a crash) at the QPSK pairing
  step. In practice this only happens with punctured convolutional rates in
  specific combinations; the baseline (rate 1/2) is never affected. BPSK
  (1 bit/symbol) is never affected by this at any rate, since it never
  pairs bits into a symbol.
- **NRZ-L convention**: bit 1 -> +1, bit 0 -> -1; to be checked against the
  polarity expected by the receiver/tool.
- **IQ file format**: raw interleaved (float32/int16, no header) follows
  the IQ format spec provided for the target Recorder/Replayer; it still
  needs to be confirmed with an end-to-end test on real hardware whether
  RF-Catcher (TestTree)'s "IQ Converter" tool requires a different
  `.rfcatcher` format (with its own header/metadata) instead of the raw
  binary produced here.
- **Payload**: currently pseudo-random test data (or raw Transfer Frames
  from a file). A real primary header (Spacecraft ID, VCID, Master/Virtual
  Channel Frame Count) can be built in with `--vcid-list` (see above), but
  only that: no secondary header, no real CCSDS Space Packet structure
  inside the frame, and no CRC.
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
