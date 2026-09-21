"""Typical RF transmitter impairments, applied to the pulse-shaped IQ
signal -- the analog transmit chain's own imperfections, distinct from the
channel/receiver-side effects a replayer's AWGN injection covers. Real
Eb/N0 sweeps (amplitude scaling + noise, done by the replayer) validate a
receiver's sensitivity; these validate its tolerance to a real
transmitter's non-idealities, from three different physical subsystems of
the chain:

- LO/synthesizer: a residual frequency/phase error (apply_frequency_offset)
  and phase noise (PhaseNoiseGenerator).
- IQ modulator: gain/phase imbalance (apply_iq_imbalance).
- Power amplifier: saturation/compression and AM-PM conversion
  (apply_pa_nonlinearity) -- the last physical stage before the antenna.

Applied (see pipeline._apply_impairments()) in that order: frequency
offset -> phase noise -> IQ imbalance -> PA nonlinearity last, matching
where each originates along the real signal path. This chain's 0 Hz *is*
the transmitter's intended RF center frequency (there is no separate
software upconversion stage), and apply_iq_imbalance()'s mirror-image term
reflects about that 0 Hz -- so it only produces a visible, separate image
in the spectrum once frequency offset has already displaced the wanted
signal away from 0 Hz; applied first, to a signal still symmetric about
0 Hz (as this project's random-data QPSK is, with no offset), the mirror
folds invisibly back onto the same band. Frequency offset and phase noise
commute with each other (both are pure phase rotations), so their relative
order doesn't matter.
"""

import numpy as np


def apply_iq_imbalance(iq: np.ndarray, gain_imbalance_db: float, phase_imbalance_deg: float) -> np.ndarray:
    """Models an IQ modulator's gain and phase imbalance as a linear
    combination of the ideal signal and its own conjugate (the standard
    "mirror-image" impairment model): s' = A*s + B*conj(s), which for a
    single input tone produces a mirror-image tone at the negative
    frequency, at an amplitude ratio (the image rejection ratio, IRR)
    of |A|/|B| relative to the wanted tone -- see
    tests/test_impairments.py for a closed-form check of this against a
    synthetic tone's own FFT.

    `gain_imbalance_db`: I/Q branch gain mismatch, in dB (0 = none).
    `phase_imbalance_deg`: deviation from ideal 90 degree I/Q phase
    separation, in degrees (0 = none). Both 0 is an exact identity
    (A=1, B=0) -- callers skip calling this entirely in that case, so it
    never runs, but the identity holds either way.
    """
    g = 10 ** (gain_imbalance_db / 20.0)  # dB -> linear amplitude ratio
    phi = np.deg2rad(phase_imbalance_deg)
    a = (1 + g * np.cos(phi) + 1j * g * np.sin(phi)) / 2
    b = (1 - g * np.cos(phi) + 1j * g * np.sin(phi)) / 2
    return a * iq + b * np.conj(iq)


def apply_frequency_offset(iq: np.ndarray, freq_offset_hz: float, sample_rate: float, start_sample: int = 0) -> np.ndarray:
    """Rotates `iq` by a constant residual LO frequency offset: a pure
    per-sample phase ramp, computed directly from each sample's absolute
    position (`start_sample` + its index within `iq`) rather than
    accumulated incrementally -- so it is exact and reproducible
    regardless of how the signal is chunked across export_chain()'s
    batches (no float accumulation drift, no state to carry between
    calls)."""
    if freq_offset_hz == 0:
        return iq
    n = start_sample + np.arange(len(iq))
    phase = 2 * np.pi * freq_offset_hz * n / sample_rate
    return iq * np.exp(1j * phase)


class PhaseNoiseGenerator:
    """Stateful Wiener (random-walk) phase noise process, for a stream of
    IQ chunks (e.g. batched/streaming export): `apply()` on a sequence of
    chunks produces output bit-for-bit identical to one `apply()` call on
    the concatenation of those chunks, the phase random walk and its PRNG
    state both carried across calls -- the same cross-batch-consistency
    contract as `pulse_shaping.RRCPulseShaper`.

    Models a free-running oscillator with a given single-sideband 3 dB
    linewidth (`linewidth_hz`): the standard result for such an
    oscillator's phase is a Wiener process with per-sample variance
    `2*pi*linewidth_hz/sample_rate` (e.g. Demir, Mehrotra & Roychowdhury,
    "Phase Noise in Oscillators: A Unifying Theory..."), which this
    generates by cumulative-summing zero-mean Gaussian increments of that
    variance and applying exp(j*phase) to each sample.
    """

    def __init__(self, linewidth_hz: float, sample_rate: float, seed: int):
        self.linewidth_hz = linewidth_hz
        self.sample_rate = sample_rate
        self._rng = np.random.default_rng(seed)
        self._phase = 0.0

    def apply(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return iq
        step_std = np.sqrt(2 * np.pi * self.linewidth_hz / self.sample_rate)
        increments = self._rng.normal(0.0, step_std, size=len(iq))
        phase = self._phase + np.cumsum(increments)
        self._phase = phase[-1]
        return iq * np.exp(1j * phase)


def apply_pa_nonlinearity(iq: np.ndarray, backoff_db: float, smoothness: float, am_pm_deg_per_db: float = 0.0) -> np.ndarray:
    """Models a power amplifier's saturation/compression (AM-AM) and,
    optionally, its AM-PM conversion -- a memoryless nonlinearity applied
    to each sample's instantaneous envelope, the last physical stage
    before the antenna. Unlike the other impairments here (pure phase
    rotations or a linear image term), a real nonlinearity generates
    energy at harmonics of the signal's own spectral content; for a
    complex baseband/IQ representation (no physical carrier in software --
    see this module's own docstring) that shows up as *spectral regrowth*
    just outside the occupied bandwidth (odd-order intermodulation
    products folding back in-band/adjacent-band) rather than literal
    harmonics at multiples of an RF carrier this chain never actually
    synthesizes -- the same effect a spectrum analyzer's adjacent-channel
    power measurement is built to catch on a real PA. See
    tests/test_impairments.py for a check that this actually raises the
    signal's own out-of-band shoulders.

    AM-AM: the Rapp model (widely used for solid-state PAs), a smooth
    saturating curve from an exactly-linear small-signal region to an
    output envelope that asymptotically approaches `A_sat` however large
    the input gets:

        A_sat = 10^(backoff_db / 20)   (this project's reference unit
                                          amplitude is 1.0 -- an ideal
                                          QPSK/BPSK symbol's own magnitude,
                                          see mapping.py -- so backoff_db
                                          is how far above a nominal
                                          symbol's amplitude the PA
                                          saturates; the RRC-pulse-shaped
                                          envelope's own peaks routinely
                                          exceed that reference by a few
                                          dB, so a small backoff already
                                          produces visible compression)
        gain(r) = 1 / (1 + (r/A_sat)^(2p))^(1/(2p))   (p = `smoothness`;
                                                         higher = sharper
                                                         knee, lower =
                                                         softer)

    AM-PM (optional, 0 = disabled): a phase shift proportional to how much
    a sample is being compressed, in degrees per dB of AM-AM compression
    -- the same "deg/dB" figure real TWTA/SSPA datasheets quote.

    Stateless (a pure function of each sample's own instantaneous
    envelope): no cross-batch state to carry, unlike PhaseNoiseGenerator.
    """
    a_sat = 10 ** (backoff_db / 20.0)
    r = np.abs(iq)
    gain = 1.0 / (1 + (r / a_sat) ** (2 * smoothness)) ** (1 / (2 * smoothness))
    out = iq * gain
    if am_pm_deg_per_db != 0:
        compression_db = -20 * np.log10(gain)
        phase_shift = np.deg2rad(am_pm_deg_per_db) * compression_db
        out = out * np.exp(1j * phase_shift)
    return out
