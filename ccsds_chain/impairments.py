"""Typical RF transmitter impairments, applied to the pulse-shaped IQ
signal -- the analog transmit chain's own imperfections, distinct from the
channel/receiver-side effects a replayer's AWGN injection covers. Real
Eb/N0 sweeps (amplitude scaling + noise, done by the replayer) validate a
receiver's sensitivity; these validate its tolerance to a real
transmitter's non-idealities: a residual LO frequency/phase error, an
imperfect IQ modulator, and oscillator phase noise.

Applied (see pipeline._apply_impairments()) as: frequency offset -> phase
noise -> IQ imbalance last. This chain's 0 Hz *is* the transmitter's
intended RF center frequency (there is no separate software upconversion
stage), and apply_iq_imbalance()'s mirror-image term reflects about that
0 Hz -- so it only produces a visible, separate image in the spectrum once
frequency offset has already displaced the wanted signal away from 0 Hz;
applied first, to a signal still symmetric about 0 Hz (as this project's
random-data QPSK is, with no offset), the mirror folds invisibly back onto
the same band. Frequency offset and phase noise commute with each other
(both are pure phase rotations), so their relative order doesn't matter.
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
