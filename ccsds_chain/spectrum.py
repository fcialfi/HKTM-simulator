"""PSD estimation and occupied-bandwidth measurement, shared by
verify_spectrum.py (CLI) and app.py (GUI)."""

import numpy as np


def welch_psd(x: np.ndarray, nperseg: int = 4096) -> np.ndarray:
    """Average periodogram (Hanning window) PSD estimate, fftshifted."""
    window = np.hanning(nperseg)
    win_energy = np.sum(window ** 2)
    n_segs = len(x) // nperseg
    if n_segs < 1:
        raise ValueError("signal shorter than one FFT segment; reduce nperseg or generate more CADU")
    acc = np.zeros(nperseg)
    for i in range(n_segs):
        seg = x[i * nperseg:(i + 1) * nperseg] * window
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc += np.abs(spec) ** 2
    acc /= (n_segs * win_energy)
    return acc


def contiguous_bandwidth(freqs: np.ndarray, db: np.ndarray, threshold_db: float,
                          min_consecutive_below: int = 3) -> float:
    """Width of the contiguous region around DC that stays above threshold_db.

    The Welch PSD estimate is noisy, so a single bin dipping below
    threshold near the passband edge (statistical ripple, not the real
    roll-off) would otherwise stop the scan early and report a bogus
    near-zero bandwidth. A dip only ends the scan once it persists for
    `min_consecutive_below` bins in a row; shorter dips are skipped over.
    """
    center_idx = int(np.argmin(np.abs(freqs)))

    def scan(step: int) -> int:
        idx = center_idx
        last_above = center_idx
        below_run = 0
        while 0 <= idx + step < len(db):
            idx += step
            if db[idx] >= threshold_db:
                below_run = 0
                last_above = idx
            else:
                below_run += 1
                if below_run >= min_consecutive_below:
                    break
        return last_above

    left = scan(-1)
    right = scan(1)
    return freqs[right] - freqs[left]
