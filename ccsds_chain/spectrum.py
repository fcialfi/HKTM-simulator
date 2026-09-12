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


def null_to_null_bandwidth(freqs: np.ndarray, db: np.ndarray, entry_threshold_db: float = -20.0,
                            min_consecutive_below: int = 5, min_consecutive_rise: int = 5) -> float:
    """Width of the main lobe between its first null on each side of DC.

    For an (infinite) RRC-shaped signal the main lobe touches zero power
    exactly at +/-symbol_rate*(1+alpha)/2, so null-to-null bandwidth equals
    the theoretical occupied bandwidth symbol_rate*(1+alpha) -- unlike the
    -3dB point, which stays near +/-symbol_rate/2 regardless of roll-off.
    With a finite (truncated) filter the null becomes a deep but non-zero
    dip, and the noisy Welch PSD estimate has its own ripple on top of
    that, so finding it takes two passes per side: first walk out past
    the noisy passband until the PSD drops persistently below
    `entry_threshold_db` (same robust-crossing trick as
    `contiguous_bandwidth`), then track the running minimum until it rises
    persistently for `min_consecutive_rise` bins -- that running minimum
    is the null.
    """
    center_idx = int(np.argmin(np.abs(freqs)))

    def scan(step: int) -> int:
        idx = center_idx
        below_run = 0
        while 0 <= idx + step < len(db):
            idx += step
            if db[idx] < entry_threshold_db:
                below_run += 1
                if below_run >= min_consecutive_below:
                    break
            else:
                below_run = 0

        best_idx, best_val, rise_run = idx, db[idx], 0
        while 0 <= idx + step < len(db):
            idx += step
            if db[idx] <= best_val:
                best_val, best_idx, rise_run = db[idx], idx, 0
            else:
                rise_run += 1
                if rise_run >= min_consecutive_rise:
                    break
        return best_idx

    left = scan(-1)
    right = scan(1)
    return freqs[right] - freqs[left]
