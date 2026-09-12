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


def null_to_null_bandwidth(symbol_rate: float, rrc_alpha: float) -> float:
    """Theoretical null-to-null bandwidth of an RRC-shaped spectrum:
    Rs * (1 + alpha), the exact edge of the (ideal, infinite-length) filter
    support -- same value to type as RF-Catcher's replay bandwidth field."""
    return symbol_rate * (1 + rrc_alpha)


def contiguous_bandwidth(freqs: np.ndarray, db: np.ndarray, threshold_db: float) -> float:
    """Width of the contiguous region around DC that stays above threshold_db."""
    center_idx = int(np.argmin(np.abs(freqs)))
    left, right = center_idx, center_idx
    while left > 0 and db[left - 1] >= threshold_db:
        left -= 1
    while right < len(db) - 1 and db[right + 1] >= threshold_db:
        right += 1
    return freqs[right] - freqs[left]
