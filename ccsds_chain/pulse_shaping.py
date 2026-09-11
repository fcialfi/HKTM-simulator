"""Root-Raised-Cosine pulse shaping."""

import numpy as np


def rrc_taps(alpha: float, span: int, sps: int) -> np.ndarray:
    """Standard RRC impulse response, `span` symbols long, `sps` samples/symbol.
    Normalized to unit filter energy."""
    n = span * sps
    t = (np.arange(-n / 2, n / 2 + 1)) / sps  # in symbol periods
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + 4 * alpha / np.pi
        elif alpha != 0.0 and abs(abs(4 * alpha * ti) - 1.0) < 1e-8:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den

    return h / np.sqrt(np.sum(h ** 2))


def pulse_shape(symbols: np.ndarray, sps: int, taps: np.ndarray) -> np.ndarray:
    """Upsample by zero-stuffing and convolve with the RRC filter.
    Note: this introduces a filter group delay of span/2 symbols at the
    start and end of the output (transient ramp-up/down)."""
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, taps, mode="full")
