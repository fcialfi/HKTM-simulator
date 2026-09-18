"""Root-Raised-Cosine pulse shaping."""

from typing import Callable, Optional

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


_CHUNK_SYMBOLS = 200_000


def pulse_shape(
    symbols: np.ndarray,
    sps: int,
    taps: np.ndarray,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> np.ndarray:
    """Upsample by zero-stuffing and convolve with the RRC filter.
    Note: this introduces a filter group delay of span/2 symbols at the
    start and end of the output (transient ramp-up/down).

    Processed in symbol chunks via overlap-add (each chunk's zero-stuffed,
    convolved output tail overlaps the next chunk's head by len(taps)-1
    samples) so `progress_callback(fraction)` -- if given -- can report
    incremental progress on a long export; the result is bit-for-bit
    identical to a single `np.convolve` over the whole signal, since
    convolution is linear over concatenated input.
    """
    n = len(symbols)
    tail = len(taps) - 1
    out = np.zeros(n * sps + tail, dtype=complex)

    pos = 0
    for start in range(0, n, _CHUNK_SYMBOLS):
        end = min(start + _CHUNK_SYMBOLS, n)
        block = symbols[start:end]
        upsampled = np.zeros(len(block) * sps, dtype=complex)
        upsampled[::sps] = block
        conv = np.convolve(upsampled, taps, mode="full")
        out[pos:pos + len(conv)] += conv
        pos += len(block) * sps
        if progress_callback is not None:
            progress_callback(end / n)

    return out
