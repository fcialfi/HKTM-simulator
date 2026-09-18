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


class RRCPulseShaper:
    """Stateful counterpart to `pulse_shape()`, for shaping a single
    continuous symbol stream a chunk at a time (e.g. batched/streaming
    export): carries the filter's overlap tail across `shape()` calls, the
    same way `pulse_shape()`'s internal chunk loop does, but usable across
    separate top-level calls. `shape()` on a sequence of chunks followed by
    one final `flush()` produces output bit-for-bit identical to a single
    `pulse_shape()` call on the concatenation of those chunks -- with peak
    memory bounded by the chunk size instead of the whole symbol stream.
    """

    def __init__(self, sps: int, taps: np.ndarray):
        self.sps = sps
        self.taps = taps
        self._tail = np.zeros(len(taps) - 1, dtype=complex)

    def shape(self, symbols: np.ndarray) -> np.ndarray:
        """Returns this chunk's `len(symbols) * sps` finalized output
        samples (any contribution still reachable by a future chunk is
        held back internally, to be combined with that chunk instead)."""
        if len(symbols) == 0:
            return np.zeros(0, dtype=complex)
        upsampled = np.zeros(len(symbols) * self.sps, dtype=complex)
        upsampled[::self.sps] = symbols
        conv = np.convolve(upsampled, self.taps, mode="full")  # len = n_out + len(_tail)
        n_out = len(symbols) * self.sps
        # Add the incoming tail (which may itself already carry contributions
        # forwarded from earlier chunks) before splitting off the new tail,
        # so a chunk shorter than the filter's tail still accumulates
        # correctly across more than one prior chunk.
        conv[:len(self._tail)] += self._tail
        out = conv[:n_out].copy()
        self._tail = conv[n_out:].copy()
        return out

    def flush(self) -> np.ndarray:
        """Call once after the last `shape()` call to get the filter's
        remaining tail (group-delay decay) -- matches the extra
        `len(taps) - 1` samples a single non-streaming `pulse_shape()` call
        appends at the end."""
        out = self._tail
        self._tail = np.zeros(len(self.taps) - 1, dtype=complex)
        return out
