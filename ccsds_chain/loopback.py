"""Self-verification loopback: decode what the encoding chain
(reed_solomon.py + convolutional.py, as pipeline.py's _cadu_bits()/
run_chain() drive them for `input_format="transfer_frame"`) actually
produces, back to the original Transfer Frame payload, entirely in the
digital/bit domain.

Deliberately does *not* go through the analog side (pulse_shaping.py's RRC
filter, symbol mapping) -- recovering symbols from the actual IQ waveform
needs a matched filter and symbol-timing recovery, a separate, larger piece
of receiver DSP this project doesn't implement (it is a signal *generator*,
per README.md). What this module closes the loop on instead is the part of
the chain that is hand-derived from the CCSDS spec and therefore the most
likely to hide a subtle bug (CCSDS-native GF(256) Reed-Solomon, the
punctured K=7 convolutional code): given a bitstream this project's own
encoder produced, can this project's own decoder recover the exact original
payload -- including when the encoded bits have been corrupted first.
"""

import numpy as np

from .convolutional import conv_encode
from .reed_solomon import rs_decode_interleaved, rs_encode_interleaved
from .scrambler import pn_sequence
from .utils import bits_to_bytes, bytes_to_bits
from .viterbi import viterbi_decode


def encode_transfer_frame_stream(
    payload: bytes, asm: bytes, rs_k: int, rs_n: int, interleave_depth: int,
    fec_rs: bool, randomizer: str, conv_rate: str, conv_invert_g2: bool,
) -> np.ndarray:
    """Re-implements pipeline.py's per-CADU bit assembly (RS encode ->
    pseudo-randomize -> attach ASM -> convolutional-encode the CADU stream)
    directly from the public primitives, for as many whole CADUs as `payload`
    holds. Kept independent of pipeline.py's own (private) _cadu_bits()/
    _prepare_payload() so this module can validate the encoding primitives
    without depending on -- or being invalidated by -- unrelated changes to
    pipeline.py's orchestration/streaming/CADU-sync logic, which has its own
    dedicated tests in test_pipeline.py.
    """
    frame_bytes = rs_k * interleave_depth
    n_cadu = len(payload) // frame_bytes
    if n_cadu * frame_bytes != len(payload):
        raise ValueError(f"payload length {len(payload)} is not a whole multiple of {frame_bytes} bytes/CADU")

    asm_bits = bytes_to_bits(asm)
    rs_block_bytes = rs_n * interleave_depth if fec_rs else frame_bytes
    pn = pn_sequence(rs_block_bytes * 8, randomizer) if randomizer != "none" else None

    chunks = []
    for i in range(n_cadu):
        frame = payload[i * frame_bytes:(i + 1) * frame_bytes]
        rs_block = rs_encode_interleaved(frame, rs_k, rs_n, interleave_depth) if fec_rs else frame
        rs_bits = bytes_to_bits(rs_block)
        if pn is not None:
            rs_bits = np.bitwise_xor(rs_bits, pn)
        chunks.append(np.concatenate([asm_bits, rs_bits]))

    bits = np.concatenate(chunks)
    return conv_encode(bits, invert_g2=conv_invert_g2, rate=conv_rate)


def decode_transfer_frame_stream(
    coded_bits: np.ndarray, n_cadu: int, asm: bytes, rs_k: int, rs_n: int,
    interleave_depth: int, fec_rs: bool, randomizer: str, conv_rate: str,
    conv_invert_g2: bool,
) -> bytes:
    """Inverse of encode_transfer_frame_stream(): Viterbi-decodes the
    convolutional stage, then for each of `n_cadu` CADUs strips and checks
    the ASM, de-scrambles, and RS-decodes/corrects -- returning the
    recovered Transfer Frame payload (n_cadu * rs_k * interleave_depth
    bytes). Raises ValueError if a CADU's ASM doesn't come back intact
    (Viterbi decode failed to converge on the correct path) or if a CADU's
    RS codeword is uncorrectable.
    """
    frame_bytes = rs_k * interleave_depth
    rs_block_bytes = rs_n * interleave_depth if fec_rs else frame_bytes
    cadu_bits = len(asm) * 8 + rs_block_bytes * 8

    decoded_bits = viterbi_decode(coded_bits, n_cadu * cadu_bits, rate=conv_rate, invert_g2=conv_invert_g2)

    asm_bits_expected = bytes_to_bits(asm)
    pn = pn_sequence(rs_block_bytes * 8, randomizer) if randomizer != "none" else None

    frames = []
    for i in range(n_cadu):
        cadu = decoded_bits[i * cadu_bits:(i + 1) * cadu_bits]
        asm_bits_recovered, rs_bits = cadu[:len(asm) * 8], cadu[len(asm) * 8:]
        if not np.array_equal(asm_bits_recovered, asm_bits_expected):
            raise ValueError(f"CADU {i}: ASM did not decode intact -- convolutional decode diverged")
        if pn is not None:
            rs_bits = np.bitwise_xor(rs_bits, pn)
        rs_block = bits_to_bytes(rs_bits)
        frame = rs_decode_interleaved(rs_block, rs_k, rs_n, interleave_depth) if fec_rs else rs_block
        frames.append(frame)

    return b"".join(frames)
