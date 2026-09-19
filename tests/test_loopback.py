"""End-to-end self-verification loopback tests: generate a bitstream the way
pipeline.py's chain would (RS encode -> pseudo-randomize -> ASM -> punctured
convolutional encode), corrupt it, decode it back with viterbi.py +
reed_solomon.py's decoders, and confirm the exact original Transfer Frame
payload comes back out. This is the "genera -> decodifica -> confronta"
loop closing the digital side of the chain: the part hand-derived from the
CCSDS spec (custom GF(256) Reed-Solomon, punctured K=7 convolutional code)
and therefore most likely to hide a subtle bug that only shows up once
something actually has to be *corrected*, not just produced.

Kept to small n_cadu -- Viterbi decoding a whole burst is O(n_bits) with a
real per-step cost (64-state trellis), so this suite deliberately doesn't
decode CADU counts anywhere near a real export's size; that scale is
covered by run_chain()/export_chain()'s own tests, which never need to
decode anything.
"""

import numpy as np
import pytest

from ccsds_chain.loopback import decode_transfer_frame_stream, encode_transfer_frame_stream

ASM = bytes.fromhex("1ACFFC1D")

_CONFIGS = {
    "baseline": dict(rs_k=223, rs_n=255, interleave_depth=5, fec_rs=True,
                      randomizer="long", conv_rate="1/2", conv_invert_g2=True),
    "high_throughput": dict(rs_k=239, rs_n=255, interleave_depth=1, fec_rs=True,
                             randomizer="long", conv_rate="7/8", conv_invert_g2=False),
    "no_randomizer": dict(rs_k=223, rs_n=255, interleave_depth=8, fec_rs=True,
                           randomizer="none", conv_rate="1/2", conv_invert_g2=True),
    "legacy_randomizer": dict(rs_k=223, rs_n=255, interleave_depth=5, fec_rs=True,
                               randomizer="short", conv_rate="3/4", conv_invert_g2=False),
}

# Fixed, explicit integer seeds -- *not* Python's built-in hash() of a
# string, which is randomized per interpreter process (PYTHONHASHSEED)
# unless disabled, and so isn't actually reproducible across runs.
_SEED_BY_NAME = {name: i for i, name in enumerate(_CONFIGS)}


def _random_payload(cfg, n_cadu, seed):
    rng = np.random.default_rng(seed)
    frame_bytes = cfg["rs_k"] * cfg["interleave_depth"]
    return rng, bytes(int(b) for b in rng.integers(0, 256, size=frame_bytes * n_cadu, dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_clean_loopback_recovers_exact_payload(name):
    cfg = _CONFIGS[name]
    n_cadu = 2
    _, payload = _random_payload(cfg, n_cadu, seed=_SEED_BY_NAME[name])

    coded = encode_transfer_frame_stream(payload, ASM, **cfg)
    decoded = decode_transfer_frame_stream(coded, n_cadu, ASM, **cfg)
    assert decoded == payload


def test_recovers_exact_payload_after_bit_flips():
    cfg = _CONFIGS["baseline"]
    n_cadu = 2
    rng, payload = _random_payload(cfg, n_cadu, seed=123)

    coded = encode_transfer_frame_stream(payload, ASM, **cfg)
    corrupted = coded.copy()
    flip_positions = rng.choice(len(corrupted), size=3, replace=False)
    corrupted[flip_positions] ^= 1

    decoded = decode_transfer_frame_stream(corrupted, n_cadu, ASM, **cfg)
    assert decoded == payload


def test_wrong_cadu_count_is_rejected_rather_than_silently_misdecoded():
    # Decoding as though there were more CADUs than actually exist expects
    # more coded bits than the stream actually has -- must be reported
    # (depuncture()'s own length check), not silently produce garbage.
    cfg = _CONFIGS["baseline"]
    n_cadu = 2
    _, payload = _random_payload(cfg, n_cadu, seed=7)
    coded = encode_transfer_frame_stream(payload, ASM, **cfg)
    with pytest.raises(ValueError):
        decode_transfer_frame_stream(coded, n_cadu + 1, ASM, **cfg)
