"""RS(255,223) encoding with CCSDS-style byte interleaving.

NOTE (see README "Limitazioni"): this uses `reedsolo`'s default GF(256)
parameters (prim=0x11d, fcr=0, generator=2), i.e. a generic *conventional*
Reed-Solomon code. It is NOT verified to be bit-exact with the CCSDS
"conventional (Beta)" field/generator-polynomial convention used by some
ground equipment, and is further removed from the "dual-basis (Alpha)"
representation some receivers expect. See the README TODO list.
"""

import reedsolo

RS_FCR = 0
RS_PRIM = 0x11D
RS_GENERATOR = 2
RS_C_EXP = 8


def rs_encode_interleaved(data: bytes, k: int = 223, n: int = 255, depth: int = 5) -> bytes:
    """RS-encode `data` (must be exactly k*depth bytes) as `depth`
    interleaved RS(n,k) codewords, per CCSDS 131.0-B byte interleaving:
    substream j (j=0..depth-1) is data[j::depth], and the interleaved
    output byte at position i*depth+j is codeword_j[i].
    """
    if len(data) != k * depth:
        raise ValueError(f"expected {k * depth} bytes, got {len(data)}")

    parity = n - k
    rsc = reedsolo.RSCodec(parity, nsize=n, fcr=RS_FCR, prim=RS_PRIM,
                            generator=RS_GENERATOR, c_exp=RS_C_EXP)

    codewords = []
    for j in range(depth):
        substream = data[j::depth]
        codeword = bytes(rsc.encode(bytearray(substream)))
        if len(codeword) != n:
            raise RuntimeError(f"unexpected RS codeword length {len(codeword)} (expected {n})")
        codewords.append(codeword)

    out = bytearray(n * depth)
    for j, codeword in enumerate(codewords):
        out[j::depth] = codeword
    return bytes(out)
