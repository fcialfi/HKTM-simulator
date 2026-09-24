"""End-to-end check of analyze_recording.py on a signal built by this tool:
it must recover what went in (symbol rate, carrier offset, randomizer,
frame structure) and flag the idle-frame content as periodic."""

import numpy as np

import analyze_recording as ar
from ccsds_chain.pipeline import ASM, ChainParams, run_chain


def _idle_frames(n):
    word = (104 << 4) | (7 << 1)  # version 0, SCID 104, VCID 7 (idle), no OCF
    frames = []
    for i in range(n):
        header = word.to_bytes(2, "big") + bytes([i % 256, i % 256]) + (0x07FE).to_bytes(2, "big")
        frames.append(ASM + header + b"\x5a" * (1275 - len(header)))
    return b"".join(frames)


def test_recovers_signal_parameters_and_frame_content():
    n_cadu = 8
    p = ChainParams(input_format="asm_frame", randomizer="short", payload_bytes=_idle_frames(n_cadu),
                    n_cadu=n_cadu, sps=4, freq_offset_hz=5000.0)
    iq = run_chain(p).iq
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(len(iq), 2)) @ np.array([1, 1j]) * np.sqrt(np.mean(np.abs(iq) ** 2) / 2) * 0.05
    x = (iq + noise).astype(np.complex64)
    fs = p.symbol_rate * p.sps

    rs = ar.estimate_symbol_rate(x, fs, 1.785e6)
    assert abs(rs - p.symbol_rate) < 5
    assert ar.detect_modulation(x, fs, 50e3) == "QPSK"
    fc, _, _ = ar.estimate_carrier(x, fs, 50e3, power=4)
    assert abs(fc - 5000.0) < 20

    d, _ = ar.remove_carrier_phase(ar.demodulate(x, fs, rs, fc, 0.0, p.rrc_alpha), power=4)
    assert ar.symbol_metrics(d, "QPSK")["esn0_evm_db"] > 15

    bits, how = ar.decode(d, "QPSK", max_symbols=len(d))
    assert bits is not None and "G2 inverted" in how
    frames = ar.analyze_frames(bits)
    assert frames["cadu_bytes"] == 1279
    assert frames["randomizer"] == "short"
    assert frames["scid"] == [104] and set(frames["vcid_counts"]) == {7}
    assert frames["dominant_data_byte"][0] == 0x5A
    assert ar.periodicity(d, frames["cadu_bits"], "QPSK") > 0.5
    # The de-randomized records are exactly what was fed in, ready to be
    # loaded back into the generator as "asm_frame" input.
    sent = _idle_frames(n_cadu)
    got = frames["records"]
    assert len(got) == frames["n_cadu"] * 1279 and frames["n_cadu"] >= n_cadu - 2
    assert got in sent


def test_analyze_end_to_end_on_a_file(tmp_path):
    p = ChainParams(input_format="asm_frame", randomizer="short", payload_bytes=_idle_frames(6),
                    n_cadu=6, sps=4)
    iq = run_chain(p).iq
    raw = np.empty(2 * len(iq), dtype="<i2")
    raw[0::2], raw[1::2] = np.round(iq.real * 1500), np.round(iq.imag * 1500)
    path = tmp_path / "rec.iq"
    path.write_bytes(raw.tobytes())
    steps = []
    r = ar.analyze(str(path), fs=p.symbol_rate * p.sps, duration_s=1.0,
                   progress=lambda frac, msg: steps.append(frac))
    assert steps[0] == 0.0 and steps[-1] == 1.0
    assert r["frames"]["randomizer"] == "short"
    assert any("Pseudo-randomizer: short" in line for line in ar.report_lines(r))


def test_unwraps_rfcatcher_tar(tmp_path):
    import tarfile
    iq = (np.arange(40, dtype="<i2")).tobytes()
    src = tmp_path / "AWS_test.iq"
    src.write_bytes(iq)
    archive = tmp_path / "AWS_test.rfcatcher"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as t:
        t.add(src, arcname="AWS_test.iq")
    x, desc = ar.load_iq(str(archive), fs=10.0, offset_s=0.0, duration_s=2.0)
    assert "AWS_test.iq" in desc
    assert np.array_equal(x, np.arange(0, 40, 2) + 1j * np.arange(1, 40, 2))
