"""verify_export.py must call a clean export clean, and locate defects put into
a copy of it (missing samples = slip, a hole of zeros, a level change)."""

import os

import numpy as np

import verify_export as ve
from ccsds_chain import rfcatcher
from ccsds_chain.pipeline import ChainParams, export_chain


def _wrap(tmp_path, iq_path, name, fs):
    out = str(tmp_path / name)
    rfcatcher.write_rfcatcher(iq_path, out, rfcatcher.build_metadata(fs, os.path.getsize(iq_path) // 4, 727e6, None))
    return out


def test_clean_export_and_injected_defects(tmp_path):
    p = ChainParams(randomizer="long", rrc_alpha=1.0, n_cadu=360)   # ~2 s
    fs = p.symbol_rate * p.sps
    iq = str(tmp_path / "x.iq")
    export_chain(p, iq, output_dtype="int16")

    r = ve.verify(_wrap(tmp_path, iq, "clean.rfcatcher", fs), p.symbol_rate, alpha=1.0, window_s=0.5)
    assert r["asms"] == p.n_cadu
    assert r["slips"] == [] and r["zero_runs"] == []
    assert min(w["mer_db"] for w in r["windows"]) > 50

    raw = np.fromfile(iq, dtype="<i2")
    at = lambda t: int(t * fs) * 2
    bad = np.concatenate([raw[:at(0.6)], raw[at(0.6) + 2 * 3:]])     # 3 samples missing at 0.6 s
    bad[at(1.2):at(1.2) + 2 * 400] = 0                                 # hole at 1.2 s
    bad_iq = str(tmp_path / "bad.iq")
    bad.tofile(bad_iq)
    r = ve.verify(_wrap(tmp_path, bad_iq, "bad.rfcatcher", fs), p.symbol_rate, alpha=1.0, window_s=0.5)
    assert len(r["slips"]) == 1 and abs(r["slips"][0]["time_s"] - 0.6) < 0.01
    assert len(r["zero_runs"]) == 1 and abs(r["zero_runs"][0][0] - 1.2) < 0.001


def test_coded_asm_pattern_is_independent_of_the_preceding_bits():
    from ccsds_chain.convolutional import conv_encode
    from ccsds_chain.utils import bytes_to_bits
    asm = bytes_to_bits(ve.ASM)
    for rate in ("1/2", "3/4", "7/8"):
        patterns = ve.coded_asm_patterns(rate, invert_g2=True)
        rng = np.random.default_rng(1)
        for lead in (0, 1, 2, 5, 11):      # lands the ASM on different puncture phases
            prefix = rng.integers(0, 2, 30 + lead).astype(np.uint8)
            coded = conv_encode(np.concatenate([prefix, asm]), rate=rate)
            n_before = len(conv_encode(prefix, rate=rate))
            period = len(patterns)
            pat, head = patterns[len(prefix) % period]
            assert np.array_equal(coded[n_before + head:n_before + head + len(pat)], pat)


def test_framing_is_detected_and_unverified_files_are_never_called_clean(tmp_path, capsys, monkeypatch):
    # rate 1/2 WITHOUT G2 inversion, and an uncoded export: both must be detected
    for kw, coding in ((dict(conv_invert_g2=False), ("1/2", False)), (dict(fec_conv=False), ("none", False))):
        p = ChainParams(randomizer="long", n_cadu=120, **kw)
        fs = p.symbol_rate * p.sps
        iq = str(tmp_path / "x.iq")
        export_chain(p, iq, output_dtype="int16")
        f = _wrap(tmp_path, iq, "x.rfcatcher", fs)
        r = ve.verify(f, p.symbol_rate)
        assert r["coding"] == coding and r["cadu_bits"] == 1279 * 8
        assert r["asms"] == p.n_cadu and r["slips"] == []

    # wrong symbol rate: no ASM can be found, and the result must not read CLEAN
    monkeypatch.setattr("sys.argv", ["verify_export.py", f, "--symbol-rate", "0.8925e6"])
    assert ve.main() == 1
    out = capsys.readouterr().out
    assert "NOT VERIFIED" in out and "CLEAN" not in out
