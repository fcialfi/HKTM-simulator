"""RF-Catcher .rfcatcher format: headers identical to RF-Catcher's own,
archives readable by standard tar tools, metadata round trip, and the
generator -> .rfcatcher -> analysis loop."""

import json
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from ccsds_chain import rfcatcher


def test_header_matches_a_real_rfcatcher_recording():
    """Same name/size/mtime as the real AWS_2_split.rfcatcher first member
    must give the very same header, down to its checksum (011301)."""
    h = rfcatcher.tar_header("AWS_2.iq", 4802478080, 1788874074)
    assert len(h) == 512
    assert h[:8] == b"AWS_2.iq"
    assert h[100:108] == b"0000644\0"
    assert h[108:124] == b"\0" * 16  # empty uid/gid
    assert h[124:136] == b"43620000000\0"
    assert h[136:148] == b"15250006532\0"
    assert h[148:156] == b"011301\0\0"
    assert h[156:157] == b"\0"  # NUL typeflag
    assert h[257:265] == b"ustar\0\0\0"  # empty version
    assert h[265:274] == b"rfcatcher" and h[297:306] == b"rfcatcher"


def test_sizes_over_8_gib_use_base256():
    field = rfcatcher._numeric_field(9 * 2 ** 30, 12)
    assert field[0] == 0x80
    assert tarfile.nti(field) == 9 * 2 ** 30


def test_archive_round_trip(tmp_path):
    iq = np.arange(-600, 600, dtype="<i2")
    iq_path = tmp_path / "gen.iq"
    iq_path.write_bytes(iq.tobytes())
    meta = rfcatcher.build_metadata(10e6, len(iq) // 2, 1707e6, 4e6)
    out = tmp_path / "gen.rfcatcher"
    rfcatcher.write_rfcatcher(str(iq_path), str(out), meta)

    with tarfile.open(out) as t:
        members = t.getmembers()
        assert [m.name for m in members] == ["gen.iq", "gen.json"]
        assert t.extractfile(members[0]).read() == iq.tobytes()
        assert json.loads(t.extractfile(members[1]).read()) == meta
    assert rfcatcher.read_metadata(str(out)) == meta

    import analyze_recording as ar
    offset, size, desc = ar.locate_iq(str(out))
    assert (offset, size) == (512, iq.nbytes) and "gen.iq" in desc


def test_build_metadata_fields():
    template = dict(rfcatcher.DEFAULT_TEMPLATE, device="999", **{"adc.small_overload": [1.5]})
    start = __import__("datetime").datetime(2026, 9, 25, 10, 0, 0)
    m = rfcatcher.build_metadata(10e6, 10_000_000 * 90, 1707.5e6, 4e6, template=template, start=start)
    assert m["rate"] == "10.000 Msps"
    assert m["frequency"] == "1707.500 MHz"
    assert m["bandwidth"] == "4.000 MHz"
    assert m["duration"] == "0:01:30.000"
    assert m["record.start_time"] == "2026-09-25T10:00:00"
    assert m["record.stop_time"] == "2026-09-25T10:01:30"
    assert m["record.size"] == "3.35 GB" and m["record.expected_size"] == "~3.35 GB"
    assert m["device"] == "999"  # device fields come from the template
    assert m["adc.small_overload"] == []  # per-recording logs are cleared
    assert template["adc.small_overload"] == [1.5]  # template left untouched


def test_parse_quantity():
    assert rfcatcher.parse_quantity("10.000 Msps") == 10e6
    assert rfcatcher.parse_quantity("727.000 MHz") == 727e6
    assert rfcatcher.parse_quantity("4.000 MHz") == 4e6
    assert rfcatcher.parse_quantity(None) is None
    assert rfcatcher.parse_quantity("n/a") is None


def test_cli_rfcatcher_export_and_analysis(tmp_path):
    out = tmp_path / "sig.rfcatcher"
    subprocess.run([sys.executable, "generate_signal.py", "--n-cadu", "12", "--randomizer", "short", "--vcid-list", "0",
                    "--rfcatcher", "--rf-frequency-mhz", "1707", "--rf-bandwidth-mhz", "4",
                    "-o", str(out)], check=True, capture_output=True)
    assert out.exists() and not (tmp_path / "sig.iq").exists()
    meta = rfcatcher.read_metadata(str(out))
    assert meta["frequency"] == "1707.000 MHz" and meta["bandwidth"] == "4.000 MHz"
    fs = rfcatcher.parse_quantity(meta["rate"])
    assert fs == pytest.approx(1.785e6 * 4, rel=1e-6)

    import analyze_recording as ar
    r = ar.analyze(str(out), None, duration_s=1.0)  # sample rate taken from the metadata
    assert r["fs"] == fs
    assert r["frames"]["randomizer"] == "short"
    assert r["rfcatcher_meta"]["frequency"] == "1707.000 MHz"


def test_randomizer_undetermined_without_tm_headers():
    """Pseudo-random payload has no valid TM primary headers, so no
    randomizer candidate passes the header check: say so, don't guess."""
    import analyze_recording as ar
    from ccsds_chain.pipeline import ChainParams, _cadu_bits, _prepare_payload
    p = ChainParams(n_cadu=6, randomizer="short")
    prep = _prepare_payload(p)
    bits = np.concatenate([_cadu_bits(p, prep, i) for i in range(p.n_cadu)])
    assert ar.analyze_frames(bits)["randomizer"] == "undetermined"


def test_linked_bandwidth_and_parameter_checks():
    assert rfcatcher.linked_bandwidth(10e6) == pytest.approx(9.091e6)
    assert rfcatcher.linked_bandwidth(36e6) == pytest.approx(32.727e6)  # manual: 36 Msps <-> ~32 MHz
    assert rfcatcher.linked_bandwidth(0.5e6) == 1e6  # clamped to RF-Catcher's 1 MHz minimum
    assert rfcatcher.check_parameters(10e6, 1707e6, 4e6, 2.41e6) == []
    assert rfcatcher.check_parameters(10e6, 1707e6, rfcatcher.linked_bandwidth(10e6), 2.41e6) == []
    problems = rfcatcher.check_parameters(7.14e6, 50e6, 8e6, 2.41e6)
    assert any("outside" in p and "RF frequency" in p for p in problems)
    assert any("not below the sample rate" in p for p in problems)
    assert any("narrower" in p for p in rfcatcher.check_parameters(10e6, 1707e6, 2e6, 2.41e6))
    meta = rfcatcher.build_metadata(10e6, 1000, 1707e6, None)
    assert meta["bandwidth"] == "9.091 MHz"
