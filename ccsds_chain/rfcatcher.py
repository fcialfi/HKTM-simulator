"""RF-Catcher (TestTree) `.rfcatcher` recording format: read its metadata,
and wrap a generated int16 IQ file into one so the replayer picks up sample
rate, RF frequency and bandwidth by itself.

A `.rfcatcher` file is a tar archive of two members, in this order:

  <name>.iq    raw interleaved IQ, int16 little-endian (12 significant bits),
               "sample_size": 4 bytes per complex sample
  <name>.json  recorder metadata: device, "frequency", "rate", "bandwidth",
               "duration", record times/size, gain and overload logs...

Headers are written exactly as RF-Catcher itself writes them (checked
against a real recording, AWS_2_split.rfcatcher): magic "ustar" with an
empty version field, NUL typeflag, empty uid/gid, owner/group "rfcatcher",
mode 0644. Members over 8 GiB, whose size no longer fits the 11 octal
digits, use the GNU base-256 size encoding.
"""

import copy
import datetime
import json
import os
import shutil
import tarfile
from typing import Optional

BLOCK = 512

# Metadata of a real recording (AWS_2_split.rfcatcher, AWS pass of
# 2026-09-08), used as the template for fields this tool has no value of
# its own for (device identity, firmware revisions, gain, signal level).
# Pass a real recording's metadata as `template` to use that device's.
DEFAULT_TEMPLATE = {
    "device": "15225007",
    "hwrev": "2.0",
    "swrev": "25.3.0",
    "fx3rev": "4.0",
    "fpgarev": "5.1",
    "sample_size": 4,
    "link.type": "USB3",
    "link.max_sample_rate": "61.200 Msps",
    "cal.status": "CALOK",
    "frequency": "727.000 MHz",
    "gain.t0": "34.0 dB",
    "gain.agc": "slow",
    "rate": "10.000 Msps",
    "bandwidth": "4.000 MHz",
    "connector": "SMA",
    "duration": "0:02:00.000",
    "date.time": "2026-09-08T12:59:24",
    "record.start_time": "2026-09-08T12:59:24",
    "record.stop_time": "2026-09-08T13:09:08",
    "record.size": "21.75 GB",
    "record.expected_size": "~21.76 GB",
    "signal.level.t0": "-51.5 dBm",
    "bookmarks": [
        {"time": 0.0, "tag": "Bookmarks are not managed yet in this software version"},
    ],
    "markers": [],
    "samples_loss": [],
    "lmt.large_overload": [],
    "lmt.small_overload": [],
    "adc.large_overload": [],
    "adc.small_overload": [],
    "gain_info": ["76:26", "83:27"],
}

_OCTAL_SIZE_LIMIT = 8 ** 11  # 8 GiB: largest size 11 octal digits can hold


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_metadata(path: str) -> Optional[dict]:
    """The JSON metadata inside a .rfcatcher recording (or a standalone
    .json file), or None if there is none. Only headers are read: tarfile
    seeks past the (possibly multi-GB) IQ member."""
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        with open(path, "rb") as f, tarfile.open(fileobj=f, mode="r:") as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith(".json"):
                    return json.loads(tar.extractfile(member).read().decode("utf-8"))
    except (tarfile.TarError, OSError, ValueError):
        return None
    return None


def parse_quantity(text) -> Optional[float]:
    """'10.000 Msps' -> 10e6, '727.000 MHz' -> 727e6, '4.000 MHz' -> 4e6."""
    if text is None:
        return None
    parts = str(text).split()
    try:
        value = float(parts[0])
    except (ValueError, IndexError):
        return None
    unit = parts[1].lower() if len(parts) > 1 else ""
    for prefix, scale in (("g", 1e9), ("m", 1e6), ("k", 1e3)):
        if unit.startswith(prefix):
            return value * scale
    return value


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


def _format_size(n_bytes: int) -> str:
    # RF-Catcher's "21.75 GB" for 23.36e9 bytes: binary units, 2 decimals.
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / 1024 ** 3:.2f} GB"
    return f"{n_bytes / 1024 ** 2:.2f} MB"


def build_metadata(sample_rate: float, n_samples: int, rf_frequency_hz: float, bandwidth_hz: float,
                   template: Optional[dict] = None,
                   start: Optional[datetime.datetime] = None) -> dict:
    """RF-Catcher metadata for a generated IQ file: the template's device
    fields, with sample rate, RF frequency, bandwidth, duration, record
    times and sizes set for this file, and the per-recording logs
    (overloads, sample losses, markers) emptied."""
    meta = copy.deepcopy(template if template is not None else DEFAULT_TEMPLATE)
    sample_size = int(meta.get("sample_size", 4))
    duration_s = n_samples / sample_rate
    start = (start or datetime.datetime.now()).replace(microsecond=0)
    stop = start + datetime.timedelta(seconds=duration_s)
    size = n_samples * sample_size
    meta.update({
        "sample_size": sample_size,
        "frequency": f"{rf_frequency_hz / 1e6:.3f} MHz",
        "rate": f"{sample_rate / 1e6:.3f} Msps",
        "bandwidth": f"{bandwidth_hz / 1e6:.3f} MHz",
        "duration": _format_duration(duration_s),
        "date.time": start.isoformat(),
        "record.start_time": start.isoformat(),
        "record.stop_time": stop.replace(microsecond=0).isoformat(),
        "record.size": _format_size(size),
        "record.expected_size": "~" + _format_size(size),
    })
    for key in ("markers", "samples_loss", "lmt.large_overload", "lmt.small_overload",
                "adc.large_overload", "adc.small_overload"):
        if key in meta:
            meta[key] = []
    return meta


def _numeric_field(value: int, width: int) -> bytes:
    """`width`-byte tar numeric field: octal digits + NUL, or GNU base-256
    when the value doesn't fit (sizes over 8 GiB)."""
    if value < 8 ** (width - 1):
        return f"{value:0{width - 1}o}".encode() + b"\0"
    return b"\x80" + value.to_bytes(width - 1, "big")


def tar_header(name: str, size: int, mtime: int) -> bytes:
    """A 512-byte tar header laid out exactly like RF-Catcher's own."""
    encoded = name.encode("utf-8")
    if len(encoded) > 99:
        raise ValueError(f"member name too long for a tar header: {name!r}")
    h = bytearray(BLOCK)
    h[0:len(encoded)] = encoded
    h[100:108] = b"0000644\0"
    # uid (108-115) and gid (116-123) left all-NUL, as RF-Catcher does
    h[124:136] = _numeric_field(size, 12)
    h[136:148] = _numeric_field(mtime, 12)
    # typeflag (156) left NUL: regular file
    h[257:263] = b"ustar\0"
    # version (263-264) left NUL, as RF-Catcher does
    h[265:274] = b"rfcatcher"
    h[297:306] = b"rfcatcher"
    h[148:156] = b" " * 8
    h[148:156] = f"{sum(h):06o}".encode() + b"\0\0"
    return bytes(h)


def write_rfcatcher(iq_path: str, out_path: str, metadata: dict, member_basename: Optional[str] = None,
                    mtime: Optional[int] = None) -> None:
    """Wrap an existing raw int16 IQ file and its metadata into a .rfcatcher
    archive (<name>.iq, then <name>.json). The IQ data is streamed, never
    loaded into memory."""
    base = member_basename or os.path.splitext(os.path.basename(out_path))[0]
    mtime = int(mtime if mtime is not None else datetime.datetime.now().timestamp())
    iq_size = os.path.getsize(iq_path)
    meta_bytes = json.dumps(metadata, indent=4).encode("utf-8")

    def pad(n):
        return b"\0" * (-n % BLOCK)

    with open(out_path, "wb") as out:
        out.write(tar_header(base + ".iq", iq_size, mtime))
        with open(iq_path, "rb") as src:
            shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
        out.write(pad(iq_size))
        out.write(tar_header(base + ".json", len(meta_bytes), mtime))
        out.write(meta_bytes)
        out.write(pad(len(meta_bytes)))
        out.write(b"\0" * (2 * BLOCK))  # end-of-archive marker
