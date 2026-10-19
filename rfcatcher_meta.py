"""Show or change the metadata of an existing .rfcatcher recording without
regenerating it: the IQ data is left exactly as it is.

    python rfcatcher_meta.py file.rfcatcher                    # show the metadata
    python rfcatcher_meta.py file.rfcatcher --clear-gain -o file_nogain.rfcatcher
    python rfcatcher_meta.py file.rfcatcher --clear-gain --in-place
    python rfcatcher_meta.py file.rfcatcher --set gain.agc=fast -o test.rfcatcher

--clear-gain empties "gain_info" (the gain changes logged during the capture
the template came from) and sets "gain.agc" to --agc (default "off"; pass
--agc keep to leave it). Replaying the same signal with and without these
fields shows whether the replayer acts on them.
"""

import argparse
import json
import os
import sys

from ccsds_chain import rfcatcher


def _parse_set(text):
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, json.loads(value)  # lists, numbers, ""
    except ValueError:
        return key, value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help=".rfcatcher file")
    ap.add_argument("--clear-gain", action="store_true",
                    help='empty "gain_info" and set "gain.agc" to --agc')
    ap.add_argument("--agc", default="off", help='"gain.agc" with --clear-gain (default off; "keep": unchanged)')
    ap.add_argument("--set", type=_parse_set, action="append", default=[], metavar="KEY=VALUE",
                    help="set a field (VALUE parsed as JSON when it can be, e.g. [] or 34); repeatable")
    ap.add_argument("--remove", action="append", default=[], metavar="KEY", help="remove a field; repeatable")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("-o", "--output", help="write the changed recording here (a full copy)")
    out.add_argument("--in-place", action="store_true", help="change the recording itself (fast, no copy)")
    args = ap.parse_args(argv)

    meta = rfcatcher.read_metadata(args.recording)
    if meta is None:
        sys.exit(f"error: no RF-Catcher metadata found in {args.recording}")
    changing = args.clear_gain or args.set or args.remove
    if not changing:
        print(json.dumps(meta, indent=4))
        return 0
    if not (args.output or args.in_place):
        sys.exit("error: pass -o OUTPUT or --in-place to write the change")

    new = rfcatcher.clear_gain(meta, None if args.agc == "keep" else args.agc) if args.clear_gain else dict(meta)
    for key, value in args.set:
        new[key] = value
    for key in args.remove:
        new.pop(key, None)

    for key in sorted(set(meta) | set(new)):
        if meta.get(key, "<absent>") != new.get(key, "<absent>"):
            print(f"{key}: {json.dumps(meta.get(key, '<absent>'))} -> {json.dumps(new.get(key, '<absent>'))}")
    try:
        rfcatcher.rewrite_metadata(args.recording, new, None if args.in_place else args.output)
    except (ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    print(f"written: {os.path.abspath(args.recording if args.in_place else args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
