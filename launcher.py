#!/usr/bin/env python3
"""Desktop-style entry point for the packaged executable: starts the same
Streamlit GUI as `streamlit run app.py`, but as a plain double-clickable
program -- opens the default browser automatically once the server is up,
and prints the URL as a fallback in case the browser can't be opened
automatically (e.g. no default browser configured).

Not used by `streamlit run app.py` directly (that keeps working exactly as
before, unpackaged); this is only the entry point PyInstaller builds into
the standalone executable. See PACKAGING.md for how to build one.
"""

import os
import sys
import threading
import time
import webbrowser

PORT = 8501
URL = f"http://localhost:{PORT}"


def _base_dir() -> str:
    """Directory to resolve app.py/ccsds_chain/.streamlit from: PyInstaller's
    extracted bundle directory when frozen, this file's own directory
    otherwise (running from source, e.g. during development)."""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


def _open_browser_when_ready():
    # The Streamlit server takes a moment to bind its port; polling avoids
    # a fixed sleep that's either too short (browser opens to a connection
    # error) or needlessly long.
    import urllib.request

    for _ in range(100):  # up to ~20s
        try:
            urllib.request.urlopen(URL, timeout=0.5)
            break
        except Exception:
            time.sleep(0.2)
    try:
        webbrowser.open(URL)
    except Exception:
        pass  # the printed URL below is the fallback
    print(f"\nHKTM CCSDS Signal Generator is running at: {URL}")
    print("(If a browser window didn't open automatically, open that address yourself.)\n")


def main():
    base_dir = _base_dir()
    app_path = os.path.join(base_dir, "app.py")
    if not os.path.exists(app_path):
        raise SystemExit(f"Could not find app.py next to the launcher (looked in {base_dir})")

    # Deliberately does NOT chdir into base_dir: when frozen, that's
    # PyInstaller's one-file extraction directory (sys._MEIPASS), a fresh
    # temp folder deleted when the process exits -- app.py writes exported
    # .iq files to a relative "output/" folder, and confirmed (by actually
    # generating one against a packaged build) that chdir'ing there makes
    # every export vanish the moment the app closes. The process's own
    # working directory is left alone instead (for a double-clicked exe on
    # Windows, that's the folder containing it), so "output/" lands next to
    # the executable and survives.
    #
    # .streamlit/config.toml is normally discovered relative to the CWD, so
    # not chdir'ing means it would no longer be found -- its handful of
    # settings are passed as explicit CLI flags instead, which work
    # regardless of CWD. Keep these in sync with .streamlit/config.toml by
    # hand if that file's theme/server settings ever change.
    theme_and_server_flags = [
        "--theme.base=dark",
        "--theme.primaryColor=#00d4ff",
        "--theme.backgroundColor=#0b0f19",
        "--theme.secondaryBackgroundColor=#131a2a",
        "--theme.textColor=#e6edf5",
        "--theme.font=sans serif",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
    ]

    threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless=true",
        f"--server.port={PORT}",
        "--global.developmentMode=false",
        *theme_and_server_flags,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
