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

# Splash screen of the packaged executable (see hktm_simulator.spec). The
# module only exists inside a PyInstaller build that has a splash screen;
# running from source it is simply absent.
try:
    import pyi_splash  # type: ignore[import-not-found]
except ImportError:
    pyi_splash = None


def _status(message: str) -> None:
    """Progress message, both in the console window and on the splash screen."""
    print(message, flush=True)
    if pyi_splash is not None:
        try:
            pyi_splash.update_text(message)
        except Exception:
            pass


def _close_splash() -> None:
    if pyi_splash is not None:
        try:
            pyi_splash.close()
        except Exception:
            pass


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

    for _ in range(600):  # up to ~2 min on a slow first start
        try:
            urllib.request.urlopen(URL, timeout=0.5)
            break
        except Exception:
            time.sleep(0.2)
    _close_splash()
    try:
        webbrowser.open(URL)
    except Exception:
        pass  # the printed URL below is the fallback
    print(f"\nHKTM CCSDS Signal Generator is running at: {URL}")
    print("(If a browser window didn't open automatically, open that address yourself.)\n")


def main():
    print("=" * 64)
    print("  HKTM CCSDS Signal Generator")
    print("  Please wait: starting takes a few seconds (longer the first time).")
    print("  Your browser will open by itself when the GUI is ready.")
    print("  Keep this window open - closing it stops the program.")
    print("=" * 64, flush=True)
    _status("Starting...")
    import licensing

    lic = licensing.check_license()
    if not lic.valid:
        # Stop before the slow GUI start-up; the console window stays open
        # so the message can be read.
        _close_splash()
        print(f"\nLICENCE NOT VALID: {lic.message}")
        print("Contact the supplier for a new licence file.\n")
        try:
            input("Press Enter to close this window.")
        except EOFError:
            pass
        sys.exit(1)
    if lic.enforced:
        print(lic.message, flush=True)
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

    _status("Loading the GUI and signal-processing libraries...")
    from streamlit.web import cli as stcli

    _status("Starting the GUI server...")
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()

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
