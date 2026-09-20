# Packaging a standalone desktop executable

`streamlit run app.py` is the normal, unpackaged way to run the GUI (see
README.md) -- this document is only for building a single-file executable
that runs the same GUI on a machine with no Python installed at all.

**PyInstaller does not cross-compile.** A Windows `.exe` must be built by
running these steps *on a Windows machine* (or a Windows VM); a Linux
binary must be built on Linux, and so on. You cannot produce a Windows
executable from Linux or macOS.

## Build steps (Windows)

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/)
   (check "Add python.exe to PATH" during install).

2. Open a Command Prompt in this repository's folder and install the
   project's dependencies plus PyInstaller:

   ```cmd
   pip install -r requirements.txt pyinstaller
   ```

3. Build:

   ```cmd
   pyinstaller hktm_simulator.spec
   ```

   This takes a few minutes (PyInstaller has to trace every module numpy,
   scipy, streamlit and plotly might import). When it finishes, the
   executable is at:

   ```
   dist\HKTM-CCSDS-Signal-Generator.exe
   ```

4. Run it (double-click, or from a terminal):

   ```cmd
   dist\HKTM-CCSDS-Signal-Generator.exe
   ```

   A console window opens (so you can see any error messages), the GUI
   server starts, and your default browser opens automatically to
   `http://localhost:8501`. If it doesn't open by itself, the console
   window prints that same address to open by hand. Closing the console
   window stops the server.

5. To hand the tool to someone else, copy just that one `.exe` file --
   it's fully self-contained (Python interpreter and all dependencies are
   bundled inside it), so it doesn't need Python installed on the machine
   that runs it. It's a few hundred MB, all numpy/scipy/streamlit/plotly
   bundled in.

## If the build fails or the app errors on first launch

Report the exact error back (the PyInstaller build log, or the console
window's traceback if the `.exe` itself fails) so `hktm_simulator.spec`
can be adjusted -- this is the kind of thing that only shows up by
actually running the packaged executable, not by reading the spec file.
One such case is already handled: `scipy.signal` needed an explicit
`collect_all("scipy")` entry, found by building and running the result on
Linux, not by inspecting the spec beforehand. A different environment
(Windows, a different Python version, a newer/older library version) can
easily surface another missing submodule the same way; the fix is always
the same shape -- add the missing package to the `collect_all(...)` loop
near the top of `hktm_simulator.spec`, or add the specific missing dotted
name to `hiddenimports` if pulling in the whole package is overkill.

## Rebuilding after a code change

Re-run `pyinstaller hktm_simulator.spec` -- it always does a clean build
from the current source (there's no incremental/watch mode). Delete the
`build/` and `dist/` folders first if you want to be sure nothing stale is
reused (both are already git-ignored).

## Packaging the CLI instead

`generate_signal.py` is a plain command-line script with no GUI/browser
involved, so it packages far more simply with PyInstaller directly (no
spec file needed):

```cmd
pyinstaller --onefile generate_signal.py
```

`dist\generate_signal.exe` then works exactly like
`python generate_signal.py` did, with the same command-line flags
(`--preset baseline --n-cadu 100 -o test.raw`, etc.), just without needing
Python installed.
