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

   A splash screen with the logo appears at once and shows the start-up
   progress, and a console window opens (so you can see any error
   messages). The first start can take a minute or two: the one-file
   executable first unpacks its few hundred MB of libraries to a temporary
   folder, and on Windows the antivirus usually scans them. When the GUI
   server is up, the splash closes and your default browser opens
   automatically to `http://localhost:8501`. If it doesn't open by itself, the console
   window prints that same address to open by hand. Closing the console
   window stops the server.

5. To hand the tool to someone else, copy just that one `.exe` file --
   it's fully self-contained (Python interpreter and all dependencies are
   bundled inside it), so it doesn't need Python installed on the machine
   that runs it. It's a few hundred MB, all numpy/scipy/streamlit/plotly
   bundled in.

## Logo and icon

The Exostaff logo lives in `assets/`:

- `exostaff_icon.ico` -- the executable's icon (Explorer, taskbar), set by
  `icon=` in `hktm_simulator.spec`;
- `exostaff_icon.png` -- the browser tab icon (`page_icon` in `app.py`);
- `exostaff_logo.png` -- the logo shown at the top of the sidebar;
- `splash.png` -- the start-up splash screen (`Splash` in the spec), shown
  while the executable unpacks; `launcher.py` writes its status line and
  closes it when the GUI is ready. Building it needs Tcl/Tk in the Python
  used for the build (included in the python.org Windows installer).

The whole `assets/` folder is bundled into the executable (`datas` in the
spec). To change the logo, replace these files with the same names and
rebuild. If they are missing, the app falls back to the 📡 icon and no logo.

## Time-limited licence

The executable can be restricted with a licence file that expires on a
given date (`licensing.py`). A licence is a small `license.key` file, placed
next to the `.exe`, with the customer name and the expiry date, signed with
a private key that only the issuer has. The executable contains only the
public key: it can check a licence but cannot make one, and editing the file
(e.g. a later date) invalidates it. Without a valid licence the program
stops at start-up with an explanation; within 30 days of the expiry date
the GUI shows a warning.

1. **Once:** create the key pair (needs `pip install cryptography`, already
   in `requirements.txt`):

   ```cmd
   python license_tool.py keygen --private-key C:\secure\hktm_license_private_key.pem
   ```

   - The **private key** goes where you say: keep it safe and outside the
     repository. Whoever has it can issue licences; if it is lost, new
     licences need a new key pair and a rebuild.
   - The **public key** goes to `assets/license_public_key.pem`. Commit it:
     every executable built from then on checks licences. Builds without
     this file have no licence check.

2. **For each customer:** issue a licence and send it with the `.exe`:

   ```cmd
   python license_tool.py issue --private-key C:\secure\hktm_license_private_key.pem --customer "Customer name" --expires 2027-12-31 -o license.key
   ```

   The expiry date is the last valid day. To extend, issue a new file; no
   rebuild is needed.

3. **Optional check:** `python license_tool.py verify license.key`

Limits: this is a deterrent, not copy protection. The latest date seen is
remembered in the user's profile, so setting the clock back is detected, but
a determined user can still decompile a Python executable. Running from
source (`streamlit run app.py`) is never blocked.

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
(`--n-cadu 100 -o test.raw`, etc.), just without needing Python installed.
