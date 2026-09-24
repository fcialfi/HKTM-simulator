# PyInstaller build spec for the packaged desktop executable.
#
# Build (on the target OS -- PyInstaller does not cross-compile, so a
# Windows .exe must be built ON Windows; see PACKAGING.md):
#
#   pip install -r requirements.txt pyinstaller
#   pyinstaller hktm_simulator.spec
#
# Result: dist/HKTM-CCSDS-Signal-Generator(.exe) -- a single file, no
# separate Python install needed on the machine that runs it.

from PyInstaller.utils.hooks import collect_all

# app.py is run by Streamlit as a script, not imported, so PyInstaller never
# traces its imports: every local module it imports must be listed here too
# (analyze_recording.py backs the GUI's "Analyze a real recording" panel).
datas = [
    ("app.py", "."),
    ("analyze_recording.py", "."),
    ("ccsds_chain", "ccsds_chain"),
    (".streamlit", ".streamlit"),
]
binaries = []
hiddenimports = []

# Streamlit and Plotly both ship non-.py assets (Streamlit's compiled
# frontend, Plotly's renderer data) that PyInstaller's static import
# analysis alone won't discover; scipy's C-extension submodules (e.g.
# scipy.signal, used by ccsds_chain.utils) are similarly missed and need
# the same explicit treatment -- confirmed by actually running a build:
# without this, the packaged app launches but the GUI's first script run
# fails with "No module named 'scipy.signal'".
for _pkg in ("streamlit", "plotly", "scipy"):
    _pkg_datas, _pkg_binaries, _pkg_hiddenimports = collect_all(_pkg)
    datas += _pkg_datas
    binaries += _pkg_binaries
    hiddenimports += _pkg_hiddenimports

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HKTM-CCSDS-Signal-Generator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
