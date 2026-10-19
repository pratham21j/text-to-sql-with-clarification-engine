# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "academic.db"
if not DATABASE_PATH.is_file():
    raise FileNotFoundError(
        "The synthetic database is required for packaging. "
        "Initialize it first with: python scripts\\init_db.py"
    )

datas = [(str(DATABASE_PATH), "data")]

a = Analysis(
    [str(PROJECT_ROOT / "scripts" / "run_desktop.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib",
        "notebook",
        "pandas",
        "pytest",
        "zmq",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TextToSQL",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="TextToSQL",
)
