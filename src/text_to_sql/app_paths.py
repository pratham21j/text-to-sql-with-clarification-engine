"""Resolve application and bundled-resource paths in source and frozen builds."""

import sys
from pathlib import Path


def application_directory() -> Path:
    """Return the project root in source runs or executable directory when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    """Resolve packaged data from PyInstaller's bundle or the source project."""
    bundle_directory = getattr(sys, "_MEIPASS", None)
    base_directory = (
        Path(bundle_directory)
        if isinstance(bundle_directory, str)
        else application_directory()
    )
    return base_directory.joinpath(*parts)


def environment_file_path() -> Path:
    """Locate user-owned provider configuration outside packaged resources."""
    return application_directory() / ".env"
