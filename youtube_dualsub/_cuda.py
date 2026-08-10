"""Make the nvidia-* pip wheels visible to CTranslate2 on Windows.

faster-whisper runs on CTranslate2, which dlopen()s cuDNN and cuBLAS by name.
On Windows those DLLs live inside the pip wheels at
``site-packages/nvidia/<lib>/bin`` — a directory Python does *not* put on the
DLL search path. Without this shim the first transcription dies with

    Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!

which is the single most common way this project fails to start. Import
``ensure_cuda_dlls()`` before importing faster_whisper.
"""

from __future__ import annotations

import os
import sysconfig
from functools import lru_cache
from pathlib import Path

# DLLs CTranslate2 needs at runtime, and the wheel that provides each.
REQUIRED_DLLS = {
    "cudnn_ops64_9.dll": "nvidia-cudnn-cu12",
    "cublas64_12.dll": "nvidia-cublas-cu12",
}


def _wheel_dll_dirs() -> list[Path]:
    roots = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    dirs: list[Path] = []
    for root in roots:
        nvidia = Path(root) / "nvidia"
        if nvidia.is_dir():
            dirs.extend(sorted(p for p in nvidia.glob("*/bin") if p.is_dir()))
    return dirs


@lru_cache(maxsize=1)
def ensure_cuda_dlls() -> list[str]:
    """Add nvidia wheel DLL directories to the search path. Idempotent."""
    if os.name != "nt":
        return []
    added: list[str] = []
    for d in _wheel_dll_dirs():
        try:
            os.add_dll_directory(str(d))
        except (OSError, AttributeError):
            continue
        added.append(str(d))
    if added:
        # Some CTranslate2 builds resolve via PATH rather than the DLL directory
        # list, so belt and braces.
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    return added


def diagnose() -> tuple[bool, list[str]]:
    """Return (ok, messages) describing whether the CUDA runtime DLLs are present."""
    ensure_cuda_dlls()
    if os.name != "nt":
        return True, ["Not Windows — relying on the system loader for CUDA libraries."]

    dirs = _wheel_dll_dirs()
    messages = [f"Found {len(dirs)} nvidia wheel DLL director{'y' if len(dirs) == 1 else 'ies'}."]
    present = {p.name.lower() for d in dirs for p in d.glob("*.dll")}

    missing: list[str] = []
    for dll, wheel in REQUIRED_DLLS.items():
        if dll.lower() in present:
            messages.append(f"  OK      {dll}")
        else:
            missing.append(dll)
            messages.append(f"  MISSING {dll}  ->  uv pip install '{wheel}'")

    if missing:
        messages.append(
            "CTranslate2 will fail at transcription time until these are installed. "
            "Run: uv sync   (or: uv pip install "
            + " ".join(f"'{REQUIRED_DLLS[m]}'" for m in missing)
            + ")"
        )
    return not missing, messages
