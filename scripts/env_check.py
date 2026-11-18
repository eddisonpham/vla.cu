"""Report the GPU, CUDA build, Triton, and which build/profiling tools exist.

A PyTorch wheel bundles the CUDA runtime, so GPU code runs without a CUDA
toolkit; `nvcc` and `compute-sanitizer` come from the toolkit. A missing tool is
information, not a broken install.

Task: python -m vla_cu env
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import torch

TOOLS = ("nvcc", "nsys", "ncu", "compute-sanitizer")

INSTALL_PATTERNS: dict[str, str] = {
    "ncu": "NVIDIA Corporation/Nsight Compute */target/windows-desktop-win7-x64/ncu.exe",
    "nsys": "NVIDIA Corporation/Nsight Systems */target-windows-x64/nsys.exe",
}


def program_files() -> Path:
    """Return the 64-bit Program Files directory holding NVIDIA's versioned tool folders."""
    return Path(os.environ.get("ProgramFiles", r"C:\Program Files"))


def locate(tool: str) -> str:
    """Return the tool's path from PATH, else from a known install folder, else NOT FOUND."""
    on_path = shutil.which(tool)
    if on_path:
        return on_path
    pattern = INSTALL_PATTERNS.get(tool)
    if not pattern:
        return "NOT FOUND"
    matches = sorted(glob.glob(str(program_files() / pattern)))
    return matches[-1] if matches else "NOT FOUND"


def triton_version() -> str:
    """Return the installed Triton version, or NOT INSTALLED on a plain Windows setup."""
    try:
        import triton
    except ImportError:
        return "NOT INSTALLED"
    return triton.__version__


def main() -> int:
    """Print the report and return 0; a missing tool is reported, never raised."""
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA (torch):   {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Triton:         {triton_version()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:            {props.name}")
        print(f"Memory:         {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute cap.:   {props.major}.{props.minor}")
        print(f"SM count:       {props.multi_processor_count}")

    print("\nTools (PATH first, then known install folders):")
    for tool in TOOLS:
        print(f"  {tool:<18} {locate(tool)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
