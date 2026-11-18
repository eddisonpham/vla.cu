"""Report the toolkit and host compiler a CUDA C++ build would use.

torch.utils.cpp_extension compiles every kernel with nvcc plus the host compiler
that PATH resolves, so what a build gets depends on the shell, not only on what
is installed. This task reads the same sources torch reads and states which
prerequisites a build has. It inspects, it does not compile.

Task: python -m vla_cu cuda-check
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import torch
from torch.utils import cpp_extension

CUDA_RELEASE = re.compile(r"release ([\d.]+), (V[\d.]+)")
VS_DIRS = ("ProgramFiles", "ProgramFiles(x86)")


def run(command: list[str]) -> str:
    """Return the merged stdout and stderr of a command, or an empty string on failure."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    except OSError:
        return ""
    return f"{result.stdout}{result.stderr}"


def visual_studio_dir(variable: str) -> Path:
    """Return the program files directory that holds the Visual Studio installs."""
    fallback = r"C:\Program Files (x86)" if "x86" in variable else r"C:\Program Files"
    return Path(os.environ.get(variable, fallback))


def toolsets() -> list[tuple[str, str, Path]]:
    """Return (MSVC version, Visual Studio edition, vcvars64.bat) for each installed C++ toolset."""
    found: list[tuple[str, str, Path]] = []
    for variable in VS_DIRS:
        pattern = str(visual_studio_dir(variable) / "Microsoft Visual Studio" / "*" / "*" / "VC" / "Tools" / "MSVC" / "*")
        for toolset in sorted(glob.glob(pattern)):
            path = Path(toolset)
            if not (path / "bin" / "Hostx64" / "x64" / "cl.exe").exists():
                continue
            edition = f"{path.parents[4].name}\\{path.parents[3].name}"
            found.append((path.name, edition, path.parents[2] / "Auxiliary" / "Build" / "vcvars64.bat"))
    return found


def report() -> int:
    """Print the configuration and one verdict line; return 0 unless no toolkit is found."""
    cuda_home = cpp_extension.CUDA_HOME
    nvcc = Path(cuda_home) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc") if cuda_home else None
    nvcc_text = run([str(nvcc), "--version"]) if nvcc and nvcc.exists() else ""
    release = CUDA_RELEASE.search(nvcc_text)

    print("CUDA C++ build configuration (what torch.utils.cpp_extension would use)\n")
    print(f"  PyTorch         {torch.__version__}")
    print(f"  CUDA (torch)    {torch.version.cuda}")
    print(f"  CUDA_HOME       {cuda_home or 'NOT FOUND'}")
    print(f"  nvcc            {nvcc if nvcc and nvcc.exists() else 'NOT FOUND'}")
    print(f"  nvcc version    {' '.join(release.groups()) if release else 'UNKNOWN'}")

    cl = shutil.which(cpp_extension.get_cxx_compiler())
    print(f"  cl on PATH      {cl or 'NOT ON PATH'}")
    installed = toolsets()
    print("  MSVC installs" if installed else "  MSVC installs   NOT FOUND")
    for version, edition, vcvars in installed:
        print(f"    {version}  {edition}")
        print(f"      vcvars64.bat  {vcvars if vcvars.exists() else 'NOT FOUND'}")

    ninja = cpp_extension.is_ninja_available()
    print(f"  ninja           {'AVAILABLE' if ninja else 'NOT AVAILABLE'}")

    missing = []
    if not release:
        missing.append("nvcc from the CUDA toolkit")
    if not cl:
        missing.append("cl.exe on PATH (call vcvars64.bat in this shell)")
    if not ninja:
        missing.append("ninja")
    if missing:
        print(f"\n  Build path      blocked, needs {', '.join(missing)}")
    else:
        print("\n  Build path      ready for torch.utils.cpp_extension")
    return 0 if release else 1


def main() -> int:
    """Run the report behind the task runner."""
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
