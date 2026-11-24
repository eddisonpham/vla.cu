"""Run compute-sanitizer memcheck over every CUDA kernel with one command.

Concept: a new kernel is not trusted until memcheck reports zero errors, and a gate that
lives in a copied command line stops being run. Each probe is a separate process because
the tool instruments at launch. A tool that cannot attach to a kernel is reported as
unsupported, never as clean, because the two mean different things (EXP-016, EXP-087).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROBES = (
    ("int4 gemv, three variants", "bench.probes.int4"),
    ("cooperative layer, 3, 17 and 26 tokens", "bench.probes.coop"),
    ("gated cooperative layer, 3, 17 and 26 tokens", "bench.probes.coop_gated"),
)
UNSUPPORTED = "CUDA_ERROR_INVALID_HANDLE"


def find_tool() -> Path:
    """Return the compute-sanitizer executable, from PATH or the CUDA toolkit.

    The installer does not put the tool on PATH, so CUDA_PATH is the fallback that makes
    this work in the shell a caller already uses for the build.
    """
    on_path = shutil.which("compute-sanitizer")
    if on_path:
        return Path(on_path)
    toolkit = os.environ.get("CUDA_PATH")
    if toolkit:
        candidate = Path(toolkit) / "bin" / ("compute-sanitizer.exe" if os.name == "nt" else "compute-sanitizer")
        if candidate.exists():
            return candidate
    raise SystemExit("compute-sanitizer not found: install the CUDA toolkit or put it on PATH")


def classify(stdout: str, returncode: int) -> str:
    """Return clean, unsupported or error for one instrumented run.

    The extension reports CUDA_ERROR_INVALID_HANDLE when the tool asks for a kernel
    function it cannot see, which is an attach failure, not a memory defect.
    """
    if "Invalid __global__" in stdout or "out of bounds" in stdout:
        return "error"
    if UNSUPPORTED in stdout:
        return "unsupported"
    return "clean" if returncode == 0 else "error"


def main() -> int:
    """Run each probe under memcheck and report one status per kernel."""
    tool = find_tool()
    failures = 0
    for label, module in PROBES:
        command = [str(tool), "--tool", "memcheck", "--error-exitcode", "9", sys.executable, "-m", module]
        print(f"{label}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        status = classify(result.stdout, result.returncode)
        summary = [line for line in result.stdout.splitlines() if "ERROR SUMMARY" in line]
        print(f"  {status}, {summary[-1].strip() if summary else 'no summary line'}")
        if status == "error":
            failures += 1
            details = [
                line for line in result.stdout.splitlines() if "Invalid" in line or "out of bounds" in line or "Address 0x" in line
            ]
            print("  " + "\n  ".join(details[-8:]))
        elif status == "unsupported":
            print("  the tool cannot attach to this torch extension on this platform, so it proves nothing")
    print(f"{len(PROBES) - failures} of {len(PROBES)} probes without a memory error")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
