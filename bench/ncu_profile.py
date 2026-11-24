"""Run Nsight Compute on the int4 GEMV, from a shell that has administrator rights.

Concept: profiling counters are denied to a normal user on this machine, so the shell
that starts ncu must be elevated. The vcvars wrapper is written to a batch file and ncu
is given a command with no quoted arguments, because a quoted batch path passed through
ncu and cmd is what failed to resolve on the first attempt.

Task: python -m vla_cu ncu
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.env_check import locate

ROOT = Path(__file__).resolve().parent.parent
VCVARS = ROOT / "bench" / "results" / "vcvars.bat"
WRAPPER = ROOT / "bench" / "results" / "ncu_run.bat"
REPORT = ROOT / "bench" / "results" / "ncu_int4_big"
SECTIONS = ("SpeedOfLight", "MemoryWorkloadAnalysis", "SchedulerStats", "WarpStateStats")


def write_wrapper() -> str:
    """Write the batch file that enters vcvars and runs the cuda task, then return its path."""
    WRAPPER.parent.mkdir(parents=True, exist_ok=True)
    WRAPPER.write_text(
        "@echo off\n" f'call "{VCVARS}" "{sys.executable}" -m vla_cu cuda\n',
        encoding="utf-8",
    )
    return str(WRAPPER.relative_to(ROOT))


def command() -> list[str]:
    """Return the ncu argument list, quoted nowhere, that profiles the large-shape kernels."""
    ncu = locate("ncu")
    if ncu == "NOT FOUND":
        raise FileNotFoundError("ncu not found on PATH or in the Nsight Compute install folder")
    arguments = [ncu, "--target-processes", "all", "--clock-control", "none"]
    for section in SECTIONS:
        arguments += ["--section", section]
    arguments += [
        "-k",
        "regex:int4_gemv",
        "--launch-skip",
        "200",
        "--launch-count",
        "3",
        "--export",
        str(REPORT),
        "--force-overwrite",
        "cmd",
        "/c",
        write_wrapper(),
    ]
    return arguments


def main() -> int:
    """Print the command, run it, and explain a counter-permission failure."""
    try:
        arguments = command()
    except FileNotFoundError as error:
        print(error)
        return 1
    print(" ".join(arguments))
    print()
    completed = subprocess.run(arguments, cwd=ROOT)
    if completed.returncode != 0:
        print(f"ncu exited {completed.returncode}")
        print("A counter permission error means this shell is not elevated. Open PowerShell as")
        print("administrator, change to the repo, and run the same task there.")
        return completed.returncode
    print(f"report written to {REPORT}.ncu-rep")
    print(f"read it with: ncu --import {REPORT}.ncu-rep --page details")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
