"""Task runner: `python -m vla_cu <task>`.

A short, stable entry point, so no doc names a long script path and the
implementation behind a task can move without changing any command.
"""

from __future__ import annotations

import importlib
import sys

TASKS: dict[str, str] = {
    "env": "scripts.env_check",
    "probe": "scripts.triton_probe",
    "cuda-check": "scripts.cuda_check",
    "bench": "bench.vector_add_bench",
    "launch": "bench.launch_overhead",
    "gemv": "bench.int8_gemv_bench",
    "graph": "bench.graph_chain",
    "tune": "bench.int8_gemv_tune",
    "matrix": "bench.matrix",
    "layer": "bench.layer",
    "cuda": "bench.cuda_compare",
    "stream": "bench.bandwidth_probe",
    "grouped": "bench.grouped",
    "m50": "bench.m50",
    "rotate": "bench.rotation",
    "fastpath": "bench.fastpath",
    "crossover": "bench.crossover",
    "attn": "bench.attention",
    "attncross": "bench.attention_cross",
    "pairedtail": "bench.paired_tail",
    "int8tiles": "bench.int8_tiles",
    "floor": "bench.launch_floor",
    "block": "bench.block",
    "fuzz": "bench.fuzz",
    "bundle": "bench.bundle_bench",
    "real": "bench.real_layers",
    "scale": "bench.scale",
    "plateau": "bench.plateau",
    "ncu": "bench.ncu_profile",
    "sass": "bench.sass",
    "style": "scripts.style_check",
}


def usage() -> str:
    """Return the task list as text, one task per line."""
    lines = ["usage: python -m vla_cu <task>", "", "tasks:"]
    width = max(len(name) for name in TASKS)
    lines += [f"  {name:<{width}} {module}" for name, module in TASKS.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run one task and return its exit code; 2 means bad usage, not a task failure."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(usage(), end="", file=sys.stderr)
        return 2
    if args[0] in {"-h", "--help", "help"}:
        print(usage(), end="")
        return 0
    if args[0] not in TASKS:
        print(f"unknown task: {args[0]}\n\n{usage()}", end="", file=sys.stderr)
        return 2
    return int(importlib.import_module(TASKS[args[0]]).main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
