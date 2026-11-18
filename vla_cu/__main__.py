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
}


def usage() -> str:
    """Return the task list as text, one task per line."""
    lines = ["usage: python -m vla_cu <task>", "", "tasks:"]
    lines += [f"  {name:<6} {module}" for name, module in TASKS.items()]
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
