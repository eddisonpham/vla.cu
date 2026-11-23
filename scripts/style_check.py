"""Check the no-comments rule and the documentation style invariant.

Concept: two rules from `CLAUDE.md` are cheap to state and easy to break, so they are a
script instead of a habit. Source files may carry a module docstring but no comments.
Prose is ASCII, with no em dashes, en dashes, curly quotes or emojis. Config files are
allowed comments because a manifest and a requirements file need notes.

It scans the files git tracks, so build output stays out of scope.

Task: python -m vla_cu style
"""

from __future__ import annotations

import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = (".py", ".cu", ".md", ".toml", ".yml", ".yaml", ".txt")
COMMENT_ALLOWED = {"requirements.txt", "pyproject.toml", ".gitignore", ".gitattributes"}


def tracked_files() -> list[Path]:
    """Return every file git tracks, as absolute paths."""
    output = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [ROOT / name.decode("utf-8") for name in output.split(b"\0") if name]


def python_comments(path: Path) -> list[tuple[int, str]]:
    """Return (line, text) for every `#` comment, excluding a shebang."""
    found = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type != tokenize.COMMENT:
                continue
            if token.start[0] == 1 and token.string.startswith("#!"):
                continue
            found.append((token.start[0], token.string.strip()))
    return found


def cuda_comments(path: Path) -> list[tuple[int, str]]:
    """Return (line, text) for every `//` or block comment line in a CUDA source."""
    found = []
    in_block = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if in_block:
            found.append((number, stripped))
            in_block = "*/" not in stripped
        elif stripped.startswith("/*"):
            found.append((number, stripped))
            in_block = "*/" not in stripped
        elif "//" in line:
            found.append((number, stripped))
    return found


def main() -> int:
    """Print every violation and return 1 if any was found."""
    violations = 0
    for path in tracked_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix in TEXT_SUFFIXES:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if any(ord(character) > 127 for character in line):
                    print(f"{relative}:{number}: non-ASCII character")
                    violations += 1
        if path.name in COMMENT_ALLOWED:
            continue
        found = []
        if path.suffix == ".py":
            found = python_comments(path)
        elif path.suffix == ".cu":
            found = cuda_comments(path)
        for number, comment in found:
            print(f"{relative}:{number}: comment {comment}")
            violations += 1
    print(f"{violations} style violations")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
