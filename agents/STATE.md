# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-18
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`
**Checkpoint:** 2026-11-18 at `1b19bca`.

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), `.gitignore`, `.gitattributes`, `requirements.txt`.
- Environment check (`python -m vla_cu env`) and hardware record in `HARDWARE.md`.
- Triton verified on native Windows and sm_120 (`python -m vla_cu probe`), EXP-002.
- Task runner `python -m vla_cu <task>` with tasks `env` and `probe` (DEC-011).
- `README.md`, `docs/ARCHITECTURE.md` (tables and Mermaid diagrams).
- Invariants in `CLAUDE.md`: no code comments, short docstrings, no em dashes or emojis (DEC-010, DEC-012).
- CUDA 12.9.2 toolkit from the verified archive installer, driver component unchecked (DEC-013, EXP-003).
- VS 2022 BuildTools 17.14.45 for an MSVC 14.44 host compiler; a CUDA file compiles and runs (DEC-014, EXP-004).

## What works
- `python -m vla_cu env`: RTX 5060 Laptop, sm_120, 26 SMs, 7.9 GB, torch 2.8.0+cu129.
- `python -m vla_cu probe`: masked vector add, exact match at n=1000 and n=1_000_003.
- `nvcc` V12.9.86 and `compute-sanitizer` 2025.2.1.0 under `...\CUDA\v12.9`; `ncu` 2025.2.1 and `nsys` 2025.1.3 found by the env task.
- CUDA C++ compiles when the shell has called `vcvars64.bat` for VS 2022 (MSVC 14.44).
- git: commits authored as Eddison Pham, pushed to `origin/main`.

## What's broken or missing
- Nothing compiled through `torch.utils.cpp_extension` yet, so that path is unproven.
- `vla_cu/bench_utils.py` does not exist, so nothing can be timed or capped yet.
- `tests/` is empty, so `pytest` reports no tests.
- Triton advanced features (`tl.dot`, TMA, autotuning) untested on this platform.
- No kernel has been profiled, so `ncu` support for sm_120 is unverified in practice.

## Current chunk
None in progress. Starter chunks 1 to 3 remain; all three are unblocked.

## Open questions for the human
- Next: timing utilities (`bench_utils.py`), or straight to the vector add kernel and its benchmark?
- Want a `python -m vla_cu` task that reports the CUDA C++ build settings, or leave builds to the chunks?

## Suggested next options (see `BACKLOG.md`)
1. Timing and the 7 GB cap in `vla_cu/bench_utils.py`, since every later number depends on it.
2. Vector add in Triton with a test and bandwidth benchmark, prediction first.
3. The same vector add in CUDA C++ through `torch.utils.cpp_extension`, to compare against Triton.
