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
- CUDA 12.9.2 toolkit installed from the verified archive installer, driver component unchecked (DEC-013, EXP-003).

## What works
- `python -m vla_cu env`: RTX 5060 Laptop, sm_120, 26 SMs, 7.9 GB, torch 2.8.0+cu129.
- `python -m vla_cu probe`: masked vector add, exact match at n=1000 and n=1_000_003.
- `nvcc` V12.9.86 and `compute-sanitizer` 2025.2.1.0, both under `...\CUDA\v12.9`.
- `ncu` 2025.2.1.0 and `nsys` 2025.1.3 installed under `C:\Program Files\NVIDIA Corporation`, off PATH.
- git: commits authored as Eddison Pham, pushed to `origin/main`.

## What's broken or missing
- CUDA C++ compilation is blocked: `nvcc` 12.9 rejects the only installed MSVC (14.50.35717, VS 18 BuildTools) with `unsupported Microsoft Visual Studio version`. Needs an MSVC 2017-2022 toolchain.
- `shutil.which` cannot see `ncu` or `nsys`, so `python -m vla_cu env` still prints NOT FOUND for them.
- `vla_cu/bench_utils.py` does not exist, so nothing can be timed or capped yet.
- `tests/` is empty, so `pytest` reports no tests.
- Triton advanced features (`tl.dot`, TMA, autotuning) untested on this platform.
- Profiling is unproven end to end: no kernel exists to profile yet.

## Current chunk
None in progress. Starter chunks 1 to 3 remain; all three are unblocked.

## Open questions for the human
- Add VS 2022 Build Tools for a supported host compiler, or stay Triton-only for now?
- Next: timing utilities (`bench_utils.py`), or straight to the vector add kernel and its benchmark?

## Suggested next options (see `BACKLOG.md`)
1. Timing and the 7 GB cap in `vla_cu/bench_utils.py`, since every later number depends on it.
2. Vector add with test and benchmark, with a written prediction first.
3. Fix tool discovery so `env` reports `ncu` and `nsys` by absolute path.
