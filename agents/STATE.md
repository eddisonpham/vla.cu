# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-18
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`
**Checkpoint:** 2026-11-18 at `1b19bca`. Bootstrap, Triton verification, and style invariants are done. Nothing has been measured beyond the environment and the probe; no kernel exists yet.

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), `.gitignore`, `.gitattributes`, `requirements.txt`.
- Environment check (`python -m vla_cu env`) and hardware record in `HARDWARE.md`.
- Triton verified on native Windows and sm_120 (`python -m vla_cu probe`), EXP-002.
- Task runner `python -m vla_cu <task>` with tasks `env` and `probe` (DEC-011).
- `README.md`, `docs/ARCHITECTURE.md` (tables and Mermaid diagrams).
- Invariants in `CLAUDE.md`: no code comments, short docstrings, no em dashes or emojis (DEC-010, DEC-012).

## What works
- `python -m vla_cu env`: RTX 5060 Laptop, sm_120, 26 SMs, 7.9 GB, torch 2.8.0+cu129.
- `python -m vla_cu probe`: masked vector add, exact match at n=1000 and n=1_000_003.
- git: commits authored as Eddison Pham, pushed to `origin/main`.

## What's broken or missing
- No CUDA toolkit: no `nvcc`, `nsys`, `ncu`, `compute-sanitizer`. Blocks CUDA C++ and every sanitizer run. Planned: CUDA 12.9 archive installer, not winget's 13.5.
- `vla_cu/bench_utils.py` does not exist, so nothing can be timed or capped yet.
- `tests/` is empty, so `pytest` reports no tests.
- Triton advanced features (`tl.dot`, TMA, autotuning) untested on this platform.

## Current chunk
None in progress. Starter chunks 1 to 3 remain; chunks 2 and 3 are unblocked.

## Open questions for the human
- Install the CUDA 12.9 toolkit now, or wait until the first CUDA C++ kernel?
- Next: timing utilities (`bench_utils.py`), or straight to the vector add kernel and its benchmark?

## Suggested next options (see `BACKLOG.md`)
1. Timing and the 7 GB cap in `vla_cu/bench_utils.py`, since every later number depends on it.
2. Vector add with test and benchmark, with a written prediction first.
3. Install CUDA 12.9 so `compute-sanitizer` and Nsight exist ahead of hand-written kernels.
