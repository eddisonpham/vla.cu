# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-18
**Last git commit:** see `git log -1` (scaffold commit was 9b5dee6)
**Bootstrap (see `agents/STARTER.md`):** COMPLETE 2026-11-18, commit 9b5dee6
**Session summary:** Part A done: skeleton, `.venv`, `.gitignore`, git workflow documented in `CLAUDE.md` section 6, `STARTER.md` moved into `agents/`. Chunk 0 done: `scripts/env_check.py` written and run; hardware recorded from measurement.

## Done so far
- Skeleton: `vla_cu/`, `kernels/triton/`, `csrc/`, `scripts/`, `tests/`, `bench/`, `docs/`, `agents/`.
- `.venv` (torch 2.8.0+cu129, torchvision, numpy, pytest) + `requirements.txt`.
- `.gitignore` and `.gitattributes`.
- Chunk 0: `scripts/env_check.py`.
- `docs/journal.md` stub, ready for the human's predictions.

## What works
- venv Python sees the GPU: RTX 5060 Laptop GPU, sm_120, 26 SMs, 7.9 GB.
- `python scripts/env_check.py` prints versions, device properties, tool availability.

## What's broken or missing
- **No Triton.** `pip install triton` has no Windows wheel. `triton-windows` 3.9.0.post30 exists; untested against torch 2.8.0+cu129. This blocks Chunk 2 as written.
- **No CUDA toolkit.** No `nvcc`, `nsys`, `ncu`, `compute-sanitizer`, so CUDA C++ chunks cannot build and the CLAUDE.md sanitizer rule cannot be met yet.
- `requirements.txt` was not installed as a set (torch came from the cu129 index directly); harmless for now, revisit when NumPy/pytest versions must match.

## Current chunk
- None in progress. STARTER.md Chunks 1-3 remain.

## Open questions for the human
- Triton route before Chunk 2: try `triton-windows` on native Windows, or add WSL2 (official Triton, second toolchain)?
- Install the CUDA toolkit now (nvcc + compute-sanitizer + Nsight) or when the first CUDA C++ chunk arrives?

## Suggested next options (see `BACKLOG.md`)
1. Chunk 1: `vla_cu/bench_utils.py` (CUDA-event timing + 7 GB budget) - every later chunk uses it.
2. Settle the Triton-on-Windows question so Chunks 2-3 can actually run.
3. Install the CUDA toolkit so kernels can be sanitized and profiled with counters.
