# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-22
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`
**Checkpoint:** 2026-11-18 at `1b19bca`.

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), `.gitignore`, `.gitattributes`, `requirements.txt`.
- Environment check (`python -m vla_cu env`) and hardware record in `HARDWARE.md`.
- Triton verified on native Windows and sm_120 (`python -m vla_cu probe`), EXP-002.
- Task runner `python -m vla_cu <task>` with tasks `env`, `probe`, `cuda-check` (DEC-011, DEC-015).
- `README.md`, `docs/ARCHITECTURE.md` (tables and Mermaid diagrams).
- Invariants in `CLAUDE.md`: no code comments, short docstrings, no em dashes or emojis (DEC-010, DEC-012).
- CUDA 12.9.2 toolkit, VS 2022 BuildTools for MSVC 14.44, build prerequisites inspectable (EXP-003 to EXP-005).
- Precision ladder fixed: fp32 reference, bf16 baseline, int8, int4 (DEC-016).
- torch.compile measured unusable here (`triton_key` ImportError); vLLM is Linux only (EXP-006).
- First project kernel: vector add in `kernels/triton/vector_add.py`, 7 tests pass (EXP-007).

## What works
- `python -m vla_cu env`: RTX 5060 Laptop, sm_120, 26 SMs, 7.9 GB, torch 2.8.0+cu129.
- `python -m vla_cu probe`: masked vector add, exact match at n=1000 and n=1_000_003.
- `python -m vla_cu cuda-check`: nvcc 12.9, both MSVC toolsets, and a verdict.
- `.venv/Scripts/python.exe -m pytest -q`: 7 passed, sizes 0, 1, 1023, 1024, 1025, 1_000_003, bit exact.
- `compute-sanitizer` on one 1025-element launch of the vector add: `ERROR SUMMARY: 0 errors`.
- git: commits authored as Eddison Pham; local commits not always pushed.

## What's broken or missing
- No benchmark and no timing utility yet, so the vector add has no measured GB/s.
- `compute-sanitizer` around the whole pytest run did not finish within 600 s, while a single launch is clean. Batch instrumentation is unexplained, not broken.
- `vla_cu/bench_utils.py` does not exist, so the 7 GB cap is still enforced nowhere.
- Reduction, matmul, 4-bit packing, fused GEMV, CUDA C++ port: none started.
- Triton advanced features (`tl.dot`, TMA, autotuning) untested on this platform.

## Current chunk
Vector add kernel plus tests: done and committed. Next is the benchmark, which needs a
prediction first, and the timing utility folded out of it.
Precision scope fixed: fp32 reference, bf16 baseline, int8, int4 (DEC-016).

## Open questions for the human
- Prediction for the benchmark: what fraction of 384 GB/s do you expect, and will Triton beat torch?
- Does `compute-sanitizer` matter enough here to keep chasing, or is it acceptable to leave it pending?

## Suggested next options (see `BACKLOG.md`)
1. Vector add benchmark plus `bench_utils.py`: warmup, CUDA events, GB/s against 384.
2. Launch-overhead experiment: one tiny kernel against 1000 launches, to size a CUDA graph win.
3. int8 GEMV with per-channel scales, the first rung that looks like the real project.
