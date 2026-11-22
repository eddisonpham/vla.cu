# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-22
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`
**Checkpoint:** 2026-11-18 at `1b19bca`.

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), environment check, hardware record, task runner.
- Triton verified on native Windows and sm_120 (`probe`), CUDA toolkit and host compiler (`cuda-check`), EXP-002 to EXP-005.
- Precision ladder fixed: fp32 reference, bf16 baseline, int8, int4 (DEC-016).
- torch.compile measured unusable here (`triton_key` ImportError); vLLM is Linux only (EXP-006).
- Vector add kernel `kernels/triton/vector_add.py`, 7 tests pass, sanitizer clean on one launch (EXP-007).
- Timing utilities `vla_cu/bench_utils.py`: 7 GB cap, CUDA events, bandwidth math.
- Vector add benchmark as `python -m vla_cu bench`; 86.2 percent of 384 GB/s at 16M elements (EXP-008).

## What works
- `python -m vla_cu env`, `probe`, `cuda-check`, `bench`.
- `.venv/Scripts/python.exe -m pytest -q`: 7 passed, sizes 0 to 1,000,003, bit exact.
- `python -m vla_cu bench`: 4,096 and 1,048,576 and 16,777,216 element rows, torch against Triton.
- git: commits authored as Eddison Pham; local commits are not always pushed.

## What's broken or missing
- Only one BLOCK_SIZE measured, so 1024 is a default, not a chosen value.
- The 14 percent gap to spec peak has no explanation yet: thermals, clocks, or access pattern.
- `compute-sanitizer` around the whole pytest run exceeded 600 s; single launches are clean.
- Reduction, matmul, 4-bit packing, fused GEMV, CUDA C++ port: none started.
- `docs/PRIMER.md` exists untracked, to be deleted on request.

## Current chunk
None in progress. Chunks 0 to 3 of `agents/STARTER.md` are done.
Next candidates: block-size sweep, launch-overhead experiment, int8 GEMV.

## Open questions for the human
- Which comes next: finish tuning the vector add, or move to a kernel that teaches something new?
- Is 86 percent of spec a good enough answer for this laptop, or is the missing 14 percent worth chasing?

## Suggested next options (see `BACKLOG.md`)
1. BLOCK_SIZE sweep at 16M: cheap, and it shows where occupancy stops helping.
2. Launch-overhead experiment: one tiny kernel against 1000 launches, to size a CUDA graph win.
3. int8 GEMV with per-channel scales: first rung that looks like the real project.
