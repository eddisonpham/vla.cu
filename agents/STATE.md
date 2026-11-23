# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-22
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), environment check, hardware record, task runner.
- Triton verified on native Windows and sm_120, toolkit and host compiler checked (EXP-002 to EXP-005).
- Precision ladder: fp32 reference, bf16 baseline, int8, int4 (DEC-016).
- torch.compile unusable here, vLLM Linux only (EXP-006).
- Vector add kernel, 7 tests, sanitizer clean (EXP-007); 86.2 percent of peak at 16M (EXP-008).
- Launch overhead: 16.71 us from Python, 0.63 us per graph node (EXP-009, DEC-018).
- int8 GEMV with per-channel scales, 9 tests, sanitizer clean, 92.4 percent of peak at 8192 x 16384 (EXP-010).
- Graphed vector add and a two-kernel GEMV chain, both bit exact (EXP-011).

## What works
- Tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`.
- `.venv/Scripts/python.exe -m pytest -q`: 16 passed, awkward sizes included.
- CUDA graph capture of Triton kernels on native Windows, verified exact against the plain path.
- git: commits authored as Eddison Pham; local commits are not always pushed.

## What's broken or missing
- 4-bit packing and the fused dequant GEMV: not started, and it is the project target.
- `BLOCK_K` fixed at 1024, never swept. Split-K not implemented for layers with few rows (DEC-019).
- No real SmolVLA layer shapes yet, only round numbers.
- The naive GEMV baseline is torch's int8 cast path, which measured about 171 GB/s, so the 18.6x is partly a slow baseline.
- The 14 percent gap to spec peak in EXP-008 remains unexplained.
- `compute-sanitizer` around the whole pytest run exceeds 600 s; single launches are clean.
- `docs/PRIMER.md` exists untracked, to be deleted on request.

## Current chunk
None in progress. Three chunks delivered: int8 GEMV, graphed vector add, graphed two-kernel chain.
The next real milestone is int4 packing plus the fused dequant GEMV.

## Open questions for the human
- int4 packing next, or a short BLOCK_K sweep to close the tuning question on a kernel that matters?
- Extract SmolVLA layer shapes now, so benchmarks stop using round numbers?

## Suggested next options (see `BACKLOG.md`)
1. Pack 4-bit weights with per-group scales and a reference dequantize, no kernel yet.
2. BLOCK_K sweep and split-K for the int8 GEMV, with a measured table.
3. Pull real layer shapes from SmolVLA and rerun the ladder on them.
