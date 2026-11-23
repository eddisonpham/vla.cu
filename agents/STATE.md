# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-22
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), environment check, hardware record, task runner.
- Triton verified on native Windows and sm_120, CUDA toolkit and host compiler checked (EXP-002 to EXP-005).
- Precision ladder fixed: fp32 reference, bf16 baseline, int8, int4 (DEC-016).
- torch.compile unusable here, vLLM Linux only (EXP-006).
- Vector add kernel, 7 tests pass, sanitizer clean on one launch (EXP-007).
- Benchmark utilities and vector add bandwidth: 330.9 GB/s, 86.2 percent of 384 at 16M elements (EXP-008).
- Launch overhead measured: 16.7 us from Python, 0.63 us as a graph node (EXP-009).

## What works
- `python -m vla_cu env`, `probe`, `cuda-check`, `bench`, `launch`.
- `.venv/Scripts/python.exe -m pytest -q`: 7 passed, sizes 0 to 1,000,003, bit exact.
- CUDA graph capture of Triton kernels on native Windows, verified working.
- git: commits authored as Eddison Pham; local commits are not always pushed.

## What's broken or missing
- No graph in the actual inference path yet, only in the experiment.
- Only one BLOCK_SIZE measured, so 1024 is a default, not a chosen value.
- The 14 percent gap to spec peak in EXP-008 is still unexplained.
- `compute-sanitizer` around the whole pytest run exceeded 600 s; single launches are clean.
- Reduction, matmul, 4-bit packing, fused GEMV, CUDA C++ port: none started.
- `docs/PRIMER.md` exists untracked, to be deleted on request.

## Current chunk
None in progress. Next is int8 GEMV, the first chunk with a reduction and a scale.
The launch-overhead answer means graphs belong in the design, not as an afterthought.

## Open questions for the human
- int8 GEMV next, or the block sweep now that the launch number explains the 4K row?
- Should the GEMV test suite run both plain and graphed, or keep graphs to the benchmark?

## Suggested next options (see `BACKLOG.md`)
1. int8 GEMV with per-channel scales against a dequantize-then-matmul reference.
2. Graph the vector add benchmark to confirm the real workload sees the 16 us saving.
3. BLOCK_SIZE sweep at 16M, cheap closure on the tuning question.
