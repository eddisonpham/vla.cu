# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-22
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), environment check, hardware record, task runner.
- Triton verified on Windows and sm_120, toolkit and compiler checked (EXP-002 to EXP-005). torch.compile unusable, vLLM Linux only (EXP-006).
- Vector add kernel and benchmark (EXP-007, EXP-008); launch overhead and CUDA graphs (EXP-009, EXP-011, DEC-018).
- int8 GEMV with per-channel scales, 92.4 percent of peak on a DRAM-bound shape (EXP-010, DEC-019).
- BLOCK_K sweep and split-K: 256 to 1024 within 0.2 percent, split-K 6.53x at 16 rows (EXP-012).
- 4-bit packing with per-group scales, fused dequant GEMV, full ladder matrix (EXP-013, DEC-020).

## What works
- Tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`, `matrix`.
- `.venv/Scripts/python.exe -m pytest -q`: 49 passed in 7 s, awkward sizes and rejection paths included.
- Ladder at 8192 x 16384: fp32 1.614 ms, bf16 0.803, int8 0.375, int4 0.263, relative L2 errors 0, 2.9e-3, 9.4e-3, 1.07e-1.
- `compute-sanitizer`: 0 errors on the int4 tiled kernel and the int8 split kernel; racecheck 0 hazards on atomics.
- git: commits authored as Eddison Pham; local commits are not always pushed.

## What's broken or missing
- No end-to-end model: kernels are measured on synthetic tensors, never wired into a policy or a custom op.
- The int4 group scale is gathered per byte instead of hoisted once per 64 weights, which is why int4 sits at 75 percent of DRAM peak against 93 for int8.
- Real SmolVLA layer shapes are still unknown here, so every benchmark uses round numbers.
- CUDA C++ port not started, though the toolkit and host compiler are verified.
- Run-to-run variance is unmeasured; single runs are reported.
- `compute-sanitizer` around the whole pytest run exceeds 600 s; single launches are clean.
- `docs/PRIMER.md` exists untracked, to be deleted on request.

## Current chunk
None in progress. The measured ladder is complete for Triton at batch 1.
The two obvious quality gaps are int4 kernel efficiency and model-level integration.

## Open questions for the human
- Optimize int4 further (hoisted scale, shared-memory x tile), or wire the kernels into a real forward pass first?
- Which SmolVLA layer shapes should the benchmarks adopt?

## Suggested next options (see `BACKLOG.md`)
1. Hoist the int4 group scale and retune ROWS, aiming for parity with int8 in GB/s.
2. Register the GEMV as a `torch.library` custom op and run one real layer end to end.
3. CUDA C++ port of the int4 GEMV, the comparison the repo exists for.
