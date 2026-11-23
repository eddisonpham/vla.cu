# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** 2fb487f fix: validate kernel inputs and gate cuda tests on msvc
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Done so far
- Skeleton, `.venv`, environment check, task runner, hardware record.
- Triton on Windows and sm_120; torch.compile unusable, vLLM Linux only (EXP-006).
- Vector add, launch overhead, CUDA graphs (EXP-007 to EXP-011).
- int8 GEMV at 92.4 percent of peak (EXP-010); split-K and block sweep (EXP-012).
- 4-bit format, fused GEMV, ladder matrix, group-major reduction (EXP-013, EXP-014).
- Custom ops, one MLP block, CUDA C++ port (EXP-015, EXP-016).
- Ten experiments in one session (EXP-017 to EXP-026), fast path (EXP-027, EXP-028),
  GEMM, attention, rotation, clocks (EXP-029 to EXP-032).
- Bug-fix pass (EXP-033): rotation order guard, grouped wrapper validation, power-of-two
  block size, cuda test gate on `cl`, `CalledProcessError` wrapping, bf16 byte count.

## What works
- Tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`, `matrix`, `layer`,
  `cuda`, `stream`, `grouped`, `m50`, `rotate`, `fastpath`, `crossover`, `attn`.
- `pytest -q`: 92 passed inside a vcvars shell; 69 passed, 23 skipped outside it (no `cl`).
- Fuzz (EXP-033): 175 int4 GEMV shapes, all three CUDA variants bit identical, 384 GEMM shapes,
  zero mismatches against float64.
- int4 GEMV: CUDA streaming variant 0.198 to 0.220 ms at 8192 x 16384 (89 to 99 percent).
- Fast path: one MLP block 0.015 ms graphed against fp32's 0.069; four blocks 0.051 ms; attention
  block 0.011 ms, cosine 0.98804; GEMM wins from M = 2, 25 to 51x at M = 50.

## What's broken or missing
- **Measurement is the weak point.** One probe configuration returned 164 to 356 GB/s in five
  repeats. Clocks are not pinned; a difference under 6 percent is not a result (DEC-029, DEC-035).
- **No profiler.** `ncu` fails with `ERR_NVGPUCTRPERM` (EXP-019) and `compute-sanitizer` cannot
  instrument the extension (EXP-016). The CUDA variants still have no sanitizer pass.
- The custom op path costs 0.109 to 0.224 ms on a block whose kernels take 0.053 (DEC-025).
- The CUDA baseline regression from 0.240 to 0.265 to 0.476 ms after the `dot_eight` refactor is
  still unexplained (EXP-022).
- No full model and no checkpoint, so every accuracy number is on random Gaussian weights.
- Four stacked blocks give cosine 0.975 on random weights, under the 0.98 gate.
- Group 64 does not divide the expert hidden width 720 (DEC-030), decision open.
- Two input gaps left on purpose (EXP-033): `block_n` is not checked for a power of two, and the
  grouped segment-run invariant needs a device sync to prove.
- `docs/PRIMER.md` is untracked and stays that way until the human says delete it.

## Current chunk
None in progress. Last chunk: the bug-fix pass in `2fb487f`.

## Open questions for the human
- Can you grant administrator rights for `nvidia-smi -lgc` and `ncu`?
- Is a SmolVLA checkpoint download acceptable in a future session?

## Suggested next options (see `agents/OPTIMIZATIONS.md` section 11)
1. An int8 grouped kernel: the attention block reads 0.98804 against the MLP's 0.99451.
2. A real checkpoint: three layers, cosine per layer, with and without rotation.
3. BLOCK_M = 48 for the GEMM plus a CUDA port, which takes the expert from 0.039 to about 0.020 ms.
