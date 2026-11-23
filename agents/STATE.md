# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** this commit logs EXP-017 to EXP-026 on top of 97ad2f0
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Done so far
- Skeleton, `.venv`, environment check, task runner, hardware record.
- Triton on Windows and sm_120; torch.compile unusable, vLLM Linux only (EXP-006).
- Vector add, launch overhead, CUDA graphs (EXP-007 to EXP-011).
- int8 GEMV at 92.4 percent of peak (EXP-010); split-K and block sweep (EXP-012).
- 4-bit format, fused GEMV, ladder matrix, group-major reduction (EXP-013, EXP-014).
- Custom ops, one MLP block, CUDA C++ port (EXP-015, EXP-016).
- **Ten experiments in one session (EXP-017 to EXP-026)**: ceiling probe, block graph, profiler
  attempt, cache hints, fp16 scales, CUDA variants, M = 50 bake-off, grouped launch, rotation.

## What works
- Tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`, `matrix`, `layer`,
  `cuda`, `stream`, `grouped`, `m50`, `rotate`.
- `pytest -q`: 62 passed, inside a vcvars shell. Triton tests pass without it.
- Fastest int4 GEMV: the CUDA streaming variant, 0.198 to 0.220 ms at 8192 x 16384 (89 to 99 percent
  of 384), 9 to 10 percent ahead of Triton (EXP-022).
- MLP block as a graph: 0.018 to 0.019 ms against 0.053 ms raw and 0.073 to 0.082 fp32 (EXP-018).
- Grouped launch: 2.73x at 960 x 960, bit identical (EXP-024).

## What's broken or missing
- **Measurement is the weak point.** One probe configuration returned 164 to 356 GB/s in five
  repeats. The harness warms the GPU but does not pin clocks; a difference under 6 percent is not a
  result (DEC-029).
- **No profiler at all.** `ncu` fails with `ERR_NVGPUCTRPERM` (EXP-019) and `compute-sanitizer`
  cannot instrument the extension (EXP-016). The new CUDA variants have no sanitizer pass.
- The custom op path costs 0.109 to 0.224 ms on a block whose kernels take 0.053, so it is not on any
  speed path (DEC-025).
- The CUDA baseline regressed from 0.240 (EXP-016) to 0.265 to 0.476 ms after the `dot_eight`
  refactor, unexplained (EXP-022).
- No full model, no checkpoint, and the expert needs a GEMM kernel that does not exist yet (DEC-028).
- Group 64 does not divide the expert hidden width 720 (DEC-030), decision open.
- `docs/PRIMER.md` is untracked and stays that way until the human says delete it.

## Current chunk
None in progress. The session that produced this file finished EXP-017 to EXP-026 and its logs.

## Open questions for the human
- Can you grant administrator rights for `nvidia-smi -lgc` and `ncu`? Both would turn inference into
  measurement, and both need elevation once.
- Is a SmolVLA checkpoint download acceptable in a future session? EXP-023 in particular is only
  half answered without it.

## Suggested next options (see `agents/OPTIMIZATIONS.md` section 9)
1. Pin clocks, then repeat EXP-017, EXP-020, EXP-021 and EXP-022. Settles three knobs at once.
2. Wire the three measured wins into a default path: streaming kernel, grouped launch, block graph.
3. Write the int4 GEMM at M = 50 with `tl.dot` and sweep M from 1 to 64 for the crossover.
