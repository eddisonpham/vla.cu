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
- **Fast path brought up (EXP-027, EXP-028)**: CUDA int8 streaming kernel, both kernels launched
  on the torch stream so they capture, one block at 0.015 ms against fp32's 0.069.
- **GEMM, attention, rotation, clocks (EXP-029 to EXP-032)**: int4 GEMM with `tl.dot` wins from
  M = 2 and is 25 to 51x ahead at M = 50; grouped attention block 0.011 ms graphed; rotation is
  neutral on Gaussian weights and 38 percent slower, so opt-in; clock pinning refused, paired
  ratios instead (spread 5 percent against a 2.4x swing between runs).

## What works
- Tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`, `matrix`, `layer`,
  `cuda`, `stream`, `grouped`, `m50`, `rotate`, `fastpath`, `crossover`, `attn`.
- `pytest -q`: 79 passed, inside a vcvars shell. Triton tests pass without it.
- int4 GEMM: one launch wins from M = 2, 25 to 51x against M GEMV calls at M = 50, cosine 1.00000
  against the fp32 answer on every shape including a 17 x 128 tail (EXP-029).
- Attention block: grouped qkv plus o in two launches, 0.011 ms graphed against fp32's 0.052,
  bit exact, cosine 0.98804 because every matrix there is int4 (EXP-031).
- Fast path: one block 0.015 ms (4.6x fp32), four-block stack 0.054 and 0.134 ms (6.57x and 2.72x),
  bit exact against the eager chain, cosine 0.99393 per block (EXP-028).
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
- Four stacked blocks give cosine 0.975 on random weights, under the 0.98 gate. Zero blocks have been
  run against real weights, so the gate is untested where it matters.
- Group 64 does not divide the expert hidden width 720 (DEC-030), decision open.
- `docs/PRIMER.md` is untracked and stays that way until the human says delete it.

## Current chunk
None in progress. The session that produced this file finished EXP-017 to EXP-026 and its logs.

## Open questions for the human
- Can you grant administrator rights for `nvidia-smi -lgc` and `ncu`? Both would turn inference into
  measurement, and both need elevation once.
- Is a SmolVLA checkpoint download acceptable in a future session? EXP-023 in particular is only
  half answered without it.

## Suggested next options (see `agents/OPTIMIZATIONS.md` section 11)
1. An int8 grouped kernel: the attention block reads 0.98804 against the MLP's 0.99451, and the
   difference is one int8 projection.
2. A real checkpoint: three layers, cosine per layer, with and without rotation.
3. BLOCK_M = 48 for the GEMM plus a CUDA port, which takes the expert from 0.039 to about 0.020 ms.
