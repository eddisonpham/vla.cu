# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Done so far
- Skeleton, `.venv`, environment check, hardware record, task runner.
- Triton verified on Windows and sm_120 (EXP-002 to EXP-005). torch.compile unusable, vLLM Linux only (EXP-006).
- Vector add and launch overhead (EXP-007 to EXP-009); CUDA graphs on a two-kernel chain (EXP-011, DEC-018).
- int8 GEMV at 92.4 percent of peak (EXP-010, DEC-019); block sweep and split-K (EXP-012).
- 4-bit format, fused GEMV, full ladder matrix (EXP-013, DEC-020); group-major reduction (EXP-014, DEC-021).
- Torch custom ops and one MLP block end to end (EXP-015, DEC-022).
- CUDA C++ port of the int4 GEMV, compared against Triton (EXP-016, DEC-023).

## What works
- Tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`, `matrix`, `layer`, `cuda`.
- `.venv/Scripts/python.exe -m pytest -q`: 60 passed, run inside a vcvars shell.
- Ladder at 8192 x 16384: fp32 1.587 ms, bf16 0.811, int8 0.400, int4 0.230 (85.5 percent of peak, 6.89x fp32).
- MLP block 960 -> 2560 -> 960: fp32 0.072 ms, fused 0.085 ms, cosine 0.99346.
- Triton kernels are sanitizer clean; the CUDA kernel is not (tooling failure, EXP-016).

## What's broken or missing
- No full model: the MLP block is two matrices, not an attention stack or a policy.
- The fused path is slower than fp32 on that block because it fits in L2 and pays dispatch. Only above L2 does fusion win.
- CUDA kernel has no sanitizer pass: `compute-sanitizer` cannot instrument the extension launch.
- The CUDA build needs a vcvars shell; outside it torch invalidates the cache.
- Real SmolVLA layer shapes are still assumed, not extracted from a checkpoint (DEC-024).
- Run-to-run variance is unmeasured; single runs are reported. int8 moved 0.400 to 0.375 ms between sessions.
- `compute-sanitizer` around the whole pytest run still exceeds 600 s.
- `docs/PRIMER.md` exists untracked, to be deleted on request.

## Current chunk
None in progress. All three requested follow-ups are implemented, measured and documented.

## Open questions for the human
- Next milestone: a CUDA-graphed block (cheap, fixes the dispatch loss), or a real checkpoint forward pass?
- Does the cosine 0.9935 at block level clear the 0.98 model gate once a real checkpoint is used?

## Suggested next options (see `BACKLOG.md`)
1. Graph the MLP block so dispatch stops dominating, then re-run `python -m vla_cu layer`.
2. `ncu` both int4 kernels to explain the 4 percent Triton lead and the remaining 15 percent of peak.
3. Extract real SmolVLA shapes and rerun the matrix on them.
