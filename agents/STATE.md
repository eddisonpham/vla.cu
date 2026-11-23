# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** this commit adds `agents/OPTIMIZATIONS.md` on top of f88dcf4
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Done so far
- Skeleton, `.venv`, environment check, hardware record, task runner.
- Triton on Windows and sm_120 (EXP-002 to EXP-005). torch.compile unusable, vLLM Linux only (EXP-006).
- Vector add, launch overhead, CUDA graphs (EXP-007 to EXP-009, EXP-011).
- int8 GEMV at 92.4 percent of peak (EXP-010, DEC-019); block sweep and split-K (EXP-012).
- 4-bit format, fused GEMV, ladder matrix (EXP-013, DEC-020); group-major reduction (EXP-014, DEC-021).
- Torch custom ops and one MLP block end to end (EXP-015, DEC-022).
- CUDA C++ port of the int4 GEMV compared against Triton (EXP-016, DEC-023).
- **Optimization map and experiment queue written (2026-11-23): `agents/OPTIMIZATIONS.md`.** Ten
  ranked hypotheses with refutation conditions, no experiment run, no new number measured.

## What works
- Tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`, `matrix`, `layer`, `cuda`.
- `.venv/Scripts/python.exe -m pytest -q`: 60 passed, inside a vcvars shell.
- Ladder at 8192 x 16384: fp32 1.587 ms, bf16 0.811, int8 0.400, int4 0.230 (85.5 percent, 6.89x fp32).
- MLP block 960 -> 2560 -> 960: fp32 0.072 ms, fused 0.085 ms, cosine 0.99346.

## What's broken or missing
- The attainable DRAM ceiling on this part is unmeasured, so every "% of peak" divides by a spec
  sheet. No run-to-run spread has been recorded either: int8 moved 336.1 to 354.7 GB/s between sessions.
- No profiler data at all. `compute-sanitizer` cannot instrument the extension (EXP-016); `ncu` on it is untested.
- No full model, and the MLP block is dispatch bound at 0.085 ms against fp32's 0.072.
- SmolVLA widths (960, 2560) are still assumptions (DEC-024). Paper and config facts that are
  verified: 450M parameters, about 100M expert, 16 VLM layers, expert ratio 0.75, chunk 50, 10 steps.
- `docs/PRIMER.md` is untracked and stays that way until the human says delete it.

## Current chunk
None in progress. The next chunk is whichever experiment the human picks from section 5 of
`agents/OPTIMIZATIONS.md`.

## Open questions for the human
- Is the next chunk a measurement (EXP-017 ceiling, EXP-019 profiler) or a change (EXP-018 graph, EXP-021 fp16 scales)?
- Is a real SmolVLA checkpoint available to download in a future session? It changes EXP-023 and EXP-026.

## Suggested next options (see `agents/OPTIMIZATIONS.md` section 5)
1. EXP-017: measure the attainable streaming ceiling and the run-to-run spread. Decides whether any
   micro-optimization is worth doing.
2. EXP-018: graph the MLP block, target 0.085 to about 0.055 ms, the cheapest real win on the board.
3. EXP-023: read `modeling_smolvla.py` and settle whether the action expert is M = 50. If it is, the
   tensor-core question reopens and the model plan changes.
