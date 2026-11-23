# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`; the campaign ends at the packaging pass
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`, pushed

## Campaign

Five improvement cycles are complete, EXP-034 to EXP-039. The temporary section 8 in
`CLAUDE.md` was removed after cycle 4, and `CLAUDE.md` matches its pre-campaign state.

## What works

- 22 tasks: `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`,
  `matrix`, `layer`, `cuda`, `stream`, `grouped`, `m50`, `rotate`, `fastpath`,
  `crossover`, `attn`, `block`, `fuzz`, `bundle`, `style`.
- `pytest -q`: 96 passed, 23 skipped outside vcvars; 119 passed inside it with both
  extensions rebuilt (EXP-039).
- One shared validator fronts all seven wrappers, EXP-034.
- int8 grouped attention reads 0.99994 at 0.019 ms graphed (EXP-035).
- Four action blocks at int8 everywhere read 0.99951 cosine at 0.140 ms, 3.94x the
  fp32 stack and 64.4 percent of peak (EXP-036).
- `fuzz`: 160 seeded cases, zero failures (EXP-037).
- `bundle`: bit-exact round trip, 5.81x smaller than fp32 at int4, 5.68 ms to load
  four blocks (EXP-038).
- `style`: zero violations, and an editable install works from outside the repo
  (EXP-039). CI runs the style gate and a byte-compile.

## What's broken or missing

- **Measurement is the weak point.** Clocks are not pinned; a difference under 6 percent
  is not a result (DEC-029, DEC-035).
- No profiler: `ncu` is denied (`ERR_NVGPUCTRPERM`), `compute-sanitizer` cannot instrument
  the extension (EXP-016).
- The CUDA baseline regression from 0.240 to 0.265 to 0.476 ms after `dot_eight` is still
  unexplained (EXP-022).
- No checkpoint, so every accuracy number is on random Gaussian weights.
- Group 64 does not divide the expert hidden width 720 (DEC-030), decision open.
- CI has no GPU runner, so the correctness suites are a manual gate.
- `docs/PRIMER.md` is untracked and stays that way until the human says delete it.

## Open questions for the human

- Can you grant administrator rights for `nvidia-smi -lgc` and `ncu`?
- Is a SmolVLA checkpoint download acceptable in a future session?

## Best next options

1. Convert a real checkpoint: three layers, cosine per layer, with and without rotation.
2. Add the missing rung from EXP-036, int4 attention with int8 down, to separate the
   two error terms cleanly.
3. A GPU CI runner, so `fuzz` and the native suite run on every push.
