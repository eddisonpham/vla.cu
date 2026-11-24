# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is closed except for profiling on the large shape, which needs the
human's elevated shell. The expert stack runs from two to eight tokens with the rope
parameters read from the model config, and the wide Triton mapping has now been measured
on both the load path and the reduction path, losing twice.

## Closed this pass

- **The chain runs at 2, 3, 4 and 8 tokens.** Exact at every depth, cosine holds at
  0.99622 to 0.99648 at depth 16, graphed 2.866 to 7.003 ms for 16 layers (EXP-055).
- **Theta is 100000, and the pairing is the Llama half split.** Read from the SmolVLM2
  text config, so the old adjacent pairing was wrong (EXP-055, DEC-056, RESEARCH).
- **The dot reduction lost.** 0.422 against 0.077 ms at 4096 x 8192, 0.18x, with 264 FFMA
  and 83 LDS against 18, kept as the alternative (EXP-056, DEC-058).
- **Every kernel has a correctness check and a sanitizer pass.** int4 in EXP-044, int8 in
  EXP-046, the wide Triton kernel in EXP-052, 0 errors each.
- **The Triton gap is a reduction fact, not a load width fact.** Both read weights 4 or 8
  bytes at a time; native has no SMEM, no barrier, five shuffles at the end (EXP-050).

## What works

- 27 tasks, `pytest -q` 124 passed with vcvars active, 96 passed and 23 skipped without.
- `plateau` median of 5: staged Triton 63.6 to 73.1, wide 11.7 to 16.1, native unroll 87.4
  to 88.0 percent at 4096 x 8192 across runs on different thermal states.
- `scale` prints four stages, the last one once per token count, with eager, graph,
  cosine, relative L2 and an exactness column.
- `sass` counts three Triton mappings and three native variants in one table.
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.

## Still open

- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **Attention is not causal.** Token 0 sees the last token, which the expert does not.
- **The RMSNorm epsilon was not compared** against the config value of 1e-05.
- **The wide mapping** has no 16 byte weight load and now a slower reduction; nothing
  there is worth shipping yet (EXP-052, EXP-054, EXP-056).
- **The largest shape is not trustworthy.** 16384 x 32768 spans 0.9 to 1.5 ms across
  repeats (EXP-047, EXP-055).
- **Globe drift between runs** reaches 25 percent on eager paths, so only numbers inside
  one output block can be compared.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: a causal mask, or the rms epsilon comparison.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Add a causal mask to the token chain, which is what the expert actually runs.
2. Compare `rms_norm_eps` 1e-05 against the kernel's EPS and correct the reference.
3. Measure the wide dot at `tf32` to separate precision from layout.
