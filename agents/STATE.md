# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

Every projection in the chain is now one GEMM over a chunk, the attention mix with rope
is one kernel, and a residual plus its norm is one launch. A 50 token layer went from 500
launches and 23.245 ms to 158 launches and 8.456 ms at depth 16, with every depth still
bit exact. The per-token tail that is left is the pre-attention norm and the final add.

## Closed this pass

- **A chunk's attention and MLP projections are one GEMM each.** 158 launches per layer
  against 500, exact, cosine unchanged to five decimals (EXP-066, EXP-067, DEC-063,
  DEC-065).
- **The attention mix with rope is 20 to 24x the einsums**, 0.034 against 0.868 ms per
  layer at 50 tokens, paired, with the fused path shipping up to 64 tokens (EXP-065,
  DEC-064).
- **A residual and its norm are one launch.** 208 to 158 launches per layer, 10.627 to
  8.456 ms at 50 tokens, cosine identical (EXP-069, DEC-066).
- **Split K was the wrong hypothesis.** 0.57x to 0.75x the single launch on both expert
  shapes, because the reduce is a whole extra launch. The code stays behind `splits`
  with a default of 1 (EXP-068, DEC-067).
- **The int4 GEMM picks its tile from the shape**: 128 wide outputs above 1024 rows,
  64 wide below, and the narrow family stops at 32 rows because a 3 stage 64 row tile
  needs 106496 shared bytes against a 101376 limit (EXP-064).

## What works

- 28 tasks, `pytest -q` 148 passed with vcvars active.
- `scale` prints four stages plus the 50 token block, eager, graph, cosine, relative L2,
  an exactness column and the real launch count per layer.
- `m50` compares the loop, the int4 GEMM and bf16, sweeps nine tiles, and pairs 2 and 4
  way K splits against the single launch.
- `attncross` times both attention paths paired in one process at eight token counts.
- The GPU is thermally throttled in long sessions: 74 to 76 C, 28 to 33 W, and stage 1
  fell from 298 to 154 GB/s in one afternoon. Only paired ratios and within-block
  comparisons are trustworthy now.

## Still open

- **The per-token tail** is the pre-attention norm and the final residual add, one launch
  each per token. A two dimensional norm over the chunk removes both.
- **Small M is unresolved.** The 2 to 8 token columns rose and fell with clocks across
  runs, so the chunk MLP has no honest small-M comparison yet.
- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **Above 64 tokens the einsum mix is back**, 25x slower, covered by one test.
- **The narrow GEMM** sits at 6.3 to 7.1 percent of peak and needs a stream-K schedule,
  not a K split.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: the two dimensional norm, or a stream-K int4 GEMM.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Norm the whole token chunk in one launch and re-run `scale` at 8 and 50.
2. Write the stream-K int4 GEMM the negative split result points at, then repeat `m50`.
3. Pair the chunk MLP against the token loop so the small-M answer stops depending on
   the clock state.
