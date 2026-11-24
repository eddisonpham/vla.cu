# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A layer is now 11 launches at any token count: four attention GEMMs, one fused attention
mix with rope, one MLP up, one SiLU, one MLP down, one norm, one fused add-norm, one add.
At depth 16 the 50 token chain is 4.468 ms graphed and 7.561 ms eager, down from 23.245
and 360.220 at the start of the session, and every token count is still bit exact.

## Closed this pass

- **The residual tail is row kernels.** Norm, add and add-norm each run one program per
  row, so the chunk is one launch and a single token is the same kernel with one program.
  158 launches to 11 per layer, 8.456 to 4.468 ms at 50 tokens (EXP-070, DEC-068).
- **Stream K recombines in its own launch.** Partials are summed by the last arrival
  through an acquire release counter, so the reduce launch is gone: the split went from
  0.57x to 0.86x to 1.15x, still parity at best, so it stays off (EXP-071, DEC-069).
- **Every projection in a chunk is one GEMM**: attention q, k, v, o and both MLP
  matrices, with the intra-token tail no longer scaling (EXP-066, EXP-067).
- **The attention mix with rope is one kernel** for up to 64 tokens, 20 to 24x the einsums
  (EXP-065, DEC-064).
- **The int4 GEMM picks its tile from the shape** and an unswept 3 stage tile that needed
  106496 shared bytes was removed from the menu (EXP-064).

## What works

- 29 tasks, `pytest -q` 152 passed with vcvars active.
- `scale` prints four stages plus the 50 token block, with eager, graph, cosine, relative
  L2, exactness and the real launch count per layer.
- `m50` compares the loop, the int4 GEMM and bf16, sweeps nine tiles, checks that a split
  repeats bit for bit, and pairs 2 and 4 way splits against the single launch.
- `attncross` times both attention paths paired at eight token counts.
- memcheck and racecheck both read 0 errors or hazards on the GEMM and norm tests.
- The GPU throttles: 74 to 76 C and 28 to 50 W through the afternoon, stage 1 between 145
  and 298 GB/s. Only paired ratios and within-block comparisons are trustworthy.

## Still open

- **Small M has no clean answer.** The 2 to 8 token columns moved with clocks across runs,
  so the chunk MLP and the row kernels have never been measured against their predecessors
  in one process. That is a paired bench, not a new kernel.
- **The narrow GEMM** sits at 7.1 percent of peak; only a true stream-K (balanced grid,
  boundary handoff) has upside left there.
- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **Above 64 tokens the einsum mix is back**, 25x slower, covered by one test.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: the paired small-M bench, or a true stream-K GEMM.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Pair the chunk MLP and the row kernels against their predecessors in one process.
2. Write a true stream-K int4 GEMM with a balanced grid and repeat `m50`.
3. Take the 11 launch layer to a graph and compare it against the eager path across depths.
