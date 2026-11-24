# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A token chunk now runs as a chunk: the four attention projections are int8 GEMMs above
one token, the causal mix with rope is one Triton kernel up to 64 tokens, and the
projections that remain are the MLP pair. At 50 tokens a layer issues 305 launches
instead of 500, and the graphed chain is 18.946 ms at depth 16 instead of 23.245.

## Closed this pass

- **The attention mix with rope is one kernel, 20 to 24x the einsums.** 0.034 ms against
  0.868 per layer at 50 tokens, paired in one process, cosine 1.000000, max diff 8.34e-07
  (EXP-065, DEC-064). The threshold moved from 4 to 64 tokens.
- **Chunk projections run as one int8 GEMM each.** Depth 16 graph at 2, 3, 4, 8, 50
  tokens: 1.172, 1.419, 1.639, 2.712, 18.946 ms against 2.016, 2.585, 3.186, 5.934,
  23.245. Exact everywhere, cosine unchanged (EXP-066, DEC-063).
- **The int4 GEMM is pipelined and picks its tile from the shape.** 0.048 and 0.054 ms at
  M = 50 against 0.054 and 0.094, 8.7 and 6.3 percent of peak, within 0.008 ms of the
  best swept row (EXP-064).
- **The causal mask, the score layout, the epsilon and the rope constants** are settled
  (DEC-056, DEC-059, DEC-060, EXP-057, EXP-058).
- **The production survey** maps 11 systems onto this repo in `docs/PRODUCTION.md`.

## What works

- 28 tasks, `pytest -q` 142 passed with vcvars active.
- `scale` prints four stages plus the 50 token block with eager, graph, cosine, relative
  L2 and an exactness column, and now prints its real launch count per layer.
- `m50` compares the GEMV loop, the int4 GEMM and cuBLAS bf16, then sweeps nine tiles and
  marks the menu's pick.
- `attncross` times both attention paths paired in one process at eight token counts.
- `plateau` median of 5: staged Triton 63.6 to 73.1, wide ieee 11.3 to 16.1, wide tf32
  28.3, native unroll 86.8 to 88.0 percent at 4096 x 8192 across runs.

## Still open

- **The MLP is still two GEMVs per token.** At 50 tokens that is about 100 of the 305
  launches per layer, and it needs the same GEMM dispatch (EXP-066).
- **The narrow expert shape is parallelism bound.** 720 x 1920 runs 12 to 24 programs
  against 26 SMs at 6.3 percent of peak, so it wants a K split (EXP-064).
- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **Run drift reaches 25 percent** on eager paths, so only numbers inside one output
  block compare. `attncross` and `time_pair` exist because of it.
- **Above 64 tokens the einsums are back**, 25x slower, and only one test covers them.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: MLP GEMM dispatch, or the K split for the narrow shape.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Wire the MLP up and down projections to the GEMMs, then re-run `scale` at 8 and 50.
2. Split K across programs in the int4 GEMM for the 720-wide shape and repeat `m50`.
3. Fuse the two norms and the residual adds so the per-token launch tail shrinks.
