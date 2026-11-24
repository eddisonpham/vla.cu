# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A layer is 11 launches at any token count. The paired bench settled small M: at 2 tokens
the chunk kernels are 2.42x (norm) to 2.57x (add-norm) and 1.97x (MLP) faster than the
per-token loops they replaced, at 50 tokens 61.5x and 48.7x. A graph is 3.18x to 5.66x
eager at 2 to 8 tokens and 1.41x to 1.86x at 50, at every depth 1 to 16.

## Closed this pass

- **Small M has a trustworthy answer.** `pairedtail` alternates the old loop and the new
  chunk inside one process, so the throttling cancels: rows are bit exact, the MLP within
  1e-3 (EXP-072, DEC-071).
- **The balanced stream K is parity.** 26 programs against 15 tiles bought 4 percent on
  the 1920 output shape and lost 11 percent on the 720 output one, bit exact, so it stays
  off (EXP-073, DEC-070).
- **The graph win holds at every depth.** `scale` sweeps 1 to 16 at all five token counts,
  exact at all 80 pairs (EXP-074, DEC-072).

## What works

- 30 tasks, `pytest -q` 157 passed with vcvars active, `style` at zero.
- `pairedtail`, `m50`, `attncross` and `scale` all run from `python -m vla_cu <task>`.
- memcheck and racecheck read 0 errors or hazards on the GEMM and norm tests.

## Still open

- **The fixed cost at M = 50.** Every int4 path lands at 0.048 to 0.057 ms for 1.3 to
  1.7 MB (26 to 37 GB/s) whatever the program count, while cuBLAS bf16 takes 0.021 to
  0.027 ms. Grid balance is not the limit, so the next question is what is.
- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **Above 64 tokens the einsum mix is back**, 25x slower, covered by one test.
- **The 720 output GEMM** is at 6.8 percent of peak with 60 tiles for 26 SMs, so it is
  over-subscribed, not under-subscribed.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the floor is.
2. Pick the next question: the launch floor, or one persistent kernel for the layer.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Profile the 0.048 ms int4 GEMM at 50 tokens and 1920 outputs.
2. One persistent kernel for the whole 11 launch layer, so the floor is paid once.
3. Give the int8 GEMM its own tile sweep, since it still runs the int4 menu.
