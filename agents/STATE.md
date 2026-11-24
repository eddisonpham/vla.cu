# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A layer is 11 launches at any token count. Small M is settled (EXP-072): at 2 tokens the
chunk kernels beat the per-token loops by 1.97x to 2.57x. The 0.05 ms identity of the int4
GEMM at M = 50 is named (EXP-076): about 6 us of host gap and a 9 us GPU floor, so 38 to 47
us is the kernel body, and cuBLAS bf16 reaches 18.5 us on the same shape.

## Closed this pass

- **The int8 GEMM has its own sweep.** Twelve candidates at three expert widths and
  three token counts all land in 0.041 to 0.052 ms, so the menu was kept rather than
  retuned from noise (EXP-075, DEC-073).
- **The launch floor is named.** Empty kernel 0.016 loop against 0.009 graph, the int4
  GEMM 0.064 against 0.047 (EXP-076).
- **The persistent layer is stopped by Triton.** A two program sense barrier never exits;
  atomics alone and a pre-visible flag both work. Removed, not worked around
  (EXP-077, DEC-074).

## What works

- 32 tasks, `pytest -q` 159 passed with vcvars active, `style` at zero.
- `int8tiles` and `floor` join `pairedtail`, `m50`, `attncross` and `scale`.
- memcheck and racecheck read 0 errors or hazards on the GEMM and norm tests.

## Still open

- **The GEMM body at M = 50.** 38 to 47 us for 1.3 to 1.7 MB, unchanged by program count
  and 2x off cuBLAS bf16. That needs `ncu` from the elevated shell.
- **A persistent layer needs a cooperative launch** in CUDA C++, since Triton here cannot
  hold a grid barrier. Ceiling 9 percent at M = 50.
- **Above 64 tokens the einsum mix is back**, 25x slower, covered by one test.
- **The int8 menu** has never been paired, only swept sequentially (EXP-075).

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 720 shape.
2. Choose: a CUDA C++ cooperative layer, or the profiler first.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Profile the int4 GEMM body at 50 tokens and 1920 outputs with `ncu`.
2. Port the layer to a CUDA C++ cooperative launch, one launch per layer.
3. Pair the top three int8 tiles at 720 x 1920 and 50 tokens.
