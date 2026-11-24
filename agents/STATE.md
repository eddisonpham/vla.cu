# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The attention block is causal, its scores cover positions, and its mix is now a Triton
kernel for short chunks and einsums for long ones. The 50 token chunk is measured, the
int4 GEMM is measured against fifty GEMV launches, and eleven production CUDA inference
systems are surveyed in `docs/PRODUCTION.md`. The named bottleneck is the per-token GEMV
launch, and the fix is decided (DEC-062) but not yet wired.

## Closed this pass

- **The causal mix is one kernel for short chunks.** Depth 16 graph 2.016 and 2.585 ms at
  2 and 3 tokens against 2.723 and 3.025 with einsums. It loses at 8 and 50, so the block
  dispatches at `FUSED_MAX_TOKENS = 4` (EXP-062, DEC-061).
- **An int4 GEMM at M = 50 beats the GEMV loop by 24 to 42x** on the expert matrices, and
  is still 2.5 to 3.8x behind cuBLAS bf16, at 3.6 to 7.6 percent of peak (EXP-063).
- **The chain runs at 50 tokens.** 23.245 ms graphed against 360.220 ms eager at depth 16,
  exact, cosine 0.99637. 500 launches per layer counted from the code (EXP-060).
- **`gqa_mix` was mixing over key heads, not tokens.** `torch.bmm` batched on tokens. The
  causal mask exposed it, einsums fix it (EXP-057, DEC-059).
- **Constants now come from the model config**: rope theta 100000 with the Llama pairing,
  and `rms_norm_eps` 1e-05 (EXP-055, EXP-058, DEC-056, DEC-060).
- **The real shapes are launch bound** at about 22 us while bytes move 3.6x (EXP-061).

## What works

- 27 tasks, `pytest -q` 126 passed with vcvars active, 96 passed and 23 skipped without.
- `scale` prints four stages plus the 50 token block, with eager, graph, cosine, relative
  L2 and an exactness column.
- `m50` compares the GEMV loop, the int4 GEMM and cuBLAS bf16 at the expert widths.
- `plateau` median of 5: staged Triton 63.6 to 73.1, wide ieee 11.3 to 16.1, wide tf32
  28.3, native unroll 86.8 to 88.0 percent at 4096 x 8192 across runs.
- `docs/PRODUCTION.md` maps 11 production systems onto this repo and ranks six gaps.

## Still open

- **A 50 token layer is 500 launches.** Decided: dispatch to the int4 GEMM above a small
  token count (DEC-062). Not wired yet.
- **The attention kernel reloads its key tile per query head.** Group it by kv head and
  fuse rope, then the crossover moves up.
- **The int4 GEMM is 7.6 percent of peak at M = 50.** Needs `num_stages` and a tile menu
  in the Marlin shape.
- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **Run drift reaches 25 percent**, so only numbers inside one output block compare. The
  attention crossover is the first casualty and needs a paired run.
- **The wide mapping** lost on load width, reduction and precision (EXP-052, 054, 056, 059).

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: wire the GEMM for chunks, or regroup the attention kernel.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Wire the int4 GEMM into the block above a token threshold, then re-run `scale` at 8 and 50.
2. Group the attention program by kv head and fuse rope into it.
3. Pipeline the int4 GEMM and give it a tile menu, then repeat `m50`.
