# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A real SmolVLA layer is now three lines and one launch: the builder reads gate, up and
down from safetensors, and the cooperative kernel computes the gated SwiGLU tail with
both projections in one pass. 235 device tests pass, both cooperative kernels memcheck clean.

## Closed in this pass

- **Gated cooperative kernel.** `coop_gated_kernel` keeps the two grid barriers and the
  shared memory of the two-matrix kernel; add, norm and down are shared device functions,
  and the gated stage reads gate and up for each output row. 3.49x at one token at
  720/1920, 2.36x at 960/2560 (EXP-091, DEC-084).
- **Wider gated tail crosses earlier.** First loss at 12 tokens at 720/1920 and 8 at
  960/2560, because both weight matrices are re-read for every 16 token tile.
- **Checks.** 235 passed on the GPU, 41 passed and 194 skipped with no device, style 0,
  memcheck 0 errors at 3, 17 and 26 tokens for both cooperative kernels.

## What works

- `python -m vla_cu demo`, `coop`, `coopgated`, `sanitize`, `style` all exit 0.
- `mlp_from_checkpoint` builds `GatedMlpBlock` when `gate_proj` is present (EXP-088).
- `Int4Linear.graphed()` replays one captured call bit for bit (EXP-089).
- `python -m vla_cu coop`: both widths cross at 10 tokens (EXP-090).

## Open edges

- **The gated tail still re-reads gate and up per token tile**, so it grows with tokens
  where the eager GEMM does not. Sharing those reads is the next kernel idea.
- **The builder and `GatedMlpBlock` do not call the fused kernel**; the block is eager and
  the fused path is reached from `bench/coop_gated.py` only.
- **The matrix row varies 0.073 to 0.194 ms across sessions** (BACKLOG, unexplained).
- **`ncu` is blocked** by ERR_NVGPUCTRPERM, so attribution is inference from timing.
- **The int4 CUDA extension** still cannot be instrumented on Windows; the Triton kernel
  is covered by tests only.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: wire the fused gated tail into `GatedMlpBlock`, share the weight reads across
   token tiles, or isolate the matrix row spread.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Let `GatedMlpBlock` call `coop_gated_into`, so the library path is one launch too.
2. Stage gate and up once per K step for all 16 tokens, so the fused tail stops growing.
3. Pin the crossover at 9 to 13 tokens with more repeats before quoting a threshold.
