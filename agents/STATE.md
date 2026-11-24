# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A real SmolVLA layer is three lines and one launch, and the cost of a captured call now
has a measured boundary. 236 device tests pass, both cooperative kernels memcheck clean.

## Closed in this pass

- **Where graph capture pays.** The wrapper removes a flat 0.020 ms of host dispatch and
  adds a copy plus a replay dispatch. Measured 2.20x at 0.09 MB of packed weights, 1.79x
  at 1.38 MB, 0.93x at 9.44 MB, 0.94x at 18.87 MB (EXP-092, DEC-085).
- **Gated cooperative kernel.** One launch for a three-matrix SwiGLU MLP, 3.49x at one
  token at 720/1920 and 2.36x at 960/2560 (EXP-091).
- **Checks.** 236 passed on the GPU, 42 passed and 194 skipped with no device, style 0,
  memcheck 0 errors on both cooperative kernels.

## What works

- `python -m vla_cu demo`, `coop`, `coopgated`, `graphed`, `sanitize`, `style` exit 0.
- `mlp_from_checkpoint` builds `GatedMlpBlock` when `gate_proj` is present (EXP-088).
- `Int4Linear.graphed()` replays bit for bit and pays below roughly 1.5 MB (EXP-092).
- `python -m vla_cu coop`: both widths cross at 10 tokens (EXP-090).

## Open edges

- **The graphed wrapper still copies its input**, which is the few microseconds that make
  4096 x 4096 a small loss. Capturing a producer that writes the same buffer would fix it.
- **The gated tail re-reads gate and up per token tile**, so it grows with tokens where
  the eager GEMM does not.
- **The builder and `GatedMlpBlock` do not call the fused kernel.**
- **All measured weights fit L2**, so the reported GB/s is L2-fed and no row here is a
  DRAM claim, including the 261 GB/s at 18.87 MB.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM, so attribution is inference from timing.
- **The int4 CUDA extension** still cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: a producer-captured layer with no input copy, `GatedMlpBlock` calling the fused
   kernel, or a block-level graph wrapper.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Remove the input copy so capture stops losing on the big projection.
2. Wire `GatedMlpBlock` to `coop_gated_into`, so the library path is one launch too.
3. Capture a whole block and its residual add, since the block demo is where graphs paid 3x.
