# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A real layer is one call: `layer_from_checkpoint(checkpoint, 3, part="expert")`. The
gated block can run its whole tail in one launch, and a capture bound to the caller's
buffer no longer copies. 243 device tests pass, style 0.

## Closed in this pass

- **Whole layer builder.** `LayerBlock` runs input norm, attention, residual, post norm,
  gated MLP, residual, and `layer_from_checkpoint` reads both RMSNorm weights from one
  index (EXP-094, DEC-087).
- **Fused tail in the library.** `GatedMlpBlock.tail_into` calls the cooperative gated
  kernel with the block's own packed weights, with the kernel contract checked first
  (EXP-095).
- **Bound capture.** `graphed(x)` captures against the caller's buffer, so the 5 to 10 us
  copy is gone: 4.18x down to 1.00 to 1.02x, where EXP-092 read 0.94x (EXP-096, DEC-088).
- **Gates.** 243 passed on the GPU, 44 passed and 199 skipped with no device, style 0,
  demo 4.72x on a four block stack.

## What works

- `python -m vla_cu demo`, `coop`, `coopgated`, `graphed`, `sanitize`, `style` exit 0.
- `layer_from_checkpoint(checkpoint, 3, part="expert", tokens=50)` returns a whole layer.
- `mlp_from_checkpoint(...).tail_into(...)` runs the gated tail in one launch.
- `Int4Linear.graphed(x)` reads `x` in place; `replay()` skips the copy.

## Open edges

- **`LayerBlock` still issues the six launch chain.** It does not call `tail_into`, so the
  layer builder has no fused path yet.
- **The fused tail has conditions**: int4 down, group 64, 1 to 64 tokens, hidden divisible
  by 16, intermediate by 64, so the 960/2560 backbone does not qualify as is.
- **The resolver knows two schemes**, the SmolVLM2 backbone and the expert, not arbitrary
  checkpoints.
- **The gated tail re-reads gate and up per token tile**, so it grows with tokens where
  the eager GEMM does not.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM, so attribution is inference from timing.
- **The int4 CUDA extension** still cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: `LayerBlock` on the fused tail, a multi-layer graph, or a producer writing into
   `static_in`.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Let `LayerBlock` call `tail_into` where the widths allow, so one layer is one launch.
2. Capture several layers and one residual add in a single graph, so the copy is paid once.
3. Measure the fused tail through `tail_into` at 1 to 8 tokens against the six launches.
