# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

A real layer is now a layer index and a stack name: `mlp_from_checkpoint(checkpoint, 3,
part="expert")`. 240 device tests pass, both cooperative kernels memcheck clean.

## Closed in this pass

- **Layer index resolver.** `layer_prefix` maps `layer=3` onto the backbone
  `model.vlm_with_expert.vlm.model.text_model.layers.3` or the expert
  `model.vlm_with_expert.lm_expert.layers.3`, with `part` choosing and the header checked
  (EXP-093, DEC-086). Both builders take an index or a prefix.
- **Naming bugs caught.** The first run appended `.self_attn` twice for a prefix and
  dropped `.mlp` for an index; tests now cover both paths.
- **Checks.** 240 passed on the GPU, 44 passed and 196 skipped with no device, style 0,
  memcheck 0 errors on both cooperative kernels (EXP-091).

## What works

- `python -m vla_cu demo`, `coop`, `coopgated`, `graphed`, `sanitize`, `style` exit 0.
- `mlp_from_checkpoint(checkpoint, 3, part="expert")` builds a gated SwiGLU block.
- `attention_from_checkpoint(checkpoint, 3, part="vlm")` appends `self_attn` itself.
- `Int4Linear.graphed()` pays below roughly 1.5 MB of packed weights (EXP-092).

## Open edges

- **A whole layer is still several calls.** Norms and the residual pair have no builder,
  so the layer index stops at the projections.
- **The resolver knows two schemes**, the SmolVLB backbone and the expert, not arbitrary
  checkpoints. A different model keeps the prefix path.
- **The gated tail re-reads gate and up per token tile**, so it grows with tokens where
  the eager GEMM does not.
- **The builder and `GatedMlpBlock` do not call the fused kernel.**
- **`ncu` is blocked** by ERR_NVGPUCTRPERM, so attribution is inference from timing.
- **The int4 CUDA extension** still cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: a whole layer builder, `GatedMlpBlock` calling the fused kernel, or the no-copy
   graph wrapper.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Add norms and the residual pair to the builder, so one index is one whole layer.
2. Wire `GatedMlpBlock` to `coop_gated_into`, so the library path is one launch too.
3. Remove the input copy from `graphed()`, the few microseconds that make big layers lose.
