# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

One call builds and runs a whole transformer layer, and every documented surface is
checked. The layer's fused tail is available and measured, but a captured chain is
faster, so the default stays the chain.

## Closed in this pass

- **Whole layer builder.** `LayerBlock` runs input norm, attention, residual, post norm,
  gated MLP, residual, and `layer_from_checkpoint` reads both RMSNorm weights from one
  index (EXP-094, DEC-087).
- **Fused tail reachable from the layer.** `LayerBlock.into(..., fused=True)` and
  `GatedMlpBlock.can_tail()` (EXP-095, EXP-097).
- **Bound capture.** `graphed(x)` captures against the caller's buffer, so the 5 to 10 us
  copy is gone: 4.18x down to 1.00 to 1.02x, where EXP-092 read 0.94x (EXP-096, DEC-088).
- **Release audit.** 39 tasks import, README and architecture counts corrected, no TODO or
  FIXME markers, CI workflow present (EXP-098).

## Measured truth about the fused tail

Against a captured chain at 960/2560, the fused tail reads 0.59x at 1 token, 0.50x at 8
and 0.15x at 50 (EXP-097). Its earlier 3.49x win was against an eager chain, so it was
removing host dispatch that a graph also removes. `fused=False` is the default (DEC-089).

## Gates

- GPU suite 244 passed, CPU suite 45 passed and 199 skipped, style 0, clocks pinned at 1987 MHz.
- `demo`, `graphed`, `whole`, `coopgated` exit 0.
- Graphed block 0.021 ms at 4.5x fp32, four block stack 4.7 to 5.2x, graphs bit exact.

## What works

- `python -m vla_cu demo`, `whole`, `graphed`, `coop`, `coopgated`, `sanitize`, `style`.
- `layer_from_checkpoint(checkpoint, 3, part="expert", tokens=50)` returns a whole layer.
- `mlp_from_checkpoint(...).tail_into(...)` runs the gated tail in one launch.
- `Int4Linear.graphed(x)` reads `x` in place; `replay()` skips the copy.

## Open edges

- **The fused tail is slower than a graph**, so it is not on the recommended path.
- **`LayerBlock` is square self-attention only**: it requires `input_width == hidden`, so a
  cross-attention layer needs the lower level blocks.
- **The resolver knows two schemes**, the SmolVLM2 backbone and the expert.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM, so attribution is inference from timing.
- **The int4 CUDA extension** cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on one kernel.
2. Choose a next experiment from BACKLOG, for example the int4 attention rung or the
   three layer stack on real weights.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Measure the fused tail at 1 token against a graph at 720/1920, to see if the smaller
   width changes the verdict.
2. A three layer stack on real `lerobot/smolvla_base` weights at int8 and int4.
3. Cache the VLM prefix KV for a policy loop, the one production idea nothing here covers.
