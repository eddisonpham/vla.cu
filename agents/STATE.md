# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The library builds and runs a whole layer, a real three layer stack holds its accuracy at
both precisions, and the fused tail is measured rather than assumed. 245 device tests pass.

## Closed in this pass

- **Real three layer stack.** Layers 0 to 2 of `lerobot/smolvla_base` at one token, all
  paths graphed: fp32 0.778 to 0.800 ms, int8 down 0.177 to 0.192 ms (4.2 to 4.4x),
  int4 down 0.141 to 0.154 ms (5.2 to 5.5x), cosine 0.99986 against 0.99505 (EXP-099).
- **The 720/1920 question.** The captured cooperative tail reads 0.58x the captured chain
  at one token, so the smaller width does not change the verdict (EXP-100, DEC-089).
- **Whole layer builder, fused tail, bound capture** closed earlier in the session:
  EXP-094, EXP-095, EXP-096, EXP-097, EXP-098.

## What the numbers say

| Path | MB | ms | x fp32 | cosine |
|---|---|---|---|---|
| fp32, 3 layers, graph | 117.99 | 0.778 to 0.800 | 1.00x | 1.00000 |
| int8 down, 3 layers, graph | 29.59 | 0.177 to 0.192 | 4.2 to 4.4x | 0.99986 |
| int4 down, 3 layers, graph | 26.36 | 0.141 to 0.154 | 5.2 to 5.5x | 0.99505 |

int4 per layer: 0.99860, 0.99768, 0.99505. int8 per layer: 0.99998, 0.99995, 0.99986.
int4 error compounds slowly and stays above the 0.98 gate for three layers.

## Gates

- GPU suite 245 passed, CPU suite 46 passed and 199 skipped, style 0, clocks pinned at 1987 MHz.
- `demo`, `whole`, `realstack`, `graphed`, `coopgated` exit 0; every graph replay bit exact.
- 40 tasks, every one imported by `tests/test_tasks.py`.

## What works

- `python -m vla_cu realstack` runs three real layers at int8 and int4 against fp32.
- `python -m vla_cu whole` compares the fused layer and the 720/1920 tail, all captured.
- `layer_from_checkpoint(checkpoint, 3, part="expert", tokens=50)` returns a whole layer.
- `Int4Linear.graphed(x)` reads `x` in place; `replay()` skips the copy.

## Open edges

- **Only 3 of 16 backbone layers measured**, one token, Gaussian activations, no VLM prefix.
- **The fused tail loses to a graph at both widths**, so it is not on the recommended path.
- **`LayerBlock` is square self-attention only**, so 720 wide layers need the lower blocks.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM; the int4 extension cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on one kernel.
2. Pick the next experiment from BACKLOG, for example all 16 layers or the expert at 50 tokens.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. All 16 backbone layers at int8 and int4, one token, graphed, to see where cosine lands.
2. The expert stack at 50 tokens, where the GEMM path and the attention mix change the cost.
3. int4 attention with int8 down, the rung that separates the two error sources.
