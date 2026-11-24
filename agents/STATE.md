# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The whole backbone and the whole expert are measured at both precisions, and the 4-bit
error is now attributed. int8 down is the safe default everywhere; int4 down holds cosine
but is slower than int8 at 50 tokens.

## Closed in this pass

- **All 16 backbone layers, one token, graphed.** fp32 629.27 MB and 4.334 ms; int8 down
  157.84 MB and 1.176 ms (3.69x), cosine 0.99998; int4 down 140.57 MB and 1.101 ms
  (3.94x), cosine 0.99965 (EXP-101).
- **All 16 expert layers at 50 tokens.** fp32 392.98 MB and 7.814 ms; int8 down 98.68 MB
  and 2.260 ms (3.46x), cosine 0.99996; int4 down 88.31 MB and 2.731 ms (2.86x, slower),
  cosine 0.99847 (EXP-102).
- **int4 attention with int8 down.** 0.93242 over four blocks against 0.90671 with int4
  down and 0.97606 with int8 attention, so 4-bit attention is the larger error (EXP-103).

## What the numbers say

| Measurement | Bytes | ms | cosine |
|---|---|---|---|
| backbone, 16 layers, fp32 graph | 629.27 MB | 4.334 | 1.00000 |
| backbone, int8 down | 157.84 MB | 1.176 | 0.99998 |
| backbone, int4 down | 140.57 MB | 1.101 | 0.99965 |
| expert, 50 tokens, fp32 graph | 392.98 MB | 7.814 | 1.00000 |
| expert, int8 down | 98.68 MB | 2.260 | 0.99996 |
| expert, int4 down | 88.31 MB | 2.731 | 0.99847 |

int4 attention error at four blocks: 0.93242 with int8 down, against 0.97606 for int8
attention with int4 down. Avoid int4 attention before int4 down.

## Gates

- GPU suite 246 passed, CPU suite 47 passed and 199 skipped, style 0, clocks pinned at 1987 MHz.
- `realstack`, `expertstack`, `block`, `whole`, `demo`, `graphed` exit 0; graphs bit exact.
- 41 tasks, every one imported by `tests/test_tasks.py`.

## What works

- `python -m vla_cu realstack` runs the whole backbone at int8 and int4 against fp32.
- `python -m vla_cu expertstack` runs the whole expert at 50 tokens, cross attention included.
- `python -m vla_cu block` now has all four precision mixes.
- `layer_from_checkpoint(checkpoint, 3, part="expert", tokens=50)` returns a whole layer.

## Open edges

- **The expert prefix is random** [50, 320], not a real VLM KV cache; the real prefix is 48 tokens.
- **int4 down is slower at 50 tokens** (2.731 against 2.260 ms), so it is an accuracy option only.
- **int4 attention is not in the library**, only in the single token block bench.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM; the int4 extension cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on one kernel.
2. Pick the next experiment from BACKLOG, for example the real VLM prefix or int4 attention at 50 tokens.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. The expert stack with the real 48 token prefix and a measured KV cache, instead of random keys.
2. int4 attention at 50 tokens on real weights, to see if the four block result holds.
3. A single number for the full model: 16 backbone plus 16 expert layers, int8 down, one graph.
