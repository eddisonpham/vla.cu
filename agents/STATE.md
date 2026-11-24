# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The expert now runs against the real VLM key and value cache instead of random rows. The
cosine does not move, so the 4-bit error comes from the projections, not from the prefix.
The deployed prefix is about 241 rows, and the bench covers its 48 language rows.

## Closed in this pass

- **The real VLM prefix.** Expert cross attention reads the same-index backbone layer's
  post-rope K and V, keys and values through separate projections, query only rotated.
  Cosine 0.99997 at int8 and 0.99847 at int4, the same as EXP-102 (EXP-104).
- **Cross mode in the block and the kernel.** `GqaTokenBlock` is cross when `wk` reads a
  narrower stream than `wq`; the kernel takes a key count and `rotate_keys` (DEC-090).
- **Memcheck on the new path.** 2 cross tests pass under `compute-sanitizer --tool
  memcheck`, 0 errors.

## What the numbers say

| Measurement | Bytes | ms | cosine |
|---|---|---|---|
| expert, real prefix, fp32 graph | 392.98 MB | 6.454 | 1.00000 |
| expert, real prefix, int8 down | 98.68 MB | 1.977 | 0.99997 |
| expert, real prefix, int4 down | 88.31 MB | 2.378 | 0.99847 |
| expert, random prefix, int8 down | 98.68 MB | 2.260 | 0.99996 |
| expert, random prefix, int4 down | 88.31 MB | 2.731 | 0.99847 |

int4 down stays 0.40 ms slower than int8 down at 50 tokens. Prefix content is not the
error source; it changes neither cosine at five decimals.

## Gates

- GPU suite 248 passed, CPU suite 47 passed and 201 skipped, style 0, clocks pinned at 1987 MHz.
- `expertstack` exit 0 with 48 key rows against 50 queries, graphs bit exact.
- 41 tasks, every one imported by `tests/test_tasks.py`.

## What works

- `python -m vla_cu expertstack` runs the whole expert on the real 48 row VLM cache.
- `vla_cu.prefix.prefix_cache(checkpoint)` returns post-rope K and V per backbone layer.
- `layer_from_checkpoint(..., other_tokens=48)` builds a cross layer for 48 keys.

## Open edges

- **The prefix is 48 language rows only.** The deployed prefix is about 241 rows: 192
  image, 48 language, 1 state. No tokenizer, camera frame, or vision forward here.
- **Rope wavelength is unresolved.** LeRobot rotates at 10000, the config says 100000,
  this repo uses 100000 on both sides (RESEARCH 2026-11-24).
- **int4 down is slower at 50 tokens** (2.378 against 1.977 ms), an accuracy option only.
- **int4 attention is not in the library**, only in the single token block bench.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM; the int4 extension cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on one kernel.
2. Pick the next experiment from BACKLOG, for example the full 241 row prefix or the rope question.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. The full 241 row prefix: SigLIP tower and connector over three real frames.
2. Resolve theta 10000 against 100000 and re-run one expert stack.
3. int4 attention at 50 tokens on real weights, to see if the four block result holds.
