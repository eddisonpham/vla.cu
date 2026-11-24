# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

Both precision axes of the expert are now measured on real weights and the real prefix.
4-bit down holds accuracy at a small speed cost. 4-bit attention is worse on both axes, so
int8 attention is the default and the int4 path exists to be reproduced.

## Closed in this pass

- **int4 attention in the library.** `GqaTokenBlock` takes `bits` and packs q, k, v and o
  at group 64, padding K to 768 for the 720 wide stream; `layer_from_checkpoint` takes
  `attention_bits` (DEC-092). Test at 6 tokens and a memcheck pass.
- **Four mixes at 50 tokens.** int8/int8 2.290 ms at 0.99997; int4 attention 3.290 ms at
  0.99761; int4 down 2.698 ms at 0.99847; both at int4 3.516 ms at 0.99613 (EXP-105).

## What the numbers say

| Mix | Bytes | ms | x fp32 | cosine |
|---|---|---|---|---|
| fp32 | 392.98 MB | 7.082 | 1.00x | 1.00000 |
| int8 attn, int8 down | 98.68 MB | 2.290 | 3.09x | 0.99997 |
| int4 attn, int8 down | 87.08 MB | 3.290 | 2.15x | 0.99761 |
| int8 attn, int4 down | 88.31 MB | 2.698 | 2.63x | 0.99847 |
| int4 attn, int4 down | 76.71 MB | 3.516 | 2.01x | 0.99613 |

| Decision | Evidence |
|---|---|
| int4 attention stays off | 1.000 ms slower and 0.0024 lower cosine than int8 attention |
| int4 down is an accuracy option | safe at 0.99847, but 0.408 ms slower than int8 down |

## Gates

- GPU suite 249 passed, CPU suite 47 passed and 202 skipped, style 0, clocks pinned at 1987 MHz.
- `expertstack` exit 0 with four stacks, graphs bit exact, under the 7 GB cap.
- `compute-sanitizer --tool memcheck` on the int4 attention test: 1 passed, 0 errors.
- 41 tasks, every one imported by `tests/test_tasks.py`.

## What works

- `python -m vla_cu expertstack` runs all four attention and down mixes on the real prefix.
- `layer_from_checkpoint(..., attention_bits=4)` builds a 4-bit attention layer.
- `vla_cu.prefix.prefix_cache(checkpoint)` returns post-rope K and V per backbone layer.

## Open edges

- **Cross-stack timing moves between runs.** The same int8/int8 path read 1.977 ms in
  EXP-104 and 2.290 ms here, so ratios within one run are the comparison.
- **The prefix is 48 language rows only.** The deployed prefix is about 241 rows: 192
  image, 48 language, 1 state.
- **Rope wavelength is unresolved.** LeRobot rotates at 10000, the config says 100000.
- **int4 GEMM body** reads 20 to 23 GB/s at these shapes, which is why int4 attention loses.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM; the int4 extension cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the int4 attention GEMM.
2. Pick the next experiment from BACKLOG, for example the full 241 row prefix.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. An int4 GEMM tuned at 960 by 768, 320 by 768 and 720 by 960, then repeat this table.
2. The full 241 row prefix with the SigLIP tower and three real frames.
3. int4 attention at one token, where the GEMV body may behave differently.
