# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The deployed prefix shape is built: 192 image rows from the real SigLIP tower and
connector, 48 language rows, 1 state row. The expert holds its cosine on 241 keys. What is
left is content, not structure: real frames, a numerical cross-check of the tower, and the
10 step loop.

## Closed in this pass

- **The full 241 row prefix.** `vla_cu/vision.py` runs the 12 layer SigLIP tower in fp32
  over a 512 frame and the connector's 4x4 pixel shuffle plus projection, 64 rows per
  frame. `full_prefix_inputs` adds 48 language rows and 1 state row with the state block
  mask. Cache 16 x [241, 320] fp32, 9.87 MB, build 415.64 ms (EXP-107, DEC-094).
- **Expert on 241 keys.** int8 down 2.710 ms at cosine 0.99996; int4 down 2.817 ms at
  0.99858, both against fp32, both graphs bit exact.

## What the numbers say

| Measurement | Value |
|---|---|
| prefix rows | 241 = 3 x 64 image + 48 language + 1 state |
| prefix cache | 16 layers x [241, 320] fp32, 9.87 MB |
| prefix build | 289.93 ms per call, weight reads included (415.64 on the cold first run) |
| expert on 241 keys, int8 down | 2.628 ms, cosine 0.99996, 98.68 MB |
| expert on 241 keys, int4 down | 2.920 ms, cosine 0.99858, 88.31 MB |
| expert on 48 keys, int8 down | 2.290 ms, cosine 0.99997 (EXP-105) |

Key count moves timing by about 0.42 ms and cosine by about 1e-5, so it is not an
accuracy lever.

## Gates

- GPU suite 253 passed, CPU suite 51 passed and 202 skipped, style 0, clocks pinned at 1987 MHz.
- `fullprefix`, `fullmodel`, `expertstack`, `realstack` exit 0; graphs bit exact.
- 43 tasks, every one imported by `tests/test_tasks.py`.
- Pixel shuffle order has two CPU tests; memcheck ran on the int4 attention path (EXP-105).

## What works

- `python -m vla_cu fullprefix` builds the 241 row cache and times the expert on it.
- `vla_cu.vision.image_tokens(checkpoint, frames)` returns 64 rows per frame.
- `vla_cu.prefix.full_prefix_inputs` returns the rows and the visibility mask.

## Open edges

- **Frames and state are synthetic.** Seeded noise and a zero state, so the structure is
  real and the content is not.
- **The tower has no numerical reference.** No transformers run exists to check the fp32
  SigLIP forward; the pixel shuffle order is tested, the weights are the checkpoint's.
- **One action chunk, not ten steps.** The real loop holds the prefix cache across 10
  denoising passes.
- **Rope wavelength is unresolved.** LeRobot rotates at 10000, the config says 100000.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM; the int4 extension cannot be instrumented.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the int4 attention GEMM.
2. Pick the next experiment from BACKLOG, for example the 10 step loop.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. The 10 step denoising loop with the prefix cache held, which is the real chunk cost.
2. Three real camera frames, to give the image rows scene content.
3. Quantize the SigLIP tower, which is most of the 415.64 ms build.
