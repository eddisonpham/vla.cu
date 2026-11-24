# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The prefix pipeline is priced and cheap: a warm prefill is 117.50 ms at an end to end 0.99992,
the whole model carries all 241 rows in one graph, and the 4-bit tower question is closed
from both sides. What is left is the first call in a process, and content.

## Closed in this pass

- **W4A8.** Exact activations read the same 0.97059 as the fp16 operands, so the 4-bit weight
  is the whole error; int8 activations read 0.96827, so A8 would lose 0.0023 (EXP-113).
- **The block mask.** The fused kernel takes a uint8 mask and both block levels forward it,
  so the whole model runs 241 rows at cosine 0.99990 and 11.720 ms (EXP-114, DEC-099).
- **The text weight cache.** 603.84 to 619.91 ms on the first call, 28.95 to 29.15 ms after,
  so a warm int8 prefill is 117.50 ms against 782.52 (EXP-115, DEC-100).

## What the numbers say

| Measurement | Value |
|---|---|
| warm prefill, int8 tower and text weights held | 117.50 ms = 88.36 frames + 29.15 text |
| first call, tower and text read | 253.37 ms + 603.84 to 619.91 ms |
| full prefix end to end, int8 down / int4 down | 0.99992 at 2.373 ms / 0.99852 at 2.816 ms |
| whole model, one graph / two graphs / eager | 11.720 / 11.747 / 32.231 ms, cosine 0.99990 |
| int4 tower rows, A16 / A32 / A8 | 0.97059 / 0.97059 / 0.96827 |

## Gates

- GPU suite 264 passed in a vcvars shell, CPU suite 54 passed and 210 skipped, style 0, 46 tasks.
- `tower`, `fullprefix`, `fullmodel` exit 0; every graph bit exact against eager.
- The masked kernel matches `gqa_mix` to 1e-5, and an all-true mask is bit identical to no mask.

## What works

- `python -m vla_cu tower` sweeps width, group and activation precision and splits the prefill.
- `python -m vla_cu fullprefix` reports the 241 row path end to end on the int8 tower.
- `python -m vla_cu fullmodel` runs 241 rows plus 50 action tokens in one graph.
- `python -m vla_cu denoise` prices the 10 step loop, now stale on the prefill side.

## Open edges

- The first call in a process still reads 253 ms of tower and about 600 ms of text weights;
  only a prepacked file would remove that.
- 629.27 MB of fp32 text weights are resident between calls in a one entry cache.
- Frames and state are synthetic, and the denoise loop uses seeded guesses with no timestep
  embedding, because this checkpoint carries no time projection.
- int4 is refused at groups 64, 32 and 16 and A8 does not help.
- `ncu` needs the elevated shell; racecheck never returns on the cooperative kernel; the int4
  CUDA extension cannot be instrumented; CI is CPU only. Rope wavelength still unresolved.

## What the human can do

1. Pick the next item from `agents/BACKLOG.md`, for example a prepacked file or real frames.
2. Run `python -m vla_cu ncu` from the elevated shell on the int4 attention GEMM.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Prepack the tower and text weights to disk, which removes the 0.85 s first call.
2. Three real camera frames, so the image rows carry scene content.
3. Hold the text weights at int8, 157.84 MB against 629.27, and measure the cosine.
