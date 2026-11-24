# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The fused tail is one cooperative launch, serves any layer width in its class, and is
measured from 1 to 64 tokens. It wins below 16 tokens and loses above. The tensor core
body that was supposed to widen that lost 2x to 4x at every token count and was reverted
before commit, so the negative is the finding: at these M values a GEMV body beats tiles.

## Closed in this pass

- **Widths as launch parameters.** hidden and intermediate come from the tensors, shared
  memory is sized from them, tests cover 720/1920, 256/640 and 384/768. Cost about
  7 percent at 1 token (EXP-085).
- **Tensor core body tried and dropped.** WMMA 16 x 16 tiles at two K staging depths:
  0.247 and 0.427 ms at 1 token against 0.042 for the GEMV body, 15 tests pass, memcheck
  clean. A double `up_scales` application was found and fixed first (EXP-084, DEC-079).
- **Crossover re-measured.** 2.91x at 1 token, 2.28x at 2, 1.95x at 3, 1.15x at 8,
  0.70x at 16, 0.22x at 50 (EXP-085).

## What works

- `pytest -q` with a GPU: 209 passed, including 15 cooperative tests.
- With no device: 35 passed, 169 skipped, exit 0.
- `python -m vla_cu style`: 0 violations. `compute-sanitizer --tool memcheck`: 0 errors.
- `python -m vla_cu coop` pins the whole tail to one launch at any width in its class.

## Open edges

- **Tiles are the wrong body below 16 tokens**, which is the batch size that matters. A
  64 token block with async copies and one K pipeline is the untried design.
- **The int4 dequant path** runs at 5 to 6 percent of peak and `ncu` is still blocked by
  ERR_NVGPUCTRPERM.
- **Shared memory races** in the cooperative kernel are argued from the code, since
  racecheck cannot run on a grid barrier.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: the 64 token single pipeline body, or the attention widths at 960 and 2560.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. A 64 row token block with async copies and one K pipeline.
2. Run the cooperative kernel on the attention widths, where M is larger.
3. A self-hosted GPU runner so CI runs the 169 device tests.
