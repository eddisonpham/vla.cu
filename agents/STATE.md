# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is done except for profiling on the large shape, which needs the
human's elevated shell. Every kernel now has a correctness check and a sanitizer pass,
the scaling question is answered in three directions, and the expert stack carries an
accuracy number at depth 16.

## Closed this pass

- **Both CUDA kernels have a standalone sanitizer pass.** int4 in EXP-044, int8 in EXP-046,
  0 errors and 0 leaks each, from a torch-free header included by a plain .cu harness.
- **The scale question is answered.** Shapes plateau at 70 to 75 percent for Triton and
  88 percent for the native kernel, group 8 buys the most cosine per byte, and 16 layers
  of real expert MLP cost 37.0 MB and 0.850 ms (EXP-045, EXP-047).
- **Depth 16 has an accuracy number.** With the layer's own post-attention norm and the
  residual, cosine is 0.99548 against 0.31792 on the bare projection chain (EXP-048).
- **Profiling is one command now.** `python -m vla_cu ncu` writes the vcvars wrapper and
  calls ncu with an argument list that needs no shell quoting (DEC-049).

## What works

- 26 tasks, `pytest -q` 96 passed and 23 skipped outside vcvars, 119 passed inside it.
- `plateau` sweeps seven shapes for four implementations, median of five repeats: native
  unroll 338.0 GB/s and 88.0 percent at 4096 x 8192, Triton 266.4 and 69.4 (EXP-047).
- `scale` reports shape, group size and depth, bare and normed (EXP-045, EXP-048).
- Locked `cuda` table at 8192 x 16384: stream 0.218 ms, 347.2 GB/s, 90.4 percent, which
  EXP-047 could not repeat as a median; the repeated number there is 307.2 GB/s and 80.0.
- int8 action blocks 0.99951 cosine at 0.140 ms, 3.94x fp32 (EXP-036).
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.

## Still open

- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`. A normal
  shell reaches ncu and stops at `ERR_NVGPUCTRPERM`, which is the expected outcome.
- **Triton's plateau is unexplained.** It reads 69 to 73 percent within 5 MB of the native
  kernel's 88, so the next step is SASS, not another tuning knob.
- **The largest shape is not trustworthy.** 16384 x 32768 spans 0.9 to 1.4 ms across five
  repeats, so quote it only with that range (EXP-047).
- **No three layer stack at M = 50** on real weights; EXP-043 measures one matrix at a time.
- Group 16 was not retested above 720, and the 0.98 gate has not been checked end to end on
  real weights.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next kernel question: attention in the chain, or SASS against Triton.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Add attention to the 16 deep chain, so cosine describes a layer and not a projection.
2. Compare SASS at 4096 x 8192 to name what Triton emits differently.
3. Graph the normed chain, which pays 43 us per layer in dispatch alone.
