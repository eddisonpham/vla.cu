# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is nearly complete. Five improvement cycles (EXP-034 to EXP-039) closed
the earlier items; this pass closed the checkpoint, the clock lock, the sanitizer and the
720-wide decision. Profiling on the large shape is the one thing left, and it needs the
human's elevated shell.

## Closed this pass

- **Real weights are in and measured.** 906,712,520 bytes, 500 tensors, expert hidden 720
  and expert intermediate 2048 (EXP-043, DEC-044). int8 reads 0.99995 or better on five
  real matrices, int4 group 64 reads 0.992 to 0.995.
- **Clocks are pinned.** The elevated `nvidia-smi -lgc 2000` holds at 1980 to 1987 MHz,
  0.35 percent spread, power capped at about 70 W (EXP-042).
- **The int4 kernel has a clean sanitizer pass.** Kernel moved to a torch-free header and a
  standalone harness reports 0 errors and 0 leaks (EXP-044, DEC-046).
- **The 720-wide case is decided.** Zero pad to 768 beats group 16 on bytes and time, and
  loses 0.0018 cosine (DEC-044). Rotation stays opt-in (DEC-045).

## What works

- 24 tasks, `pytest -q` 96 passed and 23 skipped outside vcvars, 119 passed inside it.
- `scale` sweeps shapes to a 75 percent plateau, group sizes to 8, and the expert stack to 16 (EXP-045).
- Locked `cuda` table at 8192 x 16384: stream 0.218 ms, 347.2 GB/s, 90.4 percent of spec.
- int8 action blocks 0.99951 cosine at 0.140 ms, 3.94x fp32 (EXP-036).
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.

## Still open

- **`ncu` on the large shape** needs the elevated shell. The existing report covers a small
  shape at 6.75 us and 26 percent of DRAM, which is the latency regime, not the headline.
- **The int8 kernel has no standalone harness**, so its sanitizer pass is still API level.
- **No three layer stack at M = 50** on real weights; EXP-043 measures one matrix at a time.
- Group 16 was not retested above 720, and the 0.98 gate has not been checked end to end on
  real weights.
- The native kernel has not been swept over the shape grid. It reads 90.4 percent where the
  Triton kernel plateaus at 75 percent (EXP-045).

## What the human can do

1. Run `ncu` for the int4 stream variant at the large shape, from the elevated shell.
2. Say what "iterative scaling" means, so it can be built next.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Three real layers end to end, cosine per layer at int8 and at int4.
2. The int8 header split and harness, closing the last sanitizer gap.
3. Sweep the native kernel over the same grid and compare its plateau with Triton's 75 percent.
