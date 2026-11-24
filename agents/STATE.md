# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The project now has three layers of proof: numbers, gates and CI. The cooperative tail
is hardened and sanitized (memcheck 0 errors, 10 tests from 1 to 64 tokens), CI runs
style, the task registry and the suite on a CPU runner (35 passed, 169 skipped), and the
fused path's crossover is measured at 8 to 16 tokens (EXP-082).

## Closed in this pass

- **Cooperative launcher hardened.** Shared memory limit configured once per device under
  a mutex, co-residency bound from the occupancy API, alignment checks, non-contiguous
  input refused. 10 tests, 1 to 64 tokens (EXP-081).
- **Sanitizer gate named.** memcheck clean; racecheck serializes blocks and deadlocks on
  any grid barrier, so it is documented as unusable, not skipped quietly (DEC-078).
- **CI added.** `.github/workflows/ci.yml` on ubuntu with CPU torch. `tests/conftest.py`
  keys the skip on `device_count()`, because `is_available()` reads true with no visible
  device (EXP-083).
- **Crossover measured.** Fused is 3.13x at 1 token, 1.26x at 8, 0.74x at 16, 0.27x at
  64. A tiled body is the only way to widen it (EXP-082).
- **README truth pass.** 35 tasks, a checks table with commands, the current status rows.

## What works

- `pytest -q` with a GPU: 204 passed. With no device: 35 passed, 169 skipped, exit 0.
- `python -m vla_cu style`: 0 violations. `python -m vla_cu --help`: 35 tasks.
- `compute-sanitizer --tool memcheck python -m vla_cu coop`: 0 errors.

## Open edges

- **The fused body is still a GEMV.** It wins at 1 to 8 tokens only.
- **The int4 dequant path** runs at 5 to 6 percent of peak and `ncu` is still blocked by
  ERR_NVGPUCTRPERM.
- **Shared memory races** in the cooperative kernel are argued from the code, not
  measured, since racecheck cannot run there.
- **CI proves style, imports and collection**, not kernel correctness. That needs a
  self-hosted GPU runner.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: tile the cooperative body, or strip the int4 dequant in a test variant.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Tile the cooperative body and rerun the 1 to 64 table for a new crossover.
2. An int4 variant without the group scale and transpose.
3. A self-hosted GPU runner so CI runs the 169 device tests.
