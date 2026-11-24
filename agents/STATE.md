# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The deployed prefix is built, packed and now numerically verified, and the 10 step loop is
measured: a chunk costs the prefill, not the steps. What is left is content (real frames, a
real state) and making the prefill read cheaper, not more structure.

## Closed in this pass

- **The tower is packed and verified.** `Tower` reads the 12 SigLIP layers once per prefill
  and holds them at 32, 8 or 4 bits through `_Projection`. transformers 5.24.1 reproduces
  the 1024 patches to 2.16e-04 max abs, cosine 1.00000 (EXP-110, DEC-095).
- **The prefill split.** Tower 393.72 ms against 388.80 ms for the 16 text layers at fp32.
  The checkpoint stores the tower as 172.87 MB of bf16 and every stored width reads it in
  full, so int8 saves compute only (EXP-108, DEC-096).
- **The 10 step loop.** Cache built once at 818.98 ms, ten graphed steps 22.393 ms, so a
  chunk is 841.37 ms against 8212.38 ms with the cache rebuilt every step (EXP-109).

## What the numbers say

| Measurement | Value |
|---|---|
| tower as bf16 in the checkpoint | 172.87 MB |
| tower held at fp32 / int8 / int4 | 392.59 / 102.73 / 60.07 MB |
| tower, 3 frames, fp32 / int8 / int4 | 170.64 / 89.96 / 174.33 ms |
| tower cosine against fp32, int8 / int4 | 0.99956 / 0.94484 |
| prefill fp32 / int8 / int4 | 782.52 / 729.55 / 812.06 ms |
| one expert step, graph / eager | 2.258 / 6.928 ms |
| chunk, cache held / rebuilt | 841.37 / 8212.38 ms |
| tower against transformers, 1024 patches | 2.16e-04 max abs, cosine 1.00000 |

## Gates

- GPU suite 258 passed in a vcvars shell, CPU suite 54 passed and 204 skipped, style 0, 46 tasks.
- `tower`, `denoise`, `visioncheck`, `fullprefix` exit 0; graphs bit exact against eager.
- A plain shell without vcvars fails one test that builds the cooperative extension (ninja), which is the documented requirement, not a code failure.

## What works

- `python -m vla_cu tower` splits the prefill into tower and text layers at three widths.
- `python -m vla_cu denoise` prices the 10 step loop against rebuilding the cache.
- `python -m vla_cu visioncheck` compares the tower with a transformers SigLIP run.

## Open edges

- Frames and state are synthetic, the loop uses seeded guesses, and no timestep embedding is
  modelled because this checkpoint carries no time projection.
- Every read measured is warm in the OS file cache; no cold prefill exists in this pass.
- The int4 tower fails the 0.98 gate at group 64.
- `ncu` needs the elevated shell; racecheck never returns on the cooperative kernel; the int4
  CUDA extension cannot be instrumented; CI is CPU only.
- Rope wavelength is unresolved: LeRobot rotates at 10000, the config says 100000.

## What the human can do

1. Pick the next item from `agents/BACKLOG.md`, for example three real frames or a cached packed tower.
2. Run `python -m vla_cu ncu` from the elevated shell on the int4 attention GEMM.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Cache the packed tower across calls, since the prefill is the weight read.
2. Three real camera frames, so the image rows carry scene content.
3. int4 down on the whole path, to see whether the model cosine tracks the expert 0.99847.
