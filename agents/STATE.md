# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The prefix pipeline is settled: the int8 tower ships at a 0.99992 end to end cosine, it is
built once per process, and int4 is refused at every group. What is left is the other half
of the prefill, the 16 text layers, and content rather than structure.

## Closed in this pass

- **int4 tower, group 64, 32 and 16.** Row cosine 0.94484, 0.95583 and 0.97059, so the
  4-bit code and not the scale granularity sets the error. Refused (EXP-111, DEC-098).
- **The tower cache.** `tower_for` holds one tower per process: 259.35 ms first call,
  0.005 ms after, so a warm int8 prefill is 471.99 ms against 710.08 (EXP-112, DEC-097).
- **int8 end to end.** `fullprefix` reads 0.99992 at int8 down against an fp32 tower; the
  whole model over 240 rows reads 0.99990 (EXP-112).

## What the numbers say

| Measurement | Value |
|---|---|
| tower patch cosine, int8 / int4 g64 / g32 / g16 | 0.99851 / 0.85119 / 0.88169 / 0.91617 |
| tower row cosine, int8 / int4 g64 / g32 / g16 | 0.99956 / 0.94484 / 0.95583 / 0.97059 |
| tower held, int8 / int4 g64 / g16 | 102.73 / 60.07 / 78.21 MB |
| tower cache, first / later call | 259.35 / 0.005 ms |
| warm prefill, fp32 / int8 | 542.42 / 471.99 ms, text layers 384.45 ms of it |
| full prefix end to end, int8 down / int4 down | 0.99992 at 2.395 ms / 0.99852 at 2.806 ms |
| whole model, one graph / two graphs / eager | 11.764 / 11.568 / 32.670 ms, cosine 0.99990 |

## Gates

- GPU suite 261 passed in a vcvars shell, CPU suite 54 passed and 207 skipped, style 0, 46 tasks.
- `tower`, `fullprefix`, `fullmodel` exit 0; every graph bit exact against eager.
- A plain shell without vcvars fails one test that builds the cooperative extension (ninja), which is the documented requirement, not a code failure.

## What works

- `python -m vla_cu tower` sweeps width and group and splits the prefill.
- `python -m vla_cu fullprefix` reports the 241 row path end to end with the int8 tower.
- `python -m vla_cu fullmodel` runs 240 prefix rows plus 50 action tokens in one graph.
- `python -m vla_cu denoise` prices the 10 step loop with the cache held.

## Open edges

- The whole model drops the state row, because the fused attention kernel takes no block mask.
- Frames and state are synthetic, and the denoise loop uses seeded guesses with no timestep
  embedding, because this checkpoint carries no time projection.
- Every read measured is warm in the OS file cache; no cold prefill exists.
- The 16 text layers are 384.45 ms of a 471.99 ms warm prefill and are not cached.
- `ncu` needs the elevated shell; racecheck never returns on the cooperative kernel; the int4
  CUDA extension cannot be instrumented; CI is CPU only. Rope wavelength still unresolved.

## What the human can do

1. Pick the next item from `agents/BACKLOG.md`, for example caching the text layers.
2. Run `python -m vla_cu ncu` from the elevated shell on the int4 attention GEMM.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Cache or prepack the 16 text layers, which are 81 percent of a warm prefill.
2. A block mask in the fused attention kernel, so the state row enters the graph.
3. Three real camera frames, so the image rows carry scene content.
