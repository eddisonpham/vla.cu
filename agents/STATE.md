# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

Five improvement cycles are complete, EXP-034 to EXP-039. The temporary `CLAUDE.md`
section was removed after cycle 4. This pass closed the limitations that could be closed.

## Closed this pass

- Model widths are read, not assumed: 960 and 2560 from the SmolVLM2-500M config,
  expert 720 and 1920 from the 0.75 multiplier (DEC-043, RESEARCH 2026-11-23).
- The CUDA baseline regression does not reproduce: baseline 0.238 ms and stream
  0.208 ms at 8192 x 16384, stream at 94.7 percent of peak (EXP-040).
- `docs/PRIMER.md` deleted. The CI workflow deleted at the human's request (DEC-042);
  `python -m vla_cu style` stays as the local gate.
- compute-sanitizer now starts on the extension with `--target-processes all`. It
  reports `CUDA_ERROR_INVALID_HANDLE` on `cuKernelGetFunction` and no memory error
  (EXP-041). Eager loading with blocking launches hung past 600 s and is dropped.

## What works

- 22 tasks, `pytest -q` 96 passed and 23 skipped outside vcvars, 119 passed inside it.
- int8 grouped attention 0.99994 at 0.019 ms graphed (EXP-035).
- Four action blocks at int8 everywhere: 0.99951 cosine, 0.140 ms, 3.94x fp32 (EXP-036).
- `fuzz` 160 seeded cases, zero failures. `bundle` bit exact, 5.81x smaller than fp32.
- `style` zero violations, editable install works from outside the repo.

## Still open

- **The checkpoint weights are not downloaded.** The configs are read and the file is
  public, 865 MiB. Needs the human's go-ahead before a download.
- **No administrator rights.** `ncu` is denied and clocks cannot be pinned, so a
  difference under 6 percent is not a result (DEC-029, DEC-035).
- **No kernel-level sanitizer pass** on the two CUDA extensions (EXP-041).
- Group 64 does not divide the expert hidden width 720 (DEC-030), decision open.
- No CI. The local gate is manual by choice (DEC-042).

## What the human can do

1. Approve the 865 MiB `model.safetensors` download, and set `HF_TOKEN` only if the
   public route fails.
2. Run an elevated shell once: `nvidia-smi -pm 1`, `nvidia-smi -lgc 2000`, then launch
   `ncu` and `compute-sanitizer` from a terminal started as Administrator.
3. Decide padding 720 to 768 against group 16 for the expert (DEC-030).

## Best next options

1. Convert three checkpoint layers, cosine per layer, with and without rotation.
2. The missing rung from EXP-036: int4 attention with int8 down.
3. Settle DEC-030 with a 720-wide quality and time comparison.
