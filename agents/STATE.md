# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is closed except for profiling on the large shape, which needs the
human's elevated shell. The expert stack now carries a graphed two-token chain with
positional scores, and both attempts to close the Triton gap from the load side are
measured negatives with listings.

## Closed this pass

- **The two-token chain is graphed and exact.** 24.880 ms eager against 2.818 ms graphed
  at depth 16, so 1.555 ms per layer was 0.176 ms of work (EXP-053, DEC-055).
- **q and k rotate by row position.** Cosine unchanged because the reference rotates the
  same way; the rotation costs 590 us per layer eager and 49 us graphed (EXP-053, DEC-054).
- **Contiguity hints did not widen the load.** Still 0 `LDG.E.128`, LDS 2 to 8, 3x slower,
  reverted (EXP-054).
- **Every kernel has a correctness check and a sanitizer pass.** int4 in EXP-044, int8 in
  EXP-046, the wide Triton kernel in EXP-052, 0 errors each.
- **The Triton gap is a reduction fact, not a load width fact.** Both read weights 4 or 8
  bytes at a time; native has no SMEM, no barrier, five shuffles at the end (EXP-050,
  corrected).

## What works

- 27 tasks, `pytest -q` 123 passed with vcvars active, 96 passed and 23 skipped without.
- `plateau` median of 5: staged Triton 66.6 to 73.1, wide 51.0 to 56.3, native unroll 86.9
  to 87.9 percent at 4096 x 8192 across two runs on different thermal states.
- `scale` prints four stages, the last with eager, graph, cosine and an exactness column.
- `sass` counts three Triton mappings and three native variants in one table.
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.

## Still open

- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **rope theta 10000 is an assumption.** The checkpoint holds weights only (DEC-054).
- **The wide mapping** has no 16 byte weight load and no satisfying rewrite yet (EXP-052,
  EXP-054).
- **Two tokens is the only multi-token case.** The score matrix is 2 x 2 (EXP-053).
- **The largest shape is not trustworthy.** 16384 x 32768 spans 0.9 to 1.4 ms across five
  repeats (EXP-047).

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: three tokens, or a reduction that needs no reshape.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Run the chain at three or more tokens, where the softmax has to choose among more keys.
2. Replace the reshape reduction in the wide mapping with a `tl.dot` against a ones vector.
3. Read the real rope theta from the released model config and correct DEC-054.
