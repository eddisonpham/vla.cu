# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is closed except for profiling on the large shape, which needs the
human's elevated shell. The expert stack now carries a cosine at one token and at two,
and the 18 point gap to Triton has a SASS explanation plus one measured attempt that
failed for a named reason.

## Closed this pass

- **Two tokens run through the stack with q, k and a real softmax.** Cosine 0.99622 at
  depth 16 over 64.59 MB, 15.450 ms eager, 0.97 ms per layer (EXP-051).
- **Every kernel has a correctness check and a sanitizer pass.** int4 in EXP-044, int8 in
  EXP-046, 0 errors and 0 leaks each.
- **Dispatch is measured.** 16 layers at one token cost 4.335 ms eager and 0.541 ms
  graphed, so 186 us of each 217 us layer is Python (EXP-049, DEC-051).
- **The Triton gap is a reduction fact.** Both kernels read weights 4 or 8 bytes at a
  time; the native one has no shared memory, no barrier and five shuffles at the end,
  while Triton stages a 4-row tile through SMEM with 4 barriers and 12 butterflies
  (EXP-050, corrected: its 16 `LDG.E.128` are activations).
- **Reading the row as int32 does not fix it.** The wide mapping reached 56.3 percent
  against 73.1 staged and 87.9 native, with 0 `LDG.E.128` and SMEM still allocated
  (EXP-052, DEC-053).

## What works

- 27 tasks, `pytest -q` 121 passed with vcvars active, 96 passed and 23 skipped without.
- `plateau` median of 5: staged Triton 73.1, wide Triton 56.3, native unroll 87.9 percent
  at 4096 x 8192.
- `scale` prints four stages: shapes, group sizes, one-token chain with a graph, two-token
  chain.
- `sass` counts both Triton mappings and all three native variants in one table.
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.
- `compute-sanitizer --tool memcheck` over the int4 tests: 0 errors (EXP-052).

## Still open

- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **The wide mapping has no 16 byte weight load** and nobody has asked the compiler why
  (EXP-052).
- **No rotary rotation** in the two-token block, so its scores carry content, not position
  (DEC-052).
- **The two-token chain is eager only** and pays 8 launches per layer (EXP-051).
- **The largest shape is not trustworthy.** 16384 x 32768 spans 0.9 to 1.4 ms across five
  repeats (EXP-047).

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: RoPE, or contiguity hints on the wide mapping.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Graph the two-token chain, since dispatch is now 8 launches per layer.
2. Add RoPE so the softmax weights match the model's positional behaviour.
3. Try `tl.multiple_of` and `tl.max_contiguous` on the wide load and re-read its listing.
