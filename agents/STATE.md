# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is closed except for profiling on the large shape, which needs the
human's elevated shell. Every kernel has a correctness check and a sanitizer pass, the
scaling question is answered three ways, the expert stack carries an accuracy number at
depth 16 with attention in the chain, and the gap to Triton now has a SASS explanation.

## Closed this pass

- **Both CUDA kernels have a standalone sanitizer pass.** int4 in EXP-044, int8 in EXP-046,
  0 errors and 0 leaks each, from a torch-free header and a plain .cu harness.
- **Depth 16 carries an accuracy number with attention.** 16 real layers, int8 attention
  and int8 up, int4 down: 50.78 MB, cosine 0.99795, graph exact (EXP-048, EXP-049).
- **Dispatch is measured, not guessed.** 16 layers cost 3.475 ms eager and 0.497 ms
  graphed, so 186 us of 217 us per layer is Python (EXP-049, DEC-051).
- **The Triton gap is a memory path fact.** Triton stages its tile through shared memory
  with 4 and 8 byte loads; the native unroll issues 16 byte read-only loads and no SMEM,
  with comparable nibble arithmetic (EXP-050).
- **Profiling is one command.** `python -m vla_cu ncu` writes the vcvars wrapper and calls
  ncu with no shell quoting (DEC-049).

## What works

- 27 tasks, `pytest -q` 96 passed and 23 skipped outside vcvars, 119 passed inside it.
- `plateau` median of 5: native unroll 88.0 percent, Triton 69.4 at 4096 x 8192 (EXP-047).
- `scale` reports shape, group size and depth, with proj, layer and graphed columns.
- `sass` dumps both kernels and counts 344 to 544 instructions per function (EXP-050).
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.

## Still open

- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`. A normal
  shell reaches ncu and stops at `ERR_NVGPUCTRPERM`, which is the expected outcome.
- **The SASS reading is static.** Counters would confirm the shared memory round trip is
  the cost, and no run has read them yet.
- **The largest shape is not trustworthy.** 16384 x 32768 spans 0.9 to 1.4 ms across five
  repeats (EXP-047).
- **The chain is not a full block.** q and k are skipped at one token, and cross layers
  read a random prefix, so cosine covers the value path only (DEC-050).
- **No three layer stack at M = 50** on real weights; EXP-043 measures one matrix at a time.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: a 16 byte Triton rewrite, or a two token chain with q and k.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Read the two SASS loop bodies line by line and name the instruction that differs.
2. Add q and k with a two token softmax, so the chain covers real attention.
3. Stage the packed weights through shared memory in the native kernel to test 88 percent.
