# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is closed except for profiling on the large shape, which needs the
human's elevated shell. The attention block is causal, its scores cover positions, the rope
and norm constants come from the model config, and the chain now runs the expert's real 50
token chunk. That last step moved the bottleneck from dispatch to the per-token GEMV.

## Closed this pass

- **The chain runs at 50 tokens, the real chunk length.** 336.872 ms eager against 25.120 ms
  graphed at depth 16, exact, cosine 0.99637. A layer is 200 launches, so 1.57 ms graphed
  is 7.85 us per launch (EXP-060).
- **`gqa_mix` was mixing over key heads, not tokens.** `torch.bmm` batched on tokens, so
  scores were [tokens, heads, kv heads]. The causal mask exposed it; einsums fix it, and
  scores are [heads, tokens, tokens] (EXP-057, DEC-059).
- **The chain is causal.** Cosine falls with the token count, 0.99783 at two and 0.99637 at
  fifty, which is what a real score matrix does (EXP-057, EXP-060).
- **The RMSNorm epsilon is the model's.** 1e-6 corrected to 1e-05 from the config, cosine
  unchanged at five decimals (EXP-058, DEC-060).
- **The real shapes are launch bound.** Every int4 and int8 path on the 720 to 2560 wide
  expert matrices lands at 0.021 to 0.031 ms while bytes move 2x, so ~22 us is dispatch
  (EXP-061).
- **Theta 100000 with the Llama pairing.** Read from the SmolVLM2 text config, so the old
  adjacent pairing was wrong (EXP-055, DEC-056).
- **Every kernel has a correctness check and a sanitizer pass.** int4 EXP-044, int8
  EXP-046, the wide Triton kernel EXP-052 and EXP-056, 0 errors each.

## What works

- 27 tasks, `pytest -q` 125 passed with vcvars active, 96 passed and 23 skipped without.
- `scale` prints four stages plus the 50 token block, with eager, graph, cosine, relative
  L2 and an exactness column.
- `plateau` median of 5: staged Triton 63.6 to 73.1, wide ieee 11.3 to 16.1, wide tf32
  28.3, native unroll 86.8 to 88.0 percent at 4096 x 8192 across runs.
- `sass` counts three Triton mappings and three native variants in one table.
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.

## Still open

- **A 50 token layer is 200 launches.** Batching the projections over tokens would make it
  4, with the same weight traffic (EXP-060).
- **Each weight is read once per token.** A real int4 GEMM at M = 50 would read it once per
  chunk, which is what the expert runs.
- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **The attention mix is torch ops, not a kernel.** Two einsums per layer inside the graph.
- **The wide mapping** lost on load width, reduction and precision; the staged mapping
  remains the one to keep (EXP-052, 054, 056, 059).
- **Run drift reaches 25 percent** on eager paths, so only numbers inside one output
  block are comparable.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: a batched GEMV over tokens, or the int4 GEMM.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Batch the int8 and int4 GEMV over the token axis, so a layer is 4 launches.
2. Write the int4 GEMM at M = 50 and compare it against 50 GEMV launches.
3. Put the causal mix in a Triton kernel, to price the torch einsums.
