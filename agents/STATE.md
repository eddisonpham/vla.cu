# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-23
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The limitations pass is closed except for profiling on the large shape, which needs the
human's elevated shell. The attention block is causal now, its scores really cover
positions, the rope and norm constants come from the model config, and the wide Triton
mapping has been measured on load, reduction and precision, losing all three times.

## Closed this pass

- **`gqa_mix` was mixing over key heads, not tokens.** `torch.bmm` batched on tokens, so
  scores were [tokens, heads, kv heads]. The causal mask exposed it; two einsums fix it,
  and scores are [heads, tokens, tokens] (EXP-057, DEC-059).
- **The chain is causal.** Exact at every depth, 2, 3, 4 and 8 tokens, cosine 0.99783 at
  two down to 0.99712 at eight, graphed 2.723 to 4.897 ms for 16 layers (EXP-057).
- **The RMSNorm epsilon is the model's.** 1e-6 corrected to 1e-05 from the config, cosine
  unchanged at five decimals (EXP-058, DEC-060).
- **The wide dot at tf32.** 0.174 ms against ieee's 0.438 and staged 0.072 at 4096 x 8192,
  with max abs diff 4.00e-03, so it is a diagnostic (EXP-059).
- **Theta 100000 with the Llama pairing.** Read from the SmolVLM2 text config, so the old
  adjacent pairing was wrong (EXP-055, DEC-056).
- **Every kernel has a correctness check and a sanitizer pass.** int4 EXP-044, int8
  EXP-046, the wide Triton kernel EXP-052 and EXP-056, 0 errors each.

## What works

- 27 tasks, `pytest -q` 125 passed with vcvars active, 96 passed and 23 skipped without.
- `scale` prints four stages, the last one per token count, with eager, graph, cosine,
  relative L2 and an exactness column.
- `plateau` median of 5: staged Triton 63.6 to 73.1, wide ieee 11.3 to 16.1, wide tf32
  28.3, native unroll 86.8 to 88.0 percent at 4096 x 8192 across runs.
- `sass` counts three Triton mappings and three native variants in one table.
- `style` zero violations, `bundle` bit exact, `fuzz` 160 seeded cases clean.

## Still open

- **`ncu` on the large shape** needs the elevated shell: `python -m vla_cu ncu`.
- **The sequence is still short.** 8 tokens is the longest case; the expert runs 50
  action tokens, where the n x n scores start to cost more than the projections.
- **The attention mix is torch ops, not a kernel.** Two einsums per layer inside the graph.
- **The wide mapping** has no 16 byte weight load, a slower reduction and an unusable
  tf32 path; the staged mapping remains the one to keep (EXP-052, 054, 056, 059).
- **Run drift reaches 25 percent** on eager paths, so only numbers inside one output
  block are comparable.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell, then say what the report shows.
2. Pick the next question: a 50 token chunk, or a fused attention kernel.
3. Reset the lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Run the causal chain at 50 tokens, the expert's real action chunk length.
2. Replace the two einsums in `gqa_mix` with a small fused attention kernel.
3. Re-time the staged int4 kernel now that the causal chain is the headline path.
