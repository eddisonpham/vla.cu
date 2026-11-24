# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The whole policy path is one measured graph: 16 backbone layers produce the prefix cache,
16 expert layers consume it. The attention and down precision axes are both settled by
measurement, and the remaining gaps are the prefix content and the denoising loop.

## Closed in this pass

- **The whole model in one graph.** 48 bidirectional prefix rows through 16 int8 backbone
  layers write post-rope K and V per layer; 16 int8 expert layers read them on 50 action
  tokens. One graph 4.899 ms, two graphs 4.938 ms, eager 19.638 ms, cosine 0.99992
  (EXP-106, DEC-093).

## What the numbers say

| Path | ms | x eager |
|---|---|---|
| eager, 32 layers of Python dispatch | 19.638 | 1.00x |
| two graphs, backbone then expert | 4.938 | 3.98x |
| one graph, whole path | 4.899 | 4.01x |

Stored 256.52 MB, end to end cosine 0.99992, rel L2 1.30e-02, every path bit exact.

| Component | Evidence |
|---|---|
| int8 attention | default; int4 costs 1.000 ms and 0.0024 cosine (EXP-105) |
| int4 down | 0.99847 at 50 tokens, 0.408 ms slower than int8 down (EXP-105) |
| prefix content | real 48 row cache and random rows give the same cosine (EXP-104) |

## Gates

- GPU suite 250 passed, CPU suite 48 passed and 202 skipped, style 0, clocks pinned at 1987 MHz.
- `fullmodel`, `expertstack`, `realstack`, `block` exit 0; graphs bit exact.
- 42 tasks, every one imported by `tests/test_tasks.py`.
- `compute-sanitizer --tool memcheck` on the int4 attention test: 0 errors (EXP-105).

## What works

- `python -m vla_cu fullmodel` runs the whole path in one graph and in two, against eager.
- `python -m vla_cu expertstack` runs four attention and down mixes on the real prefix.
- `vla_cu.prefix.prefix_inputs` and `prefix_cache` give the rows and the fp32 cache.

## Open edges

- **One action chunk, not ten steps.** The real inference runs 10 denoising steps with the
  prefix cache held, so a real chunk is about ten expert passes plus one prefix pass.
- **The prefix is 48 language rows**, not the deployed 241 (192 image, 48 language, 1 state).
- **Rope wavelength is unresolved.** LeRobot rotates at 10000, the config says 100000.
- **Timing moves between runs.** The same int8 expert path read 1.977, 2.290 and 2.290 ms
  across three runs, so compare ratios inside one run.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM; the int4 extension cannot be instrumented on Windows.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the int4 attention GEMM.
2. Pick the next experiment from BACKLOG, for example the 10 step loop.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. The 10 step loop with the prefix cache held, which is the real per action cost.
2. The full 241 row prefix with the SigLIP tower and three real frames.
3. int4 down on the whole path, to see if the 0.99847 expert cosine holds model level.
