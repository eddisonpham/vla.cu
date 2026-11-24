# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The library now goes from a real safetensors layer to a running block without a bench
file, the layer carries an optional CUDA graph, and the cooperative tail has been priced
at the attention widths. 230 device tests pass.

## Closed in this pass

- **Checkpoint builder.** `mlp_from_checkpoint` and `attention_from_checkpoint` read a
  layer prefix and cast to fp32. `GatedMlpBlock` adds the three-matrix SwiGLU shape, which
  is what SmolVLA stores, so `gate_proj` is not silently dropped (EXP-088, DEC-082).
- **Graph wrapper.** `Int4Linear.graphed()` captures one call, copies each input into the
  static buffer and replays. Opt in, output buffer reused, shape checked (EXP-089, DEC-083).
- **Coop at two widths.** 720/1920 and 960/2560 both cross at 10 tokens. Past it the
  larger width loses faster: 0.44x against 0.67x at 16 tokens (EXP-090).
- **Fixed.** The demo's fp32 stack reference resized its `out` buffer on first use.
  A deprecation warning exposed it; the buffer is now allocated at intermediate width.

## What works

- `pytest -q` with a GPU: 230 passed. No device: 40 passed, 190 skipped.
- `python -m vla_cu style`: 0 violations. `python -m vla_cu demo`: exit 0, no warnings.
- `python -m vla_cu coop`: both widths agree with the five launch path, exit 0.
- `python -m vla_cu sanitize`: cooperative kernel clean, extension unsupported.
- Demo latest run: matrix 6.07x, block graphed 3.6x, stack 4.55x, cosines 0.994, 0.995, 0.978.

## Open edges

- **Tiles are the wrong body below 16 tokens**, the batch size that matters (EXP-084).
- **The int4 CUDA extension has never been instrumented** on this platform (EXP-087).
- **The matrix row varies 0.073 to 0.194 ms across sessions**, which is too wide for one
  claim. Likely L2 residency or clock state, not yet isolated (BACKLOG).
- **`ncu` is blocked** by ERR_NVGPUCTRPERM, so all attribution is inference from timing.
- **The builder needs a prefix**, not a layer index, because two naming schemes exist.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: gated MLP inside the cooperative kernel, layer index resolver, or the L2 spread.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Put the gated MLP into the cooperative kernel, so a real layer is one launch.
2. A layer index to prefix resolver for the two SmolVLA naming schemes.
3. Isolate the matrix row spread before quoting a single speedup.
