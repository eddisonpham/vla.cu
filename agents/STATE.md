# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The kernels are measured, the fused tail is one cooperative launch, and the library now
has a public surface a stranger can use: `Int4Linear`, the block classes, the bundle and
the checkpoint reader, plus `python -m vla_cu demo` and `python -m vla_cu sanitize`.

## Closed in this pass

- **Public API.** `vla_cu` exports lazily to break the kernel import cycle. `Int4Linear`
  packs once and dispatches GEMV at one token, GEMM above it. Six tests plus three
  CPU surface tests (EXP-086, DEC-080).
- **Demo.** one matrix 2.54x, one block graphed 3.5x, four block stack 5.58x, cosines
  0.994, 0.995, 0.978, graphs bit exact (EXP-086).
- **One-command memcheck.** Cooperative kernel 0 errors. The int4 extension cannot be
  attached on Windows (`CUDA_ERROR_INVALID_HANDLE`), which the task reports as
  unsupported rather than clean or failed (EXP-087, DEC-081).
- **Fixed while testing.** The demo aliased its fp32 reference and read the wrong parity
  buffer; a pre-existing block test was flaky because it was unseeded.

## What works

- `pytest -q` with a GPU: 220 passed. No device: 40 passed, 180 skipped.
- `python -m vla_cu style`: 0 violations. `python -m vla_cu demo`: exit 0.
- `python -m vla_cu sanitize`: cooperative kernel clean, extension unsupported.
- `pip install -e .` plus `from vla_cu import Int4Linear` is the user path.

## Open edges

- **Tiles are the wrong body below 16 tokens**, the batch size that matters. A 64 token
  block with async copies and one K pipeline is the untried design (EXP-084).
- **The int4 CUDA extension has never been instrumented** on this platform (EXP-087).
- **No checkpoint to block builder**, so a real layer still needs a bench file.
- **`ncu` is blocked** by ERR_NVGPUCTRPERM, so all attribution is inference from timing.
- **CI proves style, imports and collection**, not kernel correctness.

## What the human can do

1. Run `python -m vla_cu ncu` from the elevated shell on the 1920 x 768 shape.
2. Choose: checkpoint builder, graph wrapper for `Int4Linear`, or the attention widths.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. A checkpoint to block builder, three lines from safetensors to a runnable layer.
2. A graph wrapper so the eager layer gets the 3.5x without caller code.
3. The cooperative kernel at the attention widths, 960 and 2560.
