# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-24
**Last git commit:** see `git log -1`
**Bootstrap (`agents/STARTER.md`):** complete 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git, branch `main`

## Campaign

The prefix pipeline is priced, the first call is packed, and the deployed path is behind one
object with a cosine gate. What is left is content: a tokenizer, real frames, and torchao's
missing Windows wheel.

## Closed in this pass

- **The comparison.** bitsandbytes nf4 is measured beside this repo's int4: 0.319 against
  0.254 ms at 8192 x 16384, cosine 0.99568 against 0.99439. torchao cannot run (EXP-116).
- **The pack file.** 417.83 MB holds the int8 tower and the 16 text layers as bf16, second
  read 495.20 to 192.41 ms, bit exact (EXP-117, DEC-101).
- **The backend object.** `SmolVlaPolicy` prefills once at 123.34 ms and replays a step at
  2.421 ms, cosine 0.99996 against the fp32 expert chain, gated at 0.98 (EXP-118, DEC-102).

## What the numbers say

| Measurement | Value |
|---|---|
| whole model, one graph / eager | 11.740 / 32.166 ms, cosine 0.99990 |
| warm prefill, int8 tower and text held | 117.50 to 123.34 ms |
| pack second read / checkpoint second read | 192.41 / 495.20 ms |
| first cold touch of the pack file | 6740.62 ms, a disk read, not a code cost |
| one denoising step / ten | 2.421 / 24.21 ms |
| int4 GEMV vs bitsandbytes nf4, 8192 x 16384 | 0.254 / 0.319 ms |
| torchao weight-only int4 | no Windows wheel, `fbgemm-gpu-genai` |

## Gates

- GPU suite 271 passed in a vcvars shell, CPU suite 57 passed and 214 skipped, style 0, 49 tasks.
- The pack round trip is bit exact over 32 cache tensors and one 64 x 960 row block.
- `backend` exits nonzero below cosine 0.98; it reads 0.99996.

## What works

- `python -m vla_cu baselines` compares five paths on one matrix, error beside speed.
- `python -m vla_cu prepack` writes the pack file that `tower_for` and `backbone_weights` use.
- `python -m vla_cu backend` runs prefill plus a step and enforces the gate.
- `python -m vla_cu fullmodel` still runs 241 rows and 50 tokens in one graph.

## Open edges

- A truly cold start is a disk read either way: the pack helps only in the page cache.
- Text weights are 629.27 MB fp32 resident; the pack stores them as bf16, the cache does not.
- Frames and state are synthetic, there is no tokenizer, and the instruction is ignored.
- torchao needs WSL or Linux to join the comparison table.
- `ncu` needs the elevated shell; racecheck never returns on the cooperative kernel; the int4
  CUDA extension cannot be instrumented; rope wavelength still unresolved.

## What the human can do

1. Pick the next item from `agents/BACKLOG.md`.
2. Paste the GitHub description on the repo page: the old one claims a torchao benchmark that
   cannot run on Windows, and a title/description edit is outside git.
3. Reset the clock lock with `nvidia-smi -rgc` when the session ends.

## Best next options

1. Text weights at int8 in the cache and the pack, 157.84 MB against 629.27, and measure it.
2. A tokenizer and three real camera frames, so the 48 language rows and 192 image rows carry
   the observation rather than a fixed slice and noise.
3. torchao on WSL2, so the comparison table has both libraries.
