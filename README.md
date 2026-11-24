# vla.cu

Hand-written GPU kernels for fused 4-bit dequantize and matmul, aimed at small
vision-language-action policies (SmolVLA scale, about 450M parameters) running at
batch size 1 on one consumer GPU.

This is a learning project. The goal is that the author can explain every line and
every number, not that the repo grows large.

## Why this problem

At batch 1 each matrix becomes a matrix-vector product: every weight is read once
and used once. So the kernel is memory-bound, and the only thing that matters is
how fast weights stream from memory.

4-bit weights help because they shrink the bytes that must be streamed.

| Precision | Weight bytes | Read per forward pass | Floor at 384 GB/s |
|---|---|---|---|
| bf16 | 2 B per param | about 900 MB | about 2.3 ms |
| 4-bit plus group scales | about 0.5 B per param | about 225 MB | about 0.6 ms |

Those numbers are arithmetic, not measurements: bytes divided by 384 GB/s. The
source for 384 GB/s is in `agents/HARDWARE.md`.

The catch is that 4-bit weights are packed, two values per byte, with a scale per
group. Something has to unpack them. Naive code dequantizes into a temporary
buffer, which writes the weights again and reads them back, so the traffic saving
disappears. The kernel worth writing unpacks inside registers and multiplies
immediately. That fusion is the whole project.

## Target hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 7.9 GB |
| Compute capability | 12.0 (sm_120, Blackwell) |
| SMs | 26 |
| Peak memory bandwidth | 384 GB/s, GDDR7, 128-bit |
| Memory budget | 7 GB per process, enforced in code |
| Stack | Windows, Python 3.13, PyTorch 2.8.0+cu129, Triton 3.9.0 |

Details and sources: `agents/HARDWARE.md`.

## Status

The ladder is measured end to end. On 8192 x 16384 weights, batch 1, the fused 4-bit
dequant GEMV runs in 0.230 ms against 1.587 ms for the fp32 path, a 6.89x speedup at
85.5 percent of DRAM peak and a relative L2 error of 1.07e-1. The same kernel exists
twice, in Triton and in CUDA C++, and the two agree on every tested shape.

| Piece | State |
|---|---|
| Repo skeleton, venv, git workflow | done |
| Environment check | done |
| Triton on native Windows, sm_120 | done, verified by a real kernel |
| Task runner (`python -m vla_cu`) | done, 38 tasks, `python -m vla_cu --help` lists every one |
| Timing and 7 GB cap (`bench_utils`) | done, warmup plus CUDA events, 7 GB cap |
| Vector add kernel and tests | done, exact match from 0 to 1,000,003 elements |
| Vector add benchmark | done, 330.9 GB/s, 86.2 percent of 384, at 16M elements |
| Launch overhead, graph against Python | 16.71 us per launch from Python, 0.63 us per graph node |
| int8 GEMV with per-channel scales | done, 354.7 GB/s, 92.4 percent of 384, at 8192 x 16384 |
| Split-K and block size sweep | done, split-K 6.53x at 16 rows, neutral above 256 rows |
| 4-bit packing and reference dequantize | done, low nibble first, one scale per 64 weights |
| Fused 4-bit dequant GEMV | done, 0.230 ms at 8192 x 16384, 6.89x the fp32 path, 85.5 percent of peak |
| Full ladder matrix, fp32 to int4 | done, one table on identical weights and activations |
| Torch custom ops for both GEMVs | done, `vla_cu/ops.py`, opcheck clean, no GPU memory for the fake |
| SmolVLA-shaped MLP block end to end | done, 960 to 2560 to 960, cos 0.9935 against fp32 |
| CUDA C++ port of the int4 GEMV | done, three variants, baseline, unroll, streaming |
| Streaming variant (`__ldcs`) | fastest int4 kernel here, 0.198 to 0.220 ms at 8192 x 16384 |
| Attainable bandwidth | 346 to 360 GB/s, 90 to 94 percent of the spec sheet, so the kernel sits at 91 to 95 percent of what the part gives (EXP-017) |
| Grouped launch across projections | one launch instead of three, 2.73x at 960 x 960, bit identical (EXP-024) |
| CUDA graph of an MLP block | 0.018 to 0.019 ms against 0.053 ms for the same two launches (EXP-018) |
| Action expert at M = 50 | 50 GEMV calls 2.588 ms against one bf16 GEMM 0.018 ms, so the expert needs a GEMM (EXP-023) |
| Fast path: native kernels plus a graph | one block 0.015 ms against fp32's 0.069, 4.6x, bit exact; four blocks 2.72 to 6.57x (EXP-028) |
| CUDA int8 kernel | warp per row, 16 weights per lane, streaming loads, capturable (EXP-027) |
| int4 GEMM with `tl.dot` | one launch beats M GEMV calls from M = 2, 25 to 51x at M = 50, cosine 1.00000 against fp32 (EXP-029) |
| Attention block, grouped qkv | two launches, 0.011 ms graphed against fp32's 0.052, bit exact (EXP-031) |
| Rotation inside the fast path | neutral on Gaussian weights and 38 percent slower per block, so opt-in (EXP-030) |
| Clock pinning | refused without administrator rights, paired ratios instead (EXP-032) |
| Two GEMV kernels chained in a CUDA graph | done, 0.031 ms plain against 0.012 ms graphed, bit exact |
| Shared input validator | one module fronts all seven wrappers, four missing checks closed (EXP-034) |
| Grouped int8 kernel | attention block reads 0.99994 at 0.019 ms graphed, 1.54x the int4 time (EXP-035) |
| Action block stack | four blocks at int8 everywhere read 0.99951 cosine at 0.140 ms, 3.94x the fp32 stack (EXP-036) |
| Seeded fuzz gate | 160 cases across five kernels, zero failures (EXP-037) |
| Versioned weight bundle | bit-exact round trip, 5.81x smaller than fp32 at int4, 5.68 ms to load four blocks (EXP-038) |
| Model widths | hidden 960 and intermediate 2560 confirmed from the SmolVLM2-500M config, expert 720 and 1920 from the 0.75 multiplier (RESEARCH 2026-11-23) |
| Cooperative layer tail | one launch instead of five, widths as launch parameters: 2.77x at 1 token, 1.11x at 8, 0.67x at 16 at 720/1920; both that width and 960/2560 cross at 10 tokens, and the larger width is 0.44x at 16 (EXP-090) |
| Tensor core body | tried as 16 x 16 WMMA tiles, 2x to 4x slower than the GEMV body, reverted (EXP-084) |
| Launch floor | 6 us of host gap and a 9 us empty kernel, so the int4 GEMM body is the cost at M = 50 (EXP-076) |
| int4 body split | time is linear in bytes at 20 to 23 GB/s, 5 to 6 percent of peak; the int8 twin reads more and runs 2.7x faster (EXP-079) |
| int8 tile menu | a chunk above 32 tokens gets a 16 row tile, the winner in every repeat at 720 x 1920 and 50 tokens (EXP-080) |
| CI and gates | 230 tests on a GPU, 40 passed and 190 skipped on a CPU runner, style and task list checked in both (EXP-081, EXP-088) |
| Public API | `Int4Linear`, the block classes, the checkpoint reader and the graph wrapper import from `vla_cu`; `python -m vla_cu demo` runs all of it (EXP-086) |
| Checkpoint builder | three lines from a safetensors layer to a running block, gated SwiGLU included, at one token or a chunk (EXP-088) |
| Gated cooperative tail | one launch for a three-matrix SwiGLU MLP, gate and up computed in one pass over the staged norm: 3.49x at 1 token at 720/1920 and 2.36x at 960/2560, crossing at 12 and 8 tokens (EXP-091) |
| Graph wrapper | `Int4Linear.graphed()` replays a captured call bit for bit, opt in, output buffer reused (EXP-089) |
| One-command sanitizer gate | `python -m vla_cu sanitize` runs memcheck over every CUDA kernel and returns nonzero on an error |

The CUDA toolkit and a host compiler nvcc accepts are installed. `python -m
vla_cu cuda-check` reports which ones a build would use and what it still needs.

## Layout

```
vla_cu/           package: task runner, shared utilities
kernels/triton/   Triton kernels, one module per kernel
csrc/             CUDA C++ sources, loaded with torch.utils.cpp_extension
scripts/          quotidian checks (environment, Triton probe)
tests/            correctness tests, awkward sizes included
bench/            benchmarks; results land in bench/results/, gitignored
docs/             architecture: diagrams and tables
agents/           memory: state, experiments, research, decisions, backlog
```

## Install

```bash
pip install -e . --no-deps
pip install -e .[triton,dev]
```

The base install needs torch only. The Triton extra resolves to `triton-windows` on
Windows, because the official wheel has no Windows build.

## Use it in your model

```python
import torch
from vla_cu import Int4Linear, mlp_from_checkpoint, open_checkpoint, pack_int4, int4_gemv

weight = torch.randn((768, 1920), device="cuda") * 0.02
x = torch.randn(1920, device="cuda")

packed, scales = pack_int4(weight)
y = int4_gemv(packed, scales, x)

layer = Int4Linear(weight)
y = layer(x)
y = layer(x.expand(8, -1))

fast = layer.graphed()
y = fast(x)

checkpoint = open_checkpoint("model.safetensors")
block = mlp_from_checkpoint(checkpoint, "model.language_model.layers.0.mlp")
out = torch.empty(block.hidden_width, device="cuda")
block.into(x, out)
```

`python -m vla_cu demo` runs all three levels on synthetic weights at the SmolVLA
widths and prints time and accuracy side by side. Every object here is inference only:
fp32 activations, one CUDA device, no bias, no backward. `graphed()` is opt in and
returns the captured buffer, so clone it to keep it. The builder takes the prefix that
ends at the block; `gate_proj` in the checkpoint selects the gated block.

## How to run

```bash
python -m vla_cu --help
python -m vla_cu env
python -m vla_cu probe
python -m vla_cu gemv
python -m vla_cu layer
python -m vla_cu coop
python -m vla_cu coopgated
python -m vla_cu style
```

The `cuda` task compiles `csrc/int4_gemv.cu` on first use, so it needs `cl.exe` and
ninja on PATH: run it from a shell where `vcvars64.bat` has been called (EXP-004).

Expected: `VERDICT: triton works on this GPU`, with `max abs err 0.0e+00` for both
tensor sizes. Architecture and the kernel roadmap: `docs/ARCHITECTURE.md`.

## Checks and CI

| Gate | Command | Needs |
|---|---|---|
| Style and doc invariants | `python -m vla_cu style` | nothing |
| Task registry and help | `python -m vla_cu --help` | nothing |
| Test suite | `python -m pytest -q` | a CUDA device, except the 35 tests marked `cpu` |
| CUDA memory errors, every kernel | `python -m vla_cu sanitize` | a vcvars shell; the cooperative kernel is instrumented, the torch extension cannot be attached (EXP-016) |
| Race conditions | not usable on the cooperative kernel, see EXP-081 | |

`.github/workflows/ci.yml` runs on `ubuntu-latest` with the CPU torch wheel: the style
check, the task list and the suite. `tests/conftest.py` skips every device test there with
a reason, because `torch.cuda.is_available()` can read true while `device_count()` is zero
when no device is visible.

## Ground rules

1. 7 GB GPU memory cap on every script, enforced by `enforce_memory_budget()`.
2. Correctness before speed. No benchmark without a passing test against PyTorch,
   including awkward sizes (1, 1023, 1025).
3. Measured numbers only, each with the GPU name and the exact command. Bandwidth
   percentages are relative to 384 GB/s.
4. One kernel or one utility per commit, with the reasoning recorded in `agents/`.
5. No comments in code. Docstrings are short, and deeper explanation lives in
   `docs/`.
6. Sanitize CUDA kernels with `compute-sanitizer` before trusting them.
7. `python -m vla_cu style` is a gate: zero violations before a push.

## Where the reasoning lives

| File | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | layers, naive vs fused traffic, roofline, kernel roadmap |
| `agents/STATE.md` | where the work stands |
| `agents/EXPERIMENTS.md` | every run, failures included: prediction, result, explanation |
| `agents/RESEARCH.md` | lookups with sources and dates |
| `agents/DECISIONS.md` | why one option was chosen over another |
| `agents/BACKLOG.md` | what could come next |

## Reference points

Plain PyTorch, `bitsandbytes`, and `torchao` are the comparisons once kernels
exist. Marlin and similar projects solve the language-model version of this
problem with far deeper engineering. This repo is a from-scratch learning build,
not a competitor.
