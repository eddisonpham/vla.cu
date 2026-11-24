# vla.cu

Hand-written GPU kernels for fused 4-bit dequantize and matmul, aimed at small
vision-language-action policies (SmolVLA scale, about 450M parameters) running at
batch size 1 on one consumer GPU.

This is a learning project. The goal is that the author can explain every line and
every number, not that the repo grows large. MIT licensed.

## What it costs at batch 1

SmolVLA, three frames and a state row through the prefix, then a 50 token action chunk,
with every weight at int8 and every step in a CUDA graph:

| Measurement | Value | Entry |
|---|---|---|
| Whole model, one graph | 11.72 ms, against 32.23 ms eager | EXP-114 |
| One denoising step | 2.42 ms graphed, 24.21 ms for ten | EXP-118 |
| Warm prefix build | 117.50 ms, 88.36 for the frames and 29.15 for the text layers | EXP-115 |
| Prefix build from the pack file | 192.41 ms, against 495.20 from the checkpoint, both page cached | EXP-117 |
| End to end cosine | 0.99990 for the whole model, 0.99992 for the prefix | EXP-112, EXP-114 |
| int4 GEMV, 8192 x 16384 | 0.254 ms, against 1.473 fp32 and 0.319 bitsandbytes nf4 | EXP-116 |

Every number, including the negative results, is logged with its command in
`agents/EXPERIMENTS.md`.

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

<details>
<summary>Every measured row, the long table</summary>

| Piece | State |
|---|---|
| Repo skeleton, venv, git workflow | done |
| Environment check | done |
| Triton on native Windows, sm_120 | done, verified by a real kernel |
| Task runner (`python -m vla_cu`) | done, 49 tasks, `python -m vla_cu --help` lists every one |
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
| CI and gates | 264 tests on a GPU, 54 passed and 210 skipped on a CPU runner, style and task list checked in both (EXP-081, EXP-088, EXP-098) |
| Public API | `Int4Linear`, the block classes, the checkpoint reader and the graph wrapper import from `vla_cu`; `python -m vla_cu demo` runs all of it (EXP-086) |
| Checkpoint builder | three lines from a safetensors layer to a running block, gated SwiGLU included, at one token or a chunk; a layer index resolves the backbone or expert stack (EXP-088, EXP-093) |
| Whole layer | one index returns both RMSNorm weights, attention, the gated MLP and the residual pair (EXP-094); `fused=True` runs its tail in one launch, measured slower than a captured chain (EXP-097) |
| Real 16 layer backbone | every backbone layer at one token, every path a captured graph: fp32 629.27 MB and 4.334 ms, int8 down 157.84 MB and 1.176 ms (3.69x), int4 down 140.57 MB and 1.101 ms (3.94x), cosine 0.99998 against 0.99965 (EXP-101) |
| Real expert at 50 tokens | all 16 expert layers on the GEMM path, cross attention on the odd ones reading the real 48 row VLM key and value cache: fp32 392.98 MB and 7.082 ms, int8 attention and down 98.68 MB and 2.290 ms (3.09x, cosine 0.99997), int4 down 88.31 MB and 2.698 ms (2.63x, cosine 0.99847), int4 attention and down 76.71 MB and 3.516 ms (2.01x, cosine 0.99613) (EXP-104, EXP-105) |
| int4 attention on real weights | 4-bit q, k, v and o at 50 tokens costs 1.000 ms more than int8 attention (3.290 against 2.290 ms) for 11.60 MB less and a lower cosine (0.99761 against 0.99997), so int8 attention stays the default (EXP-105) |
| Whole model, one graph | 16 backbone layers over 241 prefix rows from the int8 tower, state row block mask included, plus 16 expert layers over 50 action tokens, int8 attention and down, captured together: 11.720 ms against 32.231 eager (2.75x), 256.52 MB, cosine 0.99990 (EXP-114) |
| Full 241 row prefix | 3 frames through the real SigLIP tower and connector (192 rows), 48 language rows, 1 state row: cache 9.87 MB, and with the tower at int8 the end to end cosine is 0.99992 at int8 down (2.373 ms) and 0.99852 at int4 down (2.816 ms), against an fp32 tower and an fp32 expert (EXP-107, EXP-112) |
| SigLIP tower precision | the tower is 172.87 MB of bf16 in the checkpoint and is read in full at any stored width: fp32 build 233.48 ms plus 157.96 ms for 3 frames, int8 build 238.09 ms plus 87.53 ms at cosine 0.99956, int4 at 0.94484, so int8 moves the prefill 782.52 to 729.55 ms and int4 loses on both axes (EXP-108) |
| Tower cache | the first call builds the tower in 253.37 ms and every later call returns it in 0.005 ms; with the 16 text layers held too, a warm int8 prefill is 117.50 ms against 328.58 on the first call and 782.52 before anything was cached (EXP-112, EXP-115) |
| Text weight cache | the 16 backbone layers are read once per process: 603.84 ms on the first call, then 29.15 ms per prefill, so the 384.45 ms it used to cost was almost all the read (EXP-115) |
| Activation precision | the 4-bit tower at group 16 reads row cosine 0.97059 with exact activations and 0.97059 with the kernel's fp16 operands, so the 4-bit weight is the whole error; int8 activations read 0.96827, so a W4A8 kernel would lose 0.0023 rather than recover it (EXP-113) |
| int4 tower groups | group 64, 32 and 16 read row cosine 0.94484, 0.95583 and 0.97059, so the error is the 4-bit code and not the scale granularity; group 16 also grows to 78.21 MB and 146.78 ms for 3 frames, both behind int8, and stays below the 0.98 gate (EXP-111) |
| Prefill split | vision 393.72 ms against 388.80 ms for the 16 text layers at fp32, so the tower is about half of a prefill and the string of 16 backbone weight reads is the other half (EXP-108) |
| 10 step denoising loop | the 241 row cache built once at 818.98 ms, then 10 graphed expert steps at 2.258 ms each, so a chunk is 841.37 ms against 8212.38 ms if the cache were rebuilt every step (EXP-109) |
| Tower cross-check | all 197 vision tensors load into transformers 5.24.1 `SiglipVisionModel` with nothing missing; patches agree to 2.16e-04 max abs and 4.42e-06 relative L2, connector rows to 6.77e-05, cosine 1.00000 (EXP-110) |
| int4 attention rung | int4 attention with int8 down reads 0.93242 over four blocks, against 0.90671 with int4 down and 0.97606 with int8 attention, so 4-bit attention costs more accuracy than 4-bit down (EXP-103) |
| Captured tail at 720 | at one token the cooperative tail reads 0.58x the captured chain at 720/1920 and 0.50x at 960/2560, so the smaller width does not change the verdict (EXP-100) |
| Fused tail in the block | `GatedMlpBlock.tail_into` runs add, norm, gate, up, down and add in one cooperative launch (EXP-095) |
| Gated cooperative tail | one launch for a three-matrix SwiGLU MLP, gate and up computed in one pass over the staged norm: 3.49x at 1 token at 720/1920 and 2.36x at 960/2560, crossing at 12 and 8 tokens (EXP-091) |
| Graph wrapper | `Int4Linear.graphed()` replays a captured call bit for bit, opt in, output buffer reused; `graphed(x)` binds the input buffer and drops the copy (EXP-089, EXP-096) |
| Where capture pays | with a copied input, 2.20x at 0.09 MB of packed weights down to 0.94x at 18.87 MB (EXP-092); bound to the caller's buffer, 4.18x down to 1.00 to 1.02x, so the copy was the loss (EXP-096) |
| One-command sanitizer gate | `python -m vla_cu sanitize` runs memcheck over every CUDA kernel and returns nonzero on an error |
| Reproducible first call | `python -m vla_cu prepack` writes the tower and text weights to one 417.83 MB file, a 2.57x faster second read, bit exact (EXP-117) |
| Comparison against bitsandbytes | nf4 reads 0.319 ms against this repo's int4 at 0.254 ms on 8192 x 16384, cosine 0.99568 against 0.99439 (EXP-116) |
| Backend object | `SmolVlaPolicy` prefills once and replays a graph per step at cosine 0.99996, gated at 0.98 (EXP-118) |

</details>

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
Windows, because the official wheel has no Windows build. `visioncheck` is the one task
that needs `transformers`, which `requirements.txt` installs and the CPU gate never
imports. `pip install -e .[compare]` brings `bitsandbytes`, which is the alternative
`python -m vla_cu baselines` measures; torchao is also a target there but has no working
Windows wheel for weight-only int4.

## Use it in your model

```python
import torch
from vla_cu import Int4Linear, layer_from_checkpoint, mlp_from_checkpoint, open_checkpoint, pack_int4, int4_gemv

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
block = mlp_from_checkpoint(checkpoint, 3, part="expert")
out = torch.empty(block.hidden_width, device="cuda")
block.into(x, out)

whole = layer_from_checkpoint(checkpoint, 3, part="expert")
```

`python -m vla_cu demo` runs all three levels on synthetic weights at the SmolVLA
widths and prints time and accuracy side by side. Every object here is inference only:
fp32 activations, one CUDA device, no bias, no backward. `graphed()` is opt in and
returns the captured buffer, so clone it to keep it. The builder takes a layer index and
a stack name, `part="vlm"` for the SmolVLM2 backbone or `part="expert"` for the action
expert, and a full prefix still works; `layer_from_checkpoint` returns the whole layer,
both norms and both residual adds included, and `gate_proj` selects the gated block.
`GatedMlpBlock.tail_into` runs the gated tail in one cooperative launch, and
`LayerBlock.into(..., fused=True)` uses it, though a captured chain is faster (EXP-097).
`graphed(x)` binds the capture to that buffer, so a caller who writes it in place skips
the copy.

## How to run

```bash
python -m vla_cu --help
python -m vla_cu env
python -m vla_cu probe
python -m vla_cu gemv
python -m vla_cu layer
python -m vla_cu coop
python -m vla_cu coopgated
python -m vla_cu whole
python -m vla_cu realstack
python -m vla_cu expertstack
python -m vla_cu fullmodel
python -m vla_cu fullprefix
python -m vla_cu tower
python -m vla_cu denoise
python -m vla_cu visioncheck
python -m vla_cu prepack
python -m vla_cu baselines
python -m vla_cu backend
python -m vla_cu graphed
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
| Test suite | `python -m pytest -q` | a CUDA device, except the 54 tests that run without one |
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
| `docs/INTEGRATION.md` | the contract a policy runtime calls: inputs, outputs, the 0.98 gate |
| `agents/STATE.md` | where the work stands |
| `agents/EXPERIMENTS.md` | every run, failures included: prediction, result, explanation |
| `agents/RESEARCH.md` | lookups with sources and dates |
| `agents/DECISIONS.md` | why one option was chosen over another |
| `agents/BACKLOG.md` | what could come next |

## Reference points

Plain PyTorch, `bitsandbytes`, and `torchao` are the comparisons. bitsandbytes nf4 is
measured beside this repo's int4 in `python -m vla_cu baselines`: on one 8192 x 16384
matrix the two read 0.319 and 0.254 ms at cosine 0.99568 and 0.99439. torchao's
weight-only int4 does not run here, because it needs `fbgemm-gpu-genai` and there is no
Windows wheel, so that row reports the ImportError instead of a time. Marlin and similar
projects solve the language-model version of this problem with far deeper engineering.
This repo is a from-scratch learning build, not a competitor.
