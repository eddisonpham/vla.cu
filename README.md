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
| Task runner (`python -m vla_cu`) | done, tasks `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph`, `tune`, `matrix`, `layer`, `cuda`, `stream`, `grouped`, `m50`, `rotate`, `fastpath`, `crossover`, `attn`, `block`, `fuzz`, `bundle`, `style` |
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

## How to run

```bash
python -m vla_cu env
python -m vla_cu probe
python -m vla_cu cuda-check
python -m vla_cu bench
python -m vla_cu launch
python -m vla_cu gemv
python -m vla_cu graph
python -m vla_cu tune
python -m vla_cu matrix
python -m vla_cu layer
python -m vla_cu cuda
python -m vla_cu stream
python -m vla_cu grouped
python -m vla_cu m50
python -m vla_cu rotate
python -m vla_cu fastpath
python -m vla_cu crossover
python -m vla_cu attn
python -m vla_cu block
python -m vla_cu fuzz
python -m vla_cu bundle
python -m vla_cu style
```

The `cuda` task compiles `csrc/int4_gemv.cu` on first use, so it needs `cl.exe` and
ninja on PATH: run it from a shell where `vcvars64.bat` has been called (EXP-004).

Expected: `VERDICT: triton works on this GPU`, with `max abs err 0.0e+00` for both
tensor sizes. Architecture and the kernel roadmap: `docs/ARCHITECTURE.md`.

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
