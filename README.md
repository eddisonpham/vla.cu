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

The environment is real and the first project kernel exists: a fused int8 GEMV with
per-channel scales, at 92.4 percent of peak memory bandwidth on a shape that does
not fit in cache. The 4-bit path, the actual target, is not started.

| Piece | State |
|---|---|
| Repo skeleton, venv, git workflow | done |
| Environment check | done |
| Triton on native Windows, sm_120 | done, verified by a real kernel |
| Task runner (`python -m vla_cu`) | done, tasks `env`, `probe`, `cuda-check`, `bench`, `launch`, `gemv`, `graph` |
| Timing and 7 GB cap (`bench_utils`) | done, warmup plus CUDA events, 7 GB cap |
| Vector add kernel and tests | done, exact match from 0 to 1,000,003 elements |
| Vector add benchmark | done, 330.9 GB/s, 86.2 percent of 384, at 16M elements |
| Launch overhead, graph against Python | 16.71 us per launch from Python, 0.63 us per graph node |
| int8 GEMV with per-channel scales | done, 354.7 GB/s, 92.4 percent of 384, at 8192 x 16384 |
| Two GEMV kernels chained in a CUDA graph | done, 0.031 ms plain against 0.012 ms graphed, bit exact |
| 4-bit packing, fused dequant GEMV | not started |

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

## How to run

```bash
python -m vla_cu env
python -m vla_cu probe
python -m vla_cu cuda-check
python -m vla_cu bench
python -m vla_cu launch
python -m vla_cu gemv
python -m vla_cu graph
```

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
