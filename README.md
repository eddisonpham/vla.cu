# vla.cu

Hand-written GPU kernels for **fused 4-bit dequantize + matmul**, targeting small
vision-language-action policies (SmolVLA-scale, ~450M parameters) running at
**batch size 1** on a single consumer GPU.

This is a learning project first. The point is that the author can explain every
line, every measurement, and every bottleneck — not that the repo is large.

## Why this problem

At batch 1 a decoder layer does one matrix-vector product per weight matrix: every
weight is read from memory exactly once and used for exactly one multiply-add.
So the kernel is **memory-bound**, not compute-bound, and the only thing that
matters is how fast weights can be streamed.

That is also why 4-bit weights help. SmolVLA-scale math:

| Precision | Weight bytes | Data read per forward pass | Floor at 384 GB/s |
|---|---|---|---|
| bf16 | 2 B/param | ~900 MB | ~2.3 ms |
| 4-bit + scales | ~0.5 B/param | ~225 MB | ~0.6 ms |

(Arithmetic, not measurements. The floor is `bytes / 384 GB/s`; see
`agents/HARDWARE.md` for where 384 GB/s comes from.)

The catch is that 4-bit weights are **packed**: two values per byte, with a scale
per group. Something has to unpack them before the multiply. Naive code
dequantizes to a temporary buffer (write it, then read it back — doubling the
traffic). The interesting kernel fuses the unpack into the matmul so the packed
bytes are read once and never materialized. That fusion is the whole project.

## Target hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 7.9 GB |
| Compute capability | 12.0 (sm_120, Blackwell) |
| SMs | 26 |
| Peak memory bandwidth | 384 GB/s (GDDR7, 128-bit) |
| Memory budget | **7 GB max** per process, enforced in code |
| OS / stack | Windows, Python 3.13, PyTorch 2.8.0+cu129, Triton 3.9.0 (via `triton-windows`) |

Details and sources: `agents/HARDWARE.md`.

## Status

Early. The environment and tooling are real; the kernels are not written yet.

| Piece | State |
|---|---|
| Repo skeleton, venv, git workflow | done |
| Chunk 0: environment check | done (`scripts/env_check.py`) |
| Triton on native Windows (sm_120) | done, verified by `scripts/triton_probe.py` |
| Chunk 1: timing + memory budget | next (`vla_cu/bench_utils.py`) |
| Chunk 2-3: first Triton kernel + benchmark | pending |
| 4-bit packing, fused dequant GEMV, CUDA C++ | not started |

The CUDA toolkit (`nvcc`, `compute-sanitizer`, Nsight) is **not installed yet**, so
no CUDA C++ kernel can be built or sanitized until it is.

## Layout

```
vla_cu/           Python package: shared utils (timing, memory budget), CLI
kernels/triton/   Triton kernels, one module per kernel
csrc/             CUDA C++ sources (loaded via torch.utils.cpp_extension)
scripts/          one-off checks (env_check.py, triton_probe.py)
tests/            correctness tests, awkward sizes included
bench/            benchmarks; results land in bench/results/ (gitignored)
agents/           memory: state, experiments, research, decisions, backlog
```

## How to run

```bash
# environment report
.venv/Scripts/python.exe scripts/env_check.py

# does Triton actually compile for this GPU?
.venv/Scripts/python.exe scripts/triton_probe.py
```

Expected: `VERDICT: triton works on this GPU`, `max abs err 0.0e+00` for both
`n=1000` and `n=1000003`.

PowerShell/CMD equivalents: `.venv\Scripts\python.exe <script>`.

## Ground rules

1. **7 GB GPU memory cap** on every script, enforced by `enforce_memory_budget()`.
2. **Correctness before speed.** No benchmark number without a passing test
   against a PyTorch reference, including awkward sizes (1, 1023, 1025, ...).
3. **Measured numbers only**, each with the GPU name and exact command. Peak
   bandwidth percentages are relative to 384 GB/s.
4. **Lesson-sized changes.** One kernel or one utility per commit, with the
   reasoning recorded in `agents/`.
5. **Sanitize CUDA kernels** with `compute-sanitizer` before trusting them.

## Layout of the docs

- `agents/STATE.md` — where the work stands.
- `agents/EXPERIMENTS.md` — every run, failures included: the prediction made
  beforehand, the measured result, and the explanation afterwards.
- `agents/RESEARCH.md` — lookups with sources; `DECISIONS.md` — why X over Y;
  `BACKLOG.md` — what could come next.

## Reference points

To compare against, once kernels exist: plain PyTorch (`torch.matmul` on bf16),
`bitsandbytes`, and `torchao`. Marlin and similar projects solve the
LLM version of this problem with much deeper engineering; this repo is a
from-scratch learning build, not a competitor.
