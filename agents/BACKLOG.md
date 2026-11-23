# BACKLOG: ideas, suggestions, open questions

> The agent adds its own suggestions here when it spots something interesting in results. The human decides what to pursue.
> For each item: why it's interesting, what the human will learn, rough effort, what could go wrong.
> When an item is finished or dropped, move it to "Done / dropped" with the date and reason; do not delete it.

## Next up (starter chunks, see `agents/STARTER.md`)
- [x] Chunk 0: environment check (`scripts/env_check.py`) - done 2026-11-18, EXP-001
- [x] Chunk 1: timing and memory-budget utilities (`vla_cu/bench_utils.py`) - done 2026-11-22, EXP-008
- [x] Chunk 2: first Triton kernel, vector add - done 2026-11-22, `kernels/triton/vector_add.py`, EXP-007
- [x] Chunk 3: correctness test and bandwidth benchmark - done 2026-11-22, EXP-008: 86.2% of peak at 16M elements

## Environment blockers (fix before the kernels that need them)
- [x] ~~Get Triton running on Windows~~ DONE 2026-11-18: `triton-windows` 3.9.0 works on sm_120, bit-exact (EXP-002, DEC-008). WSL2 kept only as a fallback.
- [x] ~~Install the CUDA toolkit~~ DONE 2026-11-18: CUDA 12.9.2 in place, `nvcc` V12.9.86 and `compute-sanitizer` present (EXP-003, DEC-013).
- [x] ~~Give `nvcc` 12.9 a supported host compiler~~ DONE 2026-11-18: VS 2022 BuildTools 17.14.45, MSVC 14.44.35207, smoke compile and run OK (EXP-004, DEC-014).
- [x] ~~Install Nsight Systems / Nsight Compute~~ DONE 2026-11-18: Nsight Compute 2025.2.1 and Nsight Systems 2025.1.3 are installed under `C:\Program Files\NVIDIA Corporation`, but their executables sit in versioned `target-*` subfolders and are not on PATH.
- [x] ~~Resolve `ncu` and `nsys` without PATH edits~~ DONE 2026-11-18: the env task falls back to the versioned install folders (EXP-004).

## Foundations
- [ ] Vector add in CUDA C++ via `torch.utils.cpp_extension.load_inline`; compare against Triton
- [ ] Block-size sweep and vectorized loads (`float4`): how does bandwidth change?
- [ ] Profile with Nsight Systems, then Nsight Compute (read "Speed of Light" and memory sections)
- [ ] Parallel sum reduction (shared memory, then warp shuffles)
- [ ] Naive matmul, then coalesced, then shared-memory tiled

## Toward the project
- [ ] Pack 4-bit weights with per-group scales plus a PyTorch reference dequantize
- [x] int8 path: one scale per output channel, one value per byte, no bit surgery - done 2026-11-22, `kernels/triton/int8_gemv.py`, EXP-010: 92.4% of peak at 8192 x 16384
- [ ] One benchmark table across the ladder: fp32, bf16, int8, int4 on identical shapes
- [ ] Fused dequantize + GEMV (batch 1), Triton first, then CUDA C++
- [ ] Extract real layer shapes from SmolVLA for benchmarks
- [ ] Compare against bitsandbytes and torchao on the same shapes
- [ ] Roofline plot: achieved GB/s vs peak, per kernel version
- [ ] Fused RMSNorm + residual; small custom attention kernel for short sequences
- [ ] Register kernels as PyTorch custom ops (`torch.library`); capture a forward pass in a CUDA graph
- [ ] Plug into LeRobot Edge as a backend, gated by cosine similarity of at least 0.98 vs FP32

Out of scope by decision (DEC-016): 3, 5 and 6-bit bitstreams, and codebook formats
such as NF4 and FP4. Both are listed in `docs/ARCHITECTURE.md` section 5 and skipped
on purpose, not forgotten.

## Infrastructure
- [ ] `bench` with a kernel argument, JSON output and a small HTML report (vector add only so far)
- [ ] GitHub Actions for lint and CPU-only tests (no GPU runner; state this in the README)
- [x] Launch-overhead experiment - done 2026-11-22, EXP-009: 16.71 us from Python, 0.63 us per graph node
- [x] CUDA graph capture of a two-kernel GEMV chain - done 2026-11-22, EXP-011: 0.031 ms plain against 0.012 ms graphed, exact. A whole layer stack is still open
- [ ] BLOCK_K sweep for the int8 GEMV, and split-K for layers with few output rows (DEC-019)
- [ ] torch.compile as a baseline: blocked, inductor fails with `triton_key` ImportError on this venv (EXP-006)
- [x] `python -m vla_cu cuda-check`: which toolkit and host compiler a build would use, plus a verdict (2026-11-18, EXP-005, DEC-015).

## Open questions
- ~~Which GPU, and what is its peak bandwidth?~~ Answered in `HARDWARE.md`: RTX 5060 Laptop, 384 GB/s.
- Native Windows Triton (`triton-windows`) or WSL2? **Answered 2026-11-18: `triton-windows` (DEC-008).**
- Whether `triton-windows` supports `tl.dot`, TMA and autotuning on sm_120 - untested; test before relying on it for the fused GEMV.
- Which SmolVLA layer shapes do we target first (hidden size, intermediate size) once we leave toy tensors?

## Done / dropped
- [x] STARTER.md Part A: repo skeleton, venv, first commit (2026-11-18, commit 9b5dee6).
- [x] STARTER.md Chunk 0: environment check (2026-11-18, EXP-001).
