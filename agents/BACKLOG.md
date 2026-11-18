# BACKLOG: ideas, suggestions, open questions

> The agent adds its own suggestions here when it spots something interesting in results. The human decides what to pursue.
> For each item: why it's interesting, what the human will learn, rough effort, what could go wrong.
> When an item is finished or dropped, move it to "Done / dropped" with the date and reason; do not delete it.

## Next up (starter chunks, see `agents/STARTER.md`)
- [x] Chunk 0: environment check (`scripts/env_check.py`) - done 2026-11-18, EXP-001
- [ ] Chunk 1: timing and memory-budget utilities (`vla_cu/bench_utils.py`) - unblocked, any time
- [ ] Chunk 2: first Triton kernel, vector add - unblocked (Triton verified, EXP-002)
- [ ] Chunk 3: correctness test and bandwidth benchmark - unblocked with Chunk 2

## Environment blockers (fix before the kernels that need them)
- [x] ~~Get Triton running on Windows~~ DONE 2026-11-18: `triton-windows` 3.9.0 works on sm_120, bit-exact (EXP-002, DEC-008). WSL2 kept only as a fallback.
- [ ] Install the CUDA toolkit (nvcc + compute-sanitizer + Nsight Systems/Compute). Why: `CLAUDE.md` requires `compute-sanitizer` on every CUDA kernel, and `ncu` counters are the only honest way to explain a memory-bound result. Risk: ~3 GB download, must match driver 617.42.

## Foundations
- [ ] Vector add in CUDA C++ via `torch.utils.cpp_extension.load_inline`; compare against Triton
- [ ] Block-size sweep and vectorized loads (`float4`): how does bandwidth change?
- [ ] Profile with Nsight Systems, then Nsight Compute (read "Speed of Light" and memory sections)
- [ ] Parallel sum reduction (shared memory, then warp shuffles)
- [ ] Naive matmul, then coalesced, then shared-memory tiled

## Toward the project
- [ ] Pack 4-bit weights with per-group scales plus a PyTorch reference dequantize
- [ ] Fused dequantize + GEMV (batch 1), Triton first, then CUDA C++
- [ ] Extract real layer shapes from SmolVLA for benchmarks
- [ ] Compare against bitsandbytes and torchao on the same shapes
- [ ] Roofline plot: achieved GB/s vs peak, per kernel version
- [ ] Fused RMSNorm + residual; small custom attention kernel for short sequences
- [ ] Register kernels as PyTorch custom ops (`torch.library`); capture a forward pass in a CUDA graph
- [ ] Plug into LeRobot Edge as a backend, gated by cosine similarity of at least 0.98 vs FP32

## Infrastructure
- [ ] `python -m vla_cu bench <kernel>` CLI with JSON output and a small HTML report
- [ ] GitHub Actions for lint and CPU-only tests (no GPU runner; state this in the README)

## Open questions
- ~~Which GPU, and what is its peak bandwidth?~~ Answered in `HARDWARE.md`: RTX 5060 Laptop, 384 GB/s.
- Native Windows Triton (`triton-windows`) or WSL2? **Answered 2026-11-18: `triton-windows` (DEC-008).**
- Whether `triton-windows` supports `tl.dot`, TMA and autotuning on sm_120 - untested; test before relying on it for the fused GEMV.
- Which SmolVLA layer shapes do we target first (hidden size, intermediate size) once we leave toy tensors?

## Done / dropped
- [x] STARTER.md Part A: repo skeleton, venv, first commit (2026-11-18, commit 9b5dee6).
- [x] STARTER.md Chunk 0: environment check (2026-11-18, EXP-001).
