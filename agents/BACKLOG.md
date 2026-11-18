# BACKLOG: ideas, suggestions, open questions

> The agent adds its own suggestions here when it spots something interesting in results. The human decides what to pursue.
> For each item: why it's interesting, what the human will learn, rough effort, what could go wrong.
> When an item is finished or dropped, move it to "Done / dropped" with the date and reason; do not delete it.

## Next up (starter chunks, see CLAUDE.md section 7)
- [ ] Chunk 0: environment check (`scripts/env_check.py`)
- [ ] Chunk 1: timing and memory-budget utilities (`vla_cu/bench_utils.py`)
- [ ] Chunk 2: first Triton kernel, vector add
- [ ] Chunk 3: correctness test and bandwidth benchmark

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
- Which GPU does the human have, and what is its peak bandwidth? (answer after Chunk 0)

## Done / dropped
- (nothing yet)
