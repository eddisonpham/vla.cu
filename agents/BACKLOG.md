# BACKLOG: ideas, suggestions, open questions

> The agent adds its own suggestions here when it spots something interesting in results. The human decides what to pursue.
> For each item: why it's interesting, what the human will learn, rough effort, what could go wrong.
> When an item is finished or dropped, move it to "Done / dropped" with the date and reason; do not delete it.

## Next up (starter chunks, see `agents/STARTER.md`)
- [x] Chunk 0: environment check (`scripts/env_check.py`) - done 2026-11-18, EXP-001
- [x] Chunk 1: timing and memory-budget utilities (`vla_cu/bench_utils.py`) - done 2026-11-22, EXP-008
- [x] Chunk 2: first Triton kernel, vector add - done 2026-11-22, `kernels/triton/vector_add.py`, EXP-007
- [x] Chunk 3: correctness test and bandwidth benchmark - done 2026-11-22, EXP-008: 86.2% of peak at 16M elements

## Ran 2026-11-23 (full entries in `agents/EXPERIMENTS.md`, verdicts in `OPTIMIZATIONS.md` section 9)
- [x] EXP-017 H1: attainable ceiling is 346 to 360 GB/s, 90 to 94 percent of spec. Spread inside one configuration reaches 2x.
- [x] EXP-018 H6: block graph 0.018 to 0.019 ms against 0.053 raw and 0.073 to 0.082 fp32, cosine unchanged.
- [ ] EXP-019 H5: blocked. `ncu` needs administrator rights, `ERR_NVGPUCTRPERM`.
- [x] EXP-020 H2: Triton eviction hints inconclusive, the sign flipped between samples.
- [x] EXP-021 H3: fp16 group scales inconclusive, no measurable gain.
- [x] EXP-022 H4: unroll not supported. The `__ldcs` streaming variant is the fastest kernel in the repo.
- [x] EXP-023 H9: the expert is M = 50. Fifty GEMV calls 2.588 ms against one bf16 GEMM 0.018 ms.
- [x] EXP-024 H8: grouped launch 2.73x at 1.57 MB and bit identical, 1.03x at 56.7 MB.
- [x] EXP-025 H7: dropped by reasoning, a graph already removes the dispatch a persistent kernel would target.
- [x] EXP-026 H11: rotation lifts outlier cosine 0.98788 to 0.99586 at no byte cost.

## Next, ranked by the evidence (detail in `OPTIMIZATIONS.md` section 9)
- [ ] Pin clocks with `nvidia-smi -lgc` (needs admin) and repeat EXP-017, EXP-020, EXP-021, EXP-022.
- [ ] Wire the three measured wins into a default path: CUDA streaming kernel, grouped launch, block graph.
- [ ] Write an int4 GEMM that dequantizes into `tl.dot`, and sweep M from 1 to 64 for the GEMV crossover.
- [ ] Decide group 16 against padding to 768 for the 720-wide expert (DEC-030).
- [ ] Download `lerobot/smolvla_base` config and index, then re-run EXP-023 and EXP-026 on real weights.
- [ ] Make every table best-of-N with the run count stated (DEC-029).
- [ ] Explain the CUDA baseline regression, 0.240 in EXP-016 against 0.265 to 0.476 in EXP-022, once counters are available.

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
- [x] Pack 4-bit weights with per-group scales plus a PyTorch reference dequantize - done 2026-11-22, `vla_cu/quant.py`, EXP-013
- [x] int8 path: one scale per output channel, one value per byte, no bit surgery - done 2026-11-22, `kernels/triton/int8_gemv.py`, EXP-010: 92.4% of peak at 8192 x 16384
- [x] One benchmark table across the ladder: fp32, bf16, int8, int4 on identical shapes - done 2026-11-22, `python -m vla_cu matrix`, EXP-013
- [x] Fused dequantize + GEMV (batch 1) in Triton - done 2026-11-22, `kernels/triton/int4_gemv.py`, EXP-013. The CUDA C++ port is still open
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
- [x] BLOCK_K sweep and split-K for the int8 GEMV - done 2026-11-22, EXP-012: 256 to 1024 within 0.2 percent, split-K 6.53x below 32 rows and a loss above 256
- [x] Hoist the int4 group scale out of the inner loop - done 2026-11-23, EXP-014: 0.263 to 0.230 ms, 75.0 to 85.5 percent of peak. The tile plateaus at 4 rows
- [x] Wrap the kernels as torch custom ops and run one layer end to end - done 2026-11-23, EXP-015, DEC-022
- [x] Port the int4 GEMV to CUDA C++ and compare - done 2026-11-23, EXP-016, DEC-023
- [ ] CUDA-graph the MLP block: the fused path loses to fp32 only through dispatch (EXP-015). Queued as EXP-018.
- [ ] Resolve the compute-sanitizer failure on the CUDA extension, or run `ncu` on both kernels instead
- [ ] Stage the activation tile in shared memory and re-sweep the int4 row tile. Lower priority since EXP-014: the row tile already plateaus at 4, so activation traffic is not what limits the kernel
- [ ] `torch.compile` the custom op path and compare against the eager launches
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
