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
- [x] Pin clocks with `nvidia-smi -lgc` (needs admin). Done 2026-11-23, EXP-042: holds at 1980 to 1987 MHz, 0.35 percent spread. The repeat of EXP-017, EXP-020, EXP-021 and EXP-022 is still open, listed below.
- [x] Wire the measured wins into a default path. Done 2026-11-23, EXP-027 and EXP-028: native int8 and int4 streaming kernels, launched on the torch stream so they capture, one block 0.015 ms against fp32's 0.069, bit exact.
- [x] Grouped launch in a real attention block. Done 2026-11-23, EXP-031: two launches, 0.011 ms graphed against fp32's 0.052, bit exact, cosine 0.98804.
- [x] An int8 grouped kernel. Done 2026-11-23, EXP-035: the attention block reads 0.99994 at int8 for 1.54x the int4 time.
- [x] Package the project, add a style gate and CPU CI. Done 2026-11-23, EXP-039: editable install works, 47 CUDA comment lines moved to `docs/ARCHITECTURE.md`, gate at zero.
- [ ] The missing rung from EXP-036: int4 attention with int8 down, which separates the attention error from the down-projection error.
- [x] CI removed 2026-11-23 at the human's request (DEC-042). The local gate is `style`, `pytest`, `fuzz`, and the vcvars suite. A GPU runner is moot until CI returns.
- [x] Read real weights from `lerobot/smolvla_base`. Done 2026-11-23, EXP-043: five matrices, `python -m vla_cu real`. A three layer stack is still open.
- [ ] BLOCK_M = 48 for the int4 GEMM, so M = 50 pads by 1.04x instead of 2x, and a CUDA port of that GEMM.
- [x] A real checkpoint with and without rotation. Done 2026-11-23, EXP-043 and DEC-045: rotation is inconsistent on real weights. A three layer stack at int8 and int4 is still open.
- [x] Explain the four-block cosine of 0.975. Done 2026-11-23, EXP-036: the MLP down projection is the limit. int4 down reads 0.976 over four blocks with int8 attention, int8 down reads 0.99951, and int4 attention on top of it reads 0.90671.
- [x] Pin clocks from an elevated shell (`nvidia-smi -lgc 2000`). Done 2026-11-23, EXP-042.
- [ ] Re-run EXP-020, EXP-021 and EXP-032 under the pinned clock, now that the spread is 0.35 percent.
- [ ] A capture-safety check in the tests: assert `torch.equal` between a graph and the eager chain for any kernel added later (the rule in DEC-031).
- [ ] Write an int4 GEMM that dequantizes into `tl.dot`, and sweep M from 1 to 64 for the GEMV crossover.
- [x] Decide group 16 against padding to 768 for the 720-wide expert. Done 2026-11-23, DEC-044: pad to 768, 26 percent fewer bytes for 0.0018 cosine.
- [ ] Download `lerobot/smolvla_base` config and index, then re-run EXP-023 and EXP-026 on real weights.
- [ ] Make every table best-of-N with the run count stated (DEC-029).
- [x] Explain the CUDA baseline regression. Done 2026-11-23, EXP-040: it does not reproduce. Baseline 0.238 ms and stream 0.208 ms at 8192 x 16384 in one process, so the 0.476 reading was a clock outlier.

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
- [x] Resolve the compute-sanitizer failure on the CUDA extension. Done 2026-11-23, EXP-044 and EXP-046: both kernels in torch-free headers, both harnesses at 0 errors and 0 leaks.
- [ ] `ncu` the int4 stream variant at 8192 x 16384 from the elevated shell. The command is now `python -m vla_cu ncu` (DEC-049); the existing report is still a 6.75 us small shape.
- [x] Sweep the native CUDA streaming kernel over the `scale` shape grid. Done 2026-11-23, EXP-047: native 88.0 percent against Triton 69.4 at 4096 x 8192, and the 90.4 in EXP-042 does not repeat as a median.
- [x] Add a fused RMSNorm and residual so the expert stack can be chained and cosine quoted at depth 16. Done 2026-11-23, EXP-048: 0.99548 at depth 16 against 0.31792 bare.
- [x] Diff the SASS of the Triton and native int4 GEMV at 4096 x 8192, to name what separates 69 percent from 88. Done 2026-11-23, EXP-050: Triton stages the tile through shared memory and loads 4 and 8 bytes at a time, the native kernel uses 16 byte read-only loads and no SMEM.
- [x] Try a Triton int4 GEMV with 16 byte loads and no shared memory staging. Done 2026-11-23, EXP-052: the wide mapping reached 56.3 percent against 73.1 staged and 87.9 native, and both Triton mappings still allocate shared memory for the reduction.
- [x] Force per-lane contiguity in the wide mapping with `tl.multiple_of` and `tl.max_contiguous`. Done 2026-11-23, EXP-054: still 0 `LDG.E.128`, LDS 2 to 8, and 3x slower, so it was reverted.
- [x] Keep the wide tile in the load's shape and reduce it with `tl.dot` against a ones vector. Done 2026-11-23, EXP-056: 0.18x of staged, 970 instructions, 264 FFMA, 83 LDS, kept as the measured alternative (DEC-058).
- [x] Apply rotary rotation to q and k in the two-token block. Done 2026-11-23, EXP-053: interleaved pairs, row index as position, cosine unchanged because the reference rotates identically.
- [x] Run the chain at three or more tokens, where the score matrix is no longer 2 x 2. Done 2026-11-23, EXP-055: 2, 3, 4 and 8 tokens, exact at every depth, cosine 0.99622 to 0.99648 at depth 16.
- [x] Graph the normed expert chain. Done 2026-11-23, EXP-049: 16 layers 3.475 ms eager against 0.497 ms graphed, exact at every depth, so 186 us of 217 us per layer was dispatch.
- [x] Add attention to the 16 deep chain. Done 2026-11-23, EXP-049: cosine 0.99795 at depth 16 over 50.78 MB, with the value path only (DEC-050).
- [x] Run the chain at two tokens so q and k decide the output and a real softmax has to execute. Done 2026-11-23, EXP-051: cosine 0.99622 at depth 16 over 64.59 MB, 15.45 ms eager.
- [x] Graph the two-token chain. Done 2026-11-23, EXP-053: 24.880 ms eager against 2.818 ms graphed at depth 16, exact at every depth.
- [x] Read the real rope theta from the model config. Done 2026-11-23, EXP-055, DEC-056: theta 100000 and `rope_interleaved` false, so the Llama half split pairing. The old adjacent pairing was wrong.
- [x] Add a causal mask to `gqa_mix`. Done 2026-11-23, EXP-057, DEC-059: the mask exposed that the score contraction batched on tokens, so it now uses einsums and scores are [heads, tokens, tokens].
- [x] Compare the config `rms_norm_eps` 1e-05 against the EPS the RMSNorm kernel uses. Done 2026-11-23, EXP-058, DEC-060: the kernel was 1e-6, now 1e-05, cosine unchanged to five decimals.
- [x] Measure the wide kernel at `input_precision="tf32"`. Done 2026-11-23, EXP-059: 0.41x of staged, 0.174 ms at 4096 x 8192, but max abs diff 4.00e-03, so it is a diagnostic only.
- [x] Re-time the staged int4 GEMV at the expert's real 720 and 960 wide shapes. Done 2026-11-23, EXP-061: every path is 0.021 to 0.031 ms while bytes move 0.39 to 2.5 MB, so the real shapes are launch bound at about 22 us.
- [x] Run the causal chain over a 50 token action chunk. Done 2026-11-23, EXP-060: 336.872 ms eager against 25.120 ms graphed at depth 16, exact, cosine 0.99637. The limit is 200 launches per layer, not the score matrix.
- [ ] Batch the projections over the token axis so a 50 token layer is 4 launches instead of 200. This keeps the same weight traffic as today and removes the dispatch, and it is the cheapest test of the batched idea (EXP-060).
- [x] Compare the int4 GEMM at M = 50 against fifty GEMV launches. Done 2026-11-23, EXP-063: 41.6x and 24.5x faster on the two expert shapes, 1.6 and 1.3 MB against 52.4 and 39.4, but still 2.5 to 3.8x behind cuBLAS bf16 (DEC-062).
- [x] Replace the two einsums in `gqa_mix` with a small fused attention kernel. Done 2026-11-23, EXP-062, DEC-061: 27 percent faster at 2 tokens, 13 percent slower at 50, so the block dispatches at 4 tokens.
- [x] Group the attention program by kv head and fuse rope into it. Done 2026-11-23, EXP-065, DEC-064: 20 to 24x faster than the einsums at every token count, so the threshold moved from 4 to 64.
- [x] Add `num_stages` and a tile menu to the int4 GEMM. Done 2026-11-23, EXP-064: 0.048 and 0.054 ms at M = 50, 8.7 and 6.3 percent of peak, still behind cuBLAS bf16.
- [x] Split K across programs for the narrow expert shape. Done 2026-11-23, EXP-068, DEC-067: a negative, 0.57x to 0.75x the single launch, because the reduce is a whole extra launch.
- [x] A one launch K split with in launch recombine. Done 2026-11-23, EXP-071, DEC-069: deterministic and racecheck clean, but parity at best, so default off.
- [x] A true stream-K: contiguous K ranges across tiles with boundary handoff, one launch and a balanced grid. Done 2026-11-23, EXP-073: 1.04x on 1920 outputs and 0.89x on 720, bit exact, so it stays off. The grid was not what limited the kernel.
- [ ] Name the fixed cost at M = 50. Every int4 path lands at 0.048 to 0.057 ms for 1.3 to 1.7 MB (26 to 37 GB/s) whatever the program count, against 0.021 to 0.027 ms for cuBLAS bf16, so the limit is neither the grid nor DRAM. Needs `ncu` from the elevated shell (EXP-073).
- [x] Give the int8 GEMM its own sweep. Done 2026-11-23, `python -m vla_cu int8tiles`, EXP-075: twelve candidates land in 0.041 to 0.052 ms at three expert widths and three token counts, and the menu's pick is inside that band, so it was kept rather than retuned.
- [x] Pair the top three int8 tiles at 720 x 1920 and 50 tokens. Done 2026-11-24, EXP-080: the 16 row tile won every repeat of three paired runs, by 1.07x to 1.53x, and the menu now uses it above 32 tokens.
- [ ] Pair that 16 row tile at 8 tokens and at 1920 x 720 before the rule spreads (EXP-080).
- [ ] Tile the cooperative body: one shared activation tile per block and tensor core dots, so the fused path can also win at 8 and 50 tokens (EXP-078).
- [ ] Skip the group scale and the transpose in an int4 test variant, which is the leading suspect for the 5 to 6 percent of peak in EXP-079.
- [x] A cooperative-launch persistent layer in CUDA C++. Done 2026-11-24, EXP-078: one launch, 3.13x at 1 token, 2.55x at 2, 2.22x at 3, 1.25x at 8, 0.24x at 50. The body, not the launch, is what is left.
- [ ] A W4A8 rung: int4 weights with int8 activations, then compare cosine at depth 16 against the current int4 down (EXP-036, CUTLASS W4A8, LiquidGEMM).
- [ ] Replace the hardcoded tile constants with a measured shape table, the way llama.cpp switches kernels per architecture and quant type (EXP-052, EXP-060, docs/PRODUCTION.md).
- [ ] Cache the VLM prefix KV across calls, which is the RadixAttention idea applied to a fixed policy prefix (docs/PRODUCTION.md).
- [x] Run both attention paths paired in one process. Done 2026-11-23, `python -m vla_cu attncross`, EXP-065: the fused kernel wins at 1 through 50 tokens by 20 to 24x.
- [x] Wire the MLP's two projections to the GEMMs. Done 2026-11-23, EXP-067, DEC-065: 208 launches per layer and 10.627 ms at 50 tokens.
- [x] Row kernels for the chunk norm, add and fused add-norm. Done 2026-11-23, EXP-070, DEC-068: 11 launches per layer at any token count, 4.468 ms at 50 tokens.
- [x] Pair the MLP chunk against the token loop the way `attncross` pairs the mix. Done 2026-11-23, EXP-072, `python -m vla_cu pairedtail`: 1.97x at 2 tokens to 48.72x at 50 for the MLP, 2.42x to 61.51x for the row kernels, rows bit exact and the MLP within 1e-3.
- [ ] Stage the activation tile in shared memory and re-sweep the int4 row tile. Lower priority since EXP-014: the row tile already plateaus at 4, so activation traffic is not what limits the kernel
- [ ] `torch.compile` the custom op path and compare against the eager launches
- [ ] torch.compile as a baseline: blocked, inductor fails with `triton_key` ImportError on this venv (EXP-006)
- [x] `python -m vla_cu cuda-check`: which toolkit and host compiler a build would use, plus a verdict (2026-11-18, EXP-005, DEC-015).
- [x] Close the two input gaps left in EXP-033. Done 2026-11-23, EXP-034: `block_n` is checked. The grouped segment-run invariant stays documented, because proving it needs a device sync.

## Open questions
- ~~Which GPU, and what is its peak bandwidth?~~ Answered in `HARDWARE.md`: RTX 5060 Laptop, 384 GB/s.
- Native Windows Triton (`triton-windows`) or WSL2? **Answered 2026-11-18: `triton-windows` (DEC-008).**
- Whether `triton-windows` supports `tl.dot`, TMA and autotuning on sm_120 - untested; test before relying on it for the fused GEMV.
- Which SmolVLA layer shapes do we target first (hidden size, intermediate size) once we leave toy tensors?

## Done / dropped
- [x] Bug-fix pass (2026-11-23, EXP-033): rotation order guard, grouped wrapper validation, power-of-two block size, cuda test gate on `cl`, `CalledProcessError` wrapping, bf16 byte count.
- [x] STARTER.md Part A: repo skeleton, venv, first commit (2026-11-18, commit 9b5dee6).
- [x] STARTER.md Chunk 0: environment check (2026-11-18, EXP-001).
