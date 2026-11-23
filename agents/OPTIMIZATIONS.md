# OPTIMIZATIONS: where the headroom is, ranked, with one falsifiable experiment each

> Planning document. **No experiment was run in the session that wrote this.** Every number is
> either quoted from `EXPERIMENTS.md` (marked measured, with the entry) or arithmetic on measured
> numbers and spec sheets (marked derived). Nothing here is a result.
>
> Written 2026-11-23, after EXP-016, commit f88dcf4. GPU: RTX 5060 Laptop, 26 SM, 384 GB/s, L2 33.6 MB.

---

## 1. The ledger: what the current kernel actually spends

Measured, EXP-014 and EXP-010 and EXP-016, 8192 x 16384, batch 1:

| kernel | bytes | ms | GB/s | % of 384 | source |
|---|---|---|---|---|---|
| fp32 `torch.mv` | 537.0 MB | 1.587 | 332.7 | 86.6 | EXP-014 |
| bf16 `torch.mv` | 268.5 MB | 0.811 | 334.5 | 87.1 | EXP-014 |
| int8 fused | 134.3 MB | 0.400 | 336.1 | 87.5 | EXP-014, same run |
| int8 fused | 134.2 MB | 0.379 | 354.7 | 92.4 | EXP-010, one day earlier |
| int4 Triton | 75.6 MB | 0.230 | 328.2 | 85.5 | EXP-014 |
| int4 CUDA | 75.6 MB | 0.240 | 315.1 | 82.1 | EXP-016 |

Read that table twice. The gap between formats (328 to 358) is no larger than the gap within one
format across two sessions (int8: 336.1 against 354.7). So the nibble unpack is not costing 15
percent of peak. We are within a few percent of what this 45 to 100 W laptop part delivers on the
day, and the rest of the "gap to peak" is the difference between a spec sheet and a power-limited
part.

Consequence for this plan: **micro-optimizing the streaming rate is worth single digit percent and
cannot be claimed until the attainable ceiling is measured.** The large wins are in bytes and in
launches, and one of them is at model level, not kernel level (section 6).

## 2. Budget arithmetic (derived, not measured)

| Quantity | Value | Derivation |
|---|---|---|
| Arithmetic intensity, int4 | 4 FLOP/byte | 2 FLOP per weight, 0.5 B per weight |
| FLOP/s needed at 384 GB/s | 1.5 TFLOP/s | intensity times peak bandwidth |
| fp32 peak of this part | about 16 TFLOP/s | 26 SM x 128 lanes x 2 flop x 2.4 GHz |
| Measured ALU use in int8 | about 4 percent of fp32 peak | 0.71 TFLOP/s from 354 G weight/s at EXP-010 |
| Bytes in flight needed | about 58 KB, 2.2 KB per SM | 384 GB/s x 360 cycles at 2.4 GHz |
| In flight at full occupancy | about 6 KB per SM | 48 warps x 128 B of weight loads per step |
| Scale bytes in the int4 row | 8.39 MB of 75.6 MB, 11.1 percent | 8192 x 256 groups x 4 B |

Two readings:

1. **Not ALU bound.** At 4 percent of fp32 peak, an instruction saving trick (lop3 dequant, dp4a)
   cannot pay off directly. It can only pay through latency and issue slots. This is why H10 is
   ranked low despite being the most interesting trick in the literature.
2. **Scales are 11 percent of the traffic**, larger than any remaining unpack inefficiency. Trading
   fp32 scales for fp16 is a straight 5.6 percent of bytes for a 5e-4 relative error on a value that
   is already the least precise part of the sum.

## 3. Candidate table

Payoff is derived unless marked measured. Effort in chunks of about 60 lines.

| # | Candidate | Lever | Derived payoff | Effort | Risk | Novel |
|---|---|---|---|---|---|---|
| C1 | Streaming cache hints on weights, keep x warm | L1 residency | 0 to 5 percent | 1 | low | no |
| C2 | fp16 group scales | bytes | 5.6 percent of traffic, 0.230 to 0.218 ms | 1 | low | no |
| C3 | Unroll and drop the tail guard in the CUDA kernel | loads in flight | 0 to 8 percent | 1 | low | no |
| C4 | Rows per warp in the CUDA kernel, x in shared memory | activation traffic | about 0 at DRAM size, 1.5x at L2 size (EXP-013 measured in Triton) | 2 | medium | no |
| C5 | `cp.async` double buffering, warp specialization | latency hiding | 0 to 3 percent | 3 | medium | no |
| C6 | Graph the MLP block | dispatch | 0.085 to about 0.055 ms, 1.5x | 1 | low | no |
| C7 | One launch per MLP block, grid sync, hidden in L2 | dispatch plus one launch | 0.085 to 0.045 to 0.055 ms | 3 | high | yes |
| C8 | Grouped GEMV: q, k, v and the MLP pair of a layer in one launch | launches and x traffic | 3 launches to 1, x read once for three matrices | 2 | medium | yes |
| C9 | Fused RMSNorm and SiLU epilogue | launches | 2 to 4 launches per block | 2 | medium | no |
| C10 | lop3 fp16 dequant (Marlin trick) | instruction count | little while ALU sits at 4 percent | 2 | low | no |
| C11 | Hadamard rotation before packing, fused into the forward | accuracy per byte | enables group 64 at low cosine error | 3 | medium | yes |
| C12 | Per-layer bit width from a sensitivity sweep | bytes | 10 to 20 percent of model bytes (hypothesis) | 4 | high | partly |
| C13 | 2:4 structured sparsity on the action expert | bytes | up to 2x on 100M of 450M parameters | 5 | high | no |
| C14 | Tensor-core int4 for the action expert at M = 50 | throughput per byte | unknown, and M=50 not M=1 (see section 6) | 5 | high | yes |

## 4. Hypotheses and the experiment for each

Each entry: the claim, the one change, the measurement, and what would refute it. Refutation
conditions are written down before the run on purpose.

### H1: the attainable streaming ceiling is 350 to 365 GB/s, not 384
Claim: a read-only kernel with no unpack reaches 350 to 365 GB/s on 67 MB of contiguous data, so
int4 at 328.2 has 6 to 10 percent of headroom, not 15. Also: run-to-run spread on this part is
2 to 7 percent, so every table so far is one sample of a distribution.
Change: `bench/bandwidth_probe.py`, task `stream`. One kernel, vectorized 128 B per warp per step,
reads a buffer and reduces it, no weights, no scales, no unpack. Report 5 repeats and the spread.
Refuted if: the probe also lands near 330 GB/s, which closes the micro-optimization branch of this
plan and moves all effort to bytes and launches.
Why first: it decides whether C1, C3, C4 and C5 can claim anything at all.

### H2: one-shot weights should not be allowed into L1
Claim: weight bytes are read exactly once, so caching them in L1 evicts x, which is reused by every
program. Marking weights streaming (`eviction_policy="evict_first"`, `cache_modifier=".cg"`) and x
`evict_last` recovers 2 to 5 percent at DRAM sizes and much more at L2 sizes.
Change: one keyword argument in the Triton wrapper, `__ldcs` in the CUDA kernel.
Refuted if: within the spread measured by H1, which would say L1 residency was never the problem.

### H3: fp32 scales are an unearned 5.6 percent
Claim: fp16 scales cut traffic from 75.6 to 71.4 MB and add less than 1e-3 relative error, because
the scale is amax/7 and fp16 carries 10 mantissa bits.
Change: `pack_int4(..., scale_dtype=torch.float16)`, kernels convert on load.
Refuted if: the time does not move (scales were L2 resident anyway) or the relative L2 error moves in
the third digit.

### H4: the CUDA loop cannot unroll, and that is its 4 percent
Claim: `if (weight >= n_cols) continue` inside the loop prevents unrolling, so each step issues its
load and then waits on it. Requiring K divisible by 1024 and unrolling 4 steps puts four loads in
flight per lane.
Change: one kernel variant, `#pragma unroll 4`, no tail guard.
Refuted if: `ncu` shows the memory pipe already near saturation, or the variant is inside the spread.
Depends on H5 for the explanation, not for the measurement.

### H5: we do not actually know why it stops at 85 percent
Claim: the difference between 328 and 358 GB/s is one of DRAM saturation, latency, or occupancy,
and no one has looked. EXP-016 showed `compute-sanitizer` cannot instrument this extension, so the
profiler path is untested here.
Change: `ncu` on both int4 kernels, SpeedOfLight plus MemoryWorkloadAnalysis plus achieved
occupancy; `nsys` on the layer task to see the dispatch gaps.
Refuted if: `ncu` also fails on the extension, in which case profile Triton only and say so.
Note: this is a measurement of the current state, not an optimization.

### H6: the block is dispatch bound and a graph proves it
Claim: the two launches in the MLP block cost about 32 us of its 85 us at the 16.71 us per launch
from EXP-009. Capturing them removes about 30 us.
Change: `torch.cuda.CUDAGraph` around the two custom ops, replay timed against the eager path.
Refuted if: the graphed block is not at least 1.4x faster, which would say the Python wrapper is
the cost and the graph does not remove it (the op validation runs inside capture too).

### H7: one launch per block, with a grid sync, beats a graph
Claim: a single cooperative kernel that computes a slice of the up projection into shared memory,
syncs the grid, then computes the down projection reading hidden from L2 removes the launch
entirely and keeps the 10 KB hidden vector hot.
Change: `csrc/mlp_fused.cu`, `cudaLaunchCooperativeKernel`, 26 persistent blocks.
Refuted if: the grid sync costs more than the launch it saves, or cooperative launch cannot be
captured in a graph (unverified, check before building the rest).
Depends on H6: if the graph already gets most of it, this buys little for a lot of complexity.

### H8: a layer's matrices should be one launch, not four
Claim: q, k, v share the same x at batch 1. Concatenating their packed rows into one buffer and
launching once cuts three launches to one and reads x once instead of three times.
Change: `kernels/triton/int4_grouped_gemv.py` over one concatenated buffer plus an int32 segment
table; a `torch.library` op that dispatches on the table.
Refuted if: the concatenated layout hurts DRAM page locality more than the saved launches help.
This is the piece that generalizes to the whole model, so it is worth doing even if the percentage
is small at 8192 x 16384.

### H9: the action expert is M = 50, and that changes the whole plan
Claim: this is the important one. SmolVLA predicts a chunk of 50 actions per forward, and the
action expert is a transformer over those 50 tokens, so its matrices are `[50, hidden] x [hidden, out]`,
not a GEMV. Weight bytes are unchanged, so the floor is the same, but 50 MACs per weight byte makes
a tensor-core kernel legal where at M = 1 it is not.
Change: extend `bench/matrix.py` with an M dimension, measure the current GEMV at M = 1 and the
`torch.mv`-free path at M = 50, and compute the derived floor per component from the real config.
Refuted if: the expert turns out to process one action token at a time (check `modeling_smolvla.py`
before building anything on this).

### H10: lop3 gets two weights into fp16 in one instruction
Claim: Marlin's trick (`lop3` to set an fp16 exponent, subtract a constant) halves the instructions
per byte and removes the int to float convert.
Change: one alternative CUDA kernel body, compared against the current one at the same shape.
Refuted if: no measurable change, which is the expected outcome at 4 percent ALU use. Ranked this
low on purpose.

### H11: a fused Hadamard rotation is what makes int4 defensible
Claim: int4 at group 64 sits at cosine 0.9935 for one random block, and the project's gate is 0.98 on
a real model. Rotating x and the weights by a fast Hadamard transform before quantization lowers
weight kurtosis, which is the mechanism behind QuaRot and SpinQuant. At batch 1 the rotation costs
K log K adds, negligible against K/2 bytes.
Change: `vla_cu/rotate.py` (pack-time weight rotation plus the in-kernel or pre-kernel x rotation),
then the ladder matrix reports cosine rather than relative L2.
Refuted if: cosine does not move, or the rotation costs more than it saves at 64 KB of activations.

## 5. Experiment queue

Ordered by information per unit of effort. EXP-017 first because it decides whether 018 to 021 mean
anything.

| EXP | Hypothesis | Task | Depends on | Effort |
|---|---|---|---|---|
| EXP-017 | H1 ceiling and repeatability | new `stream` | none | 1 |
| EXP-018 | H6 graph the block | `layer` | none | 1 |
| EXP-019 | H5 profiler truth | `ncu`, `nsys` | none | 1 |
| EXP-020 | H2 streaming hints | `matrix` | 017 | 1 |
| EXP-021 | H3 fp16 scales | `matrix` | 017 | 1 |
| EXP-022 | H4 unroll | `cuda` | 019 | 1 |
| EXP-023 | H9 M = 50 reality | `matrix` | read the model code | 2 |
| EXP-024 | H8 grouped GEMV | new `grouped` | 020 | 2 |
| EXP-025 | H7 persistent block | new `mlp` | 018 | 3 |
| EXP-026 | H11 Hadamard | `matrix` | a checkpoint eventually | 3 |

EXP-017, 018 and 019 are independent and can go in any order. Everything after 021 needs 017's
spread number to be read honestly.

## 6. The model-level picture, derived

SmolVLA facts from the paper and the LeRobot config (sources in `RESEARCH.md`, 2026-11-23): 450M
parameters, about 100M of them in the action expert, 16 VLM layers, expert width 0.75 times the VLM
width, action chunk 50, and the flow matching sampler fixed at 10 steps.

| Item | Derived value |
|---|---|
| 450M parameters at 4-bit | 225 MB |
| Full forward floor at 384 GB/s | 0.586 ms |
| VLM part, 350M | 175 MB, 0.456 ms, once per observation |
| Action expert, 100M | 50 MB per denoising step, 0.130 ms, 10 steps, 1.30 ms |
| Per action floor, expert loop plus one prefix | 1.76 ms |
| Expert hidden and intermediate at 0.75 times 960 and 2560 | 720 and 1920 |

Two consequences that outrank every kernel micro-optimization in this document:

1. **The 10-step denoising loop is 74 percent of the floor.** The VLM prefix is computed once per
   observation and can be cached when the observation does not change. The expert is re-streamed
   ten times. So the kernel that matters most is the expert's, and its only lever is bytes per
   parameter, which 4-bit has already taken.
2. **The steps are sequential, so the loop cannot be fused.** Each step's activations come from the
   previous step's output. A fused 10-step pass would cut weight traffic 10 times and it is not
   available. Recorded here so it is not proposed again by mistake.

## 7. Considered and rejected

| Idea | Why it looks good | Why not here |
|---|---|---|
| Tensor core `mma` for the VLM GEMV | Marlin's route, 4x claims | M = 1 wastes 15 of 16 accumulator rows, and the kernel is 4 percent ALU busy. Marlin itself needs batch 16 to 32 to pay, per its own README. Revisit only for the expert at M = 50 (C14, H9) |
| W4A8 with `dp4a` | 4 MACs per instruction | needs int8 activations and a per-group x quantize, and ALU is not the limit |
| fp8 or fp4 tensor cores | newest hardware | same M = 1 argument; also unverified for sm_120 here |
| Fusing the 10 denoising steps | 10x weight reuse | the steps are sequential (section 6) |
| TMA with cache hints | the modern path | `cp.async.bulk` bypasses L1 and accepts no eviction hint, so the hint experiment and TMA are mutually exclusive |
| Staging weights in shared memory | fewer L1 misses | each weight is used once per program; the only reuse is across rows, which a register tile already gives |
| 3, 5, 6-bit widths | fewer bytes | DEC-016 |
| Group 128 scales | 2.8 percent fewer bytes than fp16 at group 64 | DEC-020; revisit together with H11, since a rotation buys back the accuracy a coarser group costs |
| 2:4 sparse expert | up to 2x on the expert | needs sparsity-aware quantization and a sparse kernel, and changes the accuracy story. Ambitious, out of scope today |

## 8. What "impressive" would look like here

All targets derived until measured. Today's column is measured.

| Milestone | Derived target | Today |
|---|---|---|
| Attainable streaming ceiling, 67 MB | 350 to 365 GB/s | unmeasured (H1) |
| int4 GEMV at 8192 x 16384, hints plus fp16 scales | 0.204 to 0.215 ms | 0.230 ms (EXP-014) |
| MLP block, one launch | 0.045 to 0.055 ms | 0.085 ms (EXP-015) |
| Expert step, 100M at 4-bit | 0.130 ms | not built |
| Whole action, prefix cached, 10 steps | about 1.5 to 1.8 ms | not built |

The artifact worth putting in the README is a real checkpoint forward pass at a measured
milliseconds-per-action number, because that is the only figure that can be compared with published
VLA latencies. A third party table lists 245 ms per action step for SmolVLA on an RTX 4090; it is
unverified and not comparable until both sides are measured the same way, but it shows the order of
magnitude a 384 GB/s laptop part is aiming at.

Honesty note on novelty: **C6, C1 to C5 and C9 are standard practice** and this repo should present
them as such. The defensible novel contributions are C7 (one persistent launch per block, VLA at
batch 1), C8 (grouped GEMV keyed to transformer structure), C11 (rotation fused into a streaming
forward) and H9 (the finding that the expert is M = 50 while the VLM prefix is the only true batch 1
part). Those are what make the writeup interesting, and none of them is claimed until measured.
