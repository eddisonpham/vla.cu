# Production CUDA inference systems, and what they say about vla.cu

Eleven codebases were read for this survey, all active in 2026. Each row names the
method that matters and the source that documents it. The second table maps each method
onto this repo's current state, and the last section turns the gaps into work.

Read this as a menu, not a roadmap. The repo's value is that every line is explainable,
so an idea enters only when a measurement here can show what it buys.

## 1. What each system does

| System | Core method | Trick worth stealing | Source |
|---|---|---|---|
| vLLM | Paged KV cache with a block table, one warp per block on read | Fused reshape plus block write, and a Q block that groups query tokens and heads per work item to feed `tl.dot` and reuse cache | arXiv 2309.06180, `csrc/attention/attention_kernels.cu` (historic), Triton backend blog 2026-03-04 |
| vLLM (allreduce) | Custom one-shot and two-hop allreduce over IPC | 128-bit aligned buffers so every lane issues `LD.128`, and a graph friendly collective | `csrc/custom_all_reduce.cuh`, PR 2192 |
| TensorRT-LLM | Kernel zoo behind a graph compiler: XQA for MQA and GQA decode, FMHA for prefill, CUTLASS for GEMM | Fused RoPE into attention, FP8 KV cache dequantized on the fly, INT4 allowed only in linear layers | `cpp/kernels/xqa/mha_sm90.cu`, XQA blog, quantization docs |
| llama.cpp / ggml | One quant format per type, two kernels: MMVQ for small batch, MMQ on int8 tensor cores above a switch point | Per-architecture and per-quant switch points, and a few-row MMQ kernel for 2 to 24 columns that reads each weight once | `ggml/src/ggml-cuda/mmvq.cu`, `mmq.cu`, PR 31533, PR 26079 |
| Marlin | FP16 by INT4 mixed precision GEMM for batch 1 to 32 | `cp.async` double buffering, L2 evict-first for weights only, XOR swizzle to dodge bank conflicts, 4 stage pipeline | `marlin/marlin_cuda_kernel.cu`, Red Hat article 2024-04-17 |
| FlashInfer | Attention as a block sparse matrix over a unified KV layout | JIT specialization per request shape, a tile size menu, and merge-state to combine partial attention results | flashinfer repo, NVIDIA blog 2025-06-13 |
| SGLang | RadixAttention prefix tree over the KV cache | Per-phase CUDA graphs (full or breakable), and backend choice made from the GPU and model | arXiv 2312.07104, attention backend docs |
| llm-awq / AutoAWQ | W4A16 with per-group scales and zero points, GEMV and GEMM variants | Dequantize to fp16 in shared memory then use the tensor core, split-K for short matrices | `awq/kernels/csrc/quantization/gemm_cuda_gen.cu`, `gemv_cuda.cu` |
| bitsandbytes | Fused 4-bit dequantize plus GEMM replacing both GEMV and dequantize then linear | Three kernel tiers (SIMT, MMA m16n8k8, MMA m16n8k16) picked at runtime by shape, architecture and SM count, with a lookup table instead of branches in dequant | PR 1949, `csrc/gemm_4bit.cu`, `csrc/gemm_4bit_simt.cu` |
| torchao | Quantized tensor subclasses with explicit packing formats | Tile-packed-to-4d layout and preshuffled packing so tensor cores consume the data directly | `torchao/quantization/quantize_/workflows/int4/`, quantization overview |
| TensorRT | Explicit dequantize layer in a compiled engine with immutable weights | Weight-only INT4 as a constant plus a dequantize prologue fused into the GEMM | Torch-TensorRT INT4 weight-only tutorial |
| CUTLASS | Header-only GEMM with warp specialized W4A8 kernels | Groupwise mixed precision with the dequantize folded into the mainloop | TensorRT-LLM W4A8 CUTLASS kernels, LiquidGEMM paper |

## 2. Mapping onto this repo

| Method | What vla.cu does today | Gap | Evidence here |
|---|---|---|---|
| Batch switch, GEMV below and GEMM above | GEMV per token at every token count, one launch per token per projection | No switch: a 50 token chunk issues 200 launches per layer | EXP-060, `m50` |
| Fused dequantize plus GEMM | `kernels/triton/int4_gemm.py` with fp16 tensor core dots | Fp16 rounding at the dot, no `cp.async` pipeline, no swizzle | EXP-013, `m50` |
| Read each weight once per chunk | GEMV loop re-reads all weights 50 times | The int4 GEMM already fixes the bytes, it is 24 to 42x faster than the loop | `m50` |
| Fuse RoPE into attention | RoPE is two torch calls per layer, attention is a separate kernel | Three launches where production uses one | EXP-057, EXP-062 |
| Attention tile per work item | One program per (token, head) in the fused kernel, so key and value tiles are reloaded per query head | A Q block or per-kv-head program would cut that traffic | EXP-062 |
| Graph friendly everything | Every chain is a `torch.cuda.CUDAGraph`, checked with `torch.equal` | Already aligned with vLLM, SGLang and TensorRT-LLM practice | EXP-049, DEC-055 |
| Per-shape dispatch table | Block size fixed at 1024, rows at 4, no shape based choice | `plateau` shows 73 percent at one shape and 38 at another, with no rule | EXP-052, EXP-060 |
| L2 evict-first for weights | Triton eviction hints on by default | Matches Marlin's policy, but the pipelining that makes it pay is missing | EXP-020, EXP-022 |
| Prefix reuse | The prefix cache is rebuilt from the 48 language rows every run, and the image and state rows are absent | A real policy would cache the full VLM KV for the episode | EXP-104 |
| W4A8 rung | int8 weights and int4 weights are separate paths, never mixed in one layer | The rung is open: int4 down with int8 attention already exists, W4A8 does not | EXP-036, backlog |
| Contiguity and packing for tensor cores | Plain row-major nibbles, unpacked with shifts | A preshuffled layout would let int8 MMA consume int4 without shifts | EXP-056, EXP-059 |

## 3. Ranked work for this repo

Each item names what to measure, so a result can be reported either way.

| Rank | Change | Expected effect | Cost | Measurement |
|---|---|---|---|---|
| 1 | Switch the block between the GEMV loop and the int4 GEMM on token count | A 50 token layer drops from 500 launches to about 10 | Medium: 2D paths and a dispatch rule | `scale` at 2, 8, 50 tokens |
| 2 | Group the attention program by kv head, and fuse RoPE into it | One launch and no rotary tensors per layer, less key traffic | Medium: kernel rewrite | `scale` graph column, `sass` |
| 3 | Add `num_stages` and pipelining to the int4 GEMM, in the Marlin shape | The GEMM is 7.6 percent of peak at M = 50, so most of the headroom is time to first byte | Medium | `m50`, `plateau` |
| 4 | A W4A8 path: int4 weights, int8 activations, fp32 accumulate | Halves activation bytes and skips the fp16 rounding at the dot | Large | cosine at depth 16, `m50` |
| 5 | Shape table: pick block, rows, warps and GEMM against GEMV from the shape | Removes the guesswork behind the current constants | Small to medium | `plateau`, `tune` |
| 6 | Cache the VLM prefix KV across calls | Real win for a policy loop, not for this benchmark | Large: needs the VLM path | not measurable yet |

## 4. Generalizing to other VLAs and larger models

| Need | Current state | Change |
|---|---|---|
| Another model's shapes | 720, 960, 1920 and 2560 are constants in benches and tests | Read hidden, intermediate, heads, kv heads, layers, rope theta and epsilon from a config file, as `bench/scale.py` already does for names |
| Larger parameters | 7 GB cap enforced per call, working cap 4 GB in `scale` | A streaming packer and a per-layer budget, so a model larger than the cap can be run layer by layer from disk |
| Other quant formats | int4 with group scales and int8 per row | A format descriptor (bits, group, packing, zero point) so a kernel asks for what it needs instead of taking one dtype |
| Other inference settings | Temperature and top-p do not exist, attention is one fixed path | A sampler at the output, and attention variants for prefill against decode |
| Other token counts | Dispatch is one hardcoded threshold at 4 tokens for attention | One table of thresholds, measured per shape, replacing the constants |
| Other hardware | sm_120 only, 26 SMs, 384 GB/s | Every kernel already reads its shapes at runtime, but the tile constants assume this GPU, so a per-architecture table is needed |
