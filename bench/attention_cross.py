"""Paired timing of the fused attention mix against the einsum reference.

Concept: the earlier crossover compared two separate runs, and run drift reached 25
percent on eager paths, so the 4 token crossing could not be read. This times both paths
in alternating pairs at every token count in one process, so one clock state covers both
columns. The fused path rotates q and k inside the kernel, the einsum path rotates them
with torch, so each column is the whole mix.

Task: python -m vla_cu attncross
"""

from __future__ import annotations

import torch

from bench.real_layers import CHECKPOINT
from kernels.triton.gqa_attention import gqa_attention_into
from vla_cu.bench_utils import PEAK_BANDWIDTH_GB_S, enforce_memory_budget, time_call, time_pair, warm_gpu
from vla_cu.blocks import HEAD_DIM, gqa_mix, rotary_tables
from vla_cu.checkpoint import open_checkpoint

TOKENS = (1, 2, 3, 4, 8, 16, 32, 50)
WARMUP = 5
ITERATIONS = 20
REPEATS = 5


def main() -> int:
    """Time rope plus the causal mix both ways at every token count, and print the crossing."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    prefix = "model.vlm_with_expert.lm_expert.layers.0"
    q_width = checkpoint.read(f"{prefix}.self_attn.q_proj.weight").shape[0]
    kv_width = checkpoint.read(f"{prefix}.self_attn.k_proj.weight").shape[0]
    heads = q_width // HEAD_DIM
    kv_heads = kv_width // HEAD_DIM
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | expert layer 0")
    print(f"heads {heads} | kv heads {kv_heads} | head dim {HEAD_DIM} | warmup {WARMUP} | iterations {ITERATIONS} | repeats {REPEATS}")
    print()
    print(f"{'tokens':>6}  {'fused ms':>9}  {'einsum ms':>10}  {'fused/einsum':>13}  {'range':>13}  {'cos':>9}  {'max diff':>9}")
    crossing = None

    for tokens in TOKENS:
        generator = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn((tokens, q_width), device="cuda", generator=generator)
        k = torch.randn((tokens, kv_width), device="cuda", generator=generator)
        v = torch.randn((tokens, kv_width), device="cuda", generator=generator)
        cosine_table, sine_table = rotary_tables(tokens, HEAD_DIM)
        fused_out = torch.empty_like(q)
        einsum_out = torch.empty_like(q)

        def fused() -> None:
            gqa_attention_into(q, k, v, cosine_table, sine_table, fused_out, HEAD_DIM)

        def einsum() -> None:
            einsum_out.copy_(gqa_mix(q, k, v, HEAD_DIM))

        fused()
        einsum()
        torch.cuda.synchronize()
        fused_seconds = time_call(fused, WARMUP, ITERATIONS)
        einsum_seconds = time_call(einsum, WARMUP, ITERATIONS)
        pair = time_pair(fused, einsum, ITERATIONS, REPEATS, WARMUP)
        if crossing is None and einsum_seconds < fused_seconds:
            crossing = tokens
        similarity = torch.nn.functional.cosine_similarity(fused_out.flatten(), einsum_out.flatten(), dim=0).item()
        print(
            f"{tokens:>6}  {fused_seconds * 1e3:>9.3f}  {einsum_seconds * 1e3:>10.3f}  {pair.ratio:>12.2f}x  "
            f"{pair.ratio_low:.2f} to {pair.ratio_high:.2f}  {similarity:>9.6f}  "
            f"{(fused_out - einsum_out).abs().max().item():>9.2e}"
        )
        del q, k, v, fused_out, einsum_out

    print()
    answer = f"{crossing} tokens" if crossing is not None else f"beyond {TOKENS[-1]} tokens"
    print(f"smallest token count where the einsums beat the fused kernel: {answer}")
    print("paired ratio is fused over einsum, medians of the per repeat ratios")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
