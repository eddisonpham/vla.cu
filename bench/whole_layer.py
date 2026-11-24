"""One transformer layer as six launches against the fused tail, at three token counts.

Concept: the chain is norm, attention, residual, norm, gated MLP, residual, and the fused
option replaces the post norm, the three projections and the residual with one cooperative
launch. Both paths are captured, so the ratio is kernel time and not host dispatch, and the
result is that the fused tail loses to the chain at every token count measured.

Task: python -m vla_cu whole
"""

from __future__ import annotations

import torch

from vla_cu.bench_utils import enforce_memory_budget, time_pair, warm_gpu
from vla_cu.blocks import GatedMlpBlock, GqaTokenBlock, LayerBlock
from vla_cu.quant import GROUP_SIZE

HIDDEN = 960
INTERMEDIATE = 2560
HEAD_DIM = 64
KV_HEADS = 3
TOKENS = (1, 8, 50)
REPEATS = 5
ITERATIONS = 20
WARMUP = 5
TOLERANCE = 2e-2


def build(tokens: int, generator: torch.Generator) -> LayerBlock:
    """Return a backbone-shaped layer at `tokens`, with random fp32 weights at benchmark scale."""
    heads = HIDDEN // HEAD_DIM
    scale = 1.0 / HIDDEN**0.5
    wq = torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale
    wk = torch.randn((KV_HEADS * HEAD_DIM, HIDDEN), device="cuda", generator=generator) * scale
    wv = torch.randn((KV_HEADS * HEAD_DIM, HIDDEN), device="cuda", generator=generator) * scale
    wo = torch.randn((HIDDEN, heads * HEAD_DIM), device="cuda", generator=generator) * scale
    gate = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * scale
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * scale
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * (1.0 / INTERMEDIATE**0.5)
    input_norm = torch.rand(HIDDEN, device="cuda", generator=generator) + 0.5
    post_norm = torch.rand(HIDDEN, device="cuda", generator=generator) + 0.5
    attention = GqaTokenBlock(wq, wk, wv, wo, head_dim=HEAD_DIM, tokens=tokens)
    mlp = GatedMlpBlock(gate, up, down, down_bits=4, group_size=GROUP_SIZE, tokens=tokens)
    return LayerBlock(attention, mlp, input_norm, post_norm)


def measure(tokens: int, generator: torch.Generator) -> int:
    """Time captured chain and fused layers at `tokens`, returning nonzero if they disagree."""
    block = build(tokens, generator)
    x = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
    chain_out = torch.empty_like(x)
    fused_out = torch.empty_like(x)
    block.into(x, chain_out, fused=False)
    block.into(x, fused_out, fused=True)
    chain_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(chain_graph):
        block.into(x, chain_out, fused=False)
    fused_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fused_graph):
        block.into(x, fused_out, fused=True)
    chain_graph.replay()
    fused_graph.replay()
    torch.cuda.synchronize()
    difference = (chain_out - fused_out).abs().max().item()
    if not torch.allclose(chain_out, fused_out, atol=TOLERANCE, rtol=TOLERANCE):
        print(f"{tokens:>6}  the fused layer disagrees with the chain, max diff {difference:.2e}")
        return 1
    timing = time_pair(
        chain_graph.replay,
        fused_graph.replay,
        ITERATIONS,
        REPEATS,
        WARMUP,
    )
    print(
        f"{tokens:>6}  {timing.a * 1e3:>9.3f}  {timing.b * 1e3:>9.3f}  {timing.ratio:>7.2f}x  "
        f"{timing.ratio_low:.2f} to {timing.ratio_high:.2f}  {difference:>9.1e}"
    )
    return 0


def main() -> int:
    """Sweep the token counts and print chain against fused for one 960/2560 layer."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)}")
    print(f"one layer {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}, {KV_HEADS} kv heads, head {HEAD_DIM}, int4 down")
    print("both paths captured, so the numbers are GPU time: chain is 12 nodes, fused is 8")
    print(f"warmup {WARMUP}, repeats {REPEATS}, {ITERATIONS} iterations per repeat")
    print(f"{'tokens':>6}  {'chain ms':>9}  {'fused ms':>9}  {'speedup':>8}  {'range':>13}  {'max diff':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    for tokens in TOKENS:
        status = measure(tokens, generator)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
