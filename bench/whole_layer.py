"""The fused gated tail against a captured chain, as a layer and as a tail alone.

Concept: the layer table runs a whole 960/2560 layer two ways and the tail table isolates the
same choice at 720/1920 and 960/2560, both captured, so the ratio is kernel time and not host
dispatch. The 720 width cannot form a square attention block, which is why the tail is the
only place that width can be compared at the layer level.

Task: python -m vla_cu whole
"""

from __future__ import annotations

import torch

from kernels.triton.rmsnorm import add_into, add_norm_into
from vla_cu.bench_utils import enforce_memory_budget, time_pair, warm_gpu
from vla_cu.blocks import GatedMlpBlock, GqaTokenBlock, LayerBlock
from vla_cu.quant import GROUP_SIZE

HIDDEN = 960
INTERMEDIATE = 2560
HEAD_DIM = 64
KV_HEADS = 3
TOKENS = (1, 8, 50)
TAIL_WIDTHS = ((720, 1920), (960, 2560))
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


def measure_layer(tokens: int, generator: torch.Generator) -> int:
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
    timing = time_pair(chain_graph.replay, fused_graph.replay, ITERATIONS, REPEATS, WARMUP)
    print(
        f"{tokens:>6}  {timing.a * 1e3:>9.3f}  {timing.b * 1e3:>9.3f}  {timing.ratio:>7.2f}x  "
        f"{timing.ratio_low:.2f} to {timing.ratio_high:.2f}  {difference:>9.1e}"
    )
    return 0


def measure_tail(hidden: int, intermediate: int, generator: torch.Generator) -> int:
    """Time a captured gated tail against a captured chain at one token, returning nonzero on disagreement."""
    scale = 1.0 / hidden**0.5
    gate = torch.randn((intermediate, hidden), device="cuda", generator=generator) * scale
    up = torch.randn((intermediate, hidden), device="cuda", generator=generator) * scale
    down = torch.randn((hidden, intermediate), device="cuda", generator=generator) * (1.0 / intermediate**0.5)
    weight = torch.rand(hidden, device="cuda", generator=generator) + 0.5
    block = GatedMlpBlock(gate, up, down, down_bits=4, group_size=GROUP_SIZE, tokens=1)
    x = torch.randn((1, hidden), device="cuda", generator=generator)
    other = torch.randn((1, hidden), device="cuda", generator=generator)
    chain_added = torch.empty_like(x)
    chain_normed = torch.empty_like(x)
    chain_mlp = torch.empty_like(x)
    chain_result = torch.empty_like(x)
    fused_added = torch.empty_like(x)
    fused_normed = torch.empty_like(x)
    fused_result = torch.empty_like(x)

    def chain() -> None:
        add_norm_into(x, other, weight, chain_added, chain_normed)
        block.into(chain_normed[0], chain_mlp[0])
        add_into(chain_added, chain_mlp, chain_result)

    def fused() -> None:
        block.tail_into(x, other, weight, fused_added, fused_normed, fused_result)

    chain()
    fused()
    chain_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(chain_graph):
        chain()
    fused_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fused_graph):
        fused()
    chain_graph.replay()
    fused_graph.replay()
    torch.cuda.synchronize()
    difference = (chain_result - fused_result).abs().max().item()
    if not torch.allclose(chain_result, fused_result, atol=TOLERANCE, rtol=TOLERANCE):
        print(f"{hidden:>6}  the fused tail disagrees with the chain, max diff {difference:.2e}")
        return 1
    timing = time_pair(chain_graph.replay, fused_graph.replay, ITERATIONS, REPEATS, WARMUP)
    print(
        f"{hidden:>6}  {intermediate:>12}  {timing.a * 1e3:>9.3f}  {timing.b * 1e3:>9.3f}  "
        f"{timing.ratio:>7.2f}x  {timing.ratio_low:.2f} to {timing.ratio_high:.2f}  {difference:>9.1e}"
    )
    return 0


def main() -> int:
    """Print the layer table at three token counts, then the one token tail table at two widths."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)}")
    print(f"one layer {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}, {KV_HEADS} kv heads, head {HEAD_DIM}, int4 down")
    print("both paths captured, so the numbers are GPU time: chain is 12 nodes, fused is 8")
    print(f"warmup {WARMUP}, repeats {REPEATS}, {ITERATIONS} iterations per repeat")
    print(f"{'tokens':>6}  {'chain ms':>9}  {'fused ms':>9}  {'speedup':>8}  {'range':>13}  {'max diff':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    for tokens in TOKENS:
        status = measure_layer(tokens, generator)
        if status:
            return status
    print()
    print("gated tail alone, one token, chain is add_norm, three MLP launches and add")
    print(f"{'hidden':>6}  {'intermediate':>12}  {'chain ms':>9}  {'fused ms':>9}  {'speedup':>8}  {'range':>13}  {'max diff':>9}")
    for hidden, intermediate in TAIL_WIDTHS:
        status = measure_tail(hidden, intermediate, generator)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
