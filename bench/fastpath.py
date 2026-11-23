"""The measured fast path: native streaming kernels, grouped where possible, graphed.

Concept: EXP-018 showed a 19.7 MB block is dispatch bound and EXP-022 showed the
streaming CUDA kernel is the fastest per launch. This task puts both together on the
real block and then on a stack of four different blocks, which is the shape a model
actually has. Paths, same weights, same input:

  fp32         eight torch.mv calls, the baseline
  triton raw   the Triton kernels, no custom op layer
  cuda raw     the CUDA streaming kernels, no graph
  cuda graph   one block captured as a CUDA graph and replayed
  stack graph  four different blocks captured as one graph, reported per block

The last two rows add a block Hadamard rotation to every weight and rotate the
activation on the way in, which costs two tiny matmuls per block and changes no byte
of storage. Cosine is printed beside every path because a graph either reproduces the
eager chain bit for bit or it is wrong, and because a rotation either buys accuracy
through a stack or it does not.

Each pair is timed in alternation with the fp32 path and reported with the spread of
the ratio, since this GPU refuses clock pinning to a non-administrator.

Task: python -m vla_cu fastpath
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int8_gemv import int8_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    time_pair,
    warm_gpu,
)
from vla_cu.cuda_int4 import int4_gemv_cuda_into
from vla_cu.cuda_int8 import int8_gemv_cuda_into
from vla_cu.quant import GROUP_SIZE, pack_int4
from vla_cu.rotate import hadamard, rotate_rows, rotate_vector

HIDDEN = 960
INTERMEDIATE = 2560
STACK_BLOCKS = 4
ROTATION = 64
WARMUP = 10
ITERATIONS = 50


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N, K] fp32 weights to int8 codes with one scale per output row."""
    row_amax = w.abs().amax(dim=1)
    scales = torch.where(row_amax > 0, row_amax / 127.0, torch.ones_like(row_amax))
    return torch.round(w / scales[:, None]).clamp(-127, 127).to(torch.int8), scales


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return torch.nn.functional.cosine_similarity(approx, exact, dim=0).item()


def make_block(seed: int, matrix: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Build one MLP block at the SmolVLA widths, optionally rotated before packing.

    A rotation is applied to the weight rows along K, so the same matrix has to be
    applied to the activation vector on the way in. The product is unchanged when the
    weights and the activation are not quantized, and better than unchanged when they are.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scale = 1.0 / HIDDEN**0.5
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * scale
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * scale
    if matrix is not None:
        up = rotate_rows(up, matrix)
        down = rotate_rows(down, matrix)
    up_int8, up_scale = quantize_int8(up)
    down_packed, down_scale = pack_int4(down)
    return {
        "up": up,
        "down": down,
        "up_int8": up_int8,
        "up_scale": up_scale,
        "down_packed": down_packed,
        "down_scale": down_scale,
    }


def low_bit_bytes(blocks: int) -> int:
    """Bytes of low-bit weights and scales for `blocks` blocks."""
    per_block = (
        INTERMEDIATE * HIDDEN
        + 4 * INTERMEDIATE
        + HIDDEN * INTERMEDIATE // 2
        + 4 * HIDDEN * INTERMEDIATE // GROUP_SIZE
    )
    return blocks * per_block


def main() -> int:
    """Time the paths on one block and on the stack, with and without rotation."""
    enforce_memory_budget()
    warm_gpu()
    activation = torch.nn.functional.silu
    matrix = hadamard(ROTATION)
    blocks = [make_block(seed) for seed in range(STACK_BLOCKS)]
    rotated_blocks = [make_block(seed, matrix) for seed in range(STACK_BLOCKS)]
    x = torch.randn(HIDDEN, device="cuda")

    hidden = torch.empty(INTERMEDIATE, device="cuda")
    out = torch.empty(HIDDEN, device="cuda")
    rotated_in = torch.empty(HIDDEN, device="cuda")
    rotated_hidden = torch.empty(INTERMEDIATE, device="cuda")

    fp32_out = torch.empty(HIDDEN, device="cuda")
    fp32_hidden = torch.empty(INTERMEDIATE, device="cuda")

    def fp32_block() -> None:
        torch.mv(blocks[0]["up"], x, out=fp32_hidden)
        activation(fp32_hidden, inplace=True)
        torch.mv(blocks[0]["down"], fp32_hidden, out=fp32_out)

    def triton_block() -> None:
        int8_gemv_into(blocks[0]["up_int8"], blocks[0]["up_scale"], x, hidden)
        activation(hidden, inplace=True)
        int4_gemv_into(blocks[0]["down_packed"], blocks[0]["down_scale"], hidden, out)

    def cuda_block() -> None:
        int8_gemv_cuda_into(blocks[0]["up_int8"], blocks[0]["up_scale"], x, hidden)
        activation(hidden, inplace=True)
        int4_gemv_cuda_into(blocks[0]["down_packed"], blocks[0]["down_scale"], hidden, out, variant="stream")

    def cuda_chain(count: int) -> None:
        current = x
        for index in range(count):
            int8_gemv_cuda_into(blocks[index]["up_int8"], blocks[index]["up_scale"], current, hidden)
            activation(hidden, inplace=True)
            int4_gemv_cuda_into(blocks[index]["down_packed"], blocks[index]["down_scale"], hidden, out, variant="stream")
            current = out

    def rotated_block() -> None:
        torch.matmul(x.reshape(-1, ROTATION), matrix, out=rotated_in.reshape(-1, ROTATION))
        int8_gemv_cuda_into(rotated_blocks[0]["up_int8"], rotated_blocks[0]["up_scale"], rotated_in, hidden)
        activation(hidden, inplace=True)
        torch.matmul(hidden.reshape(-1, ROTATION), matrix, out=rotated_hidden.reshape(-1, ROTATION))
        int4_gemv_cuda_into(rotated_blocks[0]["down_packed"], rotated_blocks[0]["down_scale"], rotated_hidden, out, variant="stream")

    def rotated_chain(count: int) -> None:
        current = x
        for index in range(count):
            torch.matmul(current.reshape(-1, ROTATION), matrix, out=rotated_in.reshape(-1, ROTATION))
            int8_gemv_cuda_into(rotated_blocks[index]["up_int8"], rotated_blocks[index]["up_scale"], rotated_in, hidden)
            activation(hidden, inplace=True)
            torch.matmul(hidden.reshape(-1, ROTATION), matrix, out=rotated_hidden.reshape(-1, ROTATION))
            int4_gemv_cuda_into(rotated_blocks[index]["down_packed"], rotated_blocks[index]["down_scale"], rotated_hidden, out, variant="stream")
            current = out

    def fp32_chain(count: int) -> None:
        current = x
        for index in range(count):
            torch.mv(blocks[index]["up"], current, out=fp32_hidden)
            activation(fp32_hidden, inplace=True)
            torch.mv(blocks[index]["down"], fp32_hidden, out=fp32_out)
            current = fp32_out

    fp32_block()
    torch.cuda.synchronize()
    exact = fp32_out.clone()

    triton_block()
    torch.cuda.synchronize()
    triton_out = out.clone()

    cuda_block()
    torch.cuda.synchronize()
    cuda_out = out.clone()

    single_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single_graph):
        cuda_block()
    single_graph.replay()
    torch.cuda.synchronize()
    graph_out = out.clone()

    rotated_block()
    torch.cuda.synchronize()
    rotated_out = out.clone()

    fp32_chain(STACK_BLOCKS)
    torch.cuda.synchronize()
    fp32_stack_snapshot = fp32_out.clone()

    cuda_chain(STACK_BLOCKS)
    torch.cuda.synchronize()
    stack_out = out.clone()

    stack_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(stack_graph):
        cuda_chain(STACK_BLOCKS)
    stack_graph.replay()
    torch.cuda.synchronize()
    stack_graph_out = out.clone()

    rotated_chain(STACK_BLOCKS)
    torch.cuda.synchronize()
    rotated_stack_out = out.clone()

    rotated_stack_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rotated_stack_graph):
        rotated_chain(STACK_BLOCKS)
    rotated_stack_graph.replay()
    torch.cuda.synchronize()
    rotated_stack_graph_out = out.clone()

    paths = (
        ("fp32", 1, fp32_block, fp32_out, blocks[0]["up"].numel() * 8, exact),
        ("triton raw", 1, triton_block, triton_out, low_bit_bytes(1), exact),
        ("cuda raw", 1, cuda_block, cuda_out, low_bit_bytes(1), exact),
        ("cuda graph", 1, single_graph.replay, graph_out, low_bit_bytes(1), exact),
        ("cuda rot", 1, rotated_block, rotated_out, low_bit_bytes(1), exact),
        ("stack graph", STACK_BLOCKS, stack_graph.replay, stack_graph_out, low_bit_bytes(STACK_BLOCKS), fp32_stack_snapshot),
        ("stack rot", STACK_BLOCKS, rotated_stack_graph.replay, rotated_stack_graph_out, low_bit_bytes(STACK_BLOCKS), fp32_stack_snapshot),
    )

    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"block {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}, batch 1, stack of {STACK_BLOCKS} different blocks")
    print(f"low-bit bytes per block {low_bit_bytes(1) / 1e6:.2f} MB, fp32 {blocks[0]['up'].numel() * 8 / 1e6:.1f} MB")
    print(f"rotation: block Hadamard {ROTATION}, applied to weights at pack time and to activations per block")
    print(f"warmup {WARMUP} | iterations {ITERATIONS}")
    print()
    print(f"{'path':>12}  {'blocks':>6}  {'ms':>8}  {'ms/block':>8}  {'GB/s':>8}  {'% PEAK':>7}  {'cos':>8}")

    fp32_single = None
    timings = {}
    for name, block_count, call, result, n_bytes, reference in paths:
        seconds = time_call(call, WARMUP, ITERATIONS)
        timings[name] = seconds
        if name == "fp32":
            fp32_single = seconds
        gbs = bandwidth_gb_s(n_bytes, seconds)
        per_block = seconds / block_count
        print(
            f"{name:>12}  {block_count:>6}  {seconds * 1e3:>8.3f}  {per_block * 1e3:>8.3f}  "
            f"{gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  {cosine(result, reference):>8.5f}"
        )

    fp32_stack_seconds = time_call(lambda: fp32_chain(STACK_BLOCKS), WARMUP, ITERATIONS)
    plain_pairs = time_pair(lambda: fp32_chain(STACK_BLOCKS), stack_graph.replay, ITERATIONS, 5)
    rotated_pairs = time_pair(lambda: fp32_chain(STACK_BLOCKS), rotated_stack_graph.replay, ITERATIONS, 5)

    print()
    print(f"fp32 one block {fp32_single * 1e3:.3f} ms | graph one block {timings['cuda graph'] * 1e3:.3f} ms | rotated {timings['cuda rot'] * 1e3:.3f} ms")
    print(f"stack of {STACK_BLOCKS}: fp32 {fp32_stack_seconds * 1e3:.3f} ms, graph {timings['stack graph'] * 1e3:.3f} ms, "
          f"{fp32_stack_seconds / timings['stack graph']:.2f}x, per block {timings['stack graph'] / STACK_BLOCKS * 1e3:.3f} ms")
    print(f"stack graph reproduces the eager chain bit for bit: {torch.equal(stack_graph_out, stack_out)}")
    print(f"rotated stack graph reproduces its eager chain bit for bit: {torch.equal(rotated_stack_graph_out, rotated_stack_out)}")
    print()
    print(f"cosine through {STACK_BLOCKS} blocks against fp32: plain {cosine(stack_graph_out, fp32_stack_snapshot):.5f}, "
          f"rotated {cosine(rotated_stack_graph_out, fp32_stack_snapshot):.5f}")
    print(f"paired ratio fp32 / graph: {plain_pairs.ratio:.2f}x ({plain_pairs.ratio_low:.2f} to {plain_pairs.ratio_high:.2f}) "
          f"| rotated {rotated_pairs.ratio:.2f}x ({rotated_pairs.ratio_low:.2f} to {rotated_pairs.ratio_high:.2f})")
    print(f"per block, median of paired repeats: fp32 {plain_pairs.a / STACK_BLOCKS * 1e3:.3f} ms, "
          f"graph {plain_pairs.b / STACK_BLOCKS * 1e3:.3f} ms, rotated {rotated_pairs.b / STACK_BLOCKS * 1e3:.3f} ms")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
