"""One object that runs the int8 policy, the per observation cost, and the 0.98 gate it clears.

Concept: a serving backend calls `prefill` once per observation and `act` once per denoising
step, so this times both and checks the step against the same 16 layers in fp32. The tower's
own contribution is measured separately at 0.99956 (EXP-112), and the whole model at 0.99990
(EXP-114), so this gate covers the backbone and the expert, which is what the backend changes.

Task: python -m vla_cu backend
"""

from __future__ import annotations

import torch

from bench.expert_stack import read_reference, reference_layer
from bench.real_layers import CHECKPOINT, cosine, relative_error
from vla_cu.bench_utils import enforce_memory_budget, time_call, warm_gpu
from vla_cu.checkpoint import open_checkpoint
from vla_cu.policy import TOKENS, SmolVlaPolicy
from vla_cu.prefix import LAYERS

GATE = 0.98
WARMUP = 5
ITERATIONS = 50


def main() -> int:
    """Build the policy, time the prefill and the step, and gate the step on cosine."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    policy = SmolVlaPolicy(checkpoint, tokens=TOKENS)
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand((3, 3, 512, 512), device="cuda", generator=generator) * 2 - 1
    state = torch.zeros(32, device="cuda")
    noise = torch.randn((TOKENS, policy.hidden), device="cuda", generator=generator)
    print(f"GPU {torch.cuda.get_device_name(0)} | {LAYERS} expert layers at int8 | {TOKENS} action tokens")
    print("frames are seeded noise and the state is zeros, so the content is not a measurement")

    first_seconds = policy.prefill(frames, state)
    steady_seconds = policy.prefill(frames, state)
    print(f"prefix {policy.rows.shape[0]} rows | {policy.bytes / 1e6:.2f} MB of expert weights, attention and down at int8")
    print(f"prefill {first_seconds * 1e3:.2f} ms first, {steady_seconds * 1e3:.2f} ms after the graph is captured")

    graph_out = policy.act(noise).clone()
    torch.cuda.synchronize()
    step_seconds = time_call(lambda: policy.act(noise), WARMUP, ITERATIONS)

    reference = [read_reference(checkpoint, index) for index in range(LAYERS)]
    current = noise.clone()
    for weights, prefix in zip(reference, policy.prefixes):
        current = reference_layer(current, weights, prefix)
    exact = current.clone()
    held = cosine(graph_out.reshape(-1), exact.reshape(-1))
    print(f"step {step_seconds * 1e3:.3f} ms graphed, ten steps {step_seconds * 10e3:.2f} ms")
    print(f"cosine against the fp32 expert chain over the same prefix {held:.5f}, rel L2 {relative_error(graph_out.reshape(-1), exact.reshape(-1)):.2e}")
    print(f"gate {GATE:.2f}: {'pass' if held >= GATE else 'FAIL'}")
    print()
    return 0 if held >= GATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
