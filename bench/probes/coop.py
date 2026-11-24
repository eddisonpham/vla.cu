"""Three cooperative launches, one per token tile boundary, for memcheck.

Token counts 3, 17 and 26 land on the partial tail, the first full tile and the first
tile past the second boundary, which are the three index paths in the body.
"""

from __future__ import annotations

import torch

from vla_cu.cuda_coop import HIDDEN, INTERMEDIATE, coop_layer_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8


def main() -> int:
    """Run the fused tail at 3, 17 and 26 tokens and print when the GPU is idle."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(HIDDEN, device="cuda", generator=generator) + 1.0
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * (1.0 / HIDDEN**0.5)
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * (1.0 / INTERMEDIATE**0.5)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP_SIZE)
    for tokens in (3, 17, 26):
        x = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
        other = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
        added = torch.empty_like(x)
        normed = torch.empty_like(x)
        out = torch.empty_like(x)
        act = torch.empty((tokens, INTERMEDIATE), device="cuda")
        coop_layer_into(x, other, weight, up_codes, up_scales, down_packed, down_scales, added, normed, act, out)
    torch.cuda.synchronize()
    print("probe ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
