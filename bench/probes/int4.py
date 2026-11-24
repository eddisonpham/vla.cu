"""Three launches of the int4 CUDA kernel, one per variant, for memcheck.

Sizes include a K that is not a multiple of 512, because the scalar tail is where an
out-of-bounds read would sit.
"""

from __future__ import annotations

import torch

from vla_cu.cuda_int4 import VARIANTS, int4_gemv_cuda_into
from vla_cu.quant import GROUP_SIZE, pack_int4


def main() -> int:
    """Launch each variant once at an awkward shape and print when the GPU is idle."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    for rows, cols in ((64, 320), (17, 1024)):
        w = torch.randn((rows, cols), device="cuda", generator=generator) * 0.05
        packed, scales = pack_int4(w, group_size=GROUP_SIZE)
        x = torch.randn(cols, device="cuda", generator=generator)
        out = torch.empty(rows, device="cuda")
        for variant in VARIANTS:
            int4_gemv_cuda_into(packed, scales, x, out, variant=variant)
    torch.cuda.synchronize()
    print("probe ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
