"""Run one int4 GEMV launch and exit, so a profiler has a single clean target.

Nsight Compute attaches to a process, so the kernel has to live in a real file: Triton
compiles the source text of the decorated function, and a heredoc has none. The launch
happens three times so a profiler can skip the compile and the first touch.

Task: used by `ncu`, not by hand.
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import enforce_memory_budget
from vla_cu.quant import pack_int4

N_OUT = 8192
N_IN = 16384
LAUNCHES = 3


def main() -> int:
    """Time nothing, but launch the kernel LAUNCHES times at the DRAM-bound shape."""
    enforce_memory_budget()
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((N_OUT, N_IN), device="cuda", generator=generator) * 0.05
    x = torch.randn(N_IN, device="cuda", generator=generator)
    packed, scales = pack_int4(w)
    out = torch.empty(N_OUT, device="cuda")
    for _ in range(LAUNCHES):
        int4_gemv_into(packed, scales, x, out)
    torch.cuda.synchronize()
    print(f"launches {LAUNCHES} at {N_OUT} x {N_IN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
