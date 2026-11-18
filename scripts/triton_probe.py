"""Check that Triton compiles and runs a real kernel on this GPU.

`import triton` proves little: Triton lowers to TTIR, LLVM, PTX, then SASS at
runtime, so it needs a `ptxas` that knows sm_120. Because `@triton.jit` compiles
the function's source text, kernels must live in a `.py` file - not stdin, not a
REPL.

Task: python -m vla_cu probe
"""

from __future__ import annotations

from pathlib import Path
from shutil import which

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024
SIZES = (1000, 1_000_003)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr) -> None:
    """out[i] = x[i] + y[i] for i < n_elements, on contiguous fp32 tensors.

    Mapping: program `pid` owns [pid*BLOCK, (pid+1)*BLOCK). Each program does two
    contiguous 4 KB reads and one 4 KB write, so accesses coalesce and the kernel
    is memory-bound. The mask guards the final partial block.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def find_ptxas() -> str:
    """Return the ptxas Triton will use: its bundled copy first, then PATH."""
    bundled = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas.exe"
    if bundled.exists():
        return str(bundled)
    return which("ptxas") or "NOT FOUND (no bundled ptxas and no CUDA toolkit)"


def check(n_elements: int) -> bool:
    """Run the kernel at one size and return whether it matched PyTorch exactly."""
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](x, y, out, n_elements, BLOCK=BLOCK_SIZE)
    torch.cuda.synchronize()

    max_err = (out - (x + y)).abs().max().item()
    ok = max_err == 0.0
    print(f"n={n_elements:>9}: grid={grid[0]:>4} programs | max abs err {max_err:.1e} | {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    """Print versions and ptxas, check both sizes, return 0 only if both pass."""
    print(f"triton {triton.__version__} | torch {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)} | cc {torch.cuda.get_device_capability(0)}")
    print(f"ptxas: {find_ptxas()}")

    failures = sum(not check(n) for n in SIZES)
    print("VERDICT:", "triton works on this GPU" if failures == 0 else "triton runs but results are wrong")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
