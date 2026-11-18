"""Probe: does Triton actually compile and run on this machine?

Why this file exists: `import triton` succeeding proves almost nothing. Triton
compiles Python -> TTIR -> LLVM -> PTX -> SASS at runtime, so a working install
must also find a `ptxas` (PTX assembler) that knows the GPU's architecture
(sm_120 here). This probe runs one real kernel and checks the numbers against
PyTorch.

Concept demonstrated: Triton's `@triton.jit` decorator does not execute your
Python function. It reads the function's *source text* with `inspect`, parses it
as an AST, and compiles it. That is why a kernel defined in a REPL, in `-c`, or
on stdin fails with "could not get source code" - the source is not a file.

Run: .venv/Scripts/python.exe scripts/triton_probe.py
Expected: "OK" for both sizes, plus the ptxas path Triton used.
"""

from __future__ import annotations

from pathlib import Path

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def add_kernel(
    x_ptr,  # *f32, one element per lane
    y_ptr,  # *f32
    out_ptr,  # *f32
    n_elements,  # i32, total element count (may not divide BLOCK_SIZE)
    BLOCK: tl.constexpr,  # compile-time block size, so masks fold away
) -> None:
    """out[i] = x[i] + y[i] for i < n_elements.

    Thread/block-to-data mapping: program id `pid` owns the contiguous chunk
    [pid*BLOCK, (pid+1)*BLOCK); lane `k` of `pid` handles element pid*BLOCK + k.
    Memory access: each program reads two contiguous 4 KB segments and writes
    one - fully coalesced, so this kernel is purely memory-bound.

    The mask exists because n_elements (1_000_003) is not a multiple of BLOCK;
    without it the last program would read and write out of bounds.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def find_ptxas() -> str:
    """Return the ptxas binary Triton will use, or a note that it is missing."""
    bundled = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas.exe"
    if bundled.exists():
        return str(bundled)
    # Falling back to the CUDA toolkit requires it to be on PATH.
    from shutil import which

    return which("ptxas") or "NOT FOUND (bundled copy missing and no CUDA toolkit)"


def main() -> int:
    print(f"triton {triton.__version__} | torch {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)} | cc {torch.cuda.get_device_capability(0)}")
    print(f"ptxas: {find_ptxas()}")

    failures = 0
    # 1000 exercises the mask (not a multiple of 1024); 1_000_003 exercises
    # ~977 programs, so a launch of the wrong grid size shows up too.
    for n_elements in (1000, 1_000_003):
        x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
        y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        add_kernel[grid](x, y, out, n_elements, BLOCK=BLOCK_SIZE)
        torch.cuda.synchronize()

        max_err = (out - (x + y)).abs().max().item()
        ok = max_err == 0.0  # exact: float32 addition is deterministic
        failures += 0 if ok else 1
        print(f"n={n_elements:>9}: grid={grid[0]:>4} programs | max abs err {max_err:.1e} | {'OK' if ok else 'MISMATCH'}")

    print("VERDICT:", "triton works on this GPU" if failures == 0 else "triton runs but results are wrong")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
