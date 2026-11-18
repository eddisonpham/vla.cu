# STARTER.md: one-time repo bootstrap for vla.cu

> **Use this once, at the very beginning.** It is not read every session. `CLAUDE.md` is the per-prompt rulebook; this file only covers creating the repo and delivering the first learning chunks. When bootstrap is finished, record that in `agents/STATE.md` and ignore this file afterwards.

Follow all rules in `CLAUDE.md` while working through this file (one small chunk at a time, documented code, predict-first prompts, `agents/` updates after every chunk).

## Project in one paragraph

vla.cu is a learning project: write fused low-bit (4-bit) dequantize + matmul kernels and related kernels for small robot policies (e.g. SmolVLA, ~450M params) running at batch size 1 on a single consumer GPU, then measure them honestly against bitsandbytes, torchao, and a PyTorch baseline. It is freeform: after the starter chunks, the human steers and the agent offers options from `agents/BACKLOG.md`.

## Part A: Bootstrap the repo (agent proposes, human runs)

Deliver this as a short checklist of commands, not as one big script. The human runs each command.

1. **Create the folder and initialize git**
   ```bash
   mkdir vla.cu && cd vla.cu
   git init
   ```
2. **Create the skeleton** (empty placeholders only; real files arrive chunk by chunk)
   ```bash
   mkdir -p vla_cu kernels/triton csrc scripts tests bench docs agents
   touch vla_cu/__init__.py kernels/__init__.py kernels/triton/__init__.py bench/__init__.py
   ```
   Place `CLAUDE.md` and the `agents/` folder (provided separately) in the repo root.
3. **Python environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
   Ask the human which OS they use. Triton ships with PyTorch on Linux; native Windows needs a different route (WSL2 is the usual answer). Check, don't assume.
4. **Dependencies**: `requirements.txt` containing `torch`, `triton` (if not bundled), `pytest`, `numpy`. The agent must tell the human which PyTorch CUDA build to install for their driver and explain why (the right wheel depends on the detected driver and GPU, so Chunk 0 comes first if unsure).
5. **`.gitignore`**: `.venv/`, `__pycache__/`, `*.pt`, `build/`, `*.so`, `bench/results/`
6. **First commit** (suggest the command, human runs it):
   ```bash
   git add . && git commit -m "Initial skeleton with CLAUDE.md and agents/ memory"
   ```
7. **Record it**: update `agents/STATE.md` (bootstrap done, date, commit hash) and add the environment to `agents/HARDWARE.md` after Chunk 0.

## Part B: Starter chunks

Deliver one at a time, in order, using the response format in `CLAUDE.md`. Adapt the code only if the human's environment requires it, and say what changed and why.

### Chunk 0: environment check (`scripts/env_check.py`)

Purpose: find out what GPU, drivers, and tools actually exist before anything else.

```python
"""Environment check for vla.cu.

Prints the GPU, PyTorch/CUDA/Triton versions, and whether the profiling and
build tools (nvcc, Nsight Systems, Nsight Compute, compute-sanitizer) are on PATH.
Run this first; every later chunk assumes these tools exist.

Usage: python scripts/env_check.py
"""
import shutil
import torch


def check_tool(name: str) -> str:
    """Return the path of an executable on PATH, or a 'NOT FOUND' marker."""
    return shutil.which(name) or "NOT FOUND"


def main() -> None:
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA (torch):  {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        import triton
        print(f"Triton:        {triton.__version__}")
    except ImportError:
        print("Triton:        NOT INSTALLED")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:           {props.name}")
        print(f"Memory:        {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute cap.:  {props.major}.{props.minor}")
        print(f"SM count:      {props.multi_processor_count}")

    print("\nTools on PATH:")
    for tool in ("nvcc", "nsys", "ncu", "compute-sanitizer"):
        print(f"  {tool:<18} {check_tool(tool)}")


if __name__ == "__main__":
    main()
```

After the human pastes the output, the agent fills in `agents/HARDWARE.md` and looks up the GPU's **theoretical peak memory bandwidth** (stating the source and date), since every bandwidth percentage depends on it.

### Chunk 1: timing and memory-budget utilities (`vla_cu/bench_utils.py`)

Purpose: one trusted place for timing and the 7 GB cap.

```python
"""Shared utilities: GPU memory budget, accurate CUDA timing, bandwidth math.

Concept: GPU work is asynchronous. The CPU launches a kernel and moves on, so
naive wall-clock timing measures launch overhead, not execution. CUDA events
are recorded *on the GPU timeline*, so they time the actual work.
"""
from typing import Callable

import torch

MEMORY_BUDGET_GB = 7.0  # Hard project constraint (see CLAUDE.md).


def enforce_memory_budget(gb: float = MEMORY_BUDGET_GB) -> None:
    """Cap this process's CUDA allocations at `gb` gigabytes.

    Any allocation beyond the cap raises an out-of-memory error instead of
    silently using more than the project allows. Call once at script start.
    """
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, gb * 1024**3 / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def time_cuda(fn: Callable[[], object], warmup: int = 25, iters: int = 100) -> float:
    """Return the average runtime of `fn` in milliseconds, measured on the GPU.

    Args:
        fn: zero-argument callable that launches the GPU work to time.
        warmup: untimed runs first (JIT compile, cache warm-up, clock ramp-up).
        iters: timed runs; we average over them to reduce noise.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()  # wait until the GPU has actually finished
    return start.elapsed_time(end) / iters


def gb_per_s(bytes_moved: int, ms: float) -> float:
    """Achieved memory bandwidth in GB/s (1 GB = 1e9 bytes, matching vendor specs)."""
    return bytes_moved / (ms * 1e-3) / 1e9
```

### Chunk 2: first Triton kernel, vector add (`kernels/triton/vector_add.py`)

Purpose: the smallest possible kernel, to learn the "program id, offsets, mask" pattern.

```python
"""Vector add in Triton: out = x + y.

Concept: a Triton kernel is launched as a grid of "programs". Each program
handles one BLOCK-sized chunk of the data. `tl.program_id` says which chunk,
`tl.arange` builds the element offsets within it, and a mask protects the
final partial block from reading or writing past the end of the tensor.

Expected bottleneck: memory-bound. Each element does 1 add but moves 12 bytes
(read x, read y, write out as fp32), so speed is limited by memory bandwidth.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Which chunk of the data does this program own?
    pid = tl.program_id(axis=0)
    # Element indices for this chunk, e.g. pid=2, BLOCK=1024 -> 2048..3071.
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    # The last block may extend past the end; mask off the extra lanes.
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """Elementwise x + y for two same-shape, contiguous CUDA tensors."""
    assert x.is_cuda and x.shape == y.shape, "need same-shape CUDA tensors"
    assert x.is_contiguous() and y.is_contiguous(), "kernel assumes contiguous memory"
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block),)  # enough programs to cover all n elements
    add_kernel[grid](x, y, out, n, BLOCK=block)
    return out
```

### Chunk 3: correctness test, then benchmark

`tests/test_vector_add.py` (run with `pytest tests/`):

```python
"""Correctness tests for the Triton vector add against PyTorch's own x + y.

Sizes are deliberately awkward (1, 1023, 1025, ...) to catch masking bugs at
block boundaries, which is the most common beginner kernel bug.
"""
import pytest
import torch

from kernels.triton.vector_add import add


@pytest.mark.parametrize("n", [1, 31, 1023, 1024, 1025, 100_003, 2**20])
def test_matches_torch(n: int) -> None:
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    torch.testing.assert_close(add(x, y), x + y)
```

`bench/bench_vector_add.py` (run with `python -m bench.bench_vector_add`):

```python
"""Benchmark Triton vector add vs. PyTorch, reported as achieved bandwidth.

Concept: for a memory-bound kernel, the right question is not "how many ms?"
but "what fraction of the GPU's peak memory bandwidth did we achieve?"
"""
import torch

from kernels.triton.vector_add import add
from vla_cu.bench_utils import enforce_memory_budget, gb_per_s, time_cuda


def main() -> None:
    enforce_memory_budget()
    n = 2**26  # 64M floats = 256 MB per tensor; big enough to hide launch overhead
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    bytes_moved = 3 * n * x.element_size()  # read x, read y, write out

    for name, fn in [("torch", lambda: x + y), ("triton", lambda: add(x, y))]:
        ms = time_cuda(fn)
        print(f"{name:<8} {ms:7.3f} ms   {gb_per_s(bytes_moved, ms):8.1f} GB/s")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
```

**Predict-first prompt for the human:** what fraction of peak bandwidth do you expect, and will Triton beat torch? Write it in `docs/journal.md` before running. Then log the run in `agents/EXPERIMENTS.md` as EXP-001.

## Part C: Done criteria for bootstrap

- Repo skeleton exists, first commit made, `.venv` working.
- Chunks 0-3 delivered and run by the human; results logged in `agents/EXPERIMENTS.md`, hardware recorded in `agents/HARDWARE.md`.
- `agents/STATE.md` says "bootstrap complete" with the date and commit hash.
- The agent then offers 3-4 options from `agents/BACKLOG.md` and waits for the human to choose.
