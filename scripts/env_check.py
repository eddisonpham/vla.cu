"""Environment check for vla.cu.

Prints the GPU, PyTorch/CUDA/Triton versions, and whether the profiling and
build tools (nvcc, Nsight Systems, Nsight Compute, compute-sanitizer) are on
PATH. Run this first; every later chunk assumes these tools exist.

Concept: never guess the hardware. The compute capability decides which kernel
features compile (e.g. sm_120 for Blackwell), the memory size decides how big a
benchmark can be, and the presence of `ncu`/`nsys` decides whether a slow kernel
can be explained by counters or only by timing.

Usage: python scripts/env_check.py
"""
import shutil

import torch


def check_tool(name: str) -> str:
    """Return the path of an executable on PATH, or a 'NOT FOUND' marker.

    Args:
        name: executable name as it would be typed in a shell (no extension).

    Returns:
        Absolute path string if found, else the literal "NOT FOUND".
    """
    return shutil.which(name) or "NOT FOUND"


def main() -> None:
    """Print versions, device properties, and tool availability. No arguments."""
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA (torch):   {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        import triton

        print(f"Triton:         {triton.__version__}")
    except ImportError:
        print("Triton:         NOT INSTALLED")

    if torch.cuda.is_available():
        # Device 0 only: the project target is one consumer GPU, not a cluster.
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:            {props.name}")
        print(f"Memory:         {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute cap.:   {props.major}.{props.minor}")
        print(f"SM count:       {props.multi_processor_count}")

    print("\nTools on PATH:")
    # ncu/nsys/compute-sanitizer are how a kernel gets explained, not just timed,
    # so their absence is worth knowing before the first benchmark.
    for tool in ("nvcc", "nsys", "ncu", "compute-sanitizer"):
        print(f"  {tool:<18} {check_tool(tool)}")


if __name__ == "__main__":
    main()
