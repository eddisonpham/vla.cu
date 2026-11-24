"""Instruction mix of the Triton and native int4 GEMV at 4096 x 8192.

Concept: both kernels read the same 8 MB of packed weights and both report hundreds of
GB/s, so a 69 against 88 percent gap is not a bandwidth fact, it is a codegen fact. This
dumps the SASS of each and counts what the compiler emitted, which is what to read before
tuning either kernel. Two Triton mappings are dumped: the staged one that ships, and the
wide one that reads the row as int32. The raw listings stay in bench/results.

Task: python -m vla_cu sass
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import torch
import triton

from vla_cu.bench_utils import enforce_memory_budget, warm_gpu
from vla_cu.quant import GROUP_SIZE

N_ROWS = 4096
N_COLS = 8192
RESULTS = Path(__file__).resolve().parent / "results"
VARIANTS = {("0", "0"): "baseline", ("1", "0"): "unroll", ("1", "1"): "stream"}
COLUMNS = ("LDG", "LDG.E.128", "IMAD", "FFMA", "LOP3", "SHF", "IADD3", "PRMT", "LDS", "BRA")
INSTRUCTION = re.compile(r"\s*/\*[0-9a-f]{4}\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9.]*)")
FUNCTION = re.compile(r"Function : (.+)")


def tool(name: str) -> str:
    """Return the path of a CUDA binary tool, from PATH or from the toolkit root."""
    found = shutil.which(name)
    if found:
        return found
    root = os.environ.get("CUDA_PATH")
    if root is None:
        raise RuntimeError(f"{name} is not on PATH and CUDA_PATH is unset")
    return str(Path(root) / "bin" / f"{name}.exe")


def disassemble(binary: Path) -> str:
    """Return the SASS listing of a cubin, or of every kernel in a shared object."""
    return subprocess.run([tool("cuobjdump"), "-sass", str(binary)], capture_output=True, text=True, check=True).stdout


def histograms(text: str) -> dict[str, Counter]:
    """Return one opcode count per function in a SASS listing."""
    counts: dict[str, Counter] = {}
    current: Counter | None = None
    for line in text.splitlines():
        header = FUNCTION.match(line.strip())
        if header:
            current = Counter()
            counts[header.group(1).strip()] = current
            continue
        instruction = INSTRUCTION.match(line)
        if instruction and current is not None:
            current[instruction.group(1)] += 1
    return counts


def cubin_listing(compiled, name: str) -> str:
    """Write a compiled kernel's cubin to bench/results and return its SASS listing."""
    cubin = compiled.asm["cubin"]
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    path = RESULTS / f"{name}.cubin"
    path.write_bytes(cubin)
    return disassemble(path)


def triton_listing() -> str:
    """Compile the staged Triton kernel at the measured shape and return its SASS."""
    from kernels.triton.int4_gemv import BLOCK_K, ROWS_PER_PROGRAM, int4_gemv_kernel

    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(N_COLS, device="cuda", generator=generator)
    packed = torch.randint(0, 256, (N_ROWS, N_COLS // 2), dtype=torch.uint8, device="cuda", generator=generator)
    scales = torch.rand((N_ROWS, N_COLS // GROUP_SIZE), device="cuda", generator=generator) * 0.01
    out = torch.empty(N_ROWS, device="cuda")
    compiled = int4_gemv_kernel[(triton.cdiv(N_ROWS, ROWS_PER_PROGRAM),)](
        packed,
        scales,
        x,
        out,
        N_ROWS,
        N_COLS,
        GROUP=GROUP_SIZE,
        BLOCK_K=BLOCK_K,
        ROWS=ROWS_PER_PROGRAM,
        W_EVICT="evict_first",
        X_EVICT="evict_last",
    )
    return cubin_listing(compiled, "int4_gemv_triton")


def wide_listing(precision: str = "") -> str:
    """Compile the wide-load Triton kernel at the measured shape and return its SASS."""
    from kernels.triton.int4_gemv_wide import BLOCK_K, DOT_LANES, PRECISION, WARPS, int4_gemv_wide_kernel

    precision = precision or PRECISION

    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(N_COLS, device="cuda", generator=generator)
    packed = torch.randint(0, 256, (N_ROWS, N_COLS // 2), dtype=torch.uint8, device="cuda", generator=generator)
    scales = torch.rand((N_ROWS, N_COLS // GROUP_SIZE), device="cuda", generator=generator) * 0.01
    out = torch.empty(N_ROWS, device="cuda")
    compiled = int4_gemv_wide_kernel[(N_ROWS,)](
        packed.view(torch.int32),
        scales,
        x,
        out,
        N_ROWS,
        N_COLS,
        GROUP=GROUP_SIZE,
        BLOCK_K=BLOCK_K,
        DOT_LANES=DOT_LANES,
        PRECISION=precision,
        num_warps=WARPS,
        W_EVICT="evict_first",
        X_EVICT="evict_last",
    )
    return cubin_listing(compiled, f"int4_gemv_wide_{precision}_triton")


def cuda_listing() -> str:
    """Build the native extension and return the SASS of every kernel it holds."""
    from vla_cu.cuda_int4 import extension

    return disassemble(Path(extension().__file__))


def row(label: str, counts: Counter) -> str:
    """Format one kernel's instruction counts for the comparison table."""
    totals = [sum(value for opcode, value in counts.items() if opcode.startswith(column)) for column in COLUMNS]
    cells = "  ".join(f"{total:>{max(len(column), 5)}}" for column, total in zip(COLUMNS, totals))
    return f"{label:>22}  {sum(counts.values()):>6}  {cells}"


def main() -> int:
    """Print Triton against each native variant, opcode by opcode, and save both listings."""
    enforce_memory_budget()
    warm_gpu()
    RESULTS.mkdir(parents=True, exist_ok=True)
    triton_text = triton_listing()
    wide_text = wide_listing()
    wide_tf32_text = wide_listing("tf32")
    cuda_text = cuda_listing()
    (RESULTS / "sass_triton.txt").write_text(triton_text, encoding="utf-8")
    (RESULTS / "sass_wide.txt").write_text(wide_text, encoding="utf-8")
    (RESULTS / "sass_wide_tf32.txt").write_text(wide_tf32_text, encoding="utf-8")
    (RESULTS / "sass_cuda.txt").write_text(cuda_text, encoding="utf-8")
    print(f"GPU {torch.cuda.get_device_name(0)} | int4 GEMV {N_ROWS} x {N_COLS}, group {GROUP_SIZE}, fp32 activations")
    print("listings: bench/results/sass_triton.txt, sass_wide.txt and sass_cuda.txt")
    print()
    print(f"{'kernel':>22}  {'instr':>6}  " + "  ".join(f"{column:>{max(len(column), 5)}}" for column in COLUMNS))
    for counts in histograms(triton_text).values():
        print(row("triton staged", counts))
    for counts in histograms(wide_text).values():
        print(row("triton wide", counts))
    for counts in histograms(wide_tf32_text).values():
        print(row("triton wide tf32", counts))
    for name, counts in histograms(cuda_text).items():
        if "int4_gemv_kernel" not in name:
            continue
        flags = re.search(r"ILb([01])ELb([01])E", name)
        print(row(VARIANTS.get(flags.groups(), "native") if flags else "native", counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
