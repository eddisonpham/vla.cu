# HARDWARE: detected environment and baselines

> Filled in from Chunk 0 (`python scripts/env_check.py`, run 2026-11-18). Do not assume the GPU; use what is recorded here.
> Update when the environment changes and add a line to the change log at the bottom.

## GPU
- Name: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, consumer)
- Total memory: 7.9 GB (8151 MiB) | Project budget: 7 GB (enforced in `vla_cu/bench_utils.py`)
- Compute capability: 12.0 (sm_120) -> needs CUDA 12.8+ builds; older wheels will not run on it
- SM count: 26
- L2 cache: 33.6 MB (torch `L2_cache_size`). Any weight matrix under 32 MiB benchmarks as cache, not DRAM, which is why the int8 table carries an L2 column (EXP-010)
- Theoretical peak memory bandwidth: 384 GB/s (GDDR7, 128-bit bus, 24 Gbps effective) | Source: NVIDIA "GeForce RTX 50 Series Laptop GPU" spec table, https://www.nvidia.com/en-us/geforce/laptops/50-series (accessed 2026-11-18); cross-checked TechPowerUp, https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-mobile.c4230

Caveat: this is a 45-100 W laptop GPU, so clocks (and therefore achieved bandwidth)
move with power limits and thermals. Use 384 GB/s as the denominator for "% of
peak", not as an expectation for any single run.

## Toolchain
- OS / driver: Windows (win32), NVIDIA driver 617.42, CUDA UMD 13.4
- Python: 3.13.12; project venv at `.venv/` -> use `.venv/Scripts/python.exe`
- PyTorch / CUDA (torch): 2.8.0+cu129 / 12.9
- Triton: 3.9.0 via `triton-windows==3.9.0.post30` (cp312-abi3 wheel, works on Python 3.13). Bundles its own `ptxas.exe` at `.venv/Lib/site-packages/triton/backends/nvidia/bin/ptxas.exe`, so kernels compile without a CUDA toolkit. Verified on sm_120: EXP-002.
- CUDA toolkit: 12.9.2 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` (installed 2026-11-18, EXP-003). `CUDA_PATH` is set machine-wide and `...\v12.9\bin` is on the machine PATH, so a newly opened shell finds it
- nvcc: PRESENT, `...\v12.9\bin\nvcc.exe`, release 12.9, V12.9.86. Matches torch's cu129
- `compute-sanitizer`: PRESENT at `...\v12.9\compute-sanitizer\compute-sanitizer.exe` (plus a `bin\compute-sanitizer.bat` wrapper). Not on PATH by default, so call it by path
- Nsight Systems (`nsys`): 2025.1.3 at `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe`. Not on PATH, so call it by path
- Nsight Compute (`ncu`): 2025.2.1 at `C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1\target\windows-desktop-win7-x64\ncu.exe`, with an `ncu.bat` wrapper one level up. Not on PATH either
- `compute-sanitizer`: version 2025.2.1.0 (build 35969825). It arrived with the toolkit, while the two Nsight tools are their own installs
- Host compilers (MSVC): 14.44.35207 from Visual Studio 2022 BuildTools 17.14.45, the one nvcc accepts, and 14.50.35717 from Visual Studio 18 BuildTools, which nvcc 12.9 rejects. Windows SDK 10.0.26100.0
- Both installs live under `C:\Program Files (x86)\Microsoft Visual Studio` (`2022\BuildTools` and `18\BuildTools`); nothing MSVC lives under `C:\Program Files`
- Use 14.44 by calling `...\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat` first; it puts `cl.exe` on PATH and sets `INCLUDE` and `LIB` (EXP-004)
- ninja: absent from the default PATH, present after `vcvars64.bat`, which prepends Visual Studio's bundled copy. `torch.utils.cpp_extension` needs it, so extension builds happen inside a vcvars shell (EXP-005)
- `python -m vla_cu cuda-check` reports the toolkit nvcc, the resolved `cl`, every installed MSVC toolset, ninja, and one verdict line, without compiling anything (DEC-015)
- `python -m vla_cu env` finds `ncu` and `nsys` in their versioned install folders, so neither needs to be on PATH (EXP-004)
- Installed into the venv: torch 2.8.0+cu129, torchvision 0.23.0, numpy 2.5.5, pytest

## Measured baselines (measured only; include command and date)
| Date | Test | Result | Command | Commit |
|---|---|---|---|---|
| 2026-11-18 | env check (no kernel) | sm_120, 26 SM, 7.9 GB, torch 2.8.0+cu129 | `.venv/Scripts/python.exe scripts/env_check.py` | 9b5dee6 |
| 2026-11-22 | vector add, torch against Triton, 16M fp32, 201 MB moved | torch 309.7 GB/s, Triton 330.9 GB/s (86.2 percent of 384) | `python -m vla_cu bench` | EXP-008 |
| 2026-11-22 | vector add, 1M fp32, 12.6 MB, cache resident | torch 1186.7 GB/s, Triton 966.0 GB/s, above DRAM peak so it measures L2 | `python -m vla_cu bench` | EXP-008 |
| 2026-11-22 | launch overhead, 1-element Triton kernel, 1000 launches | 16.71 us per launch from Python, 0.63 us per graph node, 26 times cheaper | `python -m vla_cu launch` | EXP-009 |
| 2026-11-22 | int8 GEMV fused, 8192 x 16384, 134 MB weights, DRAM bound | 354.7 GB/s, 92.4 percent of peak, 0.379 ms; naive path 7.046 ms | `python -m vla_cu gemv` | EXP-010 |
| 2026-11-22 | int8 GEMV fused, 4096 x 8192, weights equal L2 | 806.2 GB/s, which is L2 bandwidth and not a DRAM result | `python -m vla_cu gemv` | EXP-010 |
| 2026-11-22 | two int8 GEMV kernels chained, 1024 -> 1024 -> 512 | plain 0.031 ms, graph replay 0.012 ms, bit exact | `python -m vla_cu graph` | EXP-011 |
| 2026-11-22 | int8 GEMV block size sweep, 8192 x 16384 | 256 to 1024 within 0.2 percent, 336.7 GB/s best, 128 loses 13 percent | `python -m vla_cu tune` | EXP-012 |
| 2026-11-22 | int8 GEMV split-K, 16 x 8388608, 134 MB | 3.430 ms at split 1, 0.525 ms at split 8, 6.53x; loss above 256 rows | `python -m vla_cu tune` | EXP-012 |
| 2026-11-22 | ladder at 8192 x 16384, batch 1 | fp32 1.614 ms, bf16 0.803, int8 0.375, int4 0.263, all with relative L2 error | `python -m vla_cu matrix` | EXP-013 |
| 2026-11-23 | ladder at 8192 x 16384 after the group-major int4 fix | fp32 1.587 ms, bf16 0.811, int8 0.400, int4 0.230 (85.5 percent of peak, 6.89x) | `python -m vla_cu matrix` | EXP-014 |
| 2026-11-23 | MLP block 960 -> 2560 -> 960, batch 1 | fp32 0.072 ms, fused 0.085 ms, cosine 0.99346; L2 resident, so dispatch bound | `python -m vla_cu layer` | EXP-015 |
| 2026-11-23 | CUDA C++ against Triton int4 GEMV | 8192 x 16384: 0.240 against 0.231 ms; 1024 x 1024: 0.008 against 0.021 ms | `python -m vla_cu cuda` | EXP-016 |

## Environment change log
- 2026-11-18: file created, nothing detected yet.
- 2026-11-18: Chunk 0 detected the environment above; `.venv` created with the cu129 wheel; no CUDA toolkit present.
- 2026-11-18: CUDA 12.9.2 toolkit installed. nvcc and compute-sanitizer present; nsys and ncu absent; bundled VS 18 MSVC 14.50 is newer than nvcc 12.9 accepts.
- 2026-11-18: VS 2022 BuildTools 17.14.45 added (MSVC 14.44.35207). nvcc 12.9 compiles and runs a CUDA file. Nsight Compute and Nsight Systems now found by the env task without PATH edits.
- 2026-11-18: recorded that both Visual Studio installs sit under `Program Files (x86)`, and that ninja appears only in a vcvars shell. Added the `cuda-check` task (EXP-005).
- 2026-11-23: installed `ninja 1.13.2` into the venv (it is now in `requirements.txt`). A `torch.utils.cpp_extension` build still needs `cl.exe`, so builds and every `cuda` task run happen inside `vcvars64.bat`; outside it, torch invalidates the cached build and fails on ninja. First successful extension build and load: EXP-016.
- 2026-11-23: `compute-sanitizer` 2025.2.1 cannot instrument this extension's launch (`CUDA_ERROR_INVALID_HANDLE` on `cuKernelGetFunction`), so the CUDA kernel has no sanitizer pass. Triton kernels remain clean (EXP-016).
## Clock control (2026-11-23)

| Fact | Value |
|---|---|
| `nvidia-smi -lgc 2000` | refused, exit 4, needs administrator |
| clocks.current.sm at idle | 870 MHz |
| clocks.max.sm | 3090 MHz |
| Consequence | a benchmark starts at 28 percent of the ceiling |
| Workaround | paired alternation, `time_pair` in `vla_cu/bench_utils.py` (DEC-035) |

To pin clocks, run an elevated shell once: `nvidia-smi -pm 1` then `nvidia-smi -lgc 2000`, and reset with `nvidia-smi -rgc`. Nothing in the repo can do this without administrator rights.

## Host compiler on PATH (2026-11-23)

| Fact | Value |
|---|---|
| `cl` outside a vcvars shell | absent |
| `clang` outside a vcvars shell | `C:\Program Files\LLVM\bin\clang.EXE` |
| nvcc 12.9 with that clang | not a supported host compiler on Windows |
| Consequence | the cuda tests skip when `cl` is absent (DEC-036); a vcvars shell is still required to build |

## Sanitizer and the extension (2026-11-23)

| Fact | Value |
|---|---|
| `compute-sanitizer --target-processes all` | starts, runs the kernels, reports 2 errors |
| Error | `CUDA_ERROR_INVALID_HANDLE` on `cuKernelGetFunction`, once per extension |
| Memory errors reported | none |
| `CUDA_MODULE_LOADING=EAGER` with `CUDA_LAUNCH_BLOCKING=1` | did not finish inside 600 s |
| Consequence | no kernel-level sanitizer pass on the CUDA ports (DEC-041, EXP-041) |

## Measured baselines (2026-11-23, added this pass)
| Date | Test | Result | Command | Commit |
|---|---|---|---|---|
| 2026-11-23 | int4 CUDA variants, 8192 x 16384 | Triton 0.221 ms, baseline 0.238, unroll 0.230, stream 0.208 (363.5 GB/s, 94.7 percent) | `python -m vla_cu cuda` | EXP-040 |
| 2026-11-23, pinned clock | int4 CUDA variants, 8192 x 16384 | Triton 0.252 ms, baseline 0.274, unroll 0.237, stream 0.218 (347.2 GB/s, 90.4 percent) | `python -m vla_cu cuda` in vcvars | EXP-042 |

## Clock and sanitizer updates (2026-11-23, second pass)

The sections above record the state before an elevated shell was used. These rows replace them.

| Fact | Value |
|---|---|
| `nvidia-smi -lgc 2000` from an elevated shell | applied: 1987 MHz at idle, 1980 to 1987 MHz under load |
| throttle reason under load | 0x4, software power cap, 64.7 to 76.4 W of a 115 W maximum |
| clock spread under load | 0.35 percent, against the 2x spread in EXP-017 |
| `compute-sanitizer` for a kernel pass | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\compute-sanitizer\compute-sanitizer.exe`, not on PATH |
| standalone int4 harness under memcheck | 0 errors, 0 leaks, worst error 1.788e-07 (EXP-044) |
| `ncu` from a normal shell | still `ERR_NVGPUCTRPERM` |
| `ncu` from the elevated shell | works, report in `bench/results/ncu_int4.ncu-rep` |
| `ncu` through the task, normal shell | reaches ncu, stops at `ERR_NVGPUCTRPERM`, exit 1 |

## Native plateau and the int8 harness (2026-11-23)

| Date | Test | Result | Command | Entry |
|---|---|---|---|---|
| 2026-11-23, pinned clock | int8 standalone harness under memcheck | 0 errors, 0 leaks, worst error 3.815e-06 at 511 x 1040 | `compute-sanitizer --tool memcheck --leak-check full bench/results/int8_check.exe` | EXP-046 |
| 2026-11-23, pinned clock | native against Triton, median of 5 repeats | unroll 338.0 GB/s, 88.0 percent at 4096 x 8192; Triton 266.4, 69.4 percent; stream 307.2, 80.0 at 8192 x 16384 | `python -m vla_cu plateau` in vcvars | EXP-047 |
| 2026-11-23, pinned clock | depth 16 normed chain | 37.0 MB, 1.534 ms, cosine 0.99548 | `python -m vla_cu scale` | EXP-048 |
| 2026-11-23, pinned clock | depth 16 layer chain, attention plus MLP | 50.78 MB, 3.475 ms eager, 0.497 ms graphed, cosine 0.99795, exact | `python -m vla_cu scale` | EXP-049 |
| 2026-11-23, pinned clock | int4 SASS at 4096 x 8192 | Triton 416 instructions, 12 LDG, 8 shared memory ops, 12 butterfly shuffles; native unroll 544, 32 LDG where the 16 `LDG.E.128` are fp32 activations, no SMEM, 5 shuffle-downs | `python -m vla_cu sass` in vcvars | EXP-050 |
| 2026-11-23, pinned clock | depth 16 chain at two tokens, q and k included | 64.59 MB, 15.450 ms eager, cosine 0.99622, rel L2 8.76e-02 | `python -m vla_cu scale` | EXP-051 |
| 2026-11-23, pinned clock | wide-load Triton int4 at 4096 x 8192 | 0.088 ms, 216.1 GB/s, 56.3 percent, 0.77x of the staged mapping, 448 instructions, 0 `LDG.E.128` | `python -m vla_cu plateau` in vcvars | EXP-052 |
| 2026-11-23, pinned clock | depth 16 two-token chain, graphed | 2.818 ms graphed against 24.880 ms eager, exact at every depth, cosine 0.99622 | `python -m vla_cu scale` | EXP-053 |
| 2026-11-23, pinned clock | depth 16 two-token chain, q and k rotated | cosine 0.99622, rel L2 8.76e-02, rotation costs 49 us per layer inside a graph | `python -m vla_cu scale` | EXP-053 |
| 2026-11-23, noisier run | wide Triton with contiguity hints | 0.276 ms, 17.8 percent, 0 `LDG.E.128`, LDS 8, reverted | `python -m vla_cu plateau` in vcvars | EXP-054 |
