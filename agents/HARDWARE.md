# HARDWARE: detected environment and baselines

> Filled in from Chunk 0 (`python scripts/env_check.py`, run 2026-11-18). Do not assume the GPU; use what is recorded here.
> Update when the environment changes and add a line to the change log at the bottom.

## GPU
- Name: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, consumer)
- Total memory: 7.9 GB (8151 MiB) | Project budget: 7 GB (enforced in `vla_cu/bench_utils.py`)
- Compute capability: 12.0 (sm_120) -> needs CUDA 12.8+ builds; older wheels will not run on it
- SM count: 26
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
- Host compiler (MSVC): 14.50.35717 from Visual Studio 18 BuildTools, `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools`, Windows SDK 10.0.26100.0. No VS 2019/2022, so no VS-integrated Nsight
- nvcc 12.9 refuses MSVC 14.50: `fatal error C1189: -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported`. CUDA C++ needs an MSVC 2017-2022 toolchain first (EXP-003)
- Installed into the venv: torch 2.8.0+cu129, torchvision 0.23.0, numpy 2.5.5, pytest

## Measured baselines (measured only; include command and date)
| Date | Test | Result | Command | Commit |
|---|---|---|---|---|
| 2026-11-18 | env check (no kernel) | sm_120, 26 SM, 7.9 GB, torch 2.8.0+cu129 | `.venv/Scripts/python.exe scripts/env_check.py` | 9b5dee6 |

## Environment change log
- 2026-11-18: file created, nothing detected yet.
- 2026-11-18: Chunk 0 detected the environment above; `.venv` created with the cu129 wheel; no CUDA toolkit present.
- 2026-11-18: CUDA 12.9.2 toolkit installed. nvcc and compute-sanitizer present; nsys and ncu absent; bundled VS 18 MSVC 14.50 is newer than nvcc 12.9 accepts.
