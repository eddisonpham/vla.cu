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
- nvcc: NOT FOUND (no CUDA toolkit installed)
- Nsight Systems (`nsys`): NOT FOUND
- Nsight Compute (`ncu`): NOT FOUND
- `compute-sanitizer`: NOT FOUND (ships with the CUDA toolkit, not with the driver)
- Installed into the venv: torch 2.8.0+cu129, torchvision 0.23.0, numpy 2.5.5, pytest

## Measured baselines (measured only; include command and date)
| Date | Test | Result | Command | Commit |
|---|---|---|---|---|
| 2026-11-18 | env check (no kernel) | sm_120, 26 SM, 7.9 GB, torch 2.8.0+cu129 | `.venv/Scripts/python.exe scripts/env_check.py` | 9b5dee6 |

## Environment change log
- 2026-11-18: file created, nothing detected yet.
- 2026-11-18: Chunk 0 detected the environment above; `.venv` created with the cu129 wheel; no CUDA toolkit present.
