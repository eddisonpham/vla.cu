# RESEARCH: things looked up, with sources

> Append-only, newest at the bottom. Include source and date for every entry. Mark outdated entries `OUTDATED YYYY-MM-DD: reason` instead of deleting.
> Check here before researching a topic again.

---

## Seed notes (from the planning conversation, unverified)
These came from the planning discussion, not from lookups. Verify before relying on them.

- Batch-1 inference of small models is mostly memory-bound; fusing dequantize with the multiply avoids writing full-precision weights back to global memory. (From memory, unverified for this model and GPU.)
- Prior project LeRobot Edge reported SmolVLA latency 18.07 to 8.77 ms (2.06x) and a 4-bit weight footprint 1142 to 287 MB (3.97x), with a cosine-similarity quality gate of at least 0.98 vs FP32. (Taken from the user's own resume; not re-measured in this repo.)
- Suggested learning resources: Simon Boehm's "How to Optimize a CUDA Matmul Kernel" blog post; NVIDIA CUDA C++ Programming Guide; Triton official tutorials; *Programming Massively Parallel Processors* (Hwu and Kirk). (From memory; confirm links when first used.)
- Mature fused low-bit kernels already exist (e.g. Marlin, bitsandbytes, torchao), mostly tuned for LLMs. KernelEdge is a learning project, not a replacement. (From memory, unverified.)

---

## RTX 5060 Laptop GPU peak memory bandwidth  (2026-11-18)
- Question: what is the denominator for "% of peak bandwidth" on this machine?
- Finding: 8 GB GDDR7 on a 128-bit bus at 24 Gbps effective = 384 GB/s. NVIDIA's own laptop spec table lists 384 GB/s for the RTX 5060 Laptop GPU; 26 SMs, 3328 CUDA cores, 45-100 W board power.
- Source: https://www.nvidia.com/en-us/geforce/laptops/50-series (spec table, accessed 2026-11-18); https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-mobile.c4230 (accessed 2026-11-18)
- Confidence: high for the 384 GB/s figure (two sources agree); medium for how much a 75 W laptop part can actually sustain.
- Relevance to vla.cu: every bandwidth report divides by 384 GB/s. Laptop power limits mean achieved numbers may sit well below desktop-class fractions.

## Triton on Windows  (2026-11-18)
- Question: can Triton be installed in this Python 3.13 Windows venv?
- Finding: `pip install triton` fails - "Could not find a version that satisfies the requirement triton (from versions: none)". The PyPI project has no Windows wheels. `pip install --dry-run triton-windows` resolves to `triton-windows-3.9.0.post30` (cp312-abi3 wheel, so it is ABI-compatible with 3.13). Whether it works with torch 2.8.0+cu129 and sm_120 is untested here.
- Source: local `pip install --dry-run` on 2026-11-18 (PyPI index); no official Windows support statement read yet.
- Confidence: high for the wheel situation, low for runtime compatibility.
- Relevance to vla.cu: blocks Chunks 2-3. Options: `triton-windows`, or WSL2 with official Triton.
- RESOLVED 2026-11-18: `triton-windows` works, so WSL2 is not needed. See EXP-002 and the entry below.

## triton-windows actually compiles and runs on sm_120  (2026-11-18)
- Question: is the unofficial Windows build usable, or is WSL2 required for the Triton half of the roadmap?
- Finding: `pip install triton-windows` succeeds (cp312-abi3 wheel, ABI-compatible with Python 3.13) and installs `triton` 3.9.0. It bundles `ptxas.exe` under `triton/backends/nvidia/bin/`, so kernels are assembled without any CUDA toolkit. A masked vector add matched PyTorch bit-exactly at `n=1000` and `n=1_000_003` with torch 2.8.0+cu129.
- Source: local install + run on 2026-11-18 (EXP-002); wheel from PyPI `triton-windows 3.9.0.post30`.
- Confidence: high for compile + correct results on simple elementwise kernels; **unverified** for advanced features (`tl.dot`, TMA, warp specialization, autotuning) on sm_120 under this wrapper, and for official support/debugging help.
- Relevance to vla.cu: removes a second toolchain (WSL2) from the project. If a later kernel hits a Triton-on-Windows bug, WSL2 remains the fallback.

## Triton 3.9 will flip the FP-fusion default in 3.10  (2026-11-18)
- Question: why does every Triton run print `TRITON_DEFAULT_FP_FUSION is unset`?
- Finding: Triton 3.9 defaults to FP fusion ENABLED for backward compatibility and warns that 3.10 will default it to DISABLED. The knob is `TRITON_DEFAULT_FP_FUSION=0|1`.
- Source: warning text emitted by the installed `triton/knobs.py` (v3.9.0) on 2026-11-18.
- Confidence: high (read from the installed package's own warning).
- Relevance to vla.cu: fused FP arithmetic changes rounding, so exact float comparisons against PyTorch can shift. Set the knob explicitly before trusting bit-exact tests.

## CUDA toolkit version to install for this machine  (2026-11-18)
- Question: which toolkit version builds torch extensions correctly here, and why not the newest?
- Finding: torch is built for CUDA 12.9 (`2.8.0+cu129`), so a 12.9 toolkit keeps `nvcc` matching the runtime PyTorch ships; the driver reports CUDA UMD 13.4, and CUDA minor-version compatibility allows a 12.x toolkit on a newer driver, but *not* a toolkit newer than the driver permits. `winget` only offers `Nvidia.CUDA 13.5`, which is newer than the driver's 13.4 - so the archive installer for 12.9 is the safer choice. Archives: https://developer.nvidia.com/cuda-12-9-2-download-archive (12.9.2, May 2026) or https://developer.nvidia.com/cuda-12-9-0-download-archive with `?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local` for the local installer.
- Source: NVIDIA CUDA Toolkit Archive page https://developer.nvidia.com/cuda-toolkit-archive (accessed 2026-11-18); `winget search --id Nvidia.CUDA` returned only 13.5 on 2026-11-18.
- Confidence: high that 12.9 matches torch; medium on the exact driver/toolkit forward-compat boundary (from memory of CUDA compatibility rules, not read from NVIDIA docs this session).
- Relevance to vla.cu: `compute-sanitizer` and `nvcc` only arrive with the toolkit, and a version mismatch with torch's CUDA 12.9 is a classic source of extension-build failures.

## CUDA 12.9.2 Windows installer, verified  (2026-11-18)
- Question: which exact file should be installed, and how can it be checked afterwards?
- Finding: Windows 11, x86_64, local (offline) installer is `cuda_12.9.2_576.57_windows.exe`, 3,562,943,232 bytes (3.3 GB), MD5 `3507abab1105669515f3a59f3142e2f8`. URL: https://developer.download.nvidia.com/compute/cuda/12.9.2/local_installers/cuda_12.9.2_576.57_windows.exe . A HEAD request returned 200 and the official `md5sum.txt` lists the same hash.
- Source: NVIDIA archive page for 12.9 Update 2 with Windows, x86_64, version 11, exe_local selected (accessed 2026-11-18), plus `https://developer.download.nvidia.com/compute/cuda/12.9.2/docs/sidebar/md5sum.txt`.
- Confidence: high, hash and size come from NVIDIA's own checksum file.
- Relevance to vla.cu: the machine has no CUDA toolkit, no `nvcc`, and no `compute-sanitizer`, and `CLAUDE.md` requires the sanitizer for every CUDA kernel. The installer needs one administrator approval (UAC) from the human.

## compute-sanitizer / nvcc ship with the CUDA toolkit, not the driver  (2026-11-18)
- Question: why are `nvcc`, `nsys`, `ncu`, `compute-sanitizer` missing when the GPU and PyTorch CUDA work?
- Finding: the display driver (617.42, CUDA UMD 13.4) only provides the runtime/UMD; the CUDA Toolkit provides nvcc, compute-sanitizer, and Nsight tools. PyTorch's wheels bundle the CUDA runtime libraries, so GPU work runs without a toolkit installed.
- Source: standard NVIDIA driver/toolkit split; verified only by `shutil.which` returning NOT FOUND (no toolkit directory under `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA`).
- Confidence: medium (explanation from memory; the absence itself is measured).
- Relevance to vla.cu: `CLAUDE.md` section 5 requires `compute-sanitizer` on every new CUDA kernel, so a toolkit install is mandatory before CUDA C++ chunks.

## CUDA 12.8+ is required for sm_120 (Blackwell consumer)  (2026-11-18)
- Question: which PyTorch wheel runs on compute capability 12.0?
- Finding: torch 2.8.0+cu129 is installed and reports the device correctly, so cu129 satisfies the requirement. Older cu12x wheels built before 12.8 lack sm_120 kernels and would fall back or fail.
- Source: PyTorch wheel index `https://download.pytorch.org/whl/cu129` used for the install on 2026-11-18; the sm_120/CUDA 12.8 requirement is from memory, unverified against NVIDIA docs.
- Confidence: medium-high (the working install is measured; the exact minimum version is not).
- Relevance to vla.cu: pins future dependency upgrades. Don't downgrade the CUDA wheel.

## How torch.utils.cpp_extension resolves the toolkit, host compiler, and ninja  (2026-11-18)
- Question: which settings does an extension build actually read on this Windows install?
- Finding: torch 2.8.0 exposes `cpp_extension.CUDA_HOME` (already resolved here to `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\v12.9`) and has no `CUDA_VERSION` attribute. `get_cxx_compiler()` returns the bare name `cl` on Windows, so the host compiler is whatever PATH resolves at build time. `is_ninja_available()` probes for a `ninja` executable: False in a plain shell, True after `vcvars64.bat` prepends Visual Studio's bundled ninja.
- Source: local introspection of the installed torch 2.8.0 on 2026-11-18 (`dir(torch.utils.cpp_extension)`, `get_cxx_compiler()`, `is_ninja_available()`, then the same calls behind `vcvars64.bat`). No PyTorch documentation page was read for this.
- Confidence: high for this install and version; medium for other platforms, where `get_cxx_compiler()` returns `c++` and the ninja check differs.
- Relevance to vla.cu: `cuda-check` reads exactly these, and the first real extension build has to run in a shell where `vcvars64.bat` was called until the `-ccbin` path is tested.
