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

## torch.compile and vLLM on this Windows install  (2026-11-22)
- Question: can `torch.compile` and vLLM's PagedAttention be part of this project?
- Finding, torch.compile: installed and reachable (`torch._dynamo.list_backends()` reports `inductor` and `cudagraphs`), but compiling a 1024x1024 matmul fails after 11.0 s with `BackendCompilerFailed`, cause `ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'`. `triton-windows` 3.9.0 and the Triton API that torch 2.8.0's inductor expects do not match, so inductor is unusable in this venv as installed.
- Finding, vLLM: official installation docs list Linux as the only OS and state that vLLM does not support Windows natively, pointing at WSL2 or community forks. Multiple 2026 guides agree.
- Source: local probe with torch 2.8.0+cu129 and triton 3.9.0 on 2026-11-22; https://docs.vllm.ai/en/latest/getting_started/installation/gpu (accessed 2026-11-22).
- Confidence: high for both. The torch.compile failure is measured on this machine; the vLLM statement is from its own docs.
- Relevance to vla.cu: `torch.compile` cannot serve as a baseline here without changing the Triton install or moving to WSL, and vLLM is not an option on native Windows. `torch.cuda.CUDAGraph` is a separate API and is unaffected by the inductor failure; it is untested here.

## SmolVLA layer shapes used by the end-to-end block  (2026-11-23)
- Question: which shapes should stand in for a real layer in `python -m vla_cu layer`?
- Finding: the block uses hidden 960 and intermediate 2560, the widths of the SmolVLM2-500M backbone that SmolVLA builds on. Both are divisible by the group size 64 (DEC-020). This was **not** verified against a checkpoint in this session; no `config.json` was read, and the figure comes from the repo's own assumption plus the project brief of about 450M parameters.
- Source: repo assumption (DEC-020, README target of about 450M parameters), not a lookup. Confidence: low.
- Relevance to vla.cu: every layer-level number in EXP-015 inherits this assumption. Do not present 960/2560 as extracted from SmolVLA until a config is read.

## compute-sanitizer cannot instrument the torch extension on this install  (2026-11-23)
- Question: can the CUDA C++ kernel get the sanitizer pass that `CLAUDE.md` requires?
- Finding: no. compute-sanitizer 2025.2.1 fails the run with `Program hit CUDA_ERROR_INVALID_HANDLE (error 400) due to "invalid resource handle" on CUDA API call to cuKernelGetFunction`, raised from `cudaLaunchKernel` inside `vla_cu_int4_gemv.pyd`, both with the default tool and with `--tool memcheck --target-processes all --print-limit 3`. The kernel still executes and returns values that match the reference; only the instrumentation fails. Triton kernel launches on the same machine are clean.
- Source: local runs on 2026-11-23 (EXP-016), toolkit 12.9.2, driver 617.42. No NVIDIA issue tracker entry was read.
- Confidence: high that it fails here; unknown whether a driver or sanitizer update, or linking the runtime differently, changes it.
- Relevance to vla.cu: the CUDA kernel has no memory-safety pass. Report it as unverified, never as sanitized.

## A torch extension build is invalidated outside the vcvars shell  (2026-11-23)
- Question: why does a previously working extension fail to import in a new shell?
- Finding: with `cl.exe` absent from PATH, `torch.utils.cpp_extension.load` decides to rebuild and then fails in `verify_ninja_availability` with `RuntimeError: Ninja is required`, even though `ninja` and a cached `.pyd` both exist. Inside a shell where `vcvars64.bat` ran, the same call builds once and imports. So the build and every `cuda` task belong in a vcvars shell.
- Source: local reproduction on 2026-11-23 (EXP-016); torch 2.8.0 source path `cpp_extension.py:2248`.
- Confidence: high for this install and version.
- Relevance to vla.cu: README documents the vcvars requirement, and `vla_cu/cuda_int4.py` turns the bare ninja error into a message that names the cause.

## SM_120 execution limits and cache geometry  (2026-11-23)
- Question: what limits a memory-bound kernel on this exact architecture, and what denominators belong in an occupancy or latency argument?
- Finding: compute capability 12.0 allows 48 concurrent warps per SM, 1536 threads, 32 blocks per SM, 100 KB of shared memory per SM and 99 KB per block, with 64K 32-bit registers. A separate microarchitecture study of GB202 (same capability, larger die) measures 89 KB of L1 data cache by default, 128 B lines, 44 cycles for an L1 hit, 354 cycles for L2, about 362 cycles for DRAM, and HFMA2 at 0.5 instructions per cycle, so fp16x2 FMA is not faster than two fp32 FMA.
- Source: https://docs.nvidia.com/cuda/blackwell-tuning-guide (section 1.4.1 Occupancy, accessed 2026-11-23); https://zartbot.github.io/micro_arch/nvidia/sm_120/01_architecture_overview.html and .../02_memory_subsystem.html (accessed 2026-11-23).
- Confidence: high for the occupancy table, it is NVIDIA's own document; medium for the cycle counts, which were measured on a GB202 with 96 MB of L2, not on this GB206-class part with 33.6 MB.
- Relevance to vla.cu: gives the Little's law budget in `OPTIMIZATIONS.md` section 2 (about 58 KB in flight, 2.2 KB per SM) and rules out fp16x2 arithmetic as a dequant speed trick.

## Prior art: how Marlin dequantizes int4 to fp16 in one instruction  (2026-11-23)
- Question: is there a cheaper unpack than shift, mask, convert and multiply per weight?
- Finding: Marlin turns a nibble into an fp16 value with one lop3 (mask the low four bits, OR in an fp16 exponent) and then subtracts a constant with `__hsub2`, so two nibbles in an int32 become two half values in one instruction, with the zero point folded into the subtrahend. Weights are also repacked offline into the order the tensor core fragments want, and global loads use cp.async into a circular shared memory queue. Marlin's README states the approach needs roughly 25 to 50 tensor-core MACs per 4-bit weight to reach its ideal 4x, which is why it targets batch 16 to 32.
- Source: https://arxiv.org/html/2408.11743 (Marlin paper, dequantization section, accessed 2026-11-23); https://github.com/IST-DASLab/marlin (README, techniques); https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/dequant.h (the lop3 template).
- Confidence: high for the technique, medium for exact instruction counts on sm_120, untested here.
- Relevance to vla.cu: candidate H10. At batch 1 it saves instructions in a kernel that is about 4 percent ALU busy, so it is ranked low on purpose. The transferable idea is the repack: choose the layout offline to match the read pattern.

## Triton load and store cache modifiers, and eviction policies  (2026-11-23)
- Question: can a Triton kernel mark one-shot weight loads as streaming and keep the reused activation vector resident, without dropping to CUDA C++?
- Finding: `tl.load` accepts `cache_modifier` limited to "", ".ca", ".cg", ".cv", so there is no ".cs" on loads, plus `eviction_policy` of "", "evict_first" or "evict_last". `tl.store` accepts ".wb", ".cg", ".cs" and ".wt". Triton's own fp4 matmul example loads weights `evict_first` and activations `evict_last`, and xformers pairs them the same way. TMA loads accept neither, because `cp.async.bulk.tensor` moves data from L2 to shared memory and bypasses L1.
- Source: https://triton-lang.org/main/python-api/generated/triton.language.load.html and .../store.html (accessed 2026-11-23); https://github.com/facebookresearch/xformers/blob/main/xformers/ops/_triton/rmsnorm_kernels.py; https://github.com/FlashSampling/FlashSampling/blob/main/findings/tma-cache-modifiers.md
- Confidence: high for the API surface, unverified for `triton-windows` 3.9 on sm_120, which is exactly what EXP-020 would test.
- Relevance to vla.cu: H2 is a one keyword experiment, and it is mutually exclusive with a TMA-based variant.

## SmolVLA architecture facts that change the kernel plan  (2026-11-23)
- Question: what does the real model execute, and is batch 1 the right assumption everywhere?
- Finding: 450M parameters total with about 100M in the action expert, only the first 16 VLM layers run, the expert width is 0.75 times the VLM width, the model predicts chunks of 50 actions, and inference fixes flow matching at 10 steps. The expert is a transformer over the action chunk with interleaved cross attention to the VLM prefix and causal self attention, so its linear layers see 50 tokens, not 1. LeRobot's `SmolVLAConfig` names `chunk_size: 50`, `num_vlm_layers: 16`, `expert_width_multiplier: 0.75` and `self_attn_every_n_layers: 2`.
- Source: https://arxiv.org/html/2506.01844 (SmolVLA paper, implementation details); https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py; https://huggingface.co/blog/smolvla (accessed 2026-11-23).
- Confidence: high for the four config values and the 10 steps; the widths 960 and 2560 remain the repo's assumption, not read from a `config.json` (DEC-024).
- Relevance to vla.cu: rewrites the model-level budget. Derived, 0.586 ms floor for 450M parameters at 4-bit, of which 1.30 ms comes from the ten sequential expert passes, so the expert (M = 50) is the part worth a tensor-core kernel and the VLM prefix is the only true batch 1 part. It also kills the idea of fusing the denoising loop, because the steps are sequential.
