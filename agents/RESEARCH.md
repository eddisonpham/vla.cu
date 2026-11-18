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
