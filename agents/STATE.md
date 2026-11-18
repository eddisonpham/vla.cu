# STATE: current project status

> Overwrite this file at the end of every session and after every chunk. Keep it under ~60 lines.

**Last updated:** 2026-11-18
**Last git commit:** see `git log -1` (scaffold was 9b5dee6, memory update e6b5c07)
**Bootstrap (`agents/STARTER.md`):** COMPLETE 2026-11-18, commit 9b5dee6
**Remote:** `origin` = https://github.com/eddisonpham/vla.cu.git (branch `main`)
**Session summary:** Part A + Chunk 0 done earlier. Then: **Triton now works on native Windows** (EXP-002, DEC-008), README added, repo published to `origin`.

## Done so far
- Skeleton, `.venv` (torch 2.8.0+cu129), `.gitignore`, `.gitattributes`, `requirements.txt`.
- Chunk 0: `scripts/env_check.py`; hardware recorded in `HARDWARE.md`.
- `scripts/triton_probe.py`: masked vector add, bit-exact at n=1000 and n=1_000_003 on sm_120.
- `triton-windows==3.9.0.post30` installed in the venv (bundles its own ptxas).
- `README.md`: project purpose, why batch-1 is memory-bound, status, layout, rules.

## What works
- `python scripts/env_check.py` -> RTX 5060 Laptop, sm_120, 26 SMs, 7.9 GB.
- `python scripts/triton_probe.py` -> `VERDICT: triton works on this GPU`, err 0.0.
- git: commits authored as Eddison Pham, pushed to `origin/main`.

## What's broken or missing
- **Still no CUDA toolkit**: no `nvcc`, `nsys`, `ncu`, `compute-sanitizer`. Blocks CUDA C++ kernels and the mandatory sanitizer step. Recommended: CUDA **12.9** archive installer (matches torch cu129), not winget's 13.5.
- `vla_cu/bench_utils.py` does not exist yet, so nothing can be timed or memory-capped.
- No tests yet (`pytest` reports "no tests ran"); `tests/` is empty.
- Triton advanced features (`tl.dot`, TMA, autotuning) untested on this wrapper.

## Current chunk
- None in progress. Chunks 1-3 remain; Chunks 2-3 are no longer blocked.

## Open questions for the human
- Install the CUDA 12.9 toolkit now (needs their admin click), or defer to when the first CUDA C++ kernel arrives?
- Next chunk: Chunk 1 (`bench_utils.py`) or jump to the 4-bit packing + reference dequantize?

## Suggested next options (see `BACKLOG.md`)
1. Chunk 1: `vla_cu/bench_utils.py` - timing and the 7 GB cap; everything later depends on it.
2. Chunk 2 + 3: first real Triton kernel (vector add) + correctness test + bandwidth benchmark, with a written prediction first.
3. Install CUDA 12.9 toolkit so `compute-sanitizer` and Nsight exist before hand-written CUDA.
