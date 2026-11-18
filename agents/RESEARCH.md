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
