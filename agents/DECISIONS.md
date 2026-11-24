# DECISIONS: why we chose what we chose

> Append-only. To change a decision, add a new entry and mark the old one `Status: superseded by DEC-NNN`.
> Use the template in `agents/README.md`.

---

## DEC-001: KernelEdge is a separate repo from LeRobot Edge  (2026-11-18)
- Context: KernelEdge builds on the user's existing LeRobot Edge project (quantized SmolVLA inference).
- Options considered: (a) extend LeRobot Edge in place; (b) new standalone repo that exposes PyTorch custom ops, with one small backend added to LeRobot Edge later.
- Chosen and why: (b). Different story (writing kernels vs. using existing tools), different build tooling (nvcc/compiled extensions would complicate a pure-Python repo and its CI, which has no GPU runner), and it can stand alone.
- Revisit if: the integration hook back into LeRobot Edge turns out to need heavy changes there.
- Status: active

## DEC-002: Learning-first workflow with small chunks  (2026-11-18)
- Context: the goal is for the human to understand every line, not to ship a library.
- Options considered: agent builds everything autonomously; human writes everything; agent delivers small documented chunks and the human controls each step.
- Chosen and why: small documented chunks with predict-first prompts, review-not-rewrite when the human writes code, and a persistent memory in `agents/`. See `CLAUDE.md` sections 3 and 5.
- Revisit if: the human asks for a different mode (e.g. "I'll write this one myself" for a specific kernel).
- Status: active

## DEC-003: GPU memory budget of 7 GB, enforced in code  (2026-11-18)
- Context: the user's project constraint is that everything fits in a 7 GB GPU.
- Chosen and why: `enforce_memory_budget()` in `kerneledge/bench_utils.py` caps the PyTorch allocator; every GPU script must call it first.
- Revisit if: the human changes the constraint.
- Status: active

## DEC-004: Project renamed from KernelEdge to vla.cu  (2026-11-18)
- Context: the original name was a placeholder; the human wanted a short, technical name in the style of vLLM or llama.cpp.
- Options considered: vVLA, vPolicy, vla.cu, q4VLA (earlier pun-style candidates Nibble, Popcorn, Weft, and Starved were set aside).
- Chosen and why: vla.cu. It follows the llama.cpp naming style, with .cu for CUDA kernels and VLA for the robot policies it targets.
- Naming rules: the project and repo are "vla.cu"; the Python package, imports, and CLI use "vla_cu" (a dot is not valid in a Python module name), e.g. `python -m vla_cu`, `vla_cu/bench_utils.py`.
- History note: this log, `EXPERIMENTS.md`, and `RESEARCH.md` are append-only and were not rewritten. "KernelEdge" and "kerneledge" in older entries mean this project (e.g. DEC-003's `kerneledge/bench_utils.py` is now `vla_cu/bench_utils.py`).
- Revisit if: the human wants a different name.
- Status: active

## DEC-005: The agent runs git; commits belong to the human  (2026-11-18)
- Context: the human granted permission to run git (commit, push, pull, etc.) but does not want Freebuff/Codebuff named anywhere in the history.
- Options considered: (a) agent prints commands, human runs them; (b) agent runs git itself and authors commits as the human.
- Chosen and why: (b). `CLAUDE.md` section 6 documents it: use the repo's configured `user.name`/`user.email` (Eddison Pham), never add `Co-Authored-By` or "Generated with ..." footers, one commit per chunk after its checks pass, imperative `<area>: <what changed>` subjects.
- Revisit if: the human wants to review commands before they run, or wants a separate bot identity.
- Status: active

## DEC-006: Project venv with torch 2.8.0+cu129, not the conda base  (2026-11-18)
- Context: the machine's conda Python 3.13 already had torch 2.8.0+cu129; STARTER.md Part A asks for a venv.
- Options considered: (a) reuse conda base; (b) fresh `.venv` with the same wheel; (c) venv with `--system-site-packages`.
- Chosen and why: (b). Isolates the project from unrelated conda packages, keeps the installed set recorded in `requirements.txt`/`HARDWARE.md`, and (c) makes imports depend on invisible global state while teaching nothing.
- Revisit if: CUDA toolkit or driver changes force a different wheel (sm_120 needs CUDA 12.8+).
- Status: active

## DEC-007: STARTER.md archived into `agents/`, bootstrap marked complete  (2026-11-18)
- Context: the one-time setup instructions should not live in the repo root or be re-read every session.
- Options considered: (a) delete it; (b) leave it at the root; (c) move it to `agents/STARTER.md` and mark bootstrap complete in `STATE.md`.
- Chosen and why: (c), at the human's request. History stays readable, the root stays clean, and `CLAUDE.md` now points at `agents/STARTER.md` only while bootstrap is incomplete.
- Revisit if: never expected.
- Status: active

## DEC-008: Triton on native Windows via `triton-windows`, not WSL2  (2026-11-18)
- Context: `triton` has no Windows wheel, so the Triton chunks needed either a community build or a second Linux toolchain.
- Options considered: (a) WSL2 + official Triton; (b) `triton-windows` in the existing venv; (c) skip Triton, write CUDA C++ only.
- Chosen and why: (b), because it was measured working on this exact GPU (EXP-002: bit-exact results on sm_120, bundled `ptxas`, same `import triton` API). It keeps one toolchain, one venv, one set of paths - (a) would double every environment problem, and (c) skips the fastest way to learn the block/mask mental model.
- Cost accepted: it is a community wrapper, so a bug in it is ours to debug, and advanced features are untested. WSL2 stays the documented fallback.
- Revisit if: a kernel fails inside the wrapper in a way that official Triton on Linux does not reproduce; or torch is upgraded to a version requiring a different Triton ABI.
- Status: active

## DEC-009: Publish the repo to the existing `origin`  (2026-11-18)
- Context: the human asked to publish `main` and to keep committing and pushing under their own name.
- Options considered: (a) leave it local; (b) push to the already-configured `origin`; (c) create a new remote with `gh`.
- Chosen and why: (b). `origin` already points at `https://github.com/eddisonpham/vla.cu.git`, and no `gh` CLI is installed, so the existing remote is the only path. Commits stay authored by "Eddison Pham"; no attribution footers are added.
- Revisit if: the human wants a different host or visibility setting.
- Status: active

## DEC-010: No comments in code; short docstrings; explanation in `docs/`  (2026-11-18)
- Context: the human asked for visually clean code - no `#` or `//` comments anywhere in source, with explanation carried by docstrings and `docs/` instead. This overrides the earlier CLAUDE.md rule that comments should explain *why*.
- Options considered: (a) keep why-comments; (b) no comments, long docstrings; (c) no comments, short docstrings, deep material in `docs/`.
- Chosen and why: (c), at the human's request and made an invariant in `CLAUDE.md` section 4. Short docstrings keep the file readable at a glance; `docs/ARCHITECTURE.md` holds the diagrams and tables that would otherwise bloat a function header. Kernel mapping, access pattern and bottleneck stay in the kernel docstring because they belong to that one function.
- Consequences: `scripts/env_check.py` and `scripts/triton_probe.py` were rewritten comment-free; config files (`.gitignore`, `requirements.txt`, `.gitattributes`) keep their short notes.
- Revisit if: the human wants comments back for a specific teaching moment.
- Status: active

## DEC-011: One short command handle per action; docs are tables and diagrams  (2026-11-18)
- Context: commands were long (`.venv/Scripts/python.exe scripts/...`) and documentation was prose-heavy.
- Options considered: (a) paste full commands in docs; (b) shell aliases per machine; (c) `python -m vla_cu <task>` dispatcher in the repo.
- Chosen and why: (c). It matches the existing CLAUDE.md constraint that everything runs as `python -m vla_cu ...`, works identically on Windows and Linux, and is versioned with the code. `docs/ARCHITECTURE.md` uses tables and Mermaid diagrams rather than paragraphs.
- Revisit if: the number of tasks grows enough to need argument parsing (then `argparse` with subcommands, same names).
- Status: active

## DEC-012: Documentation style invariant  (2026-11-18)
- Context: the human asked for clean, human-written looking documentation: no em dashes, no emojis, no generated filler. Made an invariant in `CLAUDE.md` section 4.
- Options considered: (a) leave style to judgement; (b) encode explicit rules in `CLAUDE.md` and sweep existing files.
- Chosen and why: (b). Style rules that live only in chat are lost next session; rules in `CLAUDE.md` survive, and the sweep keeps the repo consistent so new text has a pattern to match.
- Consequences: `README.md` and `docs/ARCHITECTURE.md` rewritten by hand; every tracked file swept for em dashes, en dashes, curly quotes and smart apostrophes. Hyphens in compound words stay.
- Revisit if: the human wants a different voice, or a linter enforces this automatically.
- Status: active

## DEC-013: CUDA 12.9.2 toolkit from the archive installer, not winget's 13.5  (2026-11-18)
- Context: `CLAUDE.md` requires `compute-sanitizer` on every CUDA kernel, and no toolkit was installed. torch is built for CUDA 12.9, while the driver reports CUDA UMD 13.4.
- Options considered: (a) `winget install Nvidia.CUDA` (13.5); (b) the 12.9.2 archive local installer with Custom install, driver component unchecked.
- Chosen and why: (b). It keeps `nvcc` on the same major.minor as torch's cu129 runtime, stays at or below the driver's 13.4, and the exact file could be verified against NVIDIA's own MD5 before running it. Installing the bundled 576.57 driver over the working 617.42 was explicitly avoided.
- Consequences: 12.9.2 is in place and `compute-sanitizer` is usable. The installer did not include `nsys` or `ncu`, so profiling tools remain a separate download. The only MSVC on the machine is 14.50, which nvcc 12.9 rejects, so CUDA C++ builds are still blocked.
- Revisit if: torch is upgraded to a cu13 wheel, or the host compiler situation is resolved with a different toolchain.
- Status: active

## DEC-014: Add VS 2022 Build Tools rather than force nvcc past its version check  (2026-11-18)
- Context: `nvcc` 12.9 refused the only installed MSVC (14.50 from VS 18 BuildTools) with `unsupported Microsoft Visual Studio version`.
- Options considered: (a) `-allow-unsupported-compiler`; (b) install VS 2022 Build Tools next to VS 18; (c) move to a CUDA 13.x toolkit; (d) stay Triton-only.
- Chosen and why: (b). It gives a toolset NVIDIA supports on 12.9, keeps `nvcc` on the same CUDA as torch's cu129 wheel, and does not touch the driver. (a) risks exactly the silent miscompiles this project is learning to detect, (c) breaks the nvcc/torch match, and (d) removes the comparison the repo exists for.
- Consequences: two Visual Studio installs share the machine, so a build must select 14.44 explicitly, normally through `vcvars64.bat`. A few GB of disk.
- Revisit if: torch moves to a cu13 wheel, or `torch.utils.cpp_extension` cannot be pointed at the 2022 toolset.
- Status: active

## DEC-015: `cuda-check` inspects build settings and never compiles  (2026-11-18)
- Context: with the toolkit and two Visual Studio installs present, whether a CUDA C++ build works still depended on invisible shell state (PATH, INCLUDE, LIB), and the answer changes between a plain shell and a vcvars shell.
- Options considered: (a) document the vcvars recipe in prose; (b) a task that runs a throwaway compile; (c) a task that reads the same sources `torch.utils.cpp_extension` reads, prints them, and adds one verdict line.
- Chosen and why: (c). It answers the question in one short command in well under a second, cannot half-fail in a confusing way, and does not duplicate the first extension build, which is the step that should teach what a build is. It runs the real `nvcc` only as `--version`, so the reported toolkit is the one torch resolves, not a hardcoded path.
- Cost accepted: the verdict means "prerequisites present", not "a build succeeded". The docstring and report header say so, and EXP-005 records it as configuration only.
- Revisit if: the loader starts honoring extra variables (a pinned `CC`, `TORCH_CUDA_ARCH_LIST`), or a build fails while the task says ready.
- Status: active

## DEC-016: Precision scope is fp32, bf16, int8, int4  (2026-11-18)
- Context: the human proposed a ladder of "BF32, 16, int8, 4" to bound how far quantization is generalized, instead of leaving the bit width open.
- Naming correction: `bf32` is not a storage format. The two dense rungs are fp32 (4 B, exact) and bf16 (2 B, what plain torch runs). BF32 as an input name maps to fp32. TF32 is a compute mode inside tensor cores, not a way to store weights, so it is not a rung.
- Options considered: (a) support every bit width, including 3, 5 and 6; (b) pick the four that matter and design the unpack so a width is a parameter; (c) fix 4-bit only, hardcode the shift.
- Chosen and why: (b). fp32 and bf16 need no kernel at all, since torch ships them, so they cost nothing and give every result a reference and a baseline. int8 is one value per byte, so it isolates scale handling from bit packing and is the cheap second format that proves the abstraction. int4 is the project target. A width that is a power of two is one shift and one mask, so the ladder is a parameter, not a redesign. (a) buys formats whose packing straddles byte boundaries and whose implementations are hundreds of lines of lookup tables, which teaches little here; (c) makes int8 and int2 a rewrite later.
- Consequences: the unpacking lives in one named function taking the width as a compile-time constant, decided now and used from the first kernel. `docs/ARCHITECTURE.md` section 5 holds the ladder with per-pass traffic. 3, 5 and 6-bit bitstreams and codebook formats (NF4, FP4) are out of scope and recorded as such rather than silently ignored.
- Revisit if: the human wants a codebook format, in which case it is an add-on to the int4 path, or wants non-power-of-two widths after the fused kernel works.
- Status: active

## DEC-017: Benchmark method is preallocated buffers, CUDA events, three size regimes  (2026-11-22)
- Context: the first benchmark had to compare a Triton kernel against torch without the measurement itself becoming the result.
- Options considered: (a) time the public wrappers that allocate their output; (b) time both paths writing into one preallocated buffer; (c) wall clock timing.
- Chosen and why: (b), with 25 warmup calls, 100 timed iterations and CUDA events. Allocation inside the loop would measure the caching allocator, and wall clock would measure launches and queue depth. Byte counts are a fixed 12 per element, identical for both paths, so GB/s is comparable across kernels and sizes.
- Consequences: `vector_add_into` exists next to `vector_add` only to keep the comparison allocation-free, and the benchmark prints three sizes on purpose: launch-bound at 4K, cache-resident at 1M, memory-bound at 16M. The cache row is reported and labelled instead of being hidden, because a 251 percent of peak reading is the clearest evidence that a benchmark is measuring the wrong thing.
- Revisit if: kernels start allocating internal workspace, which must then be counted as traffic, or if a size is found where the three regimes disagree with these labels.
- Status: active

## DEC-018: CUDA graphs belong in the batch-1 design, tests stay plain  (2026-11-22)
- Context: EXP-009 measured 16.71 us per launch from Python against 0.63 us per graph node for the same 1-element kernel, a 26 times difference that is submission cost, not kernel cost.
- Options considered: (a) ignore launch cost and optimize kernels only; (b) capture everywhere, tests included; (c) keep plain launches for correctness, capture the benchmark and the inference path.
- Chosen and why: (c). A graph hides errors until replay, which makes a failing test point at the wrong line, so correctness runs stay plain. The saving also scales with chain length: a 450M parameter policy at batch 1 is hundreds of small kernels, where 16 us each is milliseconds of the forward pass.
- Consequences: every benchmark of a multi-kernel chain reports a plain column and a graphed column, and any kernel on the speed path must be capture safe, meaning no host synchronization and no allocation inside capture.
- Revisit if: a kernel needs a host-side value between launches in a way that breaks capture, or the real model turns out to be a few large kernels rather than many small ones.
- Status: active

## DEC-019: GEMV is one program per output row, scale applied after the reduction  (2026-11-22)
- Context: the first project kernel fixed a mapping that later quantized kernels will inherit.
- Options considered: (a) one program per output row walking K in BLOCK_K steps; (b) split K across programs and combine with atomics; (c) one program per output element with no reduction.
- Chosen and why: (a). A row is contiguous, so its loads coalesce; the reduction stays inside one program, so there are no atomics and no second pass; and the per-channel scale is constant along the reduction, so it multiplies the finished sum once instead of every product. (c) would re-read the whole activation vector per element. (b) is the fix for shapes where N is small relative to the SM count, which is not the case yet.
- Consequences: the grid is N, so a layer with fewer rows than SMs underuses the GPU, a known limit to revisit with real SmolVLA shapes. Every program re-reads x, which is free only while K floats stay in cache.
- Revisit if: a target layer has N below about 64, or K grows until x stops fitting in L2.
- Status: active

## DEC-020: The 4-bit format is low nibble even, zero point 8, one scale per 64 weights  (2026-11-22)
- Context: the packing had to be pinned once, in the form the kernel reads, before any kernel could be written against it.
- Options considered: (a) low nibble for element 2i with symmetric (q - 8) * scale and scale = amax / 7; (b) high nibble first; (c) unsigned codes with a separate zero point per group; (d) a codebook format.
- Chosen and why: (a). Symmetric codes need no zero-point tensor, amax / 7 keeps every weight inside [-8, 7] with no clipping, and one scale per 64 weights makes the scale table 1/16 of the weight count. (d) is out of scope by DEC-016.
- Consequences: K must divide by 64, which 960 and 2560 do, both nibbles of a byte always share one scale while the group is even, and an all-zero group stores scale 1 with codes 8 so it dequantizes to exactly zero. The kernel gathers the scale per byte; that gather is the next thing to hoist (EXP-013 shows it costs bandwidth).
- Revisit if: a real layer has K not divisible by 64, or accuracy needs groups of 32 or 16 with a larger scale table.
- Status: active

## DEC-021: The int4 reduction is group-major, one scale multiply per group  (2026-11-23)
- Context: the first fused kernel multiplied every element by its group's scale, which meant a scale load per byte even though 32 bytes share a value. int4 sat at 75 percent of peak while int8 sat at 93.
- Options considered: (a) keep the per-element scale and hope the L1 cache absorbs the gather; (b) reduce inside each group and multiply the group sum once; (c) pre-expand the scales into a per-byte table during packing.
- Chosen and why: (b). It removes 32 times the scale loads and multiplies, and it shrinks the accumulator by the group size, which is what makes a wide row tile affordable. (c) would trade DRAM traffic for bandwidth and break the format's compactness.
- Consequences: the kernel tile is 3D, `[ROWS, groups, GROUP // 2]`, which is harder to read than a flat tile but mirrors the format. Block size must be a multiple of the group size, already enforced. Measured effect: EXP-014, 0.263 to 0.230 ms.
- Revisit if: a group size smaller than 32 makes the group tile too narrow to fill a warp.
- Status: active

## DEC-022: The model-facing entry point is a torch custom op, not a bare launch  (2026-11-23)
- Context: the benchmarks call the wrappers directly, but anything upstream (compile, tracing, an nn.Module) needs an operator torch can see.
- Options considered: (a) call the Triton wrapper from model code; (b) register with `torch.library.custom_op` and provide a fake implementation; (c) write a full `autograd.Function`.
- Chosen and why: (b). Inference at batch 1 needs no backward, the fake implementation gives correct shapes under tracing, and `opcheck` validates the schema for free. (c) would be dead code.
- Consequences: `vla_cu/ops.py` must be imported before use, and the ops inherit the kernels' Python-level validation cost. EXP-015 shows that cost dominates a small layer, so the next step for small shapes is a graph, not a faster kernel.
- Revisit if: a real forward pass needs autograd, or the Python validation shows up in a profile.
- Status: active

## DEC-023: The CUDA port stays a comparison, not the default path  (2026-11-23)
- Context: the roadmap had a CUDA C++ port of the int4 GEMV. It exists now, and it needs a vcvars shell to build, which the Triton path does not.
- Options considered: (a) make the CUDA kernel the default; (b) keep both and compare on identical shapes; (c) drop the port.
- Chosen and why: (b). At the DRAM-bound shape the port is 4 percent slower and inside the noise, so there is no reason to prefer it, while it is 2.5x faster on tiny shapes purely through dispatch cost, which is a launch-overhead fact rather than a kernel fact. Keeping both is the point of the project.
- Consequences: two implementations to keep correct, and the CUDA one carries an unresolved sanitizer result (EXP-016) until the tooling problem is understood.
- Revisit if: profiling shows the Triton kernel leaving performance on the table, or the sanitizer issue is fixed.
- Status: active

## DEC-024: SmolVLA widths stay assumptions until a checkpoint config is read  (2026-11-23)
- Context: `bench/layer.py` treats hidden 960 and intermediate 2560 as "the SmolVLA MLP". The paper and the LeRobot config confirm the parameter count, the 16 VLM layers, the 0.75 expert ratio, the chunk of 50 and the 10 sampler steps, but no `config.json` has been read in this repo.
- Options considered: (a) keep assuming and say so; (b) download the base checkpoint's config and index only, no weights; (c) wait for a full checkpoint before benchmarking a block.
- Chosen and why: (a) for now, with the assumption stated in the docstring, `agents/RESEARCH.md` and this entry. The widths only set a benchmark's constants, while the verified ratio facts already carry the model-level arithmetic in `OPTIMIZATIONS.md`. (b) is small and is queued as a follow-up.
- Consequence: every layer-level number in EXP-015 and every derived model number inherits 960 and 2560. Expert hidden 720 and intermediate 1920 are derived from the verified 0.75 ratio, not read from a file.
- Revisit if: `lerobot/smolvla_base` config and index are downloaded, or the benchmark moves onto a real checkpoint.
- Status: superseded by DEC-043
- CORRECTION 2026-11-23: the two configs were read, so 960 and 2560 are no longer assumptions. See DEC-043 and `agents/RESEARCH.md`.
## DEC-025: the MLP block runs as a CUDA graph, not through custom ops  (2026-11-23)
- Context: EXP-015 left the fused block slower than fp32. EXP-018 split the cost into three paths on the same 19.7 MB block.
- Options considered: (a) keep calling the custom ops from model code; (b) call the kernel wrappers directly with preallocated buffers; (c) capture those launches in a `torch.cuda.CUDAGraph` and replay.
- Chosen and why: (c) for the benchmark and the inference path, (b) whenever a graph is not convenient, and (a) never on a hot path. Measured: ops 0.109 to 0.224 ms, raw 0.053, graph 0.018 to 0.019, all with cosine 0.99346. The custom op layer costs more than the two kernels it launches.
- Consequence: custom ops remain the honest way to expose the kernels to torch, and they stay for tracing and for tests. The speed path bypasses them.
- Revisit if: the op layer gets cheaper, or an integration needs one callable entry point more than it needs 2.8x.
- Status: active

## DEC-026: a layer's projections ship as one grouped launch  (2026-11-23)
- Context: EXP-024 measured 2.73x from concatenating three projections into one launch at 960 x 960, and 1.03x, meaning no loss, at 56.7 MB.
- Options considered: (a) one launch per projection; (b) one grouped launch with a segment table; (c) a fully fused layer kernel.
- Chosen and why: (b). It is bit identical, it needs one int32 table per row built at load time, and it keeps the weight layout unchanged, so nothing about packing or accuracy moves. (c) is EXP-025, dropped because a graph already removes the dispatch it would target.
- Consequence: every packed weight set gains a companion segment table and an activation base table. Packing becomes a per-layer operation, not a per-matrix one.
- Revisit if: a projection needs a different K, which the current kernel cannot express because K is per launch.
- Status: active

## DEC-027: `__ldcs` on the CUDA path, no claim on Triton eviction hints  (2026-11-23)
- Context: EXP-020 and EXP-022 tested the same idea, keep one-shot weights out of the way of the reused activation vector, in both languages.
- Options considered: (a) adopt both; (b) adopt neither; (c) adopt the one with a consistent measured sign.
- Chosen and why: (c). The CUDA streaming variant was fastest in both runs and 10 and 9 percent ahead of Triton; the Triton eviction policy flipped sign between two samples. The Triton hints stay enabled because they express the intent and cannot hurt, but no gain is claimed for them.
- Consequence: the fastest int4 kernel in the repo is now the CUDA streaming variant, 0.198 to 0.220 ms at 8192 x 16384, at the cost of needing a vcvars shell to build.
- Revisit if: a fixed-clock harness makes the Triton hint effect measurable, or Triton exposes `.cs` on loads.
- Status: active

## DEC-028: the action expert is a GEMM problem, the VLM prefix is the GEMV problem  (2026-11-23)
- Context: EXP-023. SmolVLA predicts 50 actions per forward and the expert is a transformer over those 50 tokens, so its linear layers have M = 50. A per-row GEMV re-reads every weight 50 times.
- Options considered: (a) keep the GEMV and loop over tokens; (b) one GEMM per matrix at M = 50; (c) an int4 weight-only GEMM that dequantizes into tensor core operands.
- Chosen and why: (b) immediately, as the baseline to beat, and (c) as the next kernel worth writing. Measured: 50 GEMV calls cost 2.588 ms against 0.018 ms for one bf16 GEMM on the same computation, byte ratio 15.5x.
- Consequence: the repo's framing needs one sentence of care. Batch 1 is true for the VLM prefix, which runs once per observation, and false for the expert, which runs 10 times per action at M = 50. The grouped launch and the whole-model budget both follow from this.
- Revisit if: a real checkpoint run shows the expert to be a small share of latency after all, for instance because the prefix dominates.
- Status: active

## DEC-029: measurement protocol, best of N in one process  (2026-11-23)
- Context: EXP-017 and EXP-018 measured the spread directly. One configuration of the read probe returned 164 to 356 GB/s in five interleaved repeats, and the first path timed in a process is systematically slow. Every table in this repo before today is one sample.
- Options considered: (a) keep single samples and note the caveat; (b) repeat inside a process and report best, median and spread; (c) require fixed clocks via `nvidia-smi -lgc` before any number is quoted.
- Chosen and why: (b) now, because it needs no administrator rights, and (c) when the human can grant it. `warm_gpu()` was added to `bench_utils` and is called at the start of every benchmark, which fixed the first-path penalty but not the general instability.
- Consequence: a difference under about 6 percent is not a result on this machine, comparisons are only valid inside one table, and a headline number is quoted as a range from repeated runs with the run count stated. Numbers measured before 2026-11-23 are samples too.
- Revisit if: clocks are pinned, or the machine stops being a power-limited laptop.
- Status: active

## DEC-030: group 64 does not fit the action expert, decide before packing real weights  (2026-11-23)
- Context: EXP-023 found the expert hidden width is 720, and 720 = 11 x 64 + 16, so group 64 cannot divide K.
- Options considered: (a) pad the width to 768 and keep group 64; (b) fall back to group 16, which divides 720 but puts a scale on every 16 weights; (c) allow a ragged last group in the kernel.
- Chosen and why: nothing chosen yet. (a) wastes 6.7 percent of the weights and keeps the fast kernel, (b) costs 25 percent scale overhead, (c) complicates the inner loop for one shape. The decision needs the real checkpoint and a sensitivity measurement.
- Consequence: `bench/m50.py` uses group 16 for the 720 shapes and says so in its output. No other benchmark is affected, since 960, 2560, 1920 and 768 all divide by 64.
- Revisit if: the checkpoint shows a different expert width, or the rotation in EXP-026 makes a coarser group affordable.
- Status: active
## DEC-031: extension kernels launch on the current torch stream, always  (2026-11-23)
- Context: EXP-027. Both CUDA kernels used `<<<blocks, threads>>>`, the legacy default stream. A CUDA graph recorded the silu kernel and silently skipped every extension launch, so the replay produced a stale intermediate and a cosine of 0.83733 where the eager chain gave 0.99468.
- Options considered: (a) legacy default stream, the form in every CUDA sample; (b) the current torch stream via `at::cuda::getCurrentCUDAStream()`; (c) a stream passed in as an argument by the caller.
- Chosen and why: (b). It is one argument in the launch configuration, it makes the kernel capture safe, and it also stops the kernel from implicitly synchronizing with torch's own streams, which the legacy stream does. (c) pushes a detail onto every caller for no benefit yet.
- Consequence: every kernel on the speed path is capturable, which DEC-018 already required and which was quietly false for the CUDA port until now. The rule is stated in both `.cu` headers so the next kernel inherits it.
- Verification: the fast path reports `torch.equal` True between a four-block graph and the same chain launched eagerly. That equality is the regression test for this rule.
- Revisit if: a kernel needs to run on its own stream, in which case (c) comes back with a reason.
- Status: active

## DEC-032: the fast path is the default way to run a block  (2026-11-23)
- Context: EXP-028. Native streaming kernels plus a graph give 0.015 ms per block against 0.069 for fp32, 0.033 for the same kernels launched eagerly, and 0.081 for Triton.
- Options considered: (a) Triton raw launches, the current teaching default; (b) native kernels, eager; (c) native kernels in a graph.
- Chosen and why: (c) as the default, with (b) as the fallback when a graph is inconvenient, for instance in a test where an error must point at a line. Measured, on one block: 0.015 (c), 0.033 (b), 0.081 (a), 0.069 fp32.
- Consequence: the repo now needs a vcvars shell for its fastest path, and Triton stays as the readable reference and the place where a new idea is tried first.
- Revisit if: Triton's launcher gets cheap enough that the gap closes, or a model integration cannot capture.
- Status: active
## DEC-033: the GEMM step width is set by shared memory, not by taste  (2026-11-23)
- Context: EXP-029. A 512-wide step at 3 stages asked for 224 KB of shared memory against a 101 KB limit and the kernel refused to launch.
- Options considered: (a) smaller tiles, (b) one stage, (c) a smaller BLOCK_N only, (d) keep 512 and drop to 2 stages.
- Chosen and why: BLOCK_K = 64 with two stages, which is 56 KB at a 64-row tile and leaves room for a 128-row tile later. Two stages still software-pipelines one step ahead, which is what a memory-bound kernel needs; a 512-wide step at one stage would have no overlap at all.
- Consequence: K is walked in 64-element steps, so a 720-wide expert pays 12 steps, the last one masked. That is the cost of fitting, and it is under a percent.
- Revisit if: BLOCK_N drops to 32, at which point a 128-wide step fits at two stages.
- Status: active

## DEC-034: rotation stays opt-in, not in the default fast path  (2026-11-23)
- Context: EXP-030. Inside the fast path the rotation costs 0.005 ms per block and moves cosine through four blocks from 0.97640 to 0.97449 on Gaussian weights.
- Options considered: (a) rotate everything, (b) rotate nothing, (c) keep it as an opt-in path and decide on real weights.
- Chosen and why: (c). EXP-026 measured a real gain on weights with outliers, and EXP-030 measured no gain and a real cost on the weights the benchmark actually builds. Both are true, so the answer depends on a distribution this repo has not loaded yet.
- Consequence: the fast path's default is unrotated, and `stack rot` stays in the table so a checkpoint result can be read against it in one command.
- Revisit if: a real SmolVLA layer stack shows the EXP-026 gain, in which case the transform wants its own fused kernel rather than two matmuls per block.
- Status: active

## DEC-035: paired alternation replaces pinned clocks  (2026-11-23)
- Context: EXP-032. `nvidia-smi -lgc` is refused without administrator rights, and the GPU idles at 870 MHz of a 3090 MHz ceiling, which is why the same table moved 2.72x and 6.57x in consecutive runs.
- Options considered: (a) elevated shell and pin, (b) longer warmup, (c) alternate the two calls in pairs and report the ratio with its spread, (d) report minima instead of means.
- Chosen and why: (c), with (a) recorded as the thing to do if administrator rights appear. A pair shares one clock state, so common drift cancels in the ratio; the spread across five pairs is the error bar the table was missing. Longest warmup alone does not help, because drift happens during measurement, not before it.
- Consequence: every head-to-head number now carries a range, and absolute milliseconds are treated as comparable within a run only. The 6 percent floor from DEC-029 stays.
- Revisit if: clocks get pinned, after which medians of many runs are enough and the ranges should tighten to a few percent.
- Status: active

## DEC-036: validate at the wrapper boundary, skip when the toolchain is absent  (2026-11-23)
- Context: EXP-033. Three guards were missing or wrong: `hadamard` silently returned the next power of two, the grouped wrapper launched on any shapes, and the cuda tests failed outside vcvars because they checked `nvcc` but not the host compiler.
- Options considered: (a) document the assumptions and leave the code, (b) check every input on the host in Python, (c) check only what would corrupt memory or launch the wrong kernel.
- Chosen and why: (c), matching the other kernels. dtype, shape, contiguity, device and the power-of-two tile are checked before the launch. The segment-run invariant stays documented because proving it needs a device sync on every call.
- Consequence: no silent out-of-bounds read from a grouped launch, and `pytest -q` is green in the default shell with the cuda tests skipped rather than failed. Kernel arithmetic is untouched.
- Revisit if: the grouped kernel becomes the default path and its per-call Python checks show up in a profile.
- Status: active

## DEC-037: one shared validator describes each format once  (2026-11-23)
- Context: EXP-034. Six wrappers repeated the same shape, dtype, device and contiguity checks, and four checks were missing after EXP-033.
- Options considered: (a) leave the checks per wrapper; (b) one module of format checks called by every wrapper; (c) validate in C++ only.
- Chosen and why: (b). It gives one description of a valid call, it keeps every rejection at the Python boundary as ValueError, and it makes the Triton and CUDA implementations of a format agree. (c) leaves the Triton path uncovered.
- Cost accepted: net +82 lines, because the module documents each format once. The six wrappers are 62 lines shorter. The win is consistency and four new checks, not size.
- Revisit if: a wrapper needs a check that does not belong to the format, such as a block size or an alignment. Those stay in that wrapper.
- Status: active

## DEC-038: attention projections default to int8, int4 stays the byte floor  (2026-11-23)
- Context: EXP-035. At 960 width, int8 attention reads 1.78x the bytes and costs 1.54x the graphed time (0.019 against 0.012 ms), while cosine moves from 0.98804 to 0.99994.
- Options considered: (a) int4 only; (b) int8 only; (c) both, int8 for attention and int4 wherever the accuracy budget allows.
- Chosen and why: (c). Seven microseconds buys two and a half digits of cosine. int4 stays the byte floor and the interesting kernel, int8 becomes the quality path.
- Revisit if: a stacked block shows int4 attention holding the 0.98 gate on real weights, or the 7 us matters at model level.
- Status: active

## DEC-039: the action block runs int8 attention and int8 down  (2026-11-23)
- Context: EXP-036. Over four blocks, int4 attention with int4 down reads 0.90671, int8 attention with int4 down 0.97606, int8 everywhere 0.99951. Graphed stack times are 0.100, 0.123 and 0.140 ms.
- Options considered: (a) int4 everywhere, the byte floor; (b) int8 attention with int4 down, the middle rung; (c) int8 everywhere, the quality path.
- Chosen and why: (c) as the default and (b) as the middle rung, because the 0.98 gate is the constraint and only (c) clears it through four blocks. (a) stays for its bytes and for the kernel comparison.
- Cost accepted: 34.5 MB per four blocks against 23.7 for int4, and 1.4x the time of int4. It is still 3.94x faster than fp32.
- Revisit if: a real checkpoint shows int4 down holding the gate, or the gate moves.
- Status: active

## DEC-040: the weight bundle is torch.save plus a JSON manifest, loaded to CPU  (2026-11-23)
- Context: EXP-038. Packed weights need a file format before a real checkpoint can be converted, and no serialization dependency exists in the repo.
- Options considered: (a) torch.save of a bare dict; (b) torch.save with a versioned manifest and per-matrix shapes, read with weights_only=True; (c) safetensors.
- Chosen and why: (b). It adds no dependency, it refuses a future version instead of decoding it, it cannot execute code on load, and the shape declaration is checked on both sides. (c) would be the right pick for a public release with untrusted files, and it is one adapter away.
- Consequence: load returns CPU tensors on purpose, so a caller moves them to a device before a launch. The bundle is 5.93 MB per int4 block against 34.4 MB of fp32 weights.
- Revisit if: untrusted bundles are ever accepted, or a host tool needs random access without loading the whole file.
- Status: active

## DEC-041: the style invariant is a script, and CI checks it without a GPU  (2026-11-23)
- Context: EXP-039. Two rules in CLAUDE.md were enforced by memory, and the CUDA sources had drifted to 47 comment lines across two files.
- Options considered: (a) keep the rules in prose; (b) a script that scans git-tracked files, wired as `style` and in CI; (c) an off-the-shelf linter.
- Chosen and why: (b). It needs no dependency, it is the same command locally and in CI, and it states the no-comments rule that no general linter carries. (c) would add a tool for one rule.
- Cost accepted: CI cannot run the GPU tests, so the workflow checks style and byte-compiles only. GPU correctness stays a manual gate on this machine.
- Revisit if: a GPU runner exists, or a linter is adopted for type and import checks.
- Status: superseded by DEC-042

## DEC-042: the CI workflow is removed at the human's request  (2026-11-23)
- Context: EXP-039 added `.github/workflows/ci.yml` for a style gate and a byte-compile. The human asked for the CI checker to be removed.
- Options considered: (a) keep the workflow; (b) delete it and keep the local `style` task; (c) delete the style task too.
- Chosen and why: (b). The workflow is gone, and `python -m vla_cu style` still checks the invariants locally. The task costs nothing, needs no network, and catches the source-comment and punctuation rules the repo actually broke.
- Consequence: no automation runs on push. The local gate is `python -m vla_cu style`, `python -m pytest -q`, `python -m vla_cu fuzz`, and the vcvars suite for the native path. `README.md` ground rule 7 says so.
- Revisit if: the human wants CI back, with or without a GPU runner.
- Status: active

## DEC-043: the model widths are read, not assumed  (2026-11-23)
- Context: DEC-024 kept hidden 960 and intermediate 2560 as assumptions because no config had been read.
- Finding: the published `lerobot/smolvla_base` config and the `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` text config confirm 960 and 2560, 16 VLM layers, the 0.75 expert multiplier, chunk 50 and 10 sampler steps. Sources and the full fact table are in `agents/RESEARCH.md`.
- Chosen and why: treat 960, 2560, 720 and 1920 as read values. The expert widths follow from 0.75 times the two confirmed widths.
- Consequence: `bench/layer.py` and `bench/m50.py` may say "published" instead of "assumed". Group 64 still cannot divide 720, so DEC-030 stays open on its own terms.
- Revisit if: a checkpoint shows a different width, which would itself be a finding.
- Status: active
- CORRECTION 2026-11-23: the checkpoint shows expert intermediate 2048, not 1920. The 0.75 multiplier sets the expert hidden width at 720, and the expert MLP then uses a 2048 intermediate, with q at 960, k and v at 320 and o at 720. The 1920 came from multiplying 2560 by 0.75 and was never read from a file. See `agents/RESEARCH.md`.

## DEC-044: a 720-wide expert matrix is zero padded to 768, not regrouped to 16  (2026-11-23)
- Context: EXP-043 measured both options on the real `expert.layers.0.mlp.up_proj` [2048, 720].
- Options considered: (a) zero pad K to 768 and keep group 64; (b) group 16 at the true width; (c) a ragged last group.
- Chosen and why: (a). 0.88 MB against 1.11 MB, 26 percent fewer bytes, 0.035 against 0.036 ms, and cosine 0.99489 against 0.99668. Padding also keeps the CUDA kernel usable, which requires K divisible by 64. The added columns are zeros and contribute exactly zero, so the pad itself costs no accuracy.
- Consequence: a packed 720-wide matrix is stored at 768 columns, and the activation vector must carry the same 48 zero columns. int8 stays the default for these projections by DEC-039, so the case appears only when int4 is requested.
- Revisit if: a wider measurement shows group 16 clearing the 0.98 gate where padding does not.
- Status: active

## DEC-045: rotation is not used on real SmolVLA weights  (2026-11-23)
- Context: DEC-034 left rotation opt-in pending a real checkpoint. EXP-043 ran it on three real matrices with K divisible by 64.
- Options considered: (a) rotate by default; (b) keep it opt-in; (c) drop it.
- Chosen and why: (b). Measured cosine moved 0.99218 to 0.99455, 0.99383 to 0.99402 and 0.99459 to 0.99384, so the sign is not consistent and the best gain is 0.0024, against two matmuls per block in the fast path (EXP-030).
- Consequence: the fast path stays unrotated, and `real` keeps the rotated row in its table so another weight distribution can be checked in one command.
- Revisit if: a layer with a known outlier channel is quantized, or a fused rotation kernel removes the cost.
- Status: active

## DEC-046: kernels live in a torch-free header, and the sanitizer runs a standalone harness  (2026-11-23)
- Context: EXP-041 blocked the kernel-level sanitizer pass because torch's launcher asks for a handle the tool cannot resolve.
- Options considered: (a) keep retrying the extension under the tool; (b) copy the kernel into a standalone file; (c) move the kernel into a header shared by the extension and a standalone harness.
- Chosen and why: (c). One copy of the kernel, two entry points, and the sanitizer runs under memcheck with zero errors (EXP-044). (b) would let the copy drift.
- Consequence: `csrc/int4_gemv_kernel.cuh` holds the kernel and its launch constants, and `csrc/standalone/int4_check.cu` is the harness. The int8 kernel does not have this split yet.
- Revisit if: the tool learns to follow torch's launcher, or a second harness makes a shared helper worthwhile.
- Status: active

## DEC-047: scaling reports time at every depth and accuracy only while the chain tracks  (2026-11-23)
- Context: EXP-045 chained the 16 real expert MLPs with no norms and no residuals. Time stays linear at 76 to 92 us per block, while cosine falls from 0.99460 at depth 1 to 0.31792 at depth 16.
- Options considered: (a) report the whole depth table including cosine; (b) stop at depth 8; (c) report time at every depth and cosine only up to the depth where the chain still resembles the model.
- Chosen and why: (c). Time is valid at any depth because every row runs the same kernel over the same bytes. Accuracy is only meaningful while the chain tracks the model, and this one stops doing so between depth 8 and 16.
- Consequence: the `scale` table is a timing result. A quality claim about the expert stack needs norms and residuals first.
- Revisit if: a norm and residual are added to the chain, or attention is included.
- Status: active
