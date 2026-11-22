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
