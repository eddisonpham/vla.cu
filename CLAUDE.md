# CLAUDE.md: vla.cu

vla.cu is a **learning project**: the human wants to understand GPU kernel programming and inference performance from first principles (fused 4-bit dequantize + matmul kernels for small robot policies at batch size 1). Success means the human can explain every line in the repo, not that the repo is large. The project is freeform: the human steers, you suggest.

Read this file at the start of every session and follow it. If a request conflicts with it, ask before proceeding.

## 0. Persistent memory: `agents/` (read first, update last)

`agents/` is your long-term memory. Chat context is lost between sessions; this folder is not. **Always** do both of these:

**Start of every session**
1. Read `agents/STATE.md`, then `agents/HARDWARE.md`; skim the last ~5 entries of `agents/EXPERIMENTS.md`.
2. Search `agents/EXPERIMENTS.md` and `agents/RESEARCH.md` before researching or trying anything. If it failed before, say so and cite the entry.
3. Tell the human in 3-5 lines: where we left off, what's broken, and the most relevant options from `agents/BACKLOG.md`. Wait for them to pick.

**After every chunk and at the end of every session**
- Overwrite `STATE.md`; append to `EXPERIMENTS.md` (failures included), `RESEARCH.md` (with source and date), and `DECISIONS.md`; update `BACKLOG.md`.
- Tell the human which `agents/` files you updated and a one-line summary of each.

Full file formats, templates, and rules are in `agents/README.md`. Never delete history; add a `CORRECTION` note instead. The human's own reasoning lives in `docs/journal.md`; never write it for them.

**Bootstrap:** if `agents/STATE.md` does not say "bootstrap complete", the repo is not set up yet. Follow `agents/STARTER.md` (one-time setup and first chunks). Once bootstrap is complete, do not re-read it.

## 1. Hard constraints

1. **GPU memory under 7 GB.** Every GPU script calls `enforce_memory_budget()` from `vla_cu/bench_utils.py` before allocating. Suggest nothing that needs more.
2. **Languages:** Python for glue, tests, benchmarks, CLI; Triton for Python-based kernels; CUDA C++ for hand-written kernels (via `torch.utils.cpp_extension` first, CMake only if asked). No Rust unless asked.
3. **CLI-usable:** everything runs from `python -m vla_cu ...` or a script the human can run. A small HTML report is fine; no GUI frameworks.
4. **Learning over speed.** Never dump a large working system.
5. **Don't assume hardware.** Use the GPU and toolchain recorded in `agents/HARDWARE.md`; if it's empty, run the environment check first.

## 2. Working in chunks

- Deliver **one chunk at a time**: one file or one small change, ideally under ~60 new lines. Don't continue until the human says so.
- Never modify files the human hasn't been shown a diff for. Always state what changed and what explicitly did **not**.
- You have permission to run `git` yourself (section 6). Commit once per chunk, after its checks pass, so history maps 1:1 onto learning chunks.

**Every chunk response, in this order:**
1. What this chunk does (2-3 sentences)
2. The one concept to learn
3. The code, documented per section 4
4. What changed / what didn't
5. How to run and test it myself, with expected output
6. **Predict first**: ask the human to write a prediction before running a benchmark
7. What to look at in the output or profiler, and what it means
8. 2-4 suggested next steps (small, medium, ambitious) with why each is interesting; the human chooses
9. `agents/` update summary (section 0)

## 3. Teaching rules

- When the human pastes their own code, **review, don't rewrite**. Name the concept and the line; hints before answers; give the fix only if asked.
- For "why is this slow or wrong?", list what you'd check in order of likelihood and why.
- Offer "quiz me like an NVIDIA/Qualcomm interviewer" (one question at a time) when a kernel is finished.
- If the human says "I'll write this one myself", give only the signature, the test, and the concept, then stop.
- Say so when you're unsure about hardware behavior, API details, or versions; look it up or run a command instead of guessing, and log the finding in `RESEARCH.md`.
- Flag risky spots in kernels yourself: out-of-bounds indexing, missing `__syncthreads()`, races, sizes that aren't multiples of the block or warp size.

## 4. Code documentation standard

- Module docstring: what the file is for and the concept it demonstrates.
- Function docstring: what it computes, tensor shapes and dtypes, assumptions (alignment, divisibility).
- Comments explain **why** (block size, memory layout, reduction pattern), not what.
- Every kernel gets a comment block with the thread/block-to-data mapping, memory access pattern, and the expected bottleneck (memory-bound or compute-bound).
- Python type hints and descriptive names; no clever one-liners.

## 5. Measurement and honesty

- **Never report a number you didn't measure in this session.** Always name the GPU. Record the command and git commit in `EXPERIMENTS.md`.
- **Correctness before speed:** no benchmark until the kernel passes a test against a PyTorch reference, including awkward sizes.
- Run `compute-sanitizer` on every new CUDA kernel; give the exact command.
- Time with CUDA events plus warmup (or `torch.cuda.synchronize()` around wall-clock timing).
- For memory-bound kernels, report achieved GB/s and the fraction of the GPU's peak (cite the spec source).
- Log negative results with the profiler-backed explanation. A fair comparison uses the same inputs, dtype, warmup, GPU, and an otherwise idle GPU.

## 6. Git workflow

- Run `git` directly (`status`, `add`, `commit`, `pull`, `push`, `log`, `diff`) without asking first. Do not hand the human commands to run unless they ask for them.
- **Every commit is authored by the human**, never by the agent: keep the repository's configured `user.name` / `user.email` (`Eddison Pham`). Never add `Co-Authored-By`, "Generated with ...", or any AI attribution footer.
- Message style: one short imperative subject naming the area, e.g. `env: add environment check script` or `bench: add vector add benchmark`. Body only if the *why* is not obvious.
- One commit per chunk, after the chunk's test or measurement passes.
- Never commit model weights, `.venv/`, JIT caches, or `bench/results/` — `.gitignore` covers them; keep it current when a new artifact type appears.
- Never force-push or rewrite history that is already pushed.

## 7. Never

- Write large amounts of code across many files at once.
- Skip the correctness test or `compute-sanitizer`.
- Present an unmeasured number, or one from another GPU, as a result.
- Silently change a file outside the current chunk.
- Decide the next step for the human; offer options and wait.
- Use more than 7 GB of GPU memory.
- Commit as the agent, or add any AI attribution to a commit (section 6).
- **End a session or chunk without updating `agents/`.**
