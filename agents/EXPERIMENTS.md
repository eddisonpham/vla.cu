# EXPERIMENTS: append-only log of everything we tried

> Never delete entries, including failures. Corrections go below the original entry as `CORRECTION YYYY-MM-DD: ...`.
> Use the template in `agents/README.md`. Number entries sequentially (EXP-001, EXP-002, ...).
> Record the git commit, the exact command, the GPU, and measured numbers only.

---

## EXP-001: environment detection (Chunk 0)  (2026-11-18, commit 9b5dee6)
- Goal / question: what GPU, driver, and toolchain actually exist before writing any kernel?
- Change (one thing): added `scripts/env_check.py`; nothing else touched.
- Prediction: none (measurement only, no timing involved).
- Command: `.venv/Scripts/python.exe scripts/env_check.py`
- GPU / environment: RTX 5060 Laptop GPU, 7.9 GB, driver 617.42.
- Result (measured): PyTorch 2.8.0+cu129, CUDA (torch) 12.9, CUDA available True, Triton NOT INSTALLED, GPU "NVIDIA GeForce RTX 5060 Laptop GPU", 7.9 GB, compute capability 12.0, 26 SM. Tools on PATH: nvcc, nsys, ncu, compute-sanitizer all NOT FOUND.
- Explanation / hypothesis: torch installs its own CUDA runtime, so `torch.version.cuda` works without the CUDA toolkit; the toolkit is what provides nvcc and compute-sanitizer, so those are absent. Triton ships with PyTorch on Linux only, hence missing on native Windows.
- Verdict: worked
- Follow-ups: decide Triton on Windows (triton-windows vs WSL2); install the CUDA toolkit before any CUDA C++ chunk.

## EXP-002: triton-windows on native Windows / sm_120  (2026-11-18, commit e6b5c07)
- Goal / question: can the community Windows build of Triton compile and run a kernel on this sm_120 GPU with torch 2.8.0+cu129?
- Change (one thing): `pip install triton-windows` into `.venv`; added `scripts/triton_probe.py` (one masked vector-add kernel + exact comparison).
- Prediction: none filed (correctness probe, no timing).
- Command: `.venv/Scripts/python.exe scripts/triton_probe.py`
- GPU / environment: RTX 5060 Laptop GPU, driver 617.42, torch 2.8.0+cu129, Python 3.13.12.
- Result (measured): installed `triton-windows-3.9.0.post30` (73.2 MB wheel). `triton.__version__` 3.9.0. `ptxas` used: `.venv\Lib\site-packages\triton\backends\nvidia\bin\ptxas.exe` (bundled; nothing on PATH). `n=1000` grid=1 program -> max abs err 0.0e+00; `n=1000003` grid=977 programs -> max abs err 0.0e+00. Verdict printed: "triton works on this GPU".
- Surprise (logged because it will bite again): the first attempt ran the kernel from a stdin heredoc and failed with `ValueError: @jit functions should be defined in a Python file` (traceback: `inspect.getsourcelines` -> `OSError: could not get source code`). Triton compiles the *source text* of the decorated function, so kernels must live in a real `.py` file - never `python -c`, stdin, or a REPL.
- Side warning: Triton 3.9 emits `TRITON_DEFAULT_FP_FUSION is unset: defaulting to FP fusion ENABLED ... will change to DISABLED in Triton 3.10`. Matters later when float results depend on fusion.
- Explanation / hypothesis: Triton's runtime stack (TTIR -> LLVM -> PTX -> SASS) only needs a `ptxas` that knows sm_120; the Windows wheel ships one, which is why a CUDA toolkit is not required for Triton (but is for nvcc/compute-sanitizer).
- Verdict: worked
- Follow-ups: benchmark this kernel against `torch` in Chunk 3; pin `TRITON_DEFAULT_FP_FUSION` explicitly once float comparisons matter; test `tl.dot` and larger tiles before trusting Triton for the real GEMV.

## EXP-003: CUDA 12.9.2 toolkit install and host compiler check  (2026-11-18)
- Goal / question: get `nvcc` and `compute-sanitizer` onto this machine without disturbing the working 617.42 driver, and find out whether a CUDA C++ kernel can be built yet.
- Change (one thing): installed CUDA 12.9.2 with a Custom install and the Display Driver component unchecked, then probed the four tools and one trivial `.cu` file.
- Prediction: none (installation, no timing).
- Command: elevated `cuda_12.9.2_576.57_windows.exe`, then `nvcc.exe --version`, `compute-sanitizer.exe --version`, `ncu.exe --version`, `nsys.exe --version`, and `nvcc -ccbin <MSVC 14.50> -arch=sm_120 probe.cu`.
- GPU / environment: RTX 5060 Laptop GPU, driver 617.42 (unchanged after install), torch 2.8.0+cu129, Windows SDK 10.0.26100.0.
- Result (measured): installer file matched NVIDIA's MD5 (`3507abab...`). Toolkit at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`, `CUDA_PATH` set machine-wide, `bin` on the machine PATH. nvcc `release 12.9, V12.9.86`. compute-sanitizer `2025.2.1.0 (build 35969825)`. Nsight Compute `2025.2.1.0 (build 35987062)` at `...\Nsight Compute 2025.2.1\target\windows-desktop-win7-x64\ncu.exe`. Nsight Systems `2025.1.3.140-251335620677v0` at `...\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe`. The smoke compile FAILED: `crt/host_config.h(170): fatal error C1189: #error: -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported!`, no binary produced.
- Explanation / hypothesis: the only MSVC on the machine is 14.50.35717 from Visual Studio 18 BuildTools, and each CUDA release whitelists host compilers up to the VS version that existed when it shipped. 14.50 is far newer than 12.9's whitelist, so `host_config.h` rejects it before compiling anything. The `-allow-unsupported-compiler` flag exists but is explicitly risk-bearing.
- Verdict: partial. Toolkit and sanitizer work; CUDA C++ is blocked.
- Follow-ups: install VS 2022 Build Tools (`winget` offers `Microsoft.VisualStudio.2022.BuildTools` 17.14.45) and rebuild the same smoke file; only then attempt `torch.utils.cpp_extension`. Profile a real kernel before claiming ncu/nsys support sm_120.

## EXP-004: supported host compiler, and ncu/nsys discovery without PATH  (2026-11-18)
- Goal / question: make CUDA C++ compilable here, and stop `python -m vla_cu env` calling installed profiling tools missing.
- Change (one thing): installed VS 2022 Build Tools (VCTools workload only) next to VS 18, and taught `scripts/env_check.py` to fall back to a known install pattern when `shutil.which` fails.
- Prediction: none (toolchain plumbing, no timing).
- Command: `winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`, then in a shell that has called `vcvars64.bat`: `nvcc -arch=sm_120 -o probe_smoke.exe probe.cu`.
- GPU / environment: RTX 5060 Laptop GPU, driver 617.42, CUDA toolkit 12.9.2, MSVC 14.44.35207, torch 2.8.0+cu129.
- Result (measured): Build Tools 17.14.45 registered by winget; `vswhere -all` lists both `2022\BuildTools` and `18\BuildTools`. The trivial `.cu` produced `SMOKE_BUILD_OK` and `SMOKE_RUN_OK` with exit 0. `python -m vla_cu env` now prints `nsys` at `...\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe` and `ncu` at `...\Nsight Compute 2025.2.1\target\windows-desktop-win7-x64\ncu.exe`, with neither on PATH.
- Failures on the way (logged because both recur): the winget install outlived a 600 second tool timeout and had to be checked separately; the first smoke batch died in `VCVARS_FAIL` because its redirect became `/dev/null`, which `cmd` reads as a path, so `vcvars64` never ran; the second had no `nvcc` on the batch PATH. Dropping the redirect and prepending the toolkit `bin` fixed both.
- Explanation / hypothesis: `vcvars64.bat` is what puts `cl.exe` on PATH and sets `INCLUDE` and `LIB` for the 14.44 toolset, so it must run in the same shell as `nvcc`. Whether a bare `nvcc -ccbin <14.44>` works without that environment was not tested.
- Verdict: worked, for a trivial file. No kernel, no sanitizer run, no profiling.
- Follow-ups: point `torch.utils.cpp_extension` at the 14.44 toolset (or require a vcvars shell) in the first CUDA C++ chunk; confirm with a real kernel, then run `compute-sanitizer` on it.

## EXP-005: cuda-check reports the toolkit and host compiler a build would get  (2026-11-18)
- Goal / question: which nvcc and which host compiler would `torch.utils.cpp_extension` actually use here, without inferring it from scattered paths?
- Change (one thing): added `scripts/cuda_check.py` and the `cuda-check` task; the task lists in `README.md` and `docs/ARCHITECTURE.md` follow it.
- Prediction: none (configuration inspection, no kernel, no timing).
- Command: `python -m vla_cu cuda-check`, then the same command from a shell that had called `vcvars64.bat` for VS 2022.
- GPU / environment: RTX 5060 Laptop GPU, driver 617.42, torch 2.8.0+cu129, toolkit 12.9.2, MSVC 14.44 and 14.50 installed.
- Result (measured), plain shell: `CUDA_HOME` = `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\v12.9`, nvcc `12.9 V12.9.86`, `cl on PATH NOT ON PATH`, toolsets `14.50.35717 18\BuildTools` and `14.44.35207 2022\BuildTools`, `ninja NOT AVAILABLE`, verdict `blocked, needs cl.exe on PATH (call vcvars64.bat in this shell), ninja`.
- Result (measured), after `vcvars64.bat` for VS 2022: `cl on PATH C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.EXE`, `ninja AVAILABLE`, verdict `ready for torch.utils.cpp_extension`.
- Explanation / hypothesis: torch resolves the toolkit itself but asks for the host compiler by the bare name `cl`, so PATH decides which compiler a build uses, not what is installed. `vcvars64.bat` also prepends Visual Studio's bundled ninja, which removes the second blocker as well.
- Surprise: ninja is reported missing in a plain shell and present after vcvars, so "ninja is installed" is a shell fact on this machine, not a machine fact.
- Verdict: worked, for configuration only. No extension was compiled and no kernel was timed in this experiment.
- Follow-ups: build the first real extension inside a vcvars shell to confirm ninja alone is sufficient; then record whether a bare `nvcc -ccbin` without vcvars is viable, and delete or keep the vcvars requirement based on that.

## EXP-006: torch.compile with triton-windows on this venv  (2026-11-22)
- Goal / question: could `torch.compile` serve as a baseline before any hand-written kernel exists?
- Change (one thing): ran a throwaway probe that asks `torch.compile` to compile a 1024x1024 matmul with the default inductor backend.
- Prediction: none (capability probe, no timing).
- Command: `.venv/Scripts/python.exe .tmp_compile_probe.py`, a temporary file deleted after the run.
- GPU / environment: RTX 5060 Laptop GPU, torch 2.8.0+cu129, triton 3.9.0 via `triton-windows`, plain shell.
- Result (measured): `torch._dynamo.list_backends()` reports `inductor` and `cudagraphs`. Compilation FAILED after 11.0 s with `BackendCompilerFailed: backend='inductor' raised: ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'`. No kernel was produced, so there was nothing to time.
- Explanation / hypothesis: inductor imports Triton internals (`triton_key` from `triton.compiler.compiler`), and `triton-windows` 3.9.0 does not expose that symbol where torch 2.8.0 looks for it. The version numbers are close, the internal APIs are not.
- Verdict: failed. `torch.compile` is unusable as installed. This says nothing about `torch.cuda.CUDAGraph`, which uses a different code path and was not tested.
- Follow-ups: if a compile baseline is ever wanted, either pin a `triton-windows` build matching torch 2.8's expected Triton ABI or move that experiment to WSL2. Do not disturb the Triton install that is verified working. Logged in `RESEARCH.md` 2026-11-22.

## EXP-007: vector add kernel, correctness before speed  (2026-11-22)
- Goal / question: does a hand-written Triton vector add match torch exactly on awkward sizes, and is it memory-safe?
- Change (one thing): added `kernels/triton/vector_add.py` and `tests/test_vector_add.py`.
- Prediction: none (correctness gate, no timing yet).
- Command: `.venv/Scripts/python.exe -m pytest -q`, then `compute-sanitizer.exe .venv/Scripts/python.exe -c '<one launch at n=1025>'`.
- GPU / environment: RTX 5060 Laptop GPU, torch 2.8.0+cu129, triton 3.9.0, plain shell.
- Result (measured): 7 passed in 3.60 s, `torch.equal` bit exact at sizes 0, 1, 1023, 1024, 1025 and 1,000,003, plus a `ValueError` on mismatched shapes. Sanitizer on a single 1025-element launch: `ERROR SUMMARY: 0 errors`, exit 0.
- Result (measured, negative): `compute-sanitizer` around the whole pytest run did not finish within 600 s. Six instrumented compiles are the likely cause. Not investigated further.
- Explanation / hypothesis: each element does one masked load pair and one masked store, so exact equality with `x + y` is expected, same operands in the same order with no reassociation. Triton warned that `TRITON_DEFAULT_FP_FUSION` is unset and defaults to enabled in 3.9; harmless for a bare add, load-bearing once a multiply is fused into it.
- Verdict: correct and memory-safe for the sizes tested.
- Follow-ups: benchmark against torch on the same buffers, prediction first; pin `TRITON_DEFAULT_FP_FUSION` before any reduction or fused multiply-add.

## EXP-008: vector add bandwidth against torch  (2026-11-22)
- Goal / question: what fraction of the 384 GB/s spec peak does a hand-written Triton vector add reach, and does it match torch?
- Change (one thing): added `vla_cu/bench_utils.py` (7 GB cap, CUDA event timing, bandwidth math) and `bench/vector_add_bench.py`, plus a `bench` task. `vector_add_into` was split out so both paths write into the same preallocated buffer.
- Prediction: none recorded. The human asked for the benchmark directly, so this is a first measurement, not a tested prediction.
- Command: `python -m vla_cu bench`.
- GPU / environment: RTX 5060 Laptop GPU, 26 SMs, 7.9 GB, torch 2.8.0+cu129, triton 3.9.0. BLOCK=1024, warmup 25, iterations 100, CUDA events, otherwise idle GPU.
- Result (measured):

| n | moved MB | torch GB/s | triton GB/s | triton % of 384 |
|---|---|---|---|---|
| 4,096 | 0.05 | 5.7 | 3.7 | 1.0 |
| 1,048,576 | 12.58 | 1186.7 | 966.0 | 251.6 |
| 16,777,216 | 201.33 | 309.7 | 330.9 | 86.2 |

- Explanation / hypothesis: at 4,096 elements there are 4 programs and about 50 KB to move, so the time is launch overhead and GB/s has no meaning. At 1M the working set is about 12.6 MB and stays in L2 across 100 iterations, so measured bandwidth exceeds DRAM peak: that row measures cache, not memory. Only the 16M row is a memory-bound measurement. There Triton reaches 86.2 percent of the spec figure and torch reaches 80.7 percent, so a hand-written kernel ties an optimized one as expected, and the remaining 14 percent is the gap between spec and reality on a 45 to 100 W part.
- Verdict: first real bandwidth number on this GPU. Triton ahead of torch at the memory-bound size, behind at the cache-resident size, both plausible.
- Follow-ups: sweep BLOCK_SIZE at 16M; check whether the missing 14 percent is thermal by watching clocks over a longer run; never quote the 251 percent row as a result.

## EXP-009: launch overhead, Python dispatch against a CUDA graph  (2026-11-22)
- Goal / question: how much of a batch 1 step goes to submitting kernels rather than running them?
- Change (one thing): added `bench/launch_overhead.py` and task `launch`, one 1-element Triton kernel timed four ways.
- Prediction: none recorded, the human said go.
- Command: `python -m vla_cu launch`.
- GPU / environment: RTX 5060 Laptop GPU, torch 2.8.0+cu129, triton 3.9.0, 1000 launches, graph replayed 20 times.
- Result (measured):

| method | us per launch | launches/s |
|---|---|---|
| python loop, queued, one sync | 16.71 | 59,861 |
| python loop, sync each launch | 24.14 | 41,418 |
| cuda events, same queued loop | 16.27 | 61,466 |
| cuda graph, 1000 nodes replayed | 0.63 | 1,578,889 |

- Explanation / hypothesis: the queued loop and the event timed loop agree at about 16.3 us, so the GPU is idle waiting for the next submission. The limit is Triton's Python launcher, not the kernel and not the driver, which is why removing Python with a graph drops the cost 26 times to 0.63 us per node. Synchronizing every launch adds 7.4 us of round trip.
- Cross-check: EXP-008's 4,096 element row read 3.7 GB/s, which is 13.3 us per call for 48 KB, the same order as the 16.7 us dispatch measured here. That row was never measuring bandwidth, and this explains it.
- Verdict: CUDA graphs are justified. A 400 kernel chain at 16.7 us costs 6.7 ms of pure dispatch, and replay removes 6.4 ms of it.
- Caveats: a 1-element kernel has no cache or occupancy effects, so this isolates submission cost only. Graph capture worked with Triton on native Windows, which was not guaranteed. Replay needs fixed shapes and addresses, which is exactly the batch 1 case.
- Follow-ups: capture a real GEMV chain once it exists; confirm nothing allocates inside capture; keep the plain path for tests, because graphs hide errors until replay.

## EXP-010: fused int8 GEMV against dequantize-then-matmul  (2026-11-22)
- Goal / question: does keeping weights int8 in memory and dequantizing in registers beat materializing fp32 weights, and by how much at batch 1?
- Change (one thing): added `kernels/triton/int8_gemv.py` (one program per output row, per-row scale applied after the reduction), `tests/test_int8_gemv.py`, `bench/int8_gemv_bench.py`, task `gemv`.
- Prediction: none recorded, the human asked for the kernel directly.
- Command: `python -m vla_cu gemv`.
- GPU / environment: RTX 5060 Laptop GPU, 26 SMs, 7.9 GB, L2 33.6 MB, peak 384 GB/s. BLOCK_K 1024, warmup 10, iterations 50.
- Correctness: `pytest -q tests/test_int8_gemv.py` 9 passed against a float64 dequantize-then-matmul reference, including K=1025 and K=257. `compute-sanitizer` on a single 17 x 257 launch: `ERROR SUMMARY: 0 errors`.
- Result (measured):

| N x K | W MB | naive ms | fused ms | speedup | fused GB/s | % of DRAM | fits L2 |
|---|---|---|---|---|---|---|---|
| 1024 x 1024 | 1.0 | 0.034 | 0.015 | 2.33 | 72.7 | 18.9 | yes |
| 4096 x 2048 | 8.4 | 0.389 | 0.020 | 19.80 | 429.2 | 111.8 | yes |
| 4096 x 8192 | 33.6 | 1.819 | 0.042 | 43.61 | 806.2 | 209.9 | yes |
| 8192 x 16384 | 134.2 | 7.046 | 0.379 | 18.60 | 354.7 | 92.4 | no |

- Explanation / hypothesis: the fused path moves N*K bytes of weights plus one scale and one output per row, so its traffic is about 9 times smaller than the naive path, which writes fp32 weights and reads them back. Rows above 100 percent of DRAM peak are L2 resident: at 33.6 MB the weight matrix exactly equals L2 and streams at 806 GB/s, which is cache bandwidth. Only the 134 MB row is a DRAM measurement, and there the kernel reaches 92.4 percent of the 384 GB/s spec. The small row is dispatch bound at 15 us, consistent with EXP-009's 16.7 us per launch.
- Verdict: fusion works and works for the stated reason. At a DRAM-bound shape the kernel sits at the memory limit, which is where a batch 1 GEMV should sit.
- Caveats: the naive baseline uses torch's int8 to fp32 cast, which managed only about 171 GB/s of effective traffic, so part of the 18.6x is a slow baseline rather than a fast kernel. It is the honest naive path, not a tuned GEMV.
- Follow-ups: sweep BLOCK_K; split K across programs when N is small compared with the SM count; revisit the x re-read once K is large enough that K floats stop fitting in L2.

## EXP-011: what a CUDA graph does to a real kernel and to a two-kernel chain  (2026-11-22)
- Goal / question: does the 16 us launch saving survive when the kernel does real work, and what does a chained pair save?
- Change (one thing): added `bench/graph_chain.py` (task `graph`), and a graphed column in `bench/vector_add_bench.py` that captures 10 launches per graph and divides the replay by 10.
- Prediction: none recorded.
- Commands: `python -m vla_cu graph`, `python -m vla_cu bench`.
- Result, vector add (measured):

| n | torch GB/s | plain GB/s | graphed GB/s |
|---|---|---|---|
| 4,096 | 6.5 | 3.5 | 40.4 |
| 1,048,576 | 1151.4 | 908.8 | 2160.7 |
| 16,777,216 | 316.5 | 317.8 | 334.2 |

- Result, two int8 GEMV kernels per chain (measured):

| in -> hidden -> out | plain ms | graph ms | speedup | exact |
|---|---|---|---|---|
| 1024 -> 1024 -> 512 | 0.031 | 0.012 | 2.64x | yes |
| 2048 -> 2048 -> 2048 | 0.028 | 0.018 | 1.59x | yes |

- Explanation / hypothesis: a graph helps exactly where submission was the bottleneck. At 4,096 elements the vector add reads 3.5 GB/s plain and 40.4 graphed, 11 times faster, because 48 KB of traffic spent 13 us in dispatch. At 16M the kernel runs about 600 us and the graph changes nothing measurable, 317.8 against 334.2, inside run-to-run noise. The 2,048 chain is faster plain than the 1,024 chain, 0.028 against 0.031 ms, because more work per kernel hides dispatch, which is the same lesson from the other side.
- Caveat: the graphed 1M row reads 2,160 GB/s, above DRAM peak, because the buffers are L2 resident. That row is a cache measurement and is not a result.
- Verdict: graphs pay for many small kernels and buy nothing for few large ones. Capture was exact, `torch.equal` against the plain chain output.
- Follow-ups: graph a layer stack once SmolVLA shapes are known; keep graphs out of the test suite (DEC-018).
