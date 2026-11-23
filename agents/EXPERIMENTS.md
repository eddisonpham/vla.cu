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

## EXP-012: BLOCK_K sweep and split-K for the int8 GEMV  (2026-11-22)
- Goal / question: which block size streams weights fastest, and which shapes are too narrow for one program per row?
- Change (one thing): K became a runtime argument so one binary serves every K; `block_k` and `split` became wrapper parameters; added `int8_gemv_split_kernel` (interleaved parts, atomic add) and `bench/int8_gemv_tune.py` with task `tune`.
- Prediction: none recorded.
- Command: `python -m vla_cu tune`.
- Result, block size sweep at 8192 x 16384, 134.3 MB moved, split 1 (measured):

| BLOCK_K | ms | GB/s | % of 384 |
|---|---|---|---|
| 128 | 0.460 | 292.4 | 76.1 |
| 256 | 0.400 | 335.7 | 87.4 |
| 512 | 0.400 | 336.2 | 87.6 |
| 1024 | 0.399 | 336.7 | 87.7 |
| 2048 | 0.415 | 323.4 | 84.2 |
| 4096 | 0.416 | 323.1 | 84.1 |

- Result, split-K with a constant 134 MB of weights per shape (measured):

| N x K | split 1 | split 2 | split 4 | split 8 | programs at 8 |
|---|---|---|---|---|---|
| 16 x 8388608 | 3.430 ms | 1.712 | 0.881 | 0.525 | 128 |
| 64 x 2097152 | 0.921 | 0.521 | 0.419 | 0.405 | 512 |
| 256 x 524288 | 0.380 | 0.392 | 0.411 | 0.388 | 2048 |
| 1024 x 131072 | 0.422 | 0.386 | 0.387 | 0.404 | 8192 |
| 4096 x 32768 | 0.380 | 0.385 | 0.410 | 0.417 | 32768 |

- Explanation / hypothesis: split-K fixes latency, not bandwidth. With 16 rows there are 16 programs on 26 SMs, so 10 SMs idle and each program streams an 8 MB row alone; eight parts per row gives 128 programs and 6.53x. From 256 rows upward the grid already fills the GPU, so splitting only adds atomic traffic plus a zero fill, which is neutral at 256 and a few percent slower at 4096. The rule of thumb this gives: split when N is below roughly twice the SM count.
- Cost: atomic order is non-deterministic, so results differ in the last bits, up to 5.7e-3 relative at 4096 x 32768. Tests use tolerances for that reason, and `compute-sanitizer --tool racecheck` reports 0 hazards.
- Block size: 256 to 1024 are within 0.2 percent of each other, 128 loses 13 percent to too few lanes in flight, and 2048 to 4096 lose 4 percent to occupancy. 1024 stays the default, now with evidence.
- Follow-ups: split ratios between 4 and 8 for the 16-row case; a two-stage reduction instead of atomics if determinism is ever needed.

## EXP-013: 4-bit packing, fused dequant GEMV, and the ladder matrix  (2026-11-22)
- Goal / question: does halving weight bytes again pay off, and what does the whole ladder look like on identical weights at batch 1?
- Change (one thing): `vla_cu/quant.py` (pack and reference dequantize), `kernels/triton/int4_gemv.py` (tiled, ROWS rows per program), `tests/test_quant.py`, `tests/test_int4_gemv.py`, `bench/matrix.py` with task `matrix`.
- Prediction: none recorded.
- Correctness: 49 tests pass in 7.03 s, with the int4 GEMV compared against float64 dequantize-then-matmul at 5 shapes including K=320, packing error bounded by half a quantization step, and 4 rejection tests. `compute-sanitizer` on the tiled int4 kernel: 0 errors. On the split int8 kernel: 0 errors, 0 race hazards.
- Result, ladder at 8192 x 16384, batch 1 (measured):

| format | MB | ms | GB/s | % of 384 | x fp32 | rel L2 err |
|---|---|---|---|---|---|---|
| fp32 | 537.0 | 1.614 | 332.7 | 86.6 | 1.00 | 0 |
| bf16 | 268.5 | 0.803 | 334.5 | 87.1 | 2.01 | 2.88e-3 |
| int8 | 134.3 | 0.375 | 358.0 | 93.2 | 4.30 | 9.37e-3 |
| int4 | 75.6 | 0.263 | 287.8 | 75.0 | 6.15 | 1.07e-1 |

- Result, int4 row tile sweep at the same shape (measured): rows 1 = 0.401 ms, 2 = 0.290, 4 = 0.236, 8 = 0.231. Tiling rows per program is worth 1.7x because every program otherwise re-reads the whole 64 KB activation vector, and there are 8192 of them.
- Explanation / hypothesis: int4 moves 56 percent of the bytes int8 does (75.6 against 134.3 MB) but is only 1.43x faster, so its effective streaming rate is 287.8 against 358.0 GB/s. The unpack costs real time: two nibble extractions, one conversion and one scale gather per byte. The precision bill is 1.07e-1 relative L2 against 9.37e-3 for int8, which is what buying half the bytes with 16 levels costs.
- Small shapes: at 1024 x 1024 every format lands between 0.016 and 0.027 ms, dispatch dominated, and the ladder says nothing there. At 4096 x 2048 the fp32 row reads 1265.8 GB/s because 33.6 MB fits L2, so only the 8192 x 16384 block is a DRAM result.
- Caveats: the matrix first reported nonsense errors because elementwise relative error explodes near zero outputs; it now reports relative L2 norm, which is the right measure for a quantization comparison but still not a quality measure. Cosine similarity against fp32 is the gate for a real model.
- Follow-ups: hoist the group scale out of the inner loop (64 weights share one value); try 8 rows with a shared-memory x tile; compare against torchao and bitsandbytes on the same shapes.

## EXP-014: Group-major reduction closes the int4 bandwidth gap  (2026-11-23)
- Goal / question: int4 streamed 287.8 GB/s against 358.0 for int8. Was that the nibble unpack, or the per-byte scale gather?
- Change (one thing): the tile became `[ROWS, groups, GROUP // 2 bytes]` and the reduction runs inside each group first, so one scale load and one multiply serve 64 weights. The accumulator shrank from `[ROWS, BLOCK_K // 2]` to `[ROWS, BLOCK_K // GROUP]`, a factor of 32. Supersedes the int4 timing in EXP-013.
- Prediction: no prediction recorded. This is a regression against the rule in CLAUDE.md, noted here rather than hidden.
- Result at 8192 x 16384, batch 1 (measured, same run for both formats):

| kernel | ms | GB/s | % of 384 | x fp32 |
|---|---|---|---|---|
| int4 per-byte scale (EXP-013) | 0.263 | 287.8 | 75.0 | 6.15 |
| int4 group-major | 0.230 | 328.2 | 85.5 | 6.89 |
| int8 reference, same run | 0.400 | 336.1 | 87.5 | 3.97 |

- Result, tile sweep at the same shape (ms, block_k 512 / 1024):

| rows | block 512 | block 1024 |
|---|---|---|
| 1 | 0.634 | 0.368 |
| 2 | 0.294 | 0.252 |
| 4 | 0.231 | 0.229 |
| 8 | 0.250 | 0.231 |
| 16 | 0.228 | 0.231 |
| 32 | 0.238 | 0.257 |

- Explanation / hypothesis: the gather was a real cost, 12.5 percent of runtime, and 32 times fewer scale loads plus 32 times fewer scale multiplies recovered it. The gap to int8 is now 2 points of peak, which is what the nibble surgery should cost. The row tile saturates at 4: below that every program re-reads the activation vector, above it the transient products tile costs more than the activation traffic it saves, and 32 rows loses 10 percent.
- Decision: keep `ROWS_PER_PROGRAM = 4`, `BLOCK_K = 1024`. Rows 4 to 16 sit within 1 percent, which is inside run-to-run noise, so there is no evidence to move the default.
- Caveats: one run per configuration, no repeats, and no profiler trace of the L2 read traffic that the model predicts.
- Follow-ups: stage the activation tile in shared memory; measure the same sweep on a shape that does not fit in L2 anyway (4096 x 2048 is L2-resident here); try `tl.dot`-style reduction instead of `tl.sum` over a 3D tile.

## EXP-015: Custom ops and one SmolVLA-shaped MLP block end to end  (2026-11-23)
- Goal / question: once the kernels are reachable as torch operators, does fusion still pay on a real layer instead of a synthetic matrix?
- Change (one thing): `vla_cu/ops.py` registers both GEMVs with `torch.library.custom_op` and a fake implementation, and `bench/layer.py` runs the MLP block 960 -> 2560 -> 960 at batch 1 on three paths.
- Prediction: no prediction recorded.
- Result (measured, warmup 10, iterations 50):

| path | ms | GB/s | x fp32 | cosine | rel L2 |
|---|---|---|---|---|---|
| fp32, two `torch.mv` | 0.072 | 273.3 | 1.00 | 1.00000 | 0 |
| dequantize, then `torch.mv` | 0.072 | 272.6 | 1.00 | 0.99346 | 1.15e-1 |
| fused, through the custom ops | 0.085 | 45.3 | 0.85 | 0.99346 | 1.15e-1 |

- Explanation / hypothesis: the block is 19.7 MB in fp32, so it fits in the 33.6 MB L2 and never touches DRAM. There are no bytes to save, and the fused path pays two Python-dispatched launches (about 16 us each, EXP-009) against kernels that take about 20 us. The 45.3 GB/s is not a bandwidth result, it is the dispatch floor divided by the byte count. Fusion wins only above L2, which is exactly the 8192 x 16384 row of the matrix.
- Both quantized paths agree to 5 decimals in cosine, which is the correctness result that matters here: the custom op path reproduces the reference chain, not just the single kernel.
- Bugs found and fixed in this chunk: the dequantize baseline first forgot to apply the int8 row scale (reported 1.4e3 relative L2), and the fused path returned `None` because the timed helper had no return value.
- Caveats: the 0.9935 cosine is for one random block, not a checkpoint, and the 0.98 gate in the prior project was measured on a whole model. Real weights could sit either side of that.
- Follow-ups: CUDA-graph the block to remove the dispatch cost (DEC-018); compare `torch.compile` on the custom op path; run a real checkpoint when one is available.

## EXP-016: CUDA C++ port of the int4 GEMV against Triton  (2026-11-23)
- Goal / question: what does hand-written CUDA buy over the Triton kernel on the same shapes?
- Change (one thing): `csrc/int4_gemv.cu`, built with `torch.utils.cpp_extension.load`. One warp per output row, each lane loads a uint32 (four bytes, eight weights) and two float4 activations, the group scale is a warp-uniform load, and the row sum is a shuffle reduction. No shared memory, no barrier.
- Prediction: no prediction recorded.
- Result (measured, warmup 5, iterations 20, both kernels on the same tensors):

| N x K | MB | Triton ms | Triton GB/s | CUDA ms | CUDA GB/s | CUDA / Triton | agree |
|---|---|---|---|---|---|---|---|
| 1024 x 1024 | 0.6 | 0.021 | 28.9 | 0.008 | 73.5 | 2.54x | yes |
| 4096 x 2048 | 4.7 | 0.021 | 227.4 | 0.014 | 331.7 | 1.46x | yes |
| 8192 x 16384 | 75.6 | 0.231 | 327.2 | 0.240 | 315.1 | 0.96x | yes |

- Explanation / hypothesis: at the DRAM-bound shape the two kernels are 4 percent apart, well inside the noise this laptop GPU produces, so the port buys nothing where bytes matter. The large small-shape gaps are dispatch cost, not kernel speed: 21 us from Python through Triton against 8 us through pybind. That is the same launch overhead EXP-009 measured, and it is why CUDA graphs exist.
- Correctness: outputs agree with Triton within `rtol=1e-4` on every shape, and both match the float64 dequantize-then-matmul reference on 5 shapes including a 3-row tail and K=320. 60 tests pass.
- Sanitizer, unresolved: `compute-sanitizer` 2025.2.1 reports `CUDA_ERROR_INVALID_HANDLE` on `cuKernelGetFunction` when it instruments this extension, both with the default tool and with `--tool memcheck --target-processes all`. That is a tooling failure, not a memory report, so the CUDA kernel has no clean sanitizer pass yet. The Triton kernels are clean. Do not read this as verified memory safety.
- Build caveat: the extension only builds inside a shell where `vcvars64.bat` has run (EXP-004, EXP-005), and outside that shell torch invalidates the cached build and fails on ninja, so the `cuda` task must run from a vcvars shell.
- Follow-ups: `ncu` on both kernels to see whether the 4 percent is occupancy or issue stalls; try two rows per warp; graph the CUDA launch to separate dispatch from kernel time.

## EXP-017: the attainable read bandwidth on this part  (2026-11-23, commit 97ad2f0)
- Goal / question: H1 in `OPTIMIZATIONS.md`. Every "% of peak" in this repo divides by a 384 GB/s spec sheet. What does a read-only kernel actually deliver, and how repeatable is it?
- Change (one thing): added `bench/bandwidth_probe.py` (task `stream`), a vectorized read with no unpacking, 5 repeats per configuration, repeats interleaved across block sizes after the first run showed order effects.
- Prediction: agent, from H1: a pure read reaches 350 to 365 GB/s, so int4 at 328 has 6 to 10 percent of headroom, not 15. No human prediction this session.
- Command: `python -m vla_cu stream`
- GPU / environment: RTX 5060 Laptop GPU, 26 SM, L2 33.6 MB, spec peak 384 GB/s, warmup 5, 20 iterations, 5 repeats.
- Result (measured), best sample per configuration:

| size MB | BLOCK 1024 | BLOCK 4096 | BLOCK 16384 |
|---|---|---|---|
| 67.1 | 355.8 GB/s (92.7%) | 333.0 (86.7%) | 355.3 (92.5%) |
| 134.2 | 346.3 (90.2%) | 347.0 (90.4%) | 360.2 (93.8%) |

- Result (measured), spread: the ratio of best to worst sample inside one configuration ranged from 53.5 to 106.8 percent. The worst samples sat near 170 to 225 GB/s while the best sat at 330 to 360 in the same configuration, minutes apart.
- Explanation / hypothesis: the ceiling is 346 to 360 GB/s, 90 to 94 percent of the spec sheet, which is what a 45 to 100 W laptop part sustains. The enormous spread is a clock and power state effect, not a kernel effect: the medians of some configurations sat at 170 GB/s while a neighbouring configuration in the same loop sat at 330, and the pattern did not follow block size or buffer size.
- Verdict: worked, H1 confirmed. int4 at 328.2 GB/s (EXP-014) is 91 to 95 percent of attainable, so single digit percent of headroom, and every single-sample table in this repo carries a spread larger than the effects it reports.
- Follow-ups: fixed clocks (`nvidia-smi -lgc`) would remove the spread but need administrator rights. Until then, report best-of-N and never compare across sessions.

## EXP-018: what a CUDA graph does to the MLP block  (2026-11-23, commit 7d8f8e3)
- Goal / question: H6. The block is 19.7 MB fp32 and fits in L2, so EXP-015 said its fused path loses only through dispatch. How much is Python, how much is the launch, how much is the kernel?
- Change (one thing): added two paths to `bench/layer.py`, "fused raw" (the two kernel launches with preallocated buffers, no custom op) and "fused graph" (a `torch.cuda.CUDAGraph` of those two launches).
- Prediction: agent, from H6: 0.085 ms becomes about 0.055 with a graph. No human prediction this session.
- Command: `python -m vla_cu layer`, four times, two before and two after the `warm_gpu` fix.
- GPU / environment: RTX 5060 Laptop GPU, L2 33.6 MB, block 960 to 2560 to 960, batch 1, warmup 10, 50 iterations.
- Result (measured), milliseconds:

| run | fp32 | dequantize | fused ops | fused raw | fused graph |
|---|---|---|---|---|---|
| A | 0.152 | 0.075 | 0.159 | 0.053 | 0.019 |
| B | 0.150 | 0.071 | 0.224 | 0.054 | 0.019 |
| C, after warm_gpu | 0.073 | 0.145 | 0.109 | 0.059 | 0.019 |
| D, after warm_gpu | 0.082 | 0.094 | 0.135 | 0.053 | 0.018 |

- Explanation / hypothesis: the graph column is the only stable one, 0.018 to 0.019 ms in all four runs, and raw launches are stable at 0.053 to 0.059. The custom op path is both slower and unstable, 0.109 to 0.224, because it validates and allocates on every call. Cosine stayed at 0.99346 on every path, so the graph reproduces the eager chain exactly.
- Surprise (the important one): in run A fp32 read 0.150 and dequantize 0.071, and in run B the two swapped places. Whichever path is timed first is slow. Adding `warm_gpu()` to `bench_utils` fixed that case and did not fix the general instability: in runs C and D the second path became the slow one. fp32 settled at 0.073 to 0.082, matching EXP-015's 0.072.
- Verdict: worked, H6 confirmed with room to spare. The graph costs 0.0185 against 0.053 raw, a 2.8x, and against fp32 at 0.073 to 0.082 it is 3.8x to 4.5x. The custom op layer is the slowest option on this block.
- Follow-ups: the instability is now the top measurement problem, see DEC-029.

## EXP-019: profiling the int4 kernel with Nsight Compute  (2026-11-23, commit 97ad2f0)
- Goal / question: H5. Is the 85 percent a DRAM limit, a latency limit or an occupancy limit?
- Change (one thing): added `scripts/profile_target.py`, three launches of the Triton int4 kernel at 8192 x 16384, so a profiler can skip two and capture one.
- Prediction: agent, from `OPTIMIZATIONS.md` section 2: ALU use about 4 percent, so the answer is expected to be memory or latency, not compute. No human prediction this session.
- Command: `ncu --launch-skip 2 --launch-count 1 --set basic --csv .venv/Scripts/python.exe -m scripts.profile_target`
- GPU / environment: RTX 5060 Laptop GPU, driver 617.42, Nsight Compute 2025.2.1.
- Result (measured): `ERR_NVGPUCTRPERM`, "The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0". No counter was collected. `compute-sanitizer` had already failed on the extension for a different reason (EXP-016).
- Explanation / hypothesis: Windows denies hardware performance counters to non-administrator users by default. It is a policy setting (`NVreg_RestrictProfilingToAdminUsers`), not a tool defect.
- Verdict: failed, as a tooling blocker. This machine currently has no way to explain a kernel's stalls, so every attribution in this repo is inference from timing, not measurement.
- Follow-ups: run the profiler from an elevated shell, or ask the human to allow non-admin profiling and reboot. Until then H2, H4 and H5 stay unproven.

## EXP-020: do Triton eviction hints move the int4 streaming rate?  (2026-11-23, commit 7d8f8e3)
- Goal / question: H2. Weights are read once and x is reused by every program, so should weights be `evict_first` and x `evict_last`?
- Change (one thing): `eviction_policy` on all three loads in `kernels/triton/int4_gemv.py`, exposed as a `hints` boolean.
- Prediction: agent, from H2: 2 to 5 percent at DRAM sizes. No human prediction this session.
- Command: `python -m vla_cu matrix`, ablation table, two samples.
- Result (measured) at 8192 x 16384, 75.6 MB:

| configuration | sample 1 ms | sample 1 GB/s | sample 2 ms | sample 2 GB/s |
|---|---|---|---|---|
| fp32 scales, no hint | 0.232 | 325.6 | 0.216 | 350.1 |
| fp32 scales, hints | 0.216 | 350.5 | 0.229 | 330.3 |

- Explanation / hypothesis: the two samples disagree in sign. The mean is 0.224 against 0.2225, a 0.7 percent difference against a 6 percent noise floor, which is what EXP-017 predicts. Relative L2 error was identical, 1.07e-1.
- Verdict: inconclusive. The hints are left on because they cannot hurt and they encode the intent, but no gain is claimed for Triton.
- Follow-ups: the CUDA `__ldcs` variant (EXP-022) is the same idea through a stronger instruction level hint and it did show a consistent sign.

## EXP-021: do fp16 group scales save anything?  (2026-11-23, commit 7d8f8e3)
- Goal / question: H3. Scales are 8.39 MB of the 75.6 MB at this shape, 11.1 percent of the bytes. fp16 halves them.
- Change (one thing): `pack_int4(..., scale_dtype=torch.float16)` and a kernel that converts the loaded scale to fp32.
- Prediction: agent, from H3: 0.230 to 0.218 ms, and error unchanged to three digits. No human prediction this session.
- Command: `python -m vla_cu matrix`, ablation table, two samples.
- Result (measured), 71.4 MB with fp16 scales: 0.219 ms (325.8 GB/s) and 0.207 ms (344.7 GB/s). Against the fp32 scale rows of 0.232, 0.216, 0.216 and 0.229.
- Explanation / hypothesis: 5.6 percent fewer bytes, no measurable time change, and the effective GB/s drops by the same 5.6 percent. Either the scale traffic was already absorbed by L2, or the narrower scale tile coalesces worse, 16 half values are 32 bytes where 16 floats are 64. The error is unchanged at 1.07e-1, as predicted, because the codes are chosen with fp32 scales and only the stored table is narrowed.
- Verdict: inconclusive, no gain. Direction agrees with the byte arithmetic, magnitude is under the noise floor. fp32 stays the default.
- Follow-ups: revisit only with a fixed-clock harness. The 4.2 MB saved is worth having in a real model even at zero time change, since it is 5.6 percent of the weight footprint.

## EXP-022: three CUDA variants of the same kernel  (2026-11-23, commit cc33ac1)
- Goal / question: H4. Does the guarded loop cost anything, and does the streaming load hint that Triton cannot express pay?
- Change (one thing): `csrc/int4_gemv.cu` now holds one templated kernel and three exports: baseline (the EXP-016 loop), unroll (split full-trip loop, `#pragma unroll 4`, guarded tail) and stream (unroll plus `__ldcs` on the packed word).
- Prediction: agent, from H4: unrolling buys 0 to 8 percent, and streaming loads are the same idea as EXP-020 so probably nothing. No human prediction this session.
- Command: `python -m vla_cu cuda`, twice.
- Result (measured) at 8192 x 16384, 75.6 MB:

| implementation | run 1 ms | run 1 GB/s | run 2 ms | run 2 GB/s |
|---|---|---|---|---|
| Triton | 0.244 | 309.8 | 0.217 | 348.7 |
| cuda baseline | 0.476 | 159.0 | 0.265 | 285.8 |
| cuda unroll | 0.234 | 322.6 | 0.269 | 281.1 |
| cuda stream | 0.220 | 343.5 | 0.198 | 381.1 |

- Result (measured), small shapes, run 2: 1024 x 1024 Triton 0.027 against cuda baseline 0.010; 4096 x 2048 Triton 0.026 against cuda unroll 0.014. Every output agreed with Triton.
- Explanation / hypothesis: `stream` is fastest in both runs, 0.220 and 0.198, and beats Triton by 10 and 9 percent. That is the only change here with a consistent sign, and it is the same idea as EXP-020 expressed as `ld.global.cs`, a stronger hint than an eviction policy. `unroll` did not separate from `baseline`: 0.234 against 0.476 in run 1 and 0.269 against 0.265 in run 2, so H4 is not supported.
- Surprise, unresolved: the baseline reads 0.476 in run 1 where EXP-016 measured 0.240 on the same kernel body. The refactor replaced the old `if (j == 0) ... else if` chain with a shared `dot_eight` helper of eight dependent `fmaf` calls, which may have serialized the accumulator. Unverified, because counters are blocked (EXP-019). Treat the baseline column as not comparable with EXP-016.
- Verdict: worked for the streaming variant, now the fastest int4 kernel in the repo. The 381.1 GB/s sample is above the probe's best 360.2, so the probe under-sampled the ceiling and absolute cross-task comparisons need care.
- Sanitizer: not run. `compute-sanitizer` cannot instrument this extension (EXP-016), so the new variants have no memory safety pass. They agree with Triton and with the float64 reference through the test suite, which is not the same thing.

## EXP-023: the action expert at its real M  (2026-11-23, commit 97ad2f0)
- Goal / question: H9, the model-level question. SmolVLA predicts 50 actions per forward, so the expert's linear layers see 50 tokens. Is a per-row GEMV still the right kernel?
- Change (one thing): added `bench/m50.py` (task `m50`), which times 50 GEMV calls against one bf16 GEMM on the same int4 weights.
- Prediction: agent, from H9: the GEMV reads the weights 50 times and should lose badly. No human prediction this session.
- Command: `python -m vla_cu m50`
- GPU / environment: RTX 5060 Laptop GPU, M = 50, expert widths 1920 x 720 and 720 x 1920, warmup 5, 20 iterations.
- Result (measured):

| N x K | group | path | bytes | ms | GB/s |
|---|---|---|---|---|---|
| 1920 x 720 | 16 | gemv loop x50 | 52.4 MB | 2.588 | 20.2 |
| 1920 x 720 | 16 | bf16 gemm x1 | 3.4 MB | 0.018 | 192.4 |
| 720 x 1920 | 64 | gemv loop x50 | 39.4 MB | 1.649 | 23.9 |
| 720 x 1920 | 64 | bf16 gemm x1 | 3.1 MB | 0.050 | 62.9 |

- Explanation / hypothesis: the byte ratio is 15.5x and 12.7x because the GEMV re-reads every weight for every token, and the measured time ratio is 147x and 33x. Part of the gap is dispatch: 2.588 ms over 50 calls is 51.8 us per call against about 34 us of launch cost, so a graphed 50-launch version would narrow the gap without closing it.
- Side finding: 720 is not a multiple of 64, so group 64 cannot represent the expert hidden width. The benchmark uses group 16 for that shape, which puts 4 bytes of scale on every 16 weights, 25 percent scale overhead instead of 6.25. Format constraint, recorded in DEC-030.
- Verdict: worked, H9 confirmed, the largest finding of the session. Below some M the per-row GEMV is right, at M = 50 it is wrong by an order of magnitude, and the crossover is unmeasured.
- Follow-ups: an int4 GEMM that dequantizes into `tl.dot` operands, which is Marlin's design and now has a reason to exist here. Sweep M from 1 to 64 to find where the GEMV stops being the right kernel.

## EXP-024: three projections in one launch  (2026-11-23, commit 97ad2f0)
- Goal / question: H8. q, k and v share one activation vector at batch 1. Does concatenating their rows into one launch pay?
- Change (one thing): added `kernels/triton/int4_grouped_gemv.py` (a segment table plus a per-segment activation base) and `bench/grouped.py` (task `grouped`).
- Prediction: agent, from H8: three launches to one, small gain at DRAM sizes, large gain where dispatch dominates. No human prediction this session.
- Command: `python -m vla_cu grouped`
- Result (measured):

| case | separate | grouped | ratio | agree |
|---|---|---|---|---|
| 3 x 960 x 960, 1.57 MB | 0.101 ms (15.5 GB/s) | 0.037 ms (42.4 GB/s) | 2.73x | yes, max abs diff 0.0 |
| 3 x 8192 x 4096, 56.7 MB | 0.268 ms (212.1 GB/s) | 0.260 ms (217.9 GB/s) | 1.03x | yes, max abs diff 0.0 |

- Explanation / hypothesis: at 1.57 MB nothing is bandwidth bound, so the only cost is three dispatches against one, which shows as 2.73x. At 56.7 MB the weights dominate and the concatenated layout costs nothing, 1.03x, which answers the page locality worry. Outputs are bit identical because the arithmetic and the order of the reduction are unchanged.
- Verdict: worked. The cheapest structural win in the repo, and it generalizes to every layer of a transformer.
- Follow-ups: group the whole layer, q, k, v and the MLP pair, with the segment table built at load time.

## EXP-025: one persistent cooperative launch per block, not attempted  (2026-11-23)
- Goal / question: H7. Would a single cooperative kernel per MLP block beat a graph of two launches?
- Change: none. Nothing was built.
- Prediction: agent: no, and the reason is measurable rather than speculative.
- Result: not run, by decision rather than failure. EXP-018 measured a graph of the two launches at 0.018 to 0.019 ms and raw launches at 0.053 ms. The 34 us difference is Python and dispatch, which a persistent kernel removes only by also removing the Python. The launch it could still save is one graph node, 0.63 us (EXP-009), against a grid sync that costs microseconds.
- Verdict: dropped, with the reasoning above. Recorded so it is not proposed again without new information.
- Follow-ups: revisit only if a block grows enough that one kernel can keep several SMs busy across a whole layer.

## EXP-026: does a Hadamard rotation rescue group 64?  (2026-11-23, commit 97ad2f0)
- Goal / question: H11. int4 at group 64 sits near cosine 0.99. Does an orthogonal block rotation, which spreads outliers across a group, buy accuracy at zero byte cost?
- Change (one thing): added `bench/rotation.py` (task `rotate`), a 64-block Sylvester Hadamard applied to weights and activations, with outlier injection.
- Prediction: agent, from H11: no change on Gaussian weights, large improvement once outliers exist. No human prediction this session.
- Command: `python -m vla_cu rotate`
- Result (measured), cosine and relative L2 against the exact fp32 product:

| shape | weights | int4 | int4 plus rotation |
|---|---|---|---|
| 2560 x 960 | gaussian | 0.99409, 1.09e-1 | 0.99425, 1.08e-1 |
| 2560 x 960 | outliers | 0.98788, 1.56e-1 | 0.99586, 9.14e-2 |
| 1920 x 768 | gaussian | 0.99429, 1.07e-1 | 0.99410, 1.09e-1 |
| 1920 x 768 | outliers | 0.98615, 1.66e-1 | 0.99492, 1.02e-1 |

- Result (measured), cost: not timed. The transform is 2 K log K operations per row against K / 2 bytes, so under a percent at these shapes, but that is derived, not measured.
- Explanation / hypothesis: the rotation does nothing on Gaussian weights, as expected since they are already incoherent, and it repairs most of the outlier damage: cosine 0.98788 to 0.99586 and relative error down 41 percent. The mechanism is that one outlier no longer sets the scale for 63 well behaved weights.
- Verdict: worked, H11 confirmed. Whether it matters depends on real checkpoint weights, which the repo does not have yet.
- Follow-ups: run this on extracted SmolVLA weights before claiming a quality win. Fusing the rotation into the kernel is a later chunk; for now it is a pack-time change plus one small kernel.
## EXP-027: a CUDA int8 kernel, and two bugs it exposed  (2026-11-23, commit pending)
- Goal / question: the block's up projection is int8 and only existed in Triton. Could it become a native streaming kernel so that a whole block runs without the Triton launcher, and can that block then be captured in a graph?
- Change (one thing): added `csrc/int8_gemv.cu` (warp per row, one uint4 load per lane is 16 weights, `__ldcs`, guarded scalar tail for K not divisible by 512), `vla_cu/cuda_int8.py`, and `tests/test_cuda_int8_gemv.py`.
- Prediction: agent: the raw path should land near the Triton int8 time and the graph should then behave like EXP-018. No human prediction this session.
- Command: `python -m pytest -q` inside a vcvars shell, then `python -m vla_cu fastpath`.
- Result (measured), first attempt: 57 tests failed with a `torch.AcceleratorError` and the fast path's graph rows read cosine 0.83733 against 0.99468 for the eager chain, and the four-block graph was not bit exact.
- Bug 1: the 16-byte weight load requires every row base to be 16-byte aligned, and K = 1025 gives row starts at odd 16-byte boundaries. The misaligned load faulted, which poisoned the CUDA context and made every later test fail. Fixed with a host check that K is a multiple of 16, plus a test that asserts the rejection, and the awkward-width test moved to K = 1040.
- Bug 2, the interesting one: both CUDA kernels launched with `<<<blocks, threads>>>`, which is the legacy default stream. A launch on that stream is invisible to stream capture, so the graph recorded the silu kernel and nothing else. That is why the graph output looked like a stale intermediate rather than like noise. Fixed by launching on `at::cuda::getCurrentCUDAStream()` in both `.cu` files. Recorded as DEC-031.
- Result (measured), after both fixes: 68 tests pass. Block on native kernels: cuda raw 0.031 to 0.033 ms, cuda graph 0.015 ms, cosine 0.99393, and the graph reproduces the eager chain bit for bit (`torch.equal` True).
- Explanation / hypothesis: the capture bug also means the int4 kernels in EXP-016 and EXP-022 were never capturable. Their numbers stand, since they were never inside a graph, but no CUDA graph of a torch op could have worked before today.
- Verdict: worked, after two genuine defects were found and fixed. The kernel itself needed no tuning: 0.031 ms for 3.85 MB of low-bit weights is 124 GB/s, which is dispatch and not bandwidth on an L2-resident block.
- Follow-ups: the same stream rule has to be checked in every future extension, so the file header now states it.

## EXP-028: the fast path, one block and a four-block stack  (2026-11-23, commit pending)
- Goal / question: put the measured wins together, native streaming kernels plus a graph, and answer the question a model actually asks: what does a block cost, and what does a stack of blocks cost?
- Change (one thing): added `bench/fastpath.py` (task `fastpath`), five paths on the same weights and input, and a stack of four different blocks.
- Prediction: agent, from EXP-018 and EXP-022: about 0.015 to 0.020 ms per block, so 4 to 5x the fp32 block. No human prediction this session.
- Command: `python -m vla_cu fastpath`, twice.
- Result (measured), milliseconds:

| path | blocks | run 1 ms | run 2 ms | ms per block | GB/s run 1 | cos |
|---|---|---|---|---|---|---|
| fp32 | 1 | 0.069 | 0.069 | 0.069 | 284.6 | 1.00000 |
| triton raw | 1 | 0.081 | 0.083 | 0.081 | 47.7 | 0.99393 |
| cuda raw | 1 | 0.033 | 0.031 | 0.033 | 118.4 | 0.99393 |
| cuda graph | 1 | 0.015 | 0.015 | 0.015 | 249.9 | 0.99393 |
| stack graph | 4 | 0.054 | 0.134 | 0.014 to 0.033 | 284.6 | 0.9747 to 0.9766 |

- Result (measured): the stack graph reproduces the eager chain bit for bit in both runs. fp32 stack of four blocks: 0.356 and 0.364 ms, so the graph is 6.57x in run 1 and 2.72x in run 2.
- Explanation / hypothesis: fp32 was stable at 0.069 ms in both runs, which the harness now manages to do, but the four-block graph moved 2.5x between runs with no code change, which is the same instability EXP-017 measured. The one-block graph was stable at 0.015. Per block, the graph is 4.6x faster than the fp32 block and 2.2x faster than the same kernels launched eagerly, and the gap is dispatch: 4 kernels at about 8 us of launch plus Python against 4 graph nodes at 0.63 us.
- Derived, not measured: sixteen blocks of this size at the best sample would be about 0.22 ms. It will not scale like that, because sixteen blocks are 62 MB of low-bit weights against a 33.6 MB L2, so a real stack streams from DRAM and the per-block floor becomes 3.85 MB divided by the 346 to 360 GB/s attainable rate, about 0.011 ms per block.
- Verdict: worked. This is the default path for the block, and the first time a quantized stack in this repo is faster than fp32 by a wide margin at layer scale.
- Caveats: cosine through four stacked blocks is 0.975, below the 0.98 gate the project targets, on random weights with no rotation and no per-layer bit choice. That is a quality question, not a speed one.
- Follow-ups: the M = 50 GEMM (DEC-028), the rotation on real weights, and per-layer bit widths.
## EXP-029: int4 GEMM with tl.dot, and where the GEMV stops winning  (2026-11-23)
- Goal / question: the action expert sees M = 50 tokens, and 50 GEMV calls read the weights 50 times (EXP-023). Write the GEMM that dequantizes into the tensor core and find the M where it beats the GEMV.
- Change (one thing): added `kernels/triton/int4_gemm.py`. A step loads a [BLOCK_N, BLOCK_K // 2] byte tile on whole groups, splits low and high nibbles, expands the group scale to one value per weight, dequantizes in fp32, rounds once to fp16, and issues two dots against the even and odd halves of the activation tile. Added `bench/crossover.py` (task `crossover`) and `tests/test_int4_gemm.py`.
- Prediction: agent, from the GEMV loop in EXP-023 and one launch costing about 0.02 ms: the GEMM should win by M = 4 to 8. No human prediction this session.
- Command: `python -m vla_cu crossover`, and `pytest -q tests/test_int4_gemm.py`.
- Bug found on the first build: `OutOfResources: shared memory, Required: 229376, limit 101376`. A 512-wide step is two 64-wide fp16 tiles per stage, and 3 stages of that does not fit. Fixed by setting BLOCK_K = 64 and `num_stages = 2`, which is 56 KB at BLOCK_M = 64.
- Result (measured), milliseconds, M GEMV calls against one GEMM, cosine for both against the fp32 answer:

| shape | M = 1 | M = 4 | M = 16 | M = 50 | M = 64 | first GEMM win | ratio at M = 50 |
|---|---|---|---|---|---|---|---|
| 1920 x 720, group 16 | 0.029 / 0.018 | 0.095 / 0.019 | 0.517 / 0.026 | 1.627 / 0.039 | 2.136 / 0.042 | M = 1 | 41.9x |
| 720 x 1920, group 64 | 0.036 / 0.045 | 0.145 / 0.034 | 0.587 / 0.036 | 1.727 / 0.067 | 2.210 / 0.069 | M = 2 | 25.8x |
| 2560 x 960, group 64 | 0.035 / 0.030 | 0.134 / 0.030 | 0.535 / 0.032 | 1.652 / 0.059 | 2.184 / 0.061 | M = 1 | 27.8x |

- Result (measured): every cosine reads 1.00000 at five decimals on both paths, including the awkward 17 x 128 with group 16 and the fp16 rounding of the dot. 11 tests pass.
- Explanation / hypothesis: the GEMV loop grows linearly with M because it is M launches, not M times the arithmetic, so the crossing is set by dispatch and not by flops. That is why the GEMM wins at M = 1 on two of three shapes: one launch against one launch, and the GEMM reads the same bytes with fewer programs. The one place the GEMV wins is M = 1 at 720 x 1920, where the GEMM's 64-row tile wastes 63 rows of tensor core work on a matrix too small to fill the machine.
- Second finding: the GEMM time jumps from 0.026 to 0.039 ms between M = 32 and M = 50 while the work only grows 56 percent. M = 50 rounds up to a 64-row tile and doubles the padded rows, so the sweep has a step at every power of two. A real expert at M = 50 pays a 2x padding tax.
- Verdict: worked. The GEMM is the right kernel for the expert at any M at or above 2, and the answer to the question is that the GEMV only ever wins at a single token on a small matrix. EXP-023's 2.588 ms against 0.018 ms was measuring 50 launches, which this sweep now separates into its two causes.
- Caveats: the GEMM path has no CUDA port, so 0.039 ms is a Triton launch and would fall with the same treatment the GEMV got in EXP-027. Not attempted.
- Follow-ups: BLOCK_M = 48 so M = 50 pads to 1.04x instead of 2x; a CUDA GEMM; the expert's 720 width still needs a group decision (DEC-030).

## EXP-030: rotation inside the fast path, and it does not pay  (2026-11-23)
- Goal / question: EXP-026 measured a rotation lifting cosine from 0.98788 to 0.99586 on weights with outliers. Does that survive inside the fast path, through a four-block stack, and what does the transform cost there?
- Change (one thing): `vla_cu/rotate.py` now holds `hadamard`, `rotate_rows`, `rotate_vector`, shared by `bench/rotation.py` and `bench/fastpath.py`. The fast path gained `cuda rot` and `stack rot` paths that rotate weights at pack time and rotate each block's activation on the way in.
- Prediction: agent: the rotation should cost one or two microseconds per block inside a graph and should not move Gaussian cosine at all, since EXP-026's gain came from outliers. No human prediction this session.
- Command: `python -m vla_cu fastpath`, inside a vcvars shell.
- Result (measured), milliseconds and cosine:

| path | ms | ms per block | cos, one block | cos, 4 blocks |
|---|---|---|---|---|
| fp32 | 0.072 | 0.072 | 1.00000 | 1.00000 |
| cuda raw | 0.028 | 0.028 | 0.99451 | |
| cuda graph | 0.015 | 0.015 | 0.99451 | |
| cuda rot | 0.057 | 0.057 | 0.99441 | |
| stack graph | 0.051 | 0.013 | | 0.97640 |
| stack rot | 0.072 | 0.018 | | 0.97449 |

- Result (measured): both stacks are bit exact against their eager chains. The rotated stack costs 0.005 ms more per block and its cosine through four blocks is 0.97449 against 0.97640, so it is 38 percent slower and 0.002 lower on these weights.
- Explanation / hypothesis: the transform is correct, since a wrong rotation would land near 0.7 rather than 0.99441, and the difference is quantization noise on Gaussian weights where a rotation has nothing to flatten. The 0.002 does not establish a regression; it does establish no gain. The cost is pure dispatch: two fp32 matmuls of [15, 64] and [40, 64] per block, which is 0.029 ms eagerly and 0.005 ms inside a graph.
- Verdict: negative for the default path, kept as an opt-in. Rotation stays where EXP-026 put it, a quality lever for checkpoints with outliers, and the fast path does not apply it.
- Follow-ups: measure both paths on real SmolVLA weights, where the outliers the rotation exists for are present. If the gain appears there, the transform wants a fused kernel, since 2 launches per block is more than the arithmetic deserves.

## EXP-031: a real attention block, q and k and v on one activation vector  (2026-11-23)
- Goal / question: the fast path was an MLP. The layer's other half is attention, where three projections read the same activation vector, which is the case the grouped launch from EXP-024 was built for.
- Change (one thing): added `bench/attention.py` (task `attn`). At one token the softmax is the identity, so the block is v = Wv x then out = Wo v. Four paths: four torch.mv, four separate int4 launches, one grouped int4 launch for q and k and v plus one for o, and the same two launches captured in a graph.
- Prediction: agent: the grouped path should save about one launch per block, and the graph should dominate everything, as in EXP-028. No human prediction this session.
- Command: `python -m vla_cu attn`.
- Result (measured), 960 x 960 four ways at batch 1:

| path | launches | ms | GB/s | % peak | cos |
|---|---|---|---|---|---|
| fp32 | 4 | 0.052 | 283.9 | 73.9% | 1.00000 |
| separate int4 | 4 | 0.086 | 24.1 | 6.3% | 0.98804 |
| grouped int4 | 2 | 0.039 | 53.1 | 13.8% | 0.98804 |
| grouped graph | 2 | 0.011 | 194.6 | 50.7% | 0.98804 |

- Result (measured): grouped agrees with separate to fp32 rounding, the graph is bit exact against the eager grouped chain, grouped is 2.20x separate, and the graph is 8.08x separate and 3.81x fp32 in paired repeats (3.74 to 4.23).
- Explanation / hypothesis: 2.08 MB of low-bit weights per block, so 0.011 ms is 194.6 GB/s and the rest is dispatch. The grouped launch saves one dispatch out of four, which is the 2.20x only because the eager path is entirely dispatch: four launches at about 0.02 ms each against two.
- Finding worth keeping: cosine is 0.98804, below the 0.98 gate, because every matrix here is int4 and there is no int8 grouped kernel. The MLP block reaches 0.99451 because its up projection is int8. So the gate is met by the MLP and missed by attention on random Gaussian weights.
- Verdict: worked, and it is the first attention block in the repo measured end to end at batch 1.
- Follow-ups: an int8 grouped kernel, which would lift this block above the gate on the same evidence as the MLP; heads at 12 x 64 rather than one 960-wide projection.

## EXP-032: clock pinning is refused, paired alternation instead  (2026-11-23)
- Goal / question: the fast-path tables moved 2.72x to 6.57x between runs with no code change (EXP-028). Can the clocks be pinned, and if not, what does the measurement protocol need?
- Change (one thing): tried `nvidia-smi -lgc 2000`, then added `time_pair` to `vla_cu/bench_utils.py`: alternate the two calls, repeat five times, report the median ratio with its min and max. Used in `fastpath` and `attn`.
- Command: `nvidia-smi -lgc 2000`, then `python -m vla_cu fastpath`.
- Result (measured): `nvidia-smi` exits 4 with "The current user does not have permission to change clocks for GPU 00000000:01:00.0". At idle, clocks.current.sm reads 870 MHz against clocks.max.sm 3090 MHz, so a benchmark starts at 28 percent of the ceiling.
- Result (measured), with pairing: fp32 over graph through four blocks is 5.57x with a spread of 5.46 to 5.76 across five pairs, and the rotated stack is 4.00x with 3.59 to 4.71. The same comparison unpaired in the same run gave 5.60x, and in EXP-028 it gave 2.72x and 6.57x on two consecutive runs.
- Explanation / hypothesis: the spread collapses from a factor of 2.4 between runs to 5 percent within a run when both paths share a clock state, because the drift is common to the pair and cancels in the ratio. Pinning would still be better: it makes absolute milliseconds comparable across sessions, which pairing cannot do.
- Verdict: blocked on administrator rights, worked around. Absolute milliseconds in this repo are only comparable within one run; ratios with a spread are the number to trust.
- Follow-ups: pin clocks from an elevated shell once and re-run EXP-017, EXP-020, EXP-021, EXP-022 and the two-hundred-series sweeps.

## EXP-033: bug-fix pass over the wrappers and the test gate  (2026-11-23, commit 2fb487f)
- Goal / question: after the kernel work landed, which defects remain in the input checks, the reported byte counts and the test gate? Does a broad shape sweep find any wrong numbers in the kernels themselves?
- Change (one thing): an audit and fuzz, then the guards they exposed. `hadamard` rejects a non-power-of-two order, the grouped wrapper validates dtype, shape, contiguity and device, and `int4_gemv_into` rejects a non-power-of-two block size. The cuda tests skip when no host compiler is on PATH, and both loaders wrap `CalledProcessError`.
- Prediction (human's, if given; else agent's, labeled): none given. Agent prediction: the kernels are correct, the defects are in the guard rails, so no measured value should move.
- Command: `python -m pytest -q` in the default shell and inside `vcvars64.bat`, then `python -m vla_cu rotate`, `python -m vla_cu attn`, `python -m vla_cu matrix`.
- GPU / environment: RTX 5060 Laptop GPU, torch 2.8.0+cu129, Triton 3.9.0, Windows.
- Result (measured): a fuzz of 175 int4 GEMV shapes (n_out 1 to 300, K 64 to 16384) against a float64 reference, all three CUDA variants bit identical to each other, 4 split factors for int8, and 384 int4 GEMM shapes (M 1 to 129, group 16 and 64): 0 mismatches. Before the fix, `pytest -q` outside vcvars was 2 failed and 77 passed; after it, 69 passed and 23 skipped. Inside vcvars, 92 passed. `attn` still agrees with the separate launches and reproduces the graph bit for bit at cosine 0.98804. `rotate` prints the EXP-026 table unchanged. `matrix` at 8192 x 16384 reports 268.5 MB for bf16 instead of 268.6 MB, because the activation read was counted at 4 bytes per element.
- Explanation / hypothesis: the kernels were right; the failures were silent wrong shapes (`hadamard(3)` returned a 4 x 4 matrix), a wrapper that trusted every grouped input, and a gate that checked `nvcc` only. `torch.utils.cpp_extension.load` raises `subprocess.CalledProcessError` when `cl` is missing, not `RuntimeError`, which is why the friendly build message never appeared.
- Verdict: worked. No kernel arithmetic changed; the largest reporting change is 0.3 percent of the bf16 byte count.
- Follow-ups: two input gaps stay open on purpose. `int4_gemm_into` accepts a non-power-of-two `block_n` and lets Triton raise, and the grouped wrapper cannot prove that every segment length is a multiple of the row tile without a device sync.
