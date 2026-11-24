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
- CORRECTION 2026-11-23: the 0.476 ms baseline did not reproduce. EXP-040 measured 0.238 ms for baseline, 0.230 for unroll, 0.208 for stream and 0.221 for Triton in one process, so the 0.476 sample was a clock outlier and the `dot_eight` serialization hypothesis is withdrawn. The sanitizer statement is also narrower: EXP-041 got the tool to start with `--target-processes all`, and it reports a handle error rather than a memory error.

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

## EXP-034: one validator for every kernel entry point  (2026-11-23, commit 0b88c1b)
- Goal / question: do the six wrappers reject the same bad inputs, and can one module describe a valid call once?
- Change (one thing): added `vla_cu/validate.py`; every Triton and CUDA wrapper now calls it; added the checks missing after EXP-033.
- Prediction (agent's, labelled): identical results on valid inputs, every wrapper rejects a wrong device, a strided activation, a wrong dtype and a ragged group before launching, and the wrappers lose about 60 lines.
- Command: `.venv/Scripts/python.exe -m pytest -q`, then the same suite with the vcvars64 environment built by `/tmp/vc_test.py`.
- GPU / environment: RTX 5060 Laptop, torch 2.8.0+cu129, Triton 3.9.0, MSVC 14.44, CUDA 12.9.
- Result (measured): 80 passed, 23 skipped in a plain shell; 103 passed inside vcvars. Before the change: 69 and 92. Four real gaps closed: `block_n` power of two, int4 GEMM scale dtype and device, CUDA fp32-scale and alignment checks. The six wrappers went from 842 to 780 lines, the new module is 144, so the net is +82 lines.
- Explanation / hypothesis: the duplicated checks were only 62 lines, so a shared module cannot pay for itself in size. Its value is one description of each format and one exception type. The prediction on net size was wrong.
- Verdict: worked on consistency, failed on size.
- Follow-ups: a new kernel must be added to `tests/test_validation.py` in the same commit that adds it. The native suite needs the vcvars environment, which this shell cannot build directly, so `/tmp/vc_test.py` is the working form.

## EXP-035: the attention block in int8 against int4  (2026-11-23, commit 7d20477)
- Goal / question: what does one extra bit buy in attention, and what does a grouped launch buy at each precision?
- Change (one thing): added `kernels/triton/int8_grouped_gemv.py`; `bench/attention.py` now times int4 and int8 on the same weights.
- Prediction (agent's, labelled): int8 cosine at least 0.994 and int8 graphed time 1.6x to 2.2x int4, because it reads 1.78x the bytes.
- Command: `python -m vla_cu attn`
- GPU / environment: RTX 5060 Laptop, 384 GB/s spec, 33.6 MB L2, torch 2.8.0+cu129.
- Result (measured): fp32 0.049 ms; int4 separate 0.110, grouped 0.056, graph 0.012, cosine 0.98804; int8 separate 0.095, grouped 0.048, graph 0.019, cosine 0.99994. Grouped is 1.97x over separate at int4 and 2.00x at int8. The graph is 9.09x and 5.11x over separate. int8 costs 1.54x the int4 time for 1.78x the bytes and reaches 51.8 percent of peak against 44.8. Grouped agrees with separate and both graphs reproduce their eager chains bit for bit.
- Explanation / hypothesis: at 2 MB the block is launch bound, which is why grouping halves it. int8 lands under the byte ratio because its outer loop does less per byte and the wider tile keeps more loads in flight.
- Verdict: worked, both predictions held.
- Follow-ups: int8 attention is the better default when 7 us is affordable. Test what it does to a stacked block in cycle 3.

## EXP-036: the accuracy limit is the MLP down projection, not attention  (2026-11-23, commit 495df03)
- Goal / question: which projection sets the stacked-block cosine, and what does int8 everywhere cost in time?
- Change (one thing): added `vla_cu/blocks.py` and `bench/block.py` (task `block`), three precision mixes over four blocks on one seeded weight set and one seeded activation.
- Prediction (agent's, labelled): int8 everywhere holds 0.99 at depth 4; both int4-down mixes fall under the 0.98 gate.
- Command: `python -m vla_cu block`
- GPU / environment: RTX 5060 Laptop, 384 GB/s spec, torch 2.8.0+cu129, Triton 3.9.0.
- Result (measured): cosine after 1, 2, 3, 4 blocks. int4 attention with int4 down: 0.98219, 0.96186, 0.94526, 0.90671. int8 attention with int4 down: 0.99401, 0.98824, 0.98336, 0.97606. int8 everywhere: 0.99985, 0.99975, 0.99965, 0.99951. Bytes and graphed time for four blocks: 23.7 MB and 0.100 ms, 30.2 MB and 0.123 ms, 34.5 MB and 0.140 ms. One block graphed is 0.025 to 0.031 ms. fp32 is 0.551 ms for the stack, so int8 everywhere is 3.94x faster at 64.4 percent of peak. Every graph reproduces its eager chain bit for bit.
- Explanation / hypothesis: int8 attention lifts the first block from 0.982 to 0.994, so attention is not the limit once it has a byte per weight. The int4 down projection then loses 0.018 over four blocks, while int8 down loses 0.0003. The extra byte on down costs 0.017 ms for four blocks, which is 13 percent of the stack.
- Verdict: worked, both predictions held.
- Follow-ups: int8 everywhere is the quality default (DEC-039). The untested rung is int4 attention with int8 down, which would separate the two effects more cleanly.

## EXP-037: a seeded fuzz gate over every kernel  (2026-11-23, commit 495df03)
- Goal / question: does a randomized sweep find a shape the fixed tests miss, and can it run as one command?
- Change (one thing): added `bench/fuzz.py` (task `fuzz`), 32 seeded cases per kernel against a float64 reference.
- Prediction (agent's, labelled): 160 cases, zero mismatches.
- Command: `python -m vla_cu fuzz`
- GPU / environment: RTX 5060 Laptop, seed 0, torch 2.8.0+cu129.
- Result (measured): 160 cases, 0 failures. Worst absolute error by kernel: int4 GEMV 5.058e-07, int8 GEMV 2.883e-05, int4 GEMM 1.573e-03, int4 grouped 0, int8 grouped 1.144e-05.
- Surprise: int8 grouped is not bit identical to per-matrix launches, 1.1e-05 absolute and 3.1e-07 relative, while int4 grouped is exactly 0. The extra segment and x_base loads change register allocation, so the compiler reassociates the last adds. That is a float fact, not a wrong result, and the gate is 1e-4.
- Explanation / hypothesis: the fixed tests cover the shapes a person chose; the sweep covers K values that break block sizes (83, 129, 249, 3401) and row counts that break tiles.
- Verdict: worked.
- Follow-ups: run the task in CI once a GPU runner exists, and add the block objects to it.

## EXP-038: a versioned bundle for packed weights  (2026-11-23, commit 340367c)
- Goal / question: can packed weights be saved and loaded without the fp32 source, and what does that cost on disk?
- Change (one thing): added `vla_cu/bundle.py` (PackedMatrix, Bundle, pack_matrix, save_bundle, load_bundle) and `bench/bundle_bench.py` (task `bundle`).
- Prediction (agent's, labelled): a per-block int4 bundle near 4.8 MB, a bit-exact round trip, and a load under 50 ms.
- Command: `python -m vla_cu bundle`
- GPU / environment: RTX 5060 Laptop, torch 2.8.0+cu129, files in `bench/results`, best of 5 wall-clock loads.
- Result (measured): four blocks at int4 attention and int4 down: 23.71 MB, 5.81x smaller than fp32 and 2.90x than fp16, save 31.87 ms, load 5.68 ms. Four blocks at int8: 34.54 MB, 3.99x and 1.99x, save 26.24 ms, load 7.90 ms. Both archives round trip codes and scales bit exactly, and a GEMV from the loaded up matrix is bit exact against the original. Load validates the version, bit width, group size and every shape.
- Explanation / hypothesis: 23.71 MB is 5.93 MB per block against a 4.84 MB estimate, because the MLP up projection stays int8 in both mixes. The prediction was per block and the table is four blocks, so the arithmetic agrees once the up projection is counted.
- Verdict: worked.
- Follow-ups: load straight into a device buffer if load time ever matters. A real checkpoint conversion is the next user of this format.

## EXP-039: packaging and a style gate  (2026-11-23, commit 9445096)
- Goal / question: can the repo install as a package and check its own style invariants without a GPU?
- Change (one thing): added `pyproject.toml`, `.github/workflows/ci.yml` and `scripts/style_check.py` (task `style`); removed 47 comment lines from the two CUDA sources and moved their content to `docs/ARCHITECTURE.md` section 9.
- Prediction (agent's, labelled): the gate reports the CUDA headers and nothing else, the editable install succeeds, and no Python file carries a comment.
- Command: `python -m vla_cu style`, `python -m pytest -q`, `python -m compileall -q vla_cu kernels bench scripts tests`, `pip install -e . --no-deps`, `python -m vla_cu --help` from outside the repo, then the vcvars suite.
- GPU / environment: RTX 5060 Laptop, torch 2.8.0+cu129, MSVC 14.44, Python 3.13.
- Result (measured): 47 violations before the strip, 0 after. 96 passed and 23 skipped in a plain shell, 119 passed inside vcvars with both extensions rebuilt from the edited sources. `compileall` and the editable install exit 0, and `--help` from `/tmp` prints every task. The gate found no Python comments, so the CUDA headers were the only offenders.
- Explanation / hypothesis: the CUDA files predate the no-comments invariant, and their mapping and access notes are documentation, so `docs/` is where they belong. The CI job checks style and byte-compiles only, because a GPU runner does not exist here.
- Verdict: worked.
- Follow-ups: add the GPU suites to CI when a runner exists; keep the gate at zero so a new comment fails the push.

## EXP-040: the CUDA baseline regression does not reproduce  (2026-11-23, docs from commit a11e321)
- Goal / question: explain the 0.240 to 0.265 to 0.476 ms baseline drift recorded in EXP-022.
- Change (one thing): none. Re-ran the existing task in one process after the extension rebuilt.
- Prediction (agent's, labelled): if the drift was code, the baseline is still near 0.47 ms; if it was clocks, it is near 0.24 ms and the variants keep their usual order.
- Command: `python -m vla_cu cuda` inside the vcvars environment.
- GPU / environment: RTX 5060 Laptop, 384 GB/s spec, MSVC 14.44, CUDA 12.9.
- Result (measured): 8192 x 16384: Triton 0.221 ms, baseline 0.238, unroll 0.230, stream 0.208. 4096 x 2048: 0.026, 0.015, 0.014, 0.014. 1024 x 1024: 0.012, 0.013, 0.012, 0.012. Stream reaches 363.5 GB/s, 94.7 percent of peak, and is the fastest variant again.
- Explanation / hypothesis: the 0.476 reading was a power and clock outlier on this 45 to 100 W part, not a code regression. The variants rank as they did in EXP-016 and EXP-022.
- Verdict: resolved, no code change.
- Follow-ups: quote ranges, not single samples (DEC-029). CORRECTION added to EXP-022.

## EXP-041: compute-sanitizer reaches the extension and fails on the kernel handle  (2026-11-23, docs from commit a11e321)
- Goal / question: get a memcheck pass over the hand-written CUDA kernels after EXP-016 could not.
- Change (one thing): none. Ran the tool with `--target-processes all`, then with eager module loading and blocking launches.
- Prediction (agent's, labelled): the handle error would disappear if lazy module loading caused it.
- Command: `compute-sanitizer --tool memcheck --target-processes all python /tmp/check_cuda.py` from a vcvars environment, over both extensions and six shapes.
- GPU / environment: RTX 5060 Laptop, RTX 5060, CUDA 12.9 sanitizer 2025.2.1.
- Result (measured): the plain run completes, prints correct results, and reports two errors, both `CUDA_ERROR_INVALID_HANDLE` on `cuKernelGetFunction`, one from `vla_cu_int4_gemv.pyd` and one from `vla_cu_int8_gemv.pyd`. With `CUDA_MODULE_LOADING=EAGER` and `CUDA_LAUNCH_BLOCKING=1` the run did not finish inside 600 seconds.
- Explanation / hypothesis: the tool instruments the API but cannot resolve the kernel handle torch's launcher asks for, so no kernel-level pass happens. No memory error was reported, which is not the same as a clean pass.
- Verdict: blocked, with a narrower cause than EXP-016.
- Follow-ups: retry from an elevated shell; a standalone CUDA harness without torch's launcher is the fallback.

## EXP-042: clocks pinned, and the locked rerun of the int4 table  (2026-11-23, commit 9c9da03)
- Goal / question: does an elevated `nvidia-smi -lgc 2000` hold on this laptop part, and how much of the 2x spread in EXP-017 was clock state?
- Change (one thing): none. The human ran `nvidia-smi -lgc 2000` from an elevated PowerShell; the harness and the benchmark are unchanged.
- Prediction (agent's, labelled): the lock holds within 3 percent under load, and the locked table moves by less than the 6 percent floor of DEC-029 between repeats.
- Command: `nvidia-smi -lgc 2000` (elevated), then `python -m vla_cu cuda` inside vcvars; clocks sampled every 2 s during a 14 s fp16 matmul loop with `nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader`.
- GPU / environment: RTX 5060 Laptop, 384 GB/s spec, driver 617.42, torch 2.8.0+cu129, MSVC 14.44.
- Result (measured): the SM clock held 1980 to 1987 MHz through the load, a 0.35 percent spread, throttle reason 0x4 (software power cap) at 64.7 to 76.4 W, and 1987 MHz at idle. One 5 s sample read 1875 MHz before the load ramped. Locked `cuda` table at 8192 x 16384: Triton 0.252 ms (299.9 GB/s, 78.1 percent), baseline 0.274 (275.6, 71.8), unroll 0.237 (319.1, 83.1), stream 0.218 (347.2, 90.4). At 4096 x 2048 stream 0.020 ms, 236.0 GB/s, 61.4 percent. At 1024 x 1024 unroll 0.019 ms.
- Explanation / hypothesis: the lock removes the boost algorithm, so the remaining variation is the power cap at work. The part sits near 70 W of a 115 W maximum and cannot raise its clock under a sustained matmul. Absolute numbers are lower than the unlocked 0.208 ms stream reading in EXP-040 because that sample ran near a 2400 MHz boost clock that the lock forbids.
- Verdict: worked. Clock pinning is available on this machine, so DEC-035's paired alternation is no longer the only method.
- Follow-ups: quote the lock in every future table, rerun EXP-020, EXP-021 and EXP-032 under it, and reset with `nvidia-smi -rgc` when the session ends.

## EXP-043: int4 and int8 accuracy on the real checkpoint  (2026-11-23, commit caba912)
- Goal / question: do the synthetic Gaussian results survive real SmolVLA weights, and what happens to group 64 when K is 720 (DEC-030, DEC-034)?
- Change (one thing): added `vla_cu/checkpoint.py`, a stdlib safetensors reader, and the `real` task.
- Prediction (agent's, labelled): int8 stays above 0.999, int4 group 64 lands near 0.99 on real weights, and group 16 buys less than the synthetic outlier test suggested.
- Command: `python -m vla_cu real`
- GPU / environment: RTX 5060 Laptop with clocks pinned near 1985 MHz, torch 2.8.0+cu129, checkpoint `models/smolvla_base/model.safetensors` at 906,712,520 bytes.
- Result (measured): int8 cosine 0.99995 to 0.99998. int4 group 64 0.99218 to 0.99504. int4 group 16 0.99604 to 0.99697 at 1.33x the int4 bytes. Padding K 720 to 768 reads 0.99489 at 0.88 MB and 0.035 ms against group 16 at 0.99668, 1.11 MB and 0.036 ms. Rotation on top of group 64 changes 0.99218 to 0.99455 on vlm down, 0.99383 to 0.99402 on vlm gate, and 0.99459 to 0.99384 on expert down, so the sign is not consistent.
- Explanation / hypothesis: real weights are close to Gaussian inside a group, so the 16 levels of int4 carry almost all of the error and only a finer group moves it. Rotation pays only where a group holds a planted outlier, which these weights do not. int4 is slower than int8 at these sizes because 4 to 9 us per call is launch bound, not bandwidth bound.
- Verdict: worked, and it settles two open decisions.
- Follow-ups: run a three layer stack rather than one matrix at a time, then compare the same layers at M = 50.

## EXP-044: compute-sanitizer passes on a standalone int4 harness  (2026-11-23, commit 9c9da03)
- Goal / question: can the int4 kernel get a kernel-level memcheck with torch's launcher out of the process?
- Change (one thing): moved the kernel into `csrc/int4_gemv_kernel.cuh`, included by both the extension and `csrc/standalone/int4_check.cu`.
- Prediction (agent's, labelled): zero errors and agreement with a host reference, because the same kernel passes every torch test.
- Command: `nvcc -std=c++17 -O3 -arch=sm_120 -o bench/results/int4_check.exe csrc/standalone/int4_check.cu` inside vcvars, then `compute-sanitizer --tool memcheck --leak-check full bench/results/int4_check.exe`.
- GPU / environment: RTX 5060 Laptop, CUDA 12.9. The executable is at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\compute-sanitizer\compute-sanitizer.exe`; `bin\compute-sanitizer.bat` is only a wrapper and neither is on PATH.
- Result (measured): 512 x 1024, three variants, `ERROR SUMMARY: 0 errors`, `LEAK SUMMARY: 0 bytes leaked in 0 allocations`, worst absolute error 1.788e-07 against a host reference. The extension still builds and the vcvars suite reads 119 passed after the refactor.
- Explanation / hypothesis: the `CUDA_ERROR_INVALID_HANDLE` of EXP-041 came from the tool instrumenting torch's `cuKernelGetFunction` call, not from the kernel, so removing torch from the process restores normal memcheck.
- Verdict: worked, EXP-041 is closed for the int4 kernel.
- Follow-ups: give the int8 kernel the same header split and harness, then put both sanitizer commands in the chunk checklist.

## EXP-045: iterative scaling across shapes, group size and stack depth  (2026-11-23, task `scale`)
- Goal / question: where does the int4 GEMV stop improving, which group size earns its bytes, and what does depth cost per token?
- Change (one thing): added `bench/scale.py` and the `scale` task. No kernel changed.
- Prediction (agent's, labelled): throughput plateaus above 8192 x 16384, a finer group buys accuracy at a byte cost, and per-block time falls with depth as launches amortize.
- Command: `python -m vla_cu scale`
- GPU / environment: RTX 5060 Laptop, clocks pinned near 1985 MHz, torch 2.8.0+cu129, Triton 3.9.0, 4 GB working cap per shape.
- Result (measured): stage 1 plateaus at 74.6 to 75.4 percent of spec from 75.5 MB upward: 289.4 GB/s at 8192 x 16384 and 286.5 GB/s at 302 MB. Below that, 4.7 MB reads 217.2 GB/s and 0.6 MB reads 25.3 GB/s at 0.023 ms, which is launch bound. Stage 2 on vlm.0.mlp.down, groups 64 to 8: 1.38, 1.54, 1.84, 2.46 MB for cosine 0.99218, 0.99530, 0.99604, 0.99716. Stage 3 on the real expert stack: 92.49 us per block at depth 1, falling to 76.25 at depth 16, cosine 0.99460 at depth 1, 0.98159 at 4, 0.96753 at 8, 0.31792 at 16.
- Explanation / hypothesis: the Triton kernel tops out near 75 percent, about 15 points below the native streaming kernel's 90.4 percent in EXP-042, so the plateau belongs to the implementation and not the part. Group size trades bytes almost linearly for cosine. The depth row mixes two effects: launches amortize, and quantization error compounds because the chain has no norms or residuals.
- Verdict: works for stages 1 and 2. Stage 3 measures projection cost only; the depth 16 collapse comes from the norm-free chain, not from the kernel.
- Follow-ups: sweep the native kernel over the same grid and compare plateaus; add a fused RMSNorm before quoting accuracy at depth; quote cosine at depth 8 or below.

## EXP-046: the int8 kernel gets a torch-free header and its own sanitizer pass  (2026-11-23)
- Goal / question: close the last sanitizer gap, since the int4 kernel had a harness and int8 only had API-level checks.
- Change (one thing): moved the kernel into `csrc/int8_gemv_kernel.cuh` and added `csrc/standalone/int8_check.cu`. No arithmetic changed.
- Prediction (agent's, labelled): the same treatment as int4, so a clean run on the first try with the host reference within 4e-06.
- Command: `nvcc -std=c++17 -O3 -arch=sm_120 -o bench\results\int8_check.exe csrc\standalone\int8_check.cu` inside vcvars, then `compute-sanitizer --tool memcheck --leak-check full bench/results/int8_check.exe`.
- GPU / environment: RTX 5060 Laptop, CUDA 12.9.2, compute-sanitizer 2025.2.1.
- Result (measured): 511 x 1040, which is 2 full steps of 512 weights plus a 16 wide tail, and 511 rows so the warp guard runs. Kernel: no error. Worst abs error 3.815e-06 at reference 23.99. `LEAK SUMMARY: 0 bytes leaked in 0 allocations`. `ERROR SUMMARY: 0 errors`.
- Explanation / hypothesis: the tail loop and the row guard are where the int8 kernel departs from int4, and both now run under memcheck.
- Verdict: worked. Both CUDA kernels have a standalone sanitizer pass (DEC-046).
- Follow-ups: none. The harness builds with the same vcvars recipe as int4.

## EXP-047: native plateau against Triton on the scaling grid  (2026-11-23, task `plateau`)
- Goal / question: does the native kernel keep its lead at every shape, or only at 8192 x 16384?
- Change (one thing): added `bench/plateau.py` and the `plateau` task. No kernel changed.
- Prediction (agent's, labelled): the native variants lead everywhere above the launch-bound sizes, and the gap stays near 15 points.
- Command: `python -m vla_cu plateau` inside vcvars, median of 5 repeats of 30 iterations, clocks pinned.
- GPU / environment: RTX 5060 Laptop, peak 384 GB/s, L2 33.6 MB.
- Result (measured), percent of spec: 2048 x 4096 Triton 50.3, baseline 65.7, unroll 66.0, stream 65.1. 4096 x 8192 Triton 69.4, baseline 87.6, unroll 88.0, stream 84.2. 8192 x 16384 Triton 67.9, baseline 74.7, unroll 75.5, stream 80.0. 16384 x 16384 Triton 67.7, baseline 74.7, unroll 79.1, stream 79.3. 16384 x 32768 Triton 63.3, baseline 69.4, unroll 76.5, stream 75.8, with a 0.9 to 1.4 ms range across the five repeats. Agreement with Triton: worst max abs diff 5.72e-06 at the largest shape, Triton relative L2 4.12e-07 at 8192 x 16384.
- Explanation / hypothesis: the gap is real and about 18 points at the plateau, 88.0 against 69.4 at 4096 x 8192. Below 5 MB both are launch bound. Above 151 MB the five-repeat range grows past 1 ms, so those rows describe a throttled part, not the kernel.
- CORRECTION to EXP-042: its 90.4 percent at 8192 x 16384 was one 20-iteration window. As a median the same shape reads 80.0 percent for stream, and the best repeated number anywhere is 88.0 percent at 4096 x 8192. Use 88 percent, not 90.4, as the native plateau.
- Verdict: worked. The native kernel is 1.2 to 2.2 times Triton from 1024 x 1024 upward and leads at every size above 5 MB.
- Follow-ups: compare SASS at 4096 x 8192 to name what Triton emits differently; re-run 16384 x 32768 on a cooler part before quoting those two rows.

## EXP-048: the expert stack with the layer's own norm and residual  (2026-11-23, task `scale`)
- Goal / question: can the 16 deep expert chain carry an accuracy number, given EXP-045 read cosine 0.31792 at depth 16?
- Change (one thing): added `kernels/triton/rmsnorm.py` and a second chain in stage 3 that runs norm, MLP, residual per layer. No quantized kernel changed.
- Prediction (agent's, labelled): the bare chain collapses because its activation scale drifts, and the normed chain holds above 0.99 at every depth at the cost of two launches per layer.
- Command: `python -m vla_cu scale`, clocks pinned.
- GPU / environment: RTX 5060 Laptop, the 16 real expert MLPs, norms read from `post_attention_layernorm.weight` [720] bf16.
- Result (measured): cosine bare against normed, depth 1 0.99460 and 0.99966, depth 2 0.98999 and 0.99930, depth 4 0.98159 and 0.99858, depth 8 0.96753 and 0.99722, depth 16 0.31792 and 0.99548. Time bare against normed: 0.090 and 0.105 ms at depth 1, 0.448 and 0.832 at depth 8, 0.850 and 1.534 at depth 16, over 37.0 MB of weights.
- Explanation / hypothesis: the bare chain applies down(silu(up(x))) to a vector whose scale shrinks each layer, so by depth 16 the signal is below the quantization error. Norm plus residual holds the scale, and cosine then tracks the weights. The extra 0.684 ms over 16 layers is 43 us per layer, which is about two Python launches at 16.7 us each (EXP-009) plus the norm kernel, so dispatch, not bytes.
- Verdict: worked. Depth 16 now carries cosine 0.99548. It is a projection chain plus the layer's norm and residual, not attention.
- Follow-ups: add attention to the chain; graph the normed chain to remove the 43 us per layer of dispatch.

## EXP-045 (duplicate header, no body; the entry above it is the record)

## EXP-049: a full expert layer in the chain, and the chain replayed as a graph  (2026-11-23, commit 4c3fb80)
- Goal / question: does cosine at depth 16 describe a layer rather than a projection pair, and how much of the layer cost is Python?
- Change (one thing): added `GqaValueBlock` to `vla_cu/blocks.py` and rewrote stage 3 of `bench/scale.py` to run three chains: MLP projections, full layer, graphed full layer.
- Prediction: none filed by the human. The earlier measurement said the extra launches cost 43 us per layer (EXP-048), so the graph was expected to remove most of that.
- Command: `python -m vla_cu scale`, clocks pinned near 1985 MHz.
- GPU / environment: RTX 5060 Laptop, 384 GB/s spec, the 16 real expert layers, layer 1 and every odd layer are cross attention with `v_proj` [320, 320].
- Result (measured), depth, MB, proj ms, layer ms, graph ms, cos proj, cos layer, graph exact: 1, 3.24, 0.072, 0.230, 0.029, 0.99501, 0.99973, yes. 2, 6.35, 0.129, 0.435, 0.057, 0.98769, 0.99953, yes. 4, 12.70, 0.265, 0.870, 0.105, 0.97800, 0.99912, yes. 8, 25.39, 0.477, 1.638, 0.213, 0.95739, 0.99846, yes. 16, 50.78, 0.967, 3.475, 0.497, 0.29974, 0.99795, yes.
- Explanation / hypothesis: the graph is exact at every depth, so it measures the same work. At depth 16 the eager layer costs 217 us and the graphed layer 31 us, so 186 us of 217 is dispatch, 86 percent. The layer cosine holds at 0.99795 because the norm and the residual keep the vector on the reference's scale; the projection chain still collapses to 0.29974.
- Caveat: at one visible token the softmax is 1, so q and k cancel out of the output and are not computed, and a cross-attention layer reads a fixed random prefix [320] rather than the VLM stream (DEC-050). The cosine covers the value path, o_proj, the MLP and both norms.
- Verdict: worked. The chain is a layer now, and its cost is dispatch bound until it is graphed.
- Follow-ups: graph the whole model path; add q and k so a two token softmax makes attention real.

## EXP-050: the SASS of both int4 GEMV kernels at 4096 x 8192  (2026-11-23, commit 83f0b3b)
- Goal / question: what does Triton emit that costs it 18.6 points of bandwidth against the native kernel (EXP-047)?
- Change (one thing): added `bench/sass.py` and the `sass` task. No kernel changed.
- Prediction: none filed by the human.
- Command: `python -m vla_cu sass` inside vcvars, clocks pinned. Listings in `bench/results/sass_triton.txt` and `bench/results/sass_cuda.txt`.
- GPU / environment: RTX 5060 Laptop, Triton 3.9.0, CUDA 12.9 `cuobjdump`, native extension built as `vla_cu_int4_gemv.pyd`.
- Result (measured), one function, static counts: Triton `int4_gemv_kernel` 416 instructions, 12 LDG and none of them 16 bytes (8 `LDG.E.EL` activation loads, 2 `LDG.E.EF.64`, 2 `LDG.E.EF`), plus 2 `STSM.16.M88.4`, 2 `STS.64` and 4 `LDS.128`, 12 `SHFL.BFLY`, 18 FFMA, 51 LOP3.LUT, 37 IMAD. Native unroll 544 instructions, 32 LDG of which 16 are `LDG.E.128`, 0 shared memory ops, 72 FFMA, 57 LOP3.LUT, 81 SHF, 63 IADD3, 13 BRA, and 5 `SHFL.DOWN` at the end. Native baseline 344 instructions, 20 LDG, 10 of them `LDG.E.128`. Stream matches unroll in count and differs only in the evict-first load qualifier.
- Explanation / hypothesis: the arithmetic is comparable (51 against 57 LOP3 per function), so the gap is the memory path. Triton reshapes its [4, 16, 32] tile through shared memory to reach the layout `tl.sum` wants, and its weight loads are 4 and 8 bytes wide. The native kernel keeps one 32-bit word per lane in registers, loads 16 bytes at a time through the read-only path (`LDG.E.*.CONSTANT`), and reduces once with 5 shuffles at the end.
- Caveat: these are static counts per function, and one Triton body covers 4 rows x 1024 columns while one native unroll body covers 1024 columns of one row, so the two instruction totals are not per-weight comparable. Counters are still needed to name the stall.
- Verdict: worked as an explanation, not as proof. The code shape difference is now named and reproducible in one command.
- Follow-ups: read the two loop bodies line by line; try a 16 byte load and no shared memory in a Triton rewrite.
- CORRECTION (2026-11-23): the line above that says the native kernel loads weights with `LDG.E.128` is wrong. `csrc/int4_gemv_kernel.cuh` loads one 4-byte word per lane for the 8 weights, and the 16 `LDG.E.128` are the two `float4` activation loads per lane per step. The counts stand, the label does not. The real native advantage is no shared memory, no barrier and one five-step shuffle at the end; Triton stages a 4-row tile through SMEM with 4 barriers and 12 butterfly shuffles. Nothing in EXP-052 depended on the wrong label.

## EXP-051: the expert stack at two tokens, with q, k and a real softmax  (2026-11-23)
- Goal / question: at one token the softmax is 1 and q and k cancel, so does the chain still hold its cosine once the tokens mix? What does a layer cost when q and k have to be computed?
- Change (one thing): added `gqa_mix` and `GqaTokenBlock` to `vla_cu/blocks.py` and a stage 4 to `bench/scale.py`. Every projection is one GEMV per token, so a layer issues eight launches.
- Prediction: none filed by the human.
- Command: `python -m vla_cu scale`, clocks pinned near 1985 MHz.
- GPU / environment: RTX 5060 Laptop, 384 GB/s spec, 16 real expert layers, two tokens, int8 q/k/v/o and int8 up, int4 down, fp32 reference with the same weights.
- Result (measured), depth, MB, ms, cos, rel L2: 1, 4.16, 0.980, 0.99970, 2.45e-02. 2, 8.07, 1.939, 0.99944, 3.34e-02. 4, 16.15, 3.855, 0.99880, 4.90e-02. 8, 32.29, 7.666, 0.99798, 6.38e-02. 16, 64.59, 15.450, 0.99622, 8.76e-02.
- Explanation / hypothesis: cosine falls from 0.99795 at one token to 0.99622 at two, because the second token adds a second quantized contribution to every residual. A layer costs 0.97 ms against 0.27 ms at one token, which is 3.6x for twice the GEMV work: the extra cost is dispatch at 8 launches per layer, the same 186 us per launch that EXP-049 graphed away.
- Caveat: no rotary rotation is applied on either side, so the scores carry content but not position (DEC-052). Cross layers read a fixed random prefix of two tokens, not the VLM stream. This stage is eager only, no graph.
- Verdict: worked. The chain now prices real attention, and the cosine is stable to 16 layers.
- Follow-ups: apply RoPE to q and k; graph this stage to remove the dispatch the table is now dominated by.

## EXP-052: a wide-load Triton int4 GEMV, and why it did not reach 88 percent  (2026-11-23)
- Goal / question: if the 18 point gap is Triton staging a tile through shared memory with 4 and 8 byte loads (EXP-050), can reading the packed row as int32 close it?
- Change (one thing): added `kernels/triton/int4_gemv_wide.py`, a second Triton mapping, plus a `wide` row in `plateau` and a second listing in `sass`.
- Prediction: none filed by the human. The hypothesis was that 16 byte loads plus no shared memory would land near the native 88 percent.
- Command: `python -m vla_cu plateau` inside vcvars, median of 5 repeats of 30 iterations, clocks pinned; `python -m vla_cu sass` for the listing.
- GPU / environment: RTX 5060 Laptop, 384 GB/s spec, Triton 3.9.0, 4096 x 8192, group 64.
- Result (measured), 4096 x 8192: staged Triton 0.067 ms, 280.9 GB/s, 73.1 percent. wide Triton 0.088 ms, 216.1 GB/s, 56.3 percent, 0.77x of staged, max abs diff 1.43e-06 against the staged output. native baseline 87.7, unroll 87.9, stream 87.5 percent. Best shape per implementation: staged 73.1 at 4096 x 8192, wide 56.3 at the same shape, native unroll 87.9.
- Result, the sweep behind that number: block 1024 and 1 warp gave 53.9 percent, 2048 and 2 warps 50.5, 4096 and 1 warp 39.8 with 254 registers and 10 spills. The shipped defaults are block 1024 and 1 warp.
- Explanation / hypothesis: the first version shaped the weight tile as [groups, words per group], and Triton then gave each lane one word: the listing shows 8 x `LDG.E.EF` and 2 x `LDG.E.EF.64` for weights, while only the activation stream reached `LDG.E.EL.128`. The second version loads one flat run of words, so a lane owns four consecutive words, and it still reports 18 LDG with 0 `LDG.E.128` in the listing: the reshape into [groups, words] puts the layout conversion back. Both versions allocate shared memory, 512 to 2048 bytes, with 4 `BAR.SYNC`, for the cross-lane reduction, so the no shared memory half of the hypothesis was not reachable with this mapping either.
- Caveat: a static count of 0 `LDG.E.128` is read from the wide listing; the compiler was not asked why. The instruction totals are not per-weight comparable, the staged body covers 4 rows x 1024 columns.
- Sanitizer: `compute-sanitizer --tool memcheck` over the int4 tests, 0 errors, 35 passed and 17 skipped. Run from `bench/results/sanitize_wide.bat`, because the toolkit ships the tool as a .bat and the path has spaces.
- Verdict: negative. Reading the row as int32 did not buy the 16 byte weight load, and the wide mapping is slower than the staged one at every shape. The gap to 88 percent is not closed by changing the element type Triton loads.
- Follow-ups: force per-lane contiguity with `tl.multiple_of` and `tl.max_contiguous` hints. Note the corrected premise in EXP-050: the native kernel reads weights 4 bytes at a time too, so matching its weight load width is not the goal, matching its reduction is.

