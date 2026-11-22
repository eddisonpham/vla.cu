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
