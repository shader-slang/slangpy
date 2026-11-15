# Explicit CUDA CUBIN compilation and caching

This ExecPlan follows .agents/PLANS.md and extends the implemented pending-pipeline experiment documented in docs/pending-pipelines-experiment.md. Work takes place in C:\projects\slangpy.

## Purpose

Move PTX-to-machine-code compilation out of CUDA module loading and into RHI's existing parallel shader workers. PTX is the intermediate assembly emitted by Slang/NVRTC; CUBIN is the target GPU binary. Cache CUBIN separately from PTX through the existing persistentPipelineCache interface. Measure cold compilation and fresh-process warm-cache reuse, without relying on cumulative Slang compiler timings.

## Progress

- [x] Inspected the virtual compileEntryPoint hook, serial artifact installation, and concurrent persistent cache contract.
- [x] Add an optional nvPTXCompiler build dependency and an opt-in CUDA device mode.
- [x] Implement CUBIN conversion, separate cache keys, diagnostics, and per-job wall traces.
- [x] Build the extension and validate the small workload in all four dispatch modes.
- [x] Validate fresh-process cache reuse, option/PTX invalidation, and the unchanged PTX path in Python; all 15 prepare/pending regression tests pass.
- [x] Build without the optional dependency and pass native pending tests (595 assertions; optional test skipped).
- [x] Build and validate outputs, cache reuse/invalidation, invalid cache handling, and the disabled path.
- [x] Run repeated serial/parallel cold and warm measurements and contemporary driver-JIT controls; validate all 24 samples against each other.
- [x] Finish final pre-commit passes in both SlangPy and slang-rhi, including untracked files; git diff --check is clean.

## Design and decisions

Keep the public API unchanged for this experiment. An explicit SLANG_RHI_NVPTXCOMPILER_ROOT CMake path enables linking NVIDIA's static compiler; the default build requires no additional dependency. SLANG_RHI_CUDA_CUBIN=1 opts a CUDA device into the experiment. Initialize configuration and hardware target before workers start. Only compute entry points use CUBIN; OptiX retains PTX.

Override cuda::ShaderProgramImpl::compileEntryPoint in external/slang-rhi/src/cuda/cuda-shader-program.cpp. Invoke the base implementation to get cached or newly generated PTX, then use an independent nvPTXCompiler handle to convert it. Store the binary in the existing owned entry-point blob. Installation remains serial; CUDA module loading recognizes the resulting ELF binary. Keep both PTX and CUBIN owned until validation/loading succeeds if a cached binary needs rebuilding.

Use persistentPipelineCache for CUBIN records, keyed by a versioned namespace, exact PTX bytes, target SM, effective compiler options, and compiler-library build fingerprint. Values have a small versioned header and checksum so malformed entries become misses. Writes remain best-effort. No cache lock may span compilation. Both cache slots already share SGL's concurrent LMDB-backed storage. A bounded asynchronous writer can delay visibility; close/flush before warm-process checks.

Use an opt-in trace to report independent PTX and CUBIN wall intervals and cache hit status, collected after worker completion where practical. Preserve existing PTX report meanings; CUBIN diagnostics and cache hits must not be confused with PTX cache hits. Instrument total artifact preparation separately from actual pipeline loading. The benchmark must verify matching outputs, compiler identity, and counts of CUBIN hits/compiles.

## Milestones and validation

First add the optional compiler integration in CUDA-specific files and CMake, leaving default builds functional. Download CUDA 12.9.86's nvPTXCompiler header/library into .temp/cuda-nvcc-12.9 and verify NVIDIA's published archive SHA256. Configure with cmake --preset windows-msvc -DSLANG_RHI_NVPTXCOMPILER_ROOT=C:/projects/slangpy/.temp/cuda-nvcc-12.9 and build slangpy_ext outside the sandbox as required by RHI instructions.

Second add subprocess-based tests under slangpy/tests/device/test_cubin_cache.py. Validate serial and parallel creation, fresh-process cache hits, option-dependent misses, and unchanged output. Inject truncated records, damaged checksums, and a checksummed ELF image with an invalid machine type through a test-owned IPersistentCache in external/slang-rhi/tests/test-parallel-pipeline-creation.cpp. Verify successful rebuilding, subsequent cache hits, and correct dispatch results. Keep OptiX behavior unchanged and test the ordinary PTX path. Follow the repository rule to build before running tests.

Finally extend tools/benchmark_pipeline_compilation.py to select CUBIN mode, record trace events, and use an optional persistent shader-cache directory. Measure 24 kernels (eight variants each of signal/network/orbit, 64 unrolled iterations) in serial and pending_dispatch modes, with NVRTC and driver caches disabled. Compare against existing raw results and a contemporary PTX baseline where useful. Separately populate an isolated persistent cache and measure fresh-process hits with driver caching still disabled. Cache-disabled runs establish parallel compilation performance; warm runs establish persistent-cache effectiveness.

## Recovery and limitations

Preserve all existing pending-pipeline changes and reports. No commits, system CUDA replacements, or changes to the user's persistent environment are required. Each benchmark uses a child process. Cache paths live under .temp and are unique per experiment. Compiler failure must surface diagnostics and must not silently fall back to PTX during timing. Additional dependencies remain optional. This is an experiment, not yet a finalized public CUDA configuration API.

## Surprises and discoveries

The installed CUDA 12.2 PTX compiler is too old to reliably consume PTX produced by NVRTC 12.9, so use a matching standalone compiler package.

NVIDIA's Windows archive contains only an /MT release library. Directly linking it into SlangPy's /MDd debug DLL fails with CRT and iterator-debug-level mismatches. The optional CMake integration therefore builds a small Windows DLL exporting the NVIDIA C API from that archive, with its own /MT runtime. The measured DLL is approximately 47 MB. This experiment places it in the build's runtime directory; production distribution/install rules remain to be designed.

The downloaded archive is cuda_nvcc-windows-x86_64-12.9.86-archive.zip from NVIDIA's CUDA 12.9.1 redistributable manifest. Its verified SHA256 is 227b109663b5e57d2718bcabb24a4ba0d9d4e52d958e327dc476f7c28691be85. The library fingerprint recorded in every trace is f969132aad649bcfc49c3a62a17872649afa11b03cadcc40276f7b283ffec9d9.

The official archive URL is https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.9.86-archive.zip. Extract include/nvPTXCompiler.h, lib/x64/nvptxcompiler_static.lib, and LICENSE from its top-level directory into .temp/cuda-nvcc-12.9. The optional CMake root points to this directory. The API reference is https://docs.nvidia.com/cuda/archive/12.9.1/ptx-compiler-api/index.html.

The first 24-kernel parallel sample took 22.092 seconds with all caches disabled, motivating the repeated measurements and contemporary driver-JIT controls below. The small three-kernel smoke run passed serial, command-list, pending-only, and pending-with-dispatch modes.

Three completed cold CUBIN runs give median 107.225 seconds serial (104.183-107.234) and 22.319 seconds parallel (22.092-22.639), a 4.80-times speedup. All outputs exactly match the previous driver-JIT results. PTX generation and CUBIN conversion overlap between jobs; their individual wall envelopes must not be added. CUDA module-load/pipeline creation spans only 0.063 seconds in the first parallel sample, compared with tens of seconds when loading PTX. This demonstrates that the expensive conversion work moved out of the driver-loading phase.

The persistent-cache population pass took 21.804 seconds. Fresh-process warm-cache medians across three repeats are 0.736 seconds serial, 0.672 seconds command-list, 0.662 seconds prepare/flush only, and 0.675 seconds prepare/flush/dispatch. Every sample hit all 24 PTX and all 24 CUBIN entries. For pending_dispatch, median preparation is 0.594 seconds, flush 0.066 seconds, and dispatch/wait 0.015 seconds. Driver caching and NVRTC caching remain disabled. Setup (device, module, input/output tensors) is outside the timing interval and takes approximately 1.9 seconds in these warm runs.

The contemporary cache-disabled driver-JIT parallel control has median 67.665 seconds (63.682-68.302, three samples), versus 22.319 seconds for explicit CUBIN: a 3.03-times reduction in total parallel startup time. Its median PTX-generation wall envelope is 8.786 seconds and its backend pipeline envelope is 58.239 seconds. With CUBIN, the backend envelope is 0.063 seconds (0.060-0.069). Standalone compiler work runs in the earlier worker phase; the PTX and CUBIN intervals overlap across jobs. The difference is not an artifact of summing overlapping compiler reports.

The contemporary driver-JIT serial control took 98.928 seconds (one sample, not a median), compared with the cold CUBIN serial median of 107.225 seconds. Thus this experiment improves parallel startup, not serial compiler throughput. Driver-JIT serial/parallel scaling is approximately 1.46 times using that serial control, compared with 4.80 times for standalone CUBIN. Different compiler implementations/versions can contribute to the serial cost; the driver implementation is not identical to nvPTXCompiler 12.9.

The PTX-only warm control hit all 24 persistent PTX entries but still took 55.148 seconds with driver caching disabled (one exploratory sample). The warm PTX+CUBIN cache median is 0.675 seconds. This confirms that caching PTX alone leaves substantial machine-code compilation work, and that the CUBIN cache avoids it. All 24 full-workload samples in this experiment have exactly equal outputs; all traced intervals fit within their independently measured elapsed intervals. Summary data is in reports/cubin-experiment-summary.json, with the raw per-sample traces in the referenced report files.

The C++ test's SKIP macro returns from the current function rather than throwing. A skip inside the batch helper lambda therefore did not exit the enclosing test; the opt-in check was moved to the outer test scope. The default no-compiler build then passed all 595 assertions and skipped the optional CUBIN test correctly.

## Outcomes

Implementation, correctness validation, and measurements are complete. The optional compiler uses an independent handle per worker, with CUBIN cached through the existing persistentPipelineCache and PTX cached through persistentShaderCache. Python cache integration passed, all 15 prepare/pending tests passed, and native cache-recovery plus OptiX tests passed 336 assertions. A separate build without the optional compiler passed 595 assertions. Cold parallel startup improves from 67.665 to 22.319 seconds (three-run medians, 3.03 times faster). Persistent PTX+CUBIN cache hits reduce startup to 0.675 seconds, while a PTX-only warm control still takes 55.148 seconds. All of these measurements disable CUDA driver and NVRTC caches.

To run the Python integration check after building:

    $env:SLANGPY_TEST_CUBIN_CUDA_PATH=(Resolve-Path .temp/nvrtc-12.9/nvidia/cuda_nvrtc).Path
    python -m pytest slangpy/tests/device/test_cubin_cache.py -q

To build and run native recovery checks, use the existing standalone build (configured with the packaged Slang include/binary paths):

    cmake -S external/slang-rhi -B build/rhi-pending -DSLANG_RHI_NVPTXCOMPILER_ROOT=C:/projects/slangpy/.temp/cuda-nvcc-12.9
    cmake --build build/rhi-pending --config Release --target slang-rhi-tests -j12
    $env:CUDA_PATH=(Resolve-Path .temp/nvrtc-12.9/nvidia/cuda_nvrtc).Path
    $env:PATH="$env:CUDA_PATH\bin;$env:PATH"
    $env:SLANG_RHI_CUDA_CUBIN='1'
    build/rhi-pending/Release/slang-rhi-tests.exe '-tc=*cuda-cubin-cache-recovery*,*parallel-pipeline-creation-cuda-ray-tracing*'

All benchmark commands run in a fresh shell with CUDA_CACHE_DISABLE=1. The cold comparison uses --nvrtc-no-cache, no --cache-path, eight variants and 64 rounds. The persistent-cache experiment first populates .temp/cubin-cache-benchmark-20261115 once with --expect-cubin-cache cold, then uses the same directory with --expect-cubin-cache warm. Both phases keep driver caching disabled. The benchmark asserts that every CUBIN record has the expected hit state and that all mode outputs agree.

Cold CUBIN benchmark:

    $env:CUDA_CACHE_DISABLE='1'
    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --nvrtc-no-cache --cubin --variants 8 --rounds 64 --repeat 3 --modes serial pending_dispatch --wall-trace --expect-cubin-cache cold --output reports/pipeline-compilation-cuda-cubin-no-cache.json

Cold driver-JIT parallel control uses the same arguments without --cubin and --expect-cubin-cache, with --modes pending_dispatch and output reports/pipeline-compilation-cuda-driver-jit-control.json. The serial driver-JIT control uses --modes serial --repeat 1. These controls use the same built extension with the experiment disabled.

Cache population and reuse (use a new cache directory for each new cold population pass):

    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --nvrtc-no-cache --cubin --variants 8 --rounds 64 --repeat 1 --modes pending_dispatch --wall-trace --cache-path .temp/cubin-cache-benchmark-20261115 --expect-cubin-cache cold --output reports/pipeline-compilation-cuda-cubin-cache-populate.json
    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --nvrtc-no-cache --cubin --variants 8 --rounds 64 --repeat 3 --wall-trace --cache-path .temp/cubin-cache-benchmark-20261115 --expect-cubin-cache warm --output reports/pipeline-compilation-cuda-cubin-cache-warm.json

The PTX-only warm control omits --cubin and --expect-cubin-cache, retains the populated cache path, and runs --modes pending_dispatch --repeat 1. The same cache therefore supplies identical PTX, but the CUDA driver must translate it again with CUDA_CACHE_DISABLE=1.

Prototype limits: only CUDA compute programs opt in; standalone compiler 12.9 uses its default optimization level 3, while the driver's compiler/version is a different implementation. Output equality does not establish equal GPU execution performance, which has not been benchmarked. Retained PTX supports recovery at a memory cost. Windows distribution of the optional compiler bridge, final public configuration, and coverage on other GPU architectures/platforms remain production work.
