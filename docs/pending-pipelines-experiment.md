# Explicit pending pipeline compilation experiment

This ExecPlan follows `.agents/PLANS.md`. It is a living record of implementation and measurements.

## Purpose / Big Picture

Allow applications to prepare several SlangPy call variants without executing them, then synchronously compile their pending pipelines together. Compare this with individual first dispatches and the existing command-encoder batch. A pipeline is the GPU backend object plus compiled target shader code; a call variant is a function with a particular argument signature and modifiers.

## Progress

- [x] (2026-11-14) Inspected the existing deferred pipeline resolver and SlangPy call-data cache.
- [x] (2026-11-14) Started configuring the Windows MSVC build; the previous build directory was absent.
- [x] (2026-11-14) Implemented non-owning pending registration, explicit RHI resolution, and SGL bindings.
- [x] (2026-11-14) Added preparation and correctness/lifetime tests; 15 Python tests pass on D3D12, Vulkan, and CUDA.
- [x] (2026-11-14) Added the multi-function experiment; a six-kernel pilot verified matching results across all four modes.
- [x] (2026-11-14) Built the extension in Debug and standalone RHI tests in Release. The selected RHI pipeline tests pass with 1,212 assertions, including the new pending-pipeline test. Root, new-file, and RHI pre-commit checks passed.
- [x] (2026-11-14) Completed 36 isolated timing samples (24 kernels, 64 unrolled iterations, three repeats per mode/backend), verified matching outputs, and recorded results.
- [x] (2026-11-14) Downloaded NVIDIA's NVRTC 12.9.86 wheel into .temp/nvrtc-12.9 without changing the system toolkit; rebuilt the extension.
- [x] (2026-11-14) Verified NVRTC 12.9 in 30 additional samples, compared compiler caching, disabled NVVM concurrency as a control, and measured two samples with CUDA caching also disabled. All outputs exactly match the original 12.2 workload results.
- [x] (2026-11-14) Re-ran preparation and pending-pipeline lifetime tests using NVRTC 12.9: all five CUDA tests pass.
- [x] (2026-11-15) Collected one serial and two parallel uncached wall traces. Eight signal variants dominate; individual serial calls peak at 11.745 seconds. Target generation scales, but the driver pipeline stage remains expensive and variable. Checked outputs and raw timing envelopes against independent Python timers.

## Surprises and Discoveries

The existing resolver runs at command encoder finish, not queue submit. SlangPy already defaults to deferred pipelines. Its debug_build_call_data method inserts into the normal call cache without dispatching.

The initial RHI checkout was older than the revision recorded by SlangPy and failed to build against ResourceHeaps/ResourceAliasing. At the user's explicit direction, all submodules were updated with `git submodule update --init --recursive`. The experiment now builds directly against external/slang-rhi at 314584012227c5d4c505b3a5907f377b8cc6dc95, with no local-RHI override. A temporary compatibility worktree was superseded by that update.

The installed Vulkan validation layer rejected a device-creation structure in standalone Debug RHI tests. The standalone Release test run passes across all three selected backends. Python tests initially mixed devices with disabled debug layers and helper-created devices enabling them, causing D3D12 device reset; using consistent device options fixed the tests. No compiler or driver changes were made.

## Decision Log

Use the requested device-wide non-owning registry. Register only deferred pipelines with fully specialized programs: generic programs requiring shader-object specialization cannot be compiled without bindings and continue to resolve through command recording. Remove registration on successful resolution or destruction. Do not keep unused pipelines alive.

Reuse PipelineResolver for both command lists and pending pipelines. Respect serial/parallel device mode and CPU/backend restrictions. The operation is blocking and requires external exclusion of other shared Slang operations and pipeline lifetime changes. Keep the Python GIL held during the explicit operation, as existing device methods do.

Use fresh child processes and disabled persistent SlangPy caches for measurements. Report setup separately, preparation plus compilation separately where possible, and the time including dispatch/wait. Use explicit output buffers so scalar readback does not dominate the serial baseline. Verify outputs outside measured intervals. Driver caching cannot be assumed disabled and must be reported as a limitation.

At the user's reminder, compilation-report durations are not used or included in results. All measured intervals use Python time.perf_counter around synchronous operations. Reports are used only for pipeline/entry-point counts, persistent cache-hit checks, and verifying that post-compilation dispatch does not create another kernel. Run final timing samples sequentially after all builds and tests have completed.

For the NVRTC follow-up, select the compiler through a process-local CUDA_PATH and prepend its bin directory to PATH. Enumerate loaded Windows DLLs and call nvrtcVersion to verify selection for each sample. Use NVIDIA's isolated NVRTC 12.9.86 wheel with runtime headers from nvidia-cuda-runtime-cu12 12.9.79. Keep the system CUDA installation and driver unchanged. NVRTC 12.9 introduces persistent caching independently of Slang's caches, so compare actual compilation using its --no-cache downstream option. Compare the same compiler with NVRTC_DISABLE_CONCURRENT_NVVM unset versus set to 1. Driver PTX JIT caching remains enabled, as in the original experiment.

## Outcomes and Retrospective

The public preparation and pending-compilation APIs are implemented and tested. Non-owning registration is removed on both successful command-list/explicit resolution and pipeline destruction. All 36 benchmark samples completed and verified matching outputs within each backend.

Explicit preparation/compilation followed by individual dispatches takes a median 0.800 seconds on Vulkan versus 1.884 seconds serial (2.35 times faster), and 0.922 seconds on D3D12 versus 2.523 seconds serial (2.74 times faster). Existing single-command-list batching takes 0.797 and 0.912 seconds, respectively. Thus the new API exposes essentially the same batching benefit without requiring a shared command list; this experiment does not establish an additional performance advantage over existing batching.

The original CUDA experiment, subsequently identified as using NVRTC 12.2, shows only a modest median improvement: 45.471 seconds serial versus 43.311 seconds with explicit compilation and dispatch (1.05 times faster), and 43.403 seconds with command-list batching. First-run costs are substantial: the first explicit-only sample took 100.131 seconds and the first serial sample 91.258 seconds. Driver caches were not controlled, so these original results must not be described as fully cold pipeline-cache measurements or interpreted as proving a CUDA backend bottleneck's cause.

The NVRTC 12.9 follow-up establishes useful CUDA compilation concurrency. With NVRTC caching disabled and driver PTX caching retained, serial compilation takes 40.249 seconds versus 8.847 seconds for explicit batching and dispatch (4.55 times faster). Disabling NVVM concurrency on the same compiler raises the batch median to 38.424 seconds. When driver caching is also disabled, single exploratory samples take 96.244 seconds serial versus 68.427 seconds batched (1.41 times faster). Thus newer NVRTC materially improves target compilation, while first-use driver JIT costs can still dominate total pipeline creation. No additional changes to the native parallel compilation implementation were needed for this follow-up.

Preparation itself takes about 0.58 seconds in all three backends. For Vulkan and D3D12 this serial work dominates much of the batched total; the new API does not parallelize SlangPy code generation, module loading, linking, or reflection. These are synthetic kernels on a Debug host build using the packaged Slang compiler, not a claim about all application workloads. Initial device/module/tensor setup is excluded from the comparison and retained separately in the JSON samples.

The initial API deliberately remains synchronous and uses the caller's existing compilation policy. With parallel device mode and deferred function pipelines, representative calls can be prepared as follows:

    module.first.prepare(x, _result=first_output)
    module.second.prepare(y, _result=second_output)
    device.create_pending_pipelines()
    module.first(x, _result=first_output)
    module.second(y, _result=second_output)

Programs requiring specialization from bound shader objects remain on the existing command-list path. Autograd backward variants must be prepared separately. Rich batch diagnostics/retry fallback handling and additional hot-reload/specialization stress coverage remain follow-up work beyond this initial experiment; the existing deferred-compilation behavior is retained.

## Context and Orientation

`external/slang-rhi/src/pipeline-resolver.cpp` gathers deferred pipelines from command lists, prepares reflection and specialization serially, compiles entry points on the task pool, installs compiled shader modules, creates backend pipelines, and publishes results. `external/slang-rhi/src/pipeline.h` defines virtual pipeline wrappers that own the final concrete pipeline. In the updated RHI, external DeviceChild references pin the owning device and internal owners must keep it valid independently. The non-owning pending registry does not extend pipeline lifetime; its entries are removed in Pipeline's destructor while the device remains valid. Concurrent pipeline lifetime changes are excluded by the documented batch contract.

`src/sgl/device/device.*` wraps RHI and `src/slangpy_ext/device/device.cpp` exposes it to Python. `slangpy/core/function.py` and `src/slangpy_ext/utils/slangpyfunction.cpp` prepare and cache CallData. `slangpy/core/calldata.py` generates source and links a program before creating its deferred pipeline.

## Plan of Work

Milestone 1 adds IDevice::createPendingPipelines and its debug-layer forwarding. Extend the resolver to collect ready-to-compile pipelines from a non-owning device set. Add registration/unregistration to virtual pipeline creation, resolution, and destruction. Expose Device.create_pending_pipelines in SGL and Python with threading and specialization documentation. The RHI explicit-pending-pipeline-creation test verifies that discarded wrappers leave the registry and live compute/render pipelines resolve before recording.

Milestone 2 adds FunctionNode.prepare returning CallData and matching normal return-type normalization. Add tests under slangpy/tests for no dispatch during prepare, cache reuse, empty/repeated flush, discarded pipelines, and subsequent normal command-list use. Both new Python test files must pass on the three selected GPU backends.

Milestone 3 adds tools/benchmark_pipeline_compilation.py and a companion Slang workload with several nontrivial functions and compile-time variants. Run serial individual dispatches, parallel single-encoder dispatches, and explicit prepare/flush with optional subsequent dispatch. Use repeated isolated runs and save JSON evidence in reports. The final workload has 24 kernels: eight link-time-constant variants of signal, network, and orbit, each with 64 unrolled iterations and 32 input elements. The benchmark verifies matching outputs across all four modes and repeats.

## Concrete Steps

From C:\projects\slangpy, configure and build outside the sandbox as required by the RHI instructions:

    git submodule update --init --recursive
    cmake --preset windows-msvc -DSGL_LOCAL_RHI=OFF
    cmake --build --preset windows-msvc-debug --target slangpy_pydoc
    cmake --build --preset windows-msvc-debug --target slangpy_ext -j 12
    python -m pytest slangpy/tests/slangpy_tests/test_prepare.py slangpy/tests/device/test_pending_pipelines.py --device-types=d3d12,vulkan,cuda -v
    python tools/benchmark_pipeline_compilation.py --backend vulkan --variants 8 --rounds 64 --repeat 3 --output reports/pipeline-compilation-vulkan.json
    python tools/benchmark_pipeline_compilation.py --backend d3d12 --variants 8 --rounds 64 --repeat 3 --output reports/pipeline-compilation-d3d12.json
    python tools/benchmark_pipeline_compilation.py --backend cuda --variants 8 --rounds 64 --repeat 3 --output reports/pipeline-compilation-cuda.json
    pre-commit run --all-files

The measurements use the Debug SlangPy/RHI build, Slang 2026.20 from the normal release package, an AMD Ryzen 9 3900X (12 cores, 24 logical CPUs), and an NVIDIA TITAN RTX with driver 617.42. GPU debug layers, RHI validation, hot reload, ray tracing, and persistent SlangPy caches are disabled for timings. Module loading and tensor/device creation are outside the elapsed interval and recorded separately as setup_s. Each sample is a fresh child process; mode order is deterministically shuffled each repeat. NVIDIA driver caches are left enabled, so results measure cold Slang state but not guaranteed cold driver state.

Standalone RHI validation used these commands, also from the repository root:

    cmake -S external/slang-rhi -B build/rhi-pending -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DSLANG_RHI_BUILD_EXAMPLES=OFF -DSLANG_RHI_BUILD_TESTS=ON -DSLANG_RHI_ENABLE_D3D11=OFF -DSLANG_RHI_ENABLE_WGPU=OFF -DSLANG_RHI_FETCH_SLANG=OFF -DSLANG_RHI_SLANG_INCLUDE_DIR=C:/projects/slangpy/build/windows-msvc/_deps/slang-src/include -DSLANG_RHI_SLANG_BINARY_DIR=C:/projects/slangpy/build/windows-msvc/_deps/slang-src
    cmake --build build/rhi-pending --config Release --target slang-rhi-tests -j 12
    build/rhi-pending/Release/slang-rhi-tests.exe --test-case="*pipeline-creation*" --select-devices=d3d12,vulkan,cuda

The Python run reports 15 passed. The RHI run reports 1,212 passing assertions and zero failures. Its test-case total includes entries that the custom reporter skips because their backend was not selected.

## Validation and Acceptance

Prepared kernels must not mutate output buffers. Explicit compilation must produce compilation reports before any dispatch, repeated flushes must not compile again, and individual subsequent calls must produce correct outputs without extra compilation. Releasing an unused pipeline before flushing must neither crash nor compile it. Tests must cover serial and parallel modes. Existing command-list resolution must still work and remove successfully resolved pipelines from pending registration. Timing results must include identical kernel sets and distinguish setup, total cold-call cost, and explicit preparation/compilation phases.

## Idempotence and Recovery

Builds and tests are repeatable. Submodules have been updated as requested; do not reset the local feature edits or modify unrelated work. Failed compilation leaves unresolved pipelines registered so callers can destroy or retry them. Benchmark child processes isolate compiler state; retained JSON results identify workload and backend so later runs can be compared.

## Artifacts and Notes

Complete samples are in reports/pipeline-compilation-vulkan.json, reports/pipeline-compilation-d3d12.json, and reports/pipeline-compilation-cuda.json. Every sample stores elapsed wall time, stage wall times where available, setup time, creation counts, workload metadata, and verified outputs. Compilation-report timing values are deliberately excluded.

Median elapsed seconds (minimum to maximum across three samples):

    Mode                 Vulkan                 D3D12                  CUDA
    serial               1.884 (1.875-1.910)    2.523 (2.497-2.735)     45.471 (45.354-91.258)
    command_list         0.797 (0.794-0.798)    0.912 (0.906-0.950)     43.403 (43.350-43.717)
    pending              0.794 (0.780-1.039)    0.884 (0.882-1.308)     43.221 (42.812-100.131)
    pending_dispatch     0.800 (0.798-0.802)    0.922 (0.912-0.929)     43.311 (42.251-43.624)

The pending interval includes prepare plus create_pending_pipelines, excluding dispatch. Other modes include dispatch and the final wait for GPU completion. Output validation is outside measured intervals. Median explicit-only preparation and compilation intervals are 0.584/0.210 seconds for Vulkan, 0.581/0.307 for D3D12, and 0.584/42.647 for CUDA. Independently computed stage medians need not sum exactly to the median total.

## CUDA follow-up: NVRTC concurrency

A follow-up probe created a CUDA SlangPy device, enumerated the process's loaded DLLs using the Windows API, and called `nvrtcVersion` on the loaded NVRTC library. It resolved to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvrtc64_120_0.dll`, reporting version 12.2. `CUDA_PATH` selects that toolkit; `NVRTC_DISABLE_CONCURRENT_NVVM` is unset. The DLL filename alone does not identify the minor version. A v12.3 toolkit directory exists but has no bin directory, so it does not provide an immediately available newer NVRTC for comparison.

NVIDIA documents three relevant milestones:

- Before CUDA 11.5, concurrent NVRTC compilations were serialized by a global lock.
- CUDA 11.5 replaced that lock with per-stage locks, allowing different compilations to overlap different stages. This did not make every stage independently concurrent. NVIDIA also improved driver PTX JIT concurrency in R495. See [NVIDIA's CUDA 11.5 compiler article](https://developer.nvidia.com/blog/reducing-application-build-times-using-cuda-c-compilation-aids/).
- CUDA 12.3 added concurrent execution of the embedded NVVM optimizer/code-generation phase. `NVRTC_DISABLE_CONCURRENT_NVVM` disables this concurrency when set. See [NVRTC 12.3 thread safety](https://docs.nvidia.com/cuda/archive/12.3.0/nvrtc/index.html#thread-safety).

Our loaded NVRTC 12.2 predates the third improvement. A serialized optimization stage is therefore a plausible explanation for the weak CUDA scaling on these heavily unrolled kernels, but the existing measurements do not establish which internal stage dominates. SlangPy requests PTX (`src/sgl/device/shader.cpp`); RHI subsequently calls `cuModuleLoadData` (`external/slang-rhi/src/cuda/cuda-pipeline.cpp`), which can involve separate driver JIT work. The explicit-flush wall interval includes both target compilation and backend pipeline creation.

The next useful comparison is the same workload with a verified NVRTC 12.3 or newer library, then the same version with `NVRTC_DISABLE_CONCURRENT_NVVM=1`. Keep the driver, workload, and cache policy fixed and repeat samples. Record the actual loaded library/version. If necessary, measure wall time around the whole parallel target-code batch and the whole backend-pipeline batch separately; do not sum compilation-report durations. A newer NVRTC requires its matching NVRTC-builtins library and a compatible driver for emitted PTX, as described in [NVIDIA's versioning documentation](https://docs.nvidia.com/cuda/archive/12.3.0/nvrtc/index.html#versioning-scheme).

If an older NVRTC must remain supported, additional threads cannot remove its internal stage locks. Process-isolated compilation could bypass process-local locks, but would require independent compiler state and transferring compiled artifacts; it is substantially more work than the pending-pipeline API and has not been measured here.

### Isolated NVRTC 12.9 experiment

The follow-up uses NVIDIA's Windows wheel packages without installing a replacement system toolkit. Reproduce the isolated setup from the repository root:

    python -m pip install --no-deps --no-cache-dir --target .temp/nvrtc-12.9 nvidia-cuda-nvrtc-cu12==12.9.86
    python -m pip install --no-deps --no-cache-dir --target .temp/cuda-runtime-12.9 nvidia-cuda-runtime-cu12==12.9.79
    Copy-Item -Path '.temp/cuda-runtime-12.9/nvidia/cuda_runtime/include/*' -Destination '.temp/nvrtc-12.9/nvidia/cuda_nvrtc/include' -Recurse -Force

RHI searches for cuda_runtime.h next to the NVRTC package, so the NVRTC wheel alone does not suffice for device initialization. The benchmark's --cuda-path option sets CUDA_PATH and prepends the isolated bin directory to PATH in each worker process; the latter allows NVRTC to find its matching builtins DLL. Each CUDA sample records the loaded DLL's actual path and nvrtcVersion result and checks that it belongs to the selected directory.

NVRTC 12.9 adds a persistent NVVM-to-PTX/SASS cache, enabled by default and implemented through the CUDA driver. This is separate from the disabled SlangPy caches. See [NVIDIA's caching documentation](https://docs.nvidia.com/cuda/archive/12.9.1/nvrtc/index.html#caching-cuda-12-9). The initial pending pilot took 66.505 seconds; subsequent cache-enabled runs were much faster. These are retained as cache behavior evidence, not evidence of uncached parallel compilation. Use --nvrtc-no-cache to pass --no-cache downstream to NVRTC while retaining driver PTX-to-machine-code caching.

    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --variants 8 --rounds 64 --repeat 3 --output reports/pipeline-compilation-cuda-nvrtc129.json
    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --nvrtc-no-cache --variants 8 --rounds 64 --repeat 3 --output reports/pipeline-compilation-cuda-nvrtc129-no-cache.json

The second command measures actual NVRTC compilation. Its NVVM concurrency control adds --disable-concurrent-nvvm and uses --modes pending_dispatch to repeat the same explicit batching path. All measurements remain synchronous wall-clock intervals; no compilation-report durations are used.

With NVRTC's cache disabled and the driver PTX cache left enabled, the three-repeat medians are 40.249 seconds serial, 8.455 seconds in one command list, 8.836 seconds for prepare plus flush, and 8.847 seconds for prepare plus flush plus individual dispatches. The latter is 4.55 times faster than serial. Ranges are 39.998-42.121, 8.298-9.524, 8.185-9.502, and 8.421-8.871 seconds, respectively. The shader source and workload are unchanged from the original NVRTC 12.2 experiment. Outputs match the 12.2 results exactly on this workload.

The separate cache-enabled medians are 4.078, 3.038, 3.031, and 3.055 seconds, respectively. Do not attribute this additional reduction to concurrency: NVRTC's persistent cache can skip optimizer/code-generation work even in a fresh process. The initial 66.505-second pilot is not included in those medians and remains in reports/pipeline-compilation-cuda-nvrtc129-pilot.json.

    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --nvrtc-no-cache --disable-concurrent-nvvm --variants 8 --rounds 64 --repeat 3 --modes pending_dispatch --output reports/pipeline-compilation-cuda-nvrtc129-serialized-nvvm.json

The concurrency-disabled control takes a median 38.424 seconds (37.905-41.594), compared with 8.847 seconds (8.421-8.871) for the identical batch with concurrency enabled. The resulting 4.34-times difference isolates the effect of permitting concurrent NVVM work much better than comparing compiler versions alone. Its time also approaches the 40.249-second serial baseline. This is strong evidence that newer NVRTC enables useful parallel target compilation through the existing RHI worker infrastructure and the new pending-pipeline API.

An additional single-sample-per-mode probe disables CUDA driver caching as well, so PTX JIT work must be repeated. These are exploratory samples, not three-repeat medians:

    $env:CUDA_CACHE_DISABLE='1'
    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --nvrtc-no-cache --variants 8 --rounds 64 --repeat 1 --modes serial pending_dispatch --output reports/pipeline-compilation-cuda-nvrtc129-all-cache-disabled.json

Run the probe in a separate shell or remove CUDA_CACHE_DISABLE afterward to restore the default cache policy. The Python benchmark records its effective value in every new result.

With CUDA caching disabled, the explicit batch plus dispatch takes 68.427 seconds and serial dispatches take 96.244 seconds, a 1.41-times improvement in this single pair. These intervals include driver PTX JIT work. This is a more conservative picture of first-use pipeline creation than the 4.55-times improvement with the driver cache available. The comparison implicates uncached driver work as a major remaining cost, but does not directly profile its internal phases or locks.

All 30 additional samples verified NVRTC version 12.9 and produced arrays exactly equal to the original NVRTC 12.2 outputs. They comprise 12 cache-enabled samples, 12 with NVRTC caching disabled, three with NVVM concurrency disabled, two with CUDA caching disabled, and the initial pilot. A subsequent CUDA-only run of test_prepare.py and test_pending_pipelines.py passed all five tests against the isolated compiler. The system toolkit and driver remain unchanged; only the experiment workers select the isolated runtime.

### Per-kernel wall trace (2026-11-15)

To distinguish a slow kernel from driver contention, --wall-trace records independent per-call wall timestamps, plus Python perf_counter durations around each serial function call. Source inspection verified that ShaderProgram::compileEntryPoint captures Timer::now() around target-code generation, and CUDA DeviceImpl::createComputePipeline2 captures separate Timer::now() values around backend pipeline creation. ShaderCompilationReporter copies these timestamps unchanged under a mutex. They are independent of Slang's cumulative compile_time, compile_slang_time, and compile_downstream_time counters, which remain excluded. Backend intervals include cuModuleLoadData, cuModuleGetFunction, and subsequent metadata queries, not just one driver API call.

Reports are returned in program creation order; this workload creates one program per function variant in signal/network/orbit order. The trace asserts those counts before assigning names. TimePoint values are nanoseconds. Analyze the union of overlapping intervals or the stage's first-start/last-end envelope, never sum overlapping durations as elapsed time. Individual concurrent intervals can include driver lock waiting.

    $env:CUDA_CACHE_DISABLE='1'
    python tools/benchmark_pipeline_compilation.py --backend cuda --cuda-path .temp/nvrtc-12.9/nvidia/cuda_nvrtc --nvrtc-no-cache --wall-trace --variants 8 --rounds 64 --repeat 1 --modes serial pending_dispatch --output reports/pipeline-compilation-cuda-nvrtc129-wall-trace.json

The serial sample took 96.031 seconds. Its target-code intervals total 40.346 seconds and its backend pipeline intervals total 55.068 seconds; these sums are valid because the intervals do not overlap. The independent serial function-call timers show:

    Function family   Variants   Per-call range (seconds)
    signal            8          10.354-11.745
    network           8           0.533-0.549
    orbit             8           0.478-0.486

The single longest full call is 11.745 seconds, about 12% of the serial total. The eight signal variants account for approximately 88 seconds, about 92% of the total. Their individual backend pipeline creation times are 6.008-6.791 seconds, accounting for 51.412 of the 55.068 seconds of backend work. Thus the workload is dominated by one function family, not one exceptional variant. signal contains 64 forced-unrolled iterations of float4 sin/cos operations; these synthetic kernels deliberately impose substantial compiler work.

The first parallel trace took 94.446 seconds: the target-code phase spans 15.522 seconds and the backend pipeline phase spans 78.108 seconds. All 24 target calls overlap. Up to 13 backend calls overlap; the first group remains outstanding for approximately 75.38 seconds and returns almost together, including network variants that take only about 0.24 seconds to create in serial. This is consistent with serialization or heavy contention inside the driver path, not intrinsic 75-second compilation for those individual kernels. The trace does not identify a specific internal driver mutex. These diagnostic samples are distinct from the previous day's 96.244/68.427-second pair and expose substantial parallel-run variability.

A second parallel trace took 59.426 seconds: target generation spans 7.866 seconds, while backend pipeline creation still spans 50.979 seconds. Again, a group of overlapping pipeline calls returns almost together after approximately 49 seconds. This supports the same bottleneck despite variation in absolute timings. Comparing this repeat with the serial trace, target generation improves by about 5.1 times, whereas backend creation improves only from 55.1 to 51.0 seconds. If that backend cost stayed fixed at 55 seconds, even eliminating all target-generation time would limit total speedup to approximately 1.7 times.

The repeat is saved in reports/pipeline-compilation-cuda-nvrtc129-wall-trace-repeat.json. A summary of the initial pair is in reports/pipeline-compilation-cuda-nvrtc129-wall-trace-summary.json. All three diagnostic samples exactly match the original CUDA outputs. Each complete trace envelope fits inside its independent Python elapsed interval, and every serial kernel's target-plus-pipeline duration fits inside its separately measured Python function-call duration. No native instrumentation or new compiler-counter timing was introduced. The original Vulkan/D3D12 measurements left driver caches enabled; they are not a controlled cache-disabled comparison against this CUDA experiment.

## Interfaces and Dependencies

Add `rhi::IDevice::createPendingPipelines() -> Result`, `sgl::Device::create_pending_pipelines() -> void`, Python `Device.create_pending_pipelines() -> None`, and `FunctionNode.prepare(*args: Any, **kwargs: Any) -> CallData`. Use existing task-pool and compilation-report infrastructure, Python standard-library timing/process/JSON facilities, and the existing NumPy dependency for correctness checks. No new dependencies.

Revision: completed after all measurements and correctness checks, with the requested submodule update, actual commands, hardware, timing methodology, full results/ranges, and limitations.

Revision: extended with isolated NVRTC 12.9 selection, per-sample DLL/version verification, compiler-cache controls, the NVVM concurrency experiment, driver-cache-disabled probes, and CUDA correctness validation.

Revision: added source-audited per-kernel wall traces, independent serial-call cross-checks, a repeated parallel diagnostic, and the distinction between a heavy function family and driver-stage contention.

Revision: the implemented standalone PTX-to-CUBIN compiler and persistent CUBIN cache experiment is documented in [cubin-cache-experiment.md](cubin-cache-experiment.md). It reduces cold parallel startup from 67.665 to 22.319 seconds in contemporary three-run comparisons, and verified PTX+CUBIN cache hits take 0.675 seconds with driver caching disabled. That optional experiment adds a separate NVIDIA compiler dependency; the original pending-pipeline API does not require it.
