# OptiX buffer visibility and CUDA load performance

Slang commit `d459d44a0` keeps ordinary buffer loads in CUDA output that contains a ray-tracing entry point. OptiX shaders can then see host updates to the same allocation between launches. Keep `__ldg` for compute-only CUDA programs. Removing it increased execution time by 29% to 47% in four of the five structured-buffer kernels measured here.

## Correctness

The incident report isolated stale reads on an RTX 4090 on Linux with drivers 535.309.01 and 580.173.02. On each driver its standalone OptiX raygen probe saw the correct value on 1 of 12 launches with `ld.global.nc.u32` and 12 of 12 with `ld.global.ca.u32`. The report also verified ordinary `ld.global.u32` on the latter driver.

The Slang regression reads a word at byte offset 16 from a `StructuredBuffer<uint>` through a helper. The unpatched compiler emitted `__ldg`. After the fix, NVRTC produced `ld.global.u32` for raygen and `ld.global.nc.u32` for a compute entry point in the same source compiled separately. The pass applies to an entire emission unit because helpers can be shared between stages. The regression also compiles closest-hit and a combined compute/raygen module.

SlangPy's `test_raygen_reads_updated_buffer` reuses the pipeline, shader table, and device buffers. It writes 0x100, 0x101, then zero to the same word, verifies each upload by host readback, and checks the shader result after each launch. Both compiler builds passed on the local Windows RTX 5090. The generated-code regression distinguishes the two builds on this machine. Verification on the affected RTX 4090 and the Falcor2 render are follow-up work.

## CUDA kernel execution time

Measured on 2026-11-20 on a GeForce RTX 5090, Windows 11, driver 616.56, CUDA toolkit 12.9, target `sm_120`. SlangPy was based on `4fa16180` and used a Release compiler built from Slang `6492e91c4` plus the OptiX fix. The compute experiment toggled only the call to `lowerImmutableBufferLoadForCUDA` in the same compiler binary. GPU clocks were unlocked.

The five kernels below come from `slangpy/benchmarks/test_benchmark_interop.py`. Each adds the same 2,096,640 floats from each of two input buffers into one output buffer. All workgroups contain 32 threads. The grid size decreases as the number of elements per thread increases.

| Elements per thread | Baseline `ld.global.nc` | Pass disabled | Explicit `ld.global.ca` | Change with pass disabled |
| --- | ---: | ---: | ---: | ---: |
| 1 | 30.61 µs | 30.61 µs | 30.61 µs | 0.0% |
| 4, explicit | 8.00 µs | 10.34 µs | 10.33 µs | +29.2% |
| 16, explicit | 21.10 µs | 30.77 µs | 30.84 µs | +45.8% |
| 16, loop | 21.10 µs | 30.83 µs | 30.78 µs | +46.1% |
| 16, forced unroll | 21.10 µs | 30.94 µs | 30.80 µs | +46.6% |

SlangPy and NVRTC produced PTX with 2, 8, and 32 non-coherent scalar loads for the 1-, 4-, and 16-element cases respectively. With the pass disabled, the same counts became ordinary scalar loads. For the third column of timings, replacing only `ld.global.nc.f32` with `ld.global.ca.f32` in the baseline PTX isolated the cache operator. That substitution gave similar times to disabling the pass on these kernels.

The CUDA driver loaded the PTX variants in one process. Each variant used the same seeded random inputs, output allocation, and grid. All five kernels and all three variants produced the expected array. Each CUDA graph replayed 64 consecutive launches. Each sample warmed up with one graph replay, timed the next replay with CUDA events, and divided by 64. The table gives the median of 36 samples per variant, rotating which variant ran first.

## Existing SlangPy benchmark fixture

The regular suite measured 100 warmups and 2,000 timestamped dispatches per case in a Debug SlangPy build. Each configuration ran in three fresh processes in the order baseline, disabled, disabled, baseline, baseline, disabled. The table gives the median of the three run medians in microseconds, with their range in parentheses. One command buffer was submitted per timed dispatch. This includes more dispatch and timing overhead than the graph measurement.

| CUDA workload | Baseline (µs) | Pass disabled (µs) |
| --- | ---: | ---: |
| Structured-buffer add, 1 element | 47 (45–52) | 45 (45–48) |
| Structured-buffer add, 4 elements | 25 (23–29) | 25 (25–31) |
| Structured-buffer add, 16 explicit | 37 (36–41) | 45 (45–51) |
| Structured-buffer add, 16 loop | 37 (36–41) | 45 (45–51) |
| Structured-buffer add, 16 forced unroll | 36 (36–41) | 50 (45–50) |
| Tensor add, 1,048,576 elements | 33 (31–36) | 36 (32–37) |
| Sum of eight tensors | 33 (33–39) | 37 (33–38) |
| Indirect sum of eight tensors | 33 (33–39) | 38 (33–39) |
| Per-element DiffTensor backward, 1,048,576 elements | 37 (31–37) | 31 (31–31) |
| PPISP forward, 100,000 pixels | 22 (21–27) | 21 (21–22) |
| PPISP forward, 1,000,000 pixels | 50 (49–55) | 49 (49–49) |
| PPISP backward, 100,000 pixels | 92 (91–97) | 91 (91–95) |

The inspected tensor, DiffTensor, and PPISP kernels already used ordinary loads in the baseline. Their timing variation illustrates the noise in single-dispatch measurements. The three 16-element source variants also compiled to the same load count and showed similar batched timings.

To repeat the fixture comparison, build Slang with the OptiX fix, point SlangPy's `SGL_LOCAL_SLANG` configuration at that build, and run the GPU cases in `test_benchmark_interop.py`, `test_benchmark_tensor.py`, `test_benchmark_bwd_diff.py`, and `test_benchmark_ppisp.py` with `--device-types=cuda`. Then temporarily skip `lowerImmutableBufferLoadForCUDA` for compute too, rebuild Slang and SlangPy, verify the loaded compiler DLL matches the rebuilt DLL, and repeat in a fresh process. The PyTorch cases used PyTorch 2.8.0+cu128 with a native bridge built from this SlangPy checkout.
