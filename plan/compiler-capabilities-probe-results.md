# Compiler capability probes: step 1

Date: 2026-09-23. This executes the first milestone of [the plan](compiler-capabilities.md). [The workaround ledger](slang-compiler-workarounds.md) separately tracks existing workarounds, tested candidate adapters, upstream gaps, and rejected approaches.

This report retains the original 80-case baseline. [Milestone 1a](compiler-profile-probe-results.md) adds 84 explicit-profile interaction cases and settles the subsequent resolver policy. Use `--suite baseline` to reproduce only this report's cases; the current runner defaults to both suites.

No production compiler behavior or public API was changed. The deliverable is a reproducible native session-API probe, a Python matrix runner, measured observations, and baseline verification.

## Environment and provenance

SlangPy revision: `a30a5626eaba8dab30979482f7c8a8fe217ad376`. RHI submodule: `82c03494bed8e2d42d65555e33b30a18d3d8f071`.

The SlangPy Debug build completed with pinned Slang **2026.17.1**. Its compiler DLL is `build/windows-msvc/Debug/slang-compiler.dll`, SHA-256 `a3a46665a22b63412396cc6327b39d920aeed5b55f0ff473ff21739173ef14a9`.

Local Slang source **b4a57b15cc47d936403cc989a628bfd25d5af5d3** was built separately under `build/slang-capabilities-master`. The loaded build tag is **2026.18.2-3-gb4a57b15c**; DLL SHA-256 is `960841ee598750c724db7aca1c5ec8b4cd3954105b2ece944a673885cbaabe32`. This uses the existing checkout's SPIR-V submodules, which differ from its recorded gitlinks: spirv-headers `29981f65241605e08b0ede4cfeb999fe3b723c6a`, spirv-tools `0d6fd73ca73830ccab5fa1f00ed5ed40124e2c55`. These checkouts were not modified. This is a local-master build, not a pristine-submodule upstream CI build. LLVM and unrelated tools were disabled for the comparison; CPU comparison is C++ source generation, not master-built host execution.

Both runs use DXC **1.9.2602.17 (21d28f727)** from SlangPy's build and NVRTC **12.2** from the installed CUDA 12.2 toolkit. The probe records the actual loaded compiler/downstream DLL paths. NVRTC reports supported numeric architectures `50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90`.

Hardware: **NVIDIA TITAN RTX**, CUDA compute capability **7.5**, driver **610.74**, Windows. Device inventories were obtained from the rebuilt SlangPy package. A buffer-write/readback smoke shader returned 7 on D3D12, Vulkan, CUDA, and CPU using production SlangPy defaults. Metal and WebGPU devices were unavailable; their probes cover source generation only. Higher-architecture PTX and SER outputs were compiled and inspected, not executed on this GPU.

## Method

`tools/compiler_capability_probe/main.cpp` dynamically loads the explicitly selected Slang library and calls `createSession`, `loadModuleFromSourceString`, composition, linking, and code generation. Each case runs in its own process. It validates capability names with `findCapability`, optionally mirrors RHI's unknown-name filtering, and can place options at session or target scope. This avoids mistaking `slangc`'s additional CLI validation for session-API behavior.

`run.py` preserves each shader, exact subprocess argument list, diagnostics, JSON result, and generated output under the chosen result directory. It extracts DXIL shader-model metadata, SPIR-V header versions/capability/extension declarations, PTX architecture/version directives, and HLSL HitObject implementation names. Library hashes and NVRTC architecture discovery are stored in `environment.json`.

The final matrix contains **80 cases per compiler**. `verify.py` checks 66 selected observations per compiler and compares every case's status and inspected output properties between versions. Expected compiler failures are assertions about the current compiler, not desired permanent behavior. A future upstream fix should prompt reevaluation of these checks and the workaround ledger.

## Measured results

Both compiler versions exhibit the same behavior in these probes.

| Question / case IDs | Observation | Consequence |
| --- | --- | --- |
| `dxil_cap_6_6`, `dxil_cap_6_9` | Whole-program output is `lib_6_3` without a profile. | Capability-only selection does not select DXC's model. |
| `dxil_profile_6_6`, `dxil_profile_6_9` | Output is `lib_6_6` / `lib_6_9`. | Internal profile synthesis works. |
| `dxil_wave_match_cap` / `dxil_wave_match_profile` | Capability-only fails DXC validation for `cs_6_0`; explicit profile produces valid `cs_6_6`. | This is an operation-level failure, not just metadata. |
| `ray_payload_6_6`, `ray_payload_6_7` | Simple TraceRay payload shaders compile to the respective library models. | Historical 6.7 workaround not reproduced; original regression still needed. |
| `spirv_cap_1_3` / `spirv_baseline_1_3` | Raw 1.3 cap alone emits 1.5; adding the 1.0 baseline profile emits 1.3. | Explicit baseline neutralizes the hidden default. |
| `spirv_baseline_1_6` | Minimal profile plus raw 1.6 emits 1.6. | Session API permits the proposed minimal-profile adapter. |
| `spirv_bundle_implicit`, `spirv_bundle_full`, `spirv_bundle_minimal` | A physical-storage-buffer requirement passes with implicit/full profile bundles, but fails strict checking with the minimal profile plus raw version. | Higher/default profiles add assumptions beyond raw version selection. |
| `strict_inferred_*`, `strict_late_*` | Permissive mode warns; restrictive mode rejects ordinary inferred and late missing requirements. | Restrictive mode is useful for these cases. |
| `strict_explicit_True` | An explicit higher entry-point requirement passes despite the lower selected model. | Strict checking is not a complete upper-bound guarantee. |
| `strict_empty_inferred` / `strict_baseline_inferred` | Empty strict selection passes; explicitly supplying `hlsl` makes it reject the missing requirement. | Always supply an explicit baseline capability when checks are intended. |
| `cuda_5_0`, `cuda_7_0`, `cuda_8_0`, `cuda_8_9`, `cuda_9_0` | PTX targets `sm_50`, `sm_70`, `sm_80`, `sm_89`, `sm_90`. | Existing upstream capability-to-NVRTC mapping works. |
| `cuda_7_5`, `cuda_8_6`, `cuda_12_0` | Names are unknown to both Slang versions. | Device/toolkit capability and compiler vocabulary differ. |
| `cuda_empty` / `cuda_half_5_0` | Default emits `sm_50`; half code with requested 5.0 emits `sm_60`. | Toolkit baseline and code requirements affect architecture independently. |
| `cuda_late_warn` / `cuda_late_strict` | Missing late 8.9 requirement warns while PTX stays `sm_80`; strict mode rejects it. | A requirement diagnostic does not itself prove the emitted architecture was raised. |
| `cuda_higher_implies_lower` | Supplying both 7.0 and 9.0 emits `sm_90`. | Adding a lower version cannot cap a higher one. |
| `cuda_bridge_86`, `cuda_duplicate_80`, `cuda_conflicting_80_75` | NVRTC rejects duplicate architecture options, even when identical. | Reject the downstream-argument bridge. |
| `ser_native`, `ser_nvapi` | Strict checking incorrectly demands the other implementation's marker. | Fix Slang requirement inference before strict SER defaults. |
| `ser_native_permissive`, `ser_nvapi_permissive` | Emits `dx::HitObject` / `NvHitObject` respectively, with a warning about the other marker. | Explicit implementation selection works in permissive mode. |
| `ser_both` | Strict checking passes, but NVAPI representation wins. | Adding both capabilities does not force native output. |
| `ser_native_dxil`, `ser_nvapi_dxil` | Both compile as `lib_6_9` with correct NVAPI include/UAV configuration. | Both paths reach the downstream compiler successfully. |
| `forward_*_legacy` | Recognized device caps, old profiles, unconditional NVAPI, and session-scope options compile the SlangPy-import shader on available backends. | Historical commented-out forwarding failure not reproduced. |
| `device_vulkan_strict_wave` | Missing `spvGroupNonUniformArithmetic`; permissive wave code compiles. | RHI reporting must improve before strict device-default compilation. |
| `device_vulkan_strict_atomic64` | Compiles, emitting SPIR-V Int64Atomics capability 12. | The suspected raw-list gap is not a demonstrated strict failure for this operation. |
| `device_cuda_simple` | Raw CC 7.5 device list contains unknown 7.2/7.5 names; recognized-name filtering yields PTX `sm_70`. | Default filtering must be visible; exact device architecture is currently lost. |

CPU wave and int64-atomic shaders are rejected as unavailable on the C++ target. Simple and SlangPy-import CPU shaders generate source successfully. Metal and WGSL basic compute shaders generate source, but this does not establish downstream compilation or runtime support.

The deliberately invalid `compute_999` downstream probe also fails at duplicate-option detection. It does not independently prove NVRTC's unsupported-architecture diagnostic path. Toolkit support is instead established by `nvrtcGetSupportedArchs`; this run does not cover CUDA 12.8/13.x minimum-architecture behavior or architecture suffixes.

## Important diagnostic excerpts

Capability-selected WaveMatch without a profile:

    Opcode WaveMatch not valid in shader model cs_6_0.

Appending the proposed CUDA override:

    nvrtc: error : --gpu-architecture (-arch) defined more than once

Native-only strict SER:

    error[E41013]: entry point uses capabilities not in specified profile
    Missing capabilities are: 'hlsl_nvapi'

Detected Vulkan capabilities with strict wave operations:

    Missing capabilities are: 'spvGroupNonUniformArithmetic'

## Reproduction

From `C:/projects/slangpy` in the configured MSVC developer environment:

    cmake --build --preset windows-msvc-debug
    cmake -S tools/compiler_capability_probe -B build/compiler-capability-probe -G Ninja -DSLANG_INCLUDE_DIR=C:/projects/slangpy/build/windows-msvc/_deps/slang-src/include
    cmake --build build/compiler-capability-probe
    python tools/compiler_capability_probe/run.py --probe build/compiler-capability-probe/compiler_capability_probe.exe --library build/windows-msvc/Debug/slang-compiler.dll --output build/compiler-capability-probe/results/release --dxc build/windows-msvc/Debug/dxcompiler.dll --nvrtc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvrtc64_120_0.dll" --suite baseline

The first run creates `device_inventory.json` and executes the production smoke shaders. Reuse that inventory when comparing compilers so device-list changes do not confound compiler differences. Source and library revisions are pinned in the provenance section above; changing compiler versions is an intentional experiment.

The independent master build used the existing local source and cached DXC package, without reconfiguring the SlangPy build or the sibling Slang build:

    cmake -S C:/projects/slang -B build/slang-capabilities-master -G Ninja -DCMAKE_BUILD_TYPE=Release -DSLANG_ENABLE_TESTS=OFF -DSLANG_ENABLE_EXAMPLES=OFF -DSLANG_ENABLE_GFX=OFF -DSLANG_ENABLE_SLANG_RHI=OFF -DSLANG_ENABLE_SLANGD=OFF -DSLANG_ENABLE_SLANGI=OFF -DSLANG_ENABLE_REPLAYER=OFF -DSLANG_ENABLE_SLANGRT=OFF -DSLANG_ENABLE_CUDA=ON -DSLANG_ENABLE_OPTIX=OFF -DSLANG_SLANG_LLVM_FLAVOR=DISABLE -DSLANG_EXCLUDE_TINT=ON -DSLANG_EXCLUDE_DAWN=ON -DSLANG_ENABLE_RELEASE_LTO=OFF -DFETCHCONTENT_SOURCE_DIR_DXC=C:/projects/slangpy/build/windows-msvc/_deps/dxc-src
    cmake --build build/slang-capabilities-master --target slang slangc -j 12
    python tools/compiler_capability_probe/run.py --probe build/compiler-capability-probe/compiler_capability_probe.exe --library build/slang-capabilities-master/Release/bin/slang-compiler.dll --output build/compiler-capability-probe/results/master --inventory build/compiler-capability-probe/results/release/device_inventory.json --dxc build/windows-msvc/Debug/dxcompiler.dll --nvrtc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvrtc64_120_0.dll" --suite baseline
    python tools/compiler_capability_probe/verify.py build/compiler-capability-probe/results/release build/compiler-capability-probe/results/master

Expected verification: 66 observation checks and 80 probes per compiler; all statuses and inspected properties match. Full commands/inputs/outputs remain in the result directories. Those large generated artifacts are build outputs, not checked-in source.

The existing shader suite also passed after building:

    pytest slangpy/tests/device/test_shader.py -q
    18 passed

Final verification completed with 66 observation checks per compiler and matching statuses/output properties for all 80 cases. Repository-wide pre-commit and explicit checks covering the new tools/documents passed. No production source files were changed.

## Consequences for step 2

Proceed with input-set semantics and observable resolution. Retain internal DX profile adaptation and use the tested minimal-profile approach for SPIR-V. Do not implement an NVRTC downstream-override bridge. Prioritize missing CUDA tiers and SER requirement inference upstream. Keep strict-default policy separate from adding the API: both Slang false positives and incomplete RHI reporting need attention first.

The forwarding failure remains an open historical question. These probes cover actual detected lists, SlangPy imports, and capability-sensitive operations; they do not establish that every functional-API-generated kernel works with forwarding. Metal/WebGPU runtime and other hardware/toolkit combinations remain future coverage, not successes inferred from this matrix.
