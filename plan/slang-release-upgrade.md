# Upgrade Slang and reassess compiler workarounds

This ExecPlan follows `.agents/PLANS.md`. Keep Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective current. It extends the completed capability migration in `plan/compiler-capabilities.md`; `plan/slang-compiler-workarounds.md` remains the authoritative workaround ledger.

## Purpose and context


Upgrade SlangPy's downloaded compiler from 2026.17.1 to the latest stable release verified on 2026-09-23, 2026.18.2. Preserve the migrated profile/capability API. Remove a workaround only when direct compiler probes and affected SlangPy tests establish that its removal gate passes. The compiler pin is in `external/CMakeLists.txt`; session adapters live in `src/sgl/device/compiler_target.cpp` and `shader.cpp`. The Windows build uses downloaded binaries in `build/windows-msvc/_deps/slang-src`, not local Slang master.

## Progress


- [x] (2026-09-23) Confirm latest stable release and review 2026.18, 2026.18.1, and 2026.18.2 release notes and package availability.
- [x] (2026-09-23) Upgrade the pin/cache to 2026.18.2, build, and verify the production DLL matches the probed official release DLL.
- [x] (2026-09-23) Rerun all 164 probes (150 unchanged observation checks), reproduce the separate ray-payload failure without W004, and remove the obsolete module digest guard while retaining the required source prefix.
- [x] (2026-09-23) Pass focused backend/cache/hot-reload and full functional tests, update the ledger, and pass repository-wide pre-commit plus explicit checks on this new plan.

## Release review and decision log


The official [2026.18.2 release](https://github.com/shader-slang/slang/releases/tag/v2026.18.2) includes link-time downstream options in entry-point hashes (#13215), enforces serialized-module version compatibility (#12905), and improves CUDA compilation performance. The intervening [2026.18](https://github.com/shader-slang/slang/releases/tag/v2026.18) and [2026.18.1](https://github.com/shader-slang/slang/releases/tag/v2026.18.1) notes include CUDA/HitObject emission fixes, but no documented complete capability/profile replacement. All package names used by our CMake configuration are present, including Linux x86_64 glibc 2.28.

Decision, 2026-09-23: use the latest stable release, keep the RHI submodule unchanged, and validate behavior rather than infer adapter obsolescence from unrelated cache or emission fixes. The new entry-point hash fix does not by itself establish exact CUDA architecture validation or downstream-toolchain cache identity. Also inspect the older source-module identity workaround referencing upstream #10996; it predates the migration ledger.

## Milestones and concrete steps


First update `SGL_SLANG_VERSION` and configure from `C:/projects/slangpy` with `cmake --preset windows-msvc '-DSGL_SLANG_VERSION=2026.18.2'`, then `cmake --build --preset windows-msvc-debug -j 8`. The explicit configure option is necessary because changing a CMake cache default does not replace an existing cached value. Confirm the actual DLL build tag through the probe executable.

Next configure/build `tools/compiler_capability_probe` with the downloaded headers and run both probe suites using the original inventory, the official compiler, SlangPy's separate DXC dependency, and NVRTC 12.2:

    cmake -S tools/compiler_capability_probe -B build/compiler-capability-probe -G Ninja '-DSLANG_INCLUDE_DIR=C:/projects/slangpy/build/windows-msvc/_deps/slang-src/include'
    cmake --build build/compiler-capability-probe
    python tools/compiler_capability_probe/run.py --probe build/compiler-capability-probe/compiler_capability_probe.exe --library build/windows-msvc/_deps/slang-src/bin/slang-compiler.dll --output build/compiler-capability-probe/results/release-2026.18.2 --inventory build/compiler-capability-probe/results/release/device_inventory.json --dxc build/windows-msvc/Debug/dxcompiler.dll --nvrtc 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvrtc64_120_0.dll' --suite all
    python tools/compiler_capability_probe/verify.py build/compiler-capability-probe/results/release-2026.18.2

The inventory path refers to the original migration's local probe artifacts. If those are absent, omit `--inventory` to collect current devices, and account for hardware-specific expectations in the verifier. Compare with `verify.py`'s documented 2026.17.1 expectations. Examine differences instead of silently updating expectations. Explicitly enable DXC payload qualifiers on the existing ray pipeline shaders to bypass SLANG-W004. For other candidate removals, inspect the released source and run focused regressions without the local adapter. Record evidence and status for every ledger entry.

Finally rebuild before tests. Run `pytest slangpy/tests/device/test_compiler_capabilities.py slangpy/tests/device/test_compiler_target_output.py slangpy/tests/device/test_shader.py slangpy/tests/device/test_shader_cache.py slangpy/tests/device/test_pipeline.py slangpy/tests/device/test_type_conformance.py slangpy/tests/device/slang -q`, CPU capability tests with `--device-types cpu`, native `sgl_tests.exe --test-suite=device,hot_reload,persistent_cache`, and `pytest slangpy/tests/slangpy_tests -q`. Run `pre-commit run --all-files` and repeat after automatic formatting changes. Preserve logs under `build/slang-upgrade-*.log`.

## Validation and acceptance


The loaded compiler must be 2026.18.2. Available D3D12/Vulkan/CUDA/CPU regressions must pass, including emitted targets, exact CUDA output, runtime execution, source-module identity, persistent caches, and hot reload. Retained adapters must have explicit current evidence in the ledger; removed adapters need a passing reproducer without them. Metal/WebGPU and other CUDA hardware/toolkits remain CI coverage limits. Do not claim that a compiler upgrade supplies an exact semantic capability ceiling.

## Surprises and discoveries


All 150 observation checks across 164 direct probes remain unchanged on 2026.18.2. The capability definition, profile, and CUDA code-generation files inspected have no changes between the two release tags, and the public header only adds a reflection accessor. W001-W007 retain their prior migration status.

Bypassing W004 with explicit `-enable-payload-qualifiers` still fails to compile the existing separate miss shader at SM 6.7 and 6.9. The initial whole-program payload probes continue to pass; the compilation boundary matters.

Removing both old source-module workarounds exposed a split result: Slang diagnoses name/content mismatches with E38202, but identical pathless source under another name triggers an internal dictionary collision. Restored the name prefix, retained the digest-guard removal, and strengthened returned-name assertions in the dedup test. This is now tracked as W008. The upstream collision fix was already in 2026.17.1, so this is overdue cleanup, not a new 2026.18.2 fix.

PowerShell requires quoting the version definition (`'-DSGL_SLANG_VERSION=2026.18.2'`) to prevent parsing the dotted value as separate arguments. The sandboxed download stalled; the authorized configure succeeded with network access. DXC is a separate SlangPy dependency and must be passed from `build/windows-msvc/Debug/dxcompiler.dll`, not from the downloaded Slang package.

## Outcomes and retrospective


The pin and active build now use Slang 2026.18.2. Removed the duplicate source-module SHA1 collision cache and delegated that diagnostic to Slang; retained the source prefix with a corrected upstream-removal criterion. No capability/profile/CUDA adapter meets its removal gate in this release. The separate ledger records current evidence for all eight entries.

Final validation: all 150 probe checks across 164 cases; 237 focused Python tests (17 skips); nine CPU tests (17 skips); 33 native device/cache/hot-reload cases with 5,620 assertions. Repository-wide pre-commit passes. The full functional suite passed 2,664 cases, with 238 skips and seven expected failures, matching the prior release. `git diff --check` passes. Binding docstrings were regenerated; only the removed private digest-cache documentation changed. The old and new compiler DLLs were not mixed: production and probe DLL hashes match, and every direct probe reports 2026.18.2.

## Recovery and artifacts


Changes are confined to the version pin, justified adapter/test changes, and documentation. Reconfiguring with `cmake --preset windows-msvc '-DSGL_SLANG_VERSION=2026.17.1'` restores the old compiler for comparison without changing submodules. Existing generated build artifacts need not be deleted. Preserve the probe environment JSON, result matrices, and regression logs so compiler and toolkit identity are reviewable.

Revision note, 2026-09-23: completed release review, upgrade, adapter-removal experiments, regression validation, and current-version ledger updates. No RHI submodule revision or public compiler-option semantics changed.
