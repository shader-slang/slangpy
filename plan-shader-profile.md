# Implement backend-derived Slang target profiles

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy currently replaces its old `ShaderModel` enum with strings, but it still discovers profiles from a hard-coded `sm_6_x` list. After this work, each device reports the genuine Slang target profiles supported by its backend: D3D12 reports DXIL-compatible shader models, Vulkan reports native SPIR-V profiles plus its emulated shader-model profiles, and Metal reports Metal library profiles supplied by slang-rhi. Users can inspect those profiles, select one through `SlangCompilerOptions.profile`, or omit the option to use a backend-specific default.

The observable result is that `device.supported_profiles`, `device.has_profile()`, and `device.default_profile` agree with one another; explicit advertised profiles successfully compile a minimal shader; invalid or cross-backend profiles fail with a clear error; and backends without genuine target profiles use an empty profile instead of a fabricated shader model.

## Progress

- [x] (2026-07-06) Reviewed the existing string-profile changes, identified the stale generated bindings, and verified that CUDA architecture names are capabilities rather than target profiles.
- [x] (2026-07-06) Chose native SPIR-V as Vulkan's default and included upstream slang-rhi Metal profile reporting in scope.
- [x] (2026-07-06) Audited the clean parent worktree, clean slang-rhi submodule, current backend capability reporting, tests, and code-generation pipeline.
- [x] (2026-07-06) Added runtime-version-derived Metal language capabilities and tests to slang-rhi and committed them as `679814c` on `codex/proper-target-profiles`.
- [x] (2026-07-06) Replaced hard-coded profile discovery with normalized, compiler-filtered, numerically ordered backend-derived profiles.
- [x] (2026-07-06) Applied genuine profiles consistently during Slang session creation and updated the Python dispatch helper to use the backend default when omitted.
- [x] (2026-07-06) Added direct profile inventory, ordering, selection, session, and compilation tests and updated existing profile-dependent tests.
- [x] (2026-07-06) Regenerated binding documentation, the Python stub, API documentation, and tutorial output; committed the tutorial refresh as samples revision `9afcac9a`.
- [x] (2026-07-06) Built before testing, ran the focused Python and full C++ suites outside the sandbox, and ran pre-commit successfully on every changed file in both repositories.

## Surprises and Discoveries

- Observation: `cuda_sm_8_9` is not accepted by Slang's profile parser, but it is accepted as a compiler capability.
  Evidence: the bundled `slangc` rejects `-profile cuda_sm_8_9` and accepts `-capability cuda_sm_8_9`. CUDA selection is therefore explicitly deferred.

- Observation: slang-rhi already reports cumulative `_spirv_1_x` capabilities for Vulkan and cumulative `_sm_6_x` capabilities for D3D12, while Vulkan exposes its approximate shader-model support as features.
  Evidence: `external/slang-rhi/src/vulkan/vk-device.cpp` and `external/slang-rhi/src/d3d12/d3d12-device.cpp` populate those respective sets.

- Observation: slang-rhi defines Metal 2.3 through 3.1 capability atoms but its Metal backend currently reports only the generic `metal` capability.
  Evidence: `external/slang-rhi/include/slang-rhi/capabilities.h` contains the atoms and `external/slang-rhi/src/metal/metal-device.cpp` does not add them.

- Observation: the normal SlangPy build does not regenerate `src/slangpy_ext/py_doc.h` automatically.
  Evidence: `slangpy_pydoc` is a standalone custom target, and the current build fails on missing documentation symbols for the new bindings.

- Observation: expected `SGL_CHECK` failures trigger an attached-debugger breakpoint in the Windows Debug build before Python can catch the translated exception.
  Evidence: the explicit invalid-profile test terminated with `0x80000003` in Debug. The test is retained with a Debug-build skip so Release CI exercises the exception path.

- Observation: `docs/generate_api.py` refreshes many previously stale API entries and embeds process-specific object addresses and build metadata.
  Evidence: regenerating `docs/generated/api.rst` produced a broad generated diff beyond the profile entries. The generated result is retained as requested; the generator itself was not changed by this work.

- Observation: an initial parent-repository `pre-commit run --all-files` attempt reported transient read-only errors from the EOF fixer for four tracked headers.
  Evidence: the final all-files invocation completed with every hook passing, so no persistent repository blocker remains.

## Decision Log

- Decision: Treat only true Slang target profiles as profiles; do not include CUDA architecture capabilities, GLSL profiles, CPU, or WGPU in this change.
  Rationale: CUDA uses compiler capabilities, and no current SlangPy backend emits GLSL. CPU and WGPU have no meaningful profile inventory here.
  Date/Author: 2026-07-06 / user and Codex

- Decision: D3D12 advertises only `sm_6_x` profiles.
  Rationale: the SGL D3D12 target is DXIL; `sm_5_1` requires the DXBC path and fails through the current DXIL target.
  Date/Author: 2026-07-06 / Codex

- Decision: Vulkan advertises both native `spirv_1_x` profiles and compatible emulated `sm_6_x` profiles, and defaults to its highest SPIR-V profile.
  Rationale: the user selected native SPIR-V default behavior despite the intentional behavior change from the old shader-model default.
  Date/Author: 2026-07-06 / user

- Decision: Metal profile availability is reported by slang-rhi, not inferred from native handles inside SlangPy.
  Rationale: backend capability detection belongs at the RHI boundary and can then be shared by every slang-rhi consumer.
  Date/Author: 2026-07-06 / user and Codex

- Decision: Metal reports cumulative language profiles by runtime macOS version: 2.3 on 11+, 2.4 on 12+, 3.0 on 13+, 3.1 on 14+, 3.2 on 15+, and 4.0 on 26+.
  Rationale: Metal library profiles describe available compiler language versions, not GPU feature families.
  Date/Author: 2026-07-06 / Codex

- Decision: Profile ordering is deterministic by family and numeric major/minor version, while default selection explicitly filters the backend's preferred family rather than relying on list position.
  Rationale: lexical ordering puts `6.10` before `6.9`, and Vulkan intentionally exposes two families.
  Date/Author: 2026-07-06 / Codex

## Outcomes and Retrospective

The implementation is complete. SlangPy now derives profiles from RHI capabilities and Vulkan features, accepts only the `sm`, `spirv`, and `metallib` families appropriate to the active backend, filters every candidate through the pinned Slang compiler, and sorts profiles by family and numeric version. D3D12 defaults to the highest supported shader model with the 6.7 workaround, Vulkan defaults to native SPIR-V, Metal defaults to the highest recognized Metal library profile, and CUDA keeps an empty profile.

slang-rhi revision `679814c` adds cumulative Metal profile reporting from the runtime macOS major version. The pure mapping test built and passed on Windows; the actual Metal device integration path remains a macOS CI validation because no Metal runtime is available on the development host. The samples revision `9afcac9a` records the refreshed tutorial output.

Final validation on Windows:

- `cmake --build --preset windows-msvc-debug` succeeded after the final formatting pass.
- The focused Python profile/device/shader matrix passed with 142 tests passed and 47 skipped. After centralizing backend and compiler filtering, the directly affected device/session subset was rerun with 54 passed and 3 intentional Debug-build skips.
- `python tools/ci.py unit-test-cpp` passed all 182 executed test cases and 14,636 assertions, with one existing skipped case.
- Standalone slang-rhi configuration and Debug build succeeded, and `slang-rhi-tests.exe -tc=metal-profile-capabilities` passed 28 assertions.
- `pre-commit run --all-files` completed with every parent-repository hook passing, and the full hook set also passed for every changed slang-rhi file.

CUDA architecture capability selection remains intentionally deferred. No CUDA capability was reinterpreted as a target profile, and the existing warning suppression now documents that boundary.

## Context and Orientation

`external/slang-rhi` is a Git submodule containing the backend implementations that determine hardware and runtime capabilities. Its public capability names are defined in `external/slang-rhi/include/slang-rhi/capabilities.h`. D3D12 and Vulkan already publish the information SlangPy needs, but Metal must be extended to publish `metallib_x_y` atoms.

`src/sgl/device/device.cpp` constructs the higher-level SGL device and currently builds `m_supported_profiles` before it queries RHI capabilities. That discovery must move after feature and capability collection. `src/sgl/device/shader.cpp` creates Slang sessions and currently validates every call against a fabricated non-empty profile and assigns a target profile only for D3D12/Vulkan. It must accept an empty profile for backends without genuine profiles and assign every non-empty validated profile, including Metal.

The Python API is bound in `src/slangpy_ext/device`, while `slangpy/testing/helpers.py` and tests under `slangpy/tests/device` exercise it. `src/slangpy_ext/py_doc.h`, `slangpy/__init__.pyi`, and `docs/generated/api.rst` are generated artifacts that must be refreshed after the native API compiles.

## Plan of Work

First, create a local slang-rhi topic branch. Extend the capability enum with `metallib_3_2` and `metallib_4_0`. Add a small pure helper that maps a macOS major version to the cumulative Metal profile capability list, use `NS::ProcessInfo::processInfo()->operatingSystemVersion()` in the Metal device initialization path, and test the mapping without requiring a GPU. A Metal integration assertion should verify that an actual device exposes the expected default-era capability. Build and test slang-rhi outside the sandbox, commit the submodule change, and leave the parent repository pointing at that commit.

Second, introduce internal profile parsing and normalization helpers in the SGL device implementation. The helper recognizes only `sm_<major>_<minor>`, `spirv_<major>_<minor>`, and `metallib_<major>_<minor>`, stores a family plus numeric version, and produces canonical lowercase names. D3D12 candidates come from `_sm_6_x` RHI capabilities. Vulkan candidates come from `_spirv_1_x` capabilities plus `sm_6_x` RHI features. Metal candidates come from `metallib_x_y` capabilities. Every candidate is checked with `IGlobalSession::findProfile`; unknown profiles are logged and omitted. Duplicates are removed and the public list is sorted by family and numeric version.

`Device::default_profile()` then explicitly selects the highest `sm` profile on D3D12, highest `spirv` profile on Vulkan, and highest `metallib` profile on Metal. The SM 6.7 workaround selects 6.6 only when 6.6 is actually advertised. CUDA, CPU, WGPU, and any device without a genuine candidate return an empty string. No fallback profile is invented.

Third, update Slang session creation. An empty compiler option resolves to `default_profile()`. Validation and `findProfile()` run only for a non-empty result, and the resolved profile is assigned to `target_desc.profile` independently of device type. Shader-target major/minor macros are emitted only for a parsed shader-model profile. Keep the CUDA warning suppression with a comment that architecture capability selection remains deferred. Change `dispatch_compute()` to accept `profile: str | None = None`; explicit unsupported profiles cause the helper to skip its feature-dependent test, while `None` leaves the option empty and lets the device choose its default.

Fourth, add focused tests. Pure C++ tests cover parsing, normalization, numeric ordering, duplicate removal, compiler filtering, and default-family selection. Python tests cover public inventory consistency, defaults, invalid profiles, explicit compilation, and omitted profiles on every enabled backend. Vulkan assertions require both native SPIR-V and emulated shader-model families and a SPIR-V default. Metal assertions compare the reported maximum with the running macOS version. Existing numeric shader tests continue to request explicit shader-model profiles where those features are being tested.

Finally, regenerate `py_doc.h`, build, allow the normal stub target to refresh `slangpy/__init__.pyi`, regenerate `docs/generated/api.rst`, and refresh the tutorial output. Search the repository for retired `ShaderModel`, `shader_model`, and `supported_shader_model` references. Run pre-commit after all generated and hand-written changes are complete.

## Concrete Steps

All CMake builds and all tests run outside the sandbox.

From `C:\src\slangpy\external\slang-rhi`, configure and verify the upstream change on supported platforms:

    cmake --preset msvc
    cmake --build build --config Debug
    .\build\Debug\slang-rhi-tests.exe

On macOS, run the corresponding default preset/build and Metal device checks:

    cmake --preset default
    cmake --build build --config Debug
    ./build/Debug/slang-rhi-tests -check-devices

From `C:\src\slangpy`, regenerate binding docs before the main build, then build:

    cmake --build --preset windows-msvc-debug --target slangpy_pydoc
    cmake --build --preset windows-msvc-debug

Run the focused Python tests only after the build succeeds:

    pytest slangpy/tests/device/test_device.py slangpy/tests/device/test_shader.py -v
    pytest slangpy/tests/device/slang/test_float16.py slangpy/tests/device/slang/test_float64.py slangpy/tests/device/slang/test_uint64.py -v

Regenerate API docs and run formatting/checks:

    python docs/generate_api.py
    pre-commit run --all-files

If pre-commit modifies files, rerun it until it exits cleanly. Then rerun the build and focused tests if any source or generated binding artifact changed.

## Validation and Acceptance

On D3D12, `supported_profiles` contains each reported DXIL-compatible `sm_6_x` profile in numeric order, including `sm_6_10` when the device reports it. `default_profile` is the highest supported shader model except that an otherwise-default `sm_6_7` resolves to `sm_6_6` when available.

On Vulkan, `supported_profiles` contains `spirv_1_x` profiles derived from RHI capabilities and `sm_6_x` profiles derived from emulated features. `default_profile` is the highest advertised SPIR-V profile. A minimal compute module compiles with every advertised profile.

On Metal, slang-rhi exposes cumulative `metallib_x_y` capabilities appropriate to the running macOS release. SlangPy filters them through the pinned compiler, advertises the recognized subset, chooses the highest as default, and compiles a minimal module for every advertised profile.

On CUDA, CPU, and WGPU, `default_profile` is empty unless a future genuine target profile is explicitly added. Session creation with an omitted profile still succeeds. Explicit invalid or cross-backend profiles fail with an error that lists the advertised profiles.

For every backend, `has_profile(name)` is true exactly when `name` appears in `supported_profiles`, every non-empty default appears in the list, and list ordering remains stable and numeric. The generated Python stub and API docs expose `profile`, `supported_profiles`, `has_profile`, and `default_profile`, with no retired shader-model API names remaining.

## Idempotence and Recovery

Profile discovery is derived on each device construction and does not persist state. Generated documentation and stubs can be safely regenerated. If the slang-rhi change cannot be validated on macOS locally, keep its isolated topic commit and rely on macOS CI for the Metal integration portion while still running the pure mapping tests on the available platform where possible.

Do not reset unrelated user changes. If code generation changes unrelated sections because of tool-version drift, inspect and separate those changes before proceeding. If a build fails after `py_doc.h` generation, preserve the generated header and fix the underlying source issue before rerunning the normal build.

## Artifacts and Notes

The existing string-profile commit removed the enum but left generated artifacts stale; the first full debug build fails on missing `__doc_sgl_Device_supported_profiles`, `__doc_sgl_Device_has_profile`, `__doc_sgl_Device_default_profile`, and `__doc_sgl_SlangCompilerOptions_profile` identifiers. Regenerating `py_doc.h` is therefore a prerequisite, not optional cleanup.

The bundled compiler accepts `spirv_1_5` through `findProfile`/`-profile`, rejects `cuda_sm_8_9` as a profile, and accepts it as a capability. This is the acceptance boundary for the current work.

## Interfaces and Dependencies

The final public interfaces remain:

    const std::vector<std::string>& Device::supported_profiles() const;
    bool Device::has_profile(std::string_view profile) const;
    std::string Device::default_profile() const;

    struct SlangCompilerOptions {
        std::string profile;
    };

Python exposes `Device.supported_profiles -> list[str]`, `Device.has_profile(profile: str) -> bool`, `Device.default_profile -> str`, and `SlangCompilerOptions.profile: str`.

The slang-rhi dependency adds Metal capability atoms for 3.2 and 4.0 and makes the Metal backend report cumulative metallib profile capabilities. SlangPy consumes only names supplied by the active RHI device and recognized by its pinned Slang global session.
