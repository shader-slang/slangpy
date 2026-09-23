# Explicit profile and capability interactions: milestone 1a

Date: 2026-09-23. This completes milestone 1a of [the capability migration plan](compiler-capabilities.md). The earlier [step-1 report](compiler-capabilities-probe-results.md) records the environment, original 80 probes, and compiler build commands. The [workaround ledger](slang-compiler-workarounds.md) tracks adapters and upstream fixes separately.

## Scope and method

The new matrix contains **84 interaction probes per compiler**, using Slang **2026.17.1** and local master **2026.18.2-3-gb4a57b15c**. Compiler library hashes remain the same as step 1. Both use DXC 1.9.2602.17 and NVRTC 12.2. SlangPy was rebuilt before the experiments; the native probe was rebuilt after adding profile-lookup reporting. The step-1 TITAN RTX device inventory is reused to hold detected inputs fixed.

The SlangPy checkout for this milestone is `50efaff058777b7201f95922e20d9cb4bdc9d0f1`, with the probe/documentation changes described here. RHI and local Slang source revisions remain those recorded in step 1.

These are direct `createSession()` experiments, without the proposed SlangPy resolver. `interaction_*` cases in `tools/compiler_capability_probe/run.py` retain their exact inputs and generated artifacts. `verify.py` checks all 84 cases individually, including relevant diagnostics and failure stages, then compares statuses and inspected output properties between the two compilers. **All 84 checks pass for each compiler, and all compared observations match.** These checks describe current behavior, including defects; they do not prescribe preserving it.

The source-level requirement probes deliberately separate capability validation from actual instruction use. A function annotated with a higher requirement but returning a constant can pass compilation to a lower model. WaveMatch and HitObject probes additionally exercise real downstream instruction requirements. The physical-storage-buffer helper checks permission to use a capability; it does not generate a physical-storage-buffer instruction. The resulting binary version still exposes how target assumptions are combined.

No production API or compiler behavior was changed. Higher-model DXIL/PTX and SER were compiled and inspected, not executed. Metal, WGSL, and C++ cross-profile cases cover source generation, not downstream/runtime behavior. This does not broaden the hardware/toolkit coverage claimed in step 1.

## Measured interactions

| Inputs / case suffixes (all prefixed `interaction_`) | Measured behavior on both compilers |
| --- | --- |
| `dx_lower_raw_*`, `dx_lower_inferred_*`: `sm_6_0` profile plus `_sm_6_6` | Simple and annotated-requirement shaders emit `cs_6_0`, including under strict checking. The capability does not update DXC's model. |
| `dx_lower_wave_*`: same inputs, actual WaveMatch | Both modes reach code generation, then DXC rejects WaveMatch for `cs_6_0`. Strict checking did not detect the configuration mismatch. |
| `dx_equal_wave_*`, `dx_higher_wave_*`, `dx_profile_only_wave_*` | `sm_6_6` compiles WaveMatch to `cs_6_6` with equal, lower, or no additional version input. A lower capability is not an upper bound. |
| `dx_6_6_with_6_9_*` / `dx_6_6_without_6_9_*` | With `_sm_6_9`, a helper requiring that model passes strict checking but emits `cs_6_6`. Without it, strict checking rejects; permissive checking warns and still emits `cs_6_6`. The raw requirement annotation also produces an unrelated internal-name warning. |
| `dx_native_implied_*`, `dx_native_alias_*` | `ser_hlsl_native` and `ser_dxr` each satisfy a native-SER requirement under a 6.6 profile, including strict mode. The helper-only shader still emits `cs_6_6`. Removing raw model inputs does not remove these implied higher requirements. |
| `dx_native_removed_*` | Removing the native marker makes the helper fail strict checking; permissive mode warns. |
| `dx_ser_lower_permissive` / `dx_ser_equal_permissive` | Actual native HitObject under `sm_6_6` fails DXC's 6.9 availability checks; under `sm_6_9` it compiles to `lib_6_9`. Both expose the existing missing-NVAPI warning. |
| `dx_ser_lower_strict`, `dx_ser_equal_strict` | Both fail module loading for the missing NVAPI marker, confirming SLANG-W003 independently of the downstream model mismatch. |
| `dx_ser_both_6_6_*` | Both markers compile to `lib_6_6`, even in strict mode. The previously observed NVAPI preference makes this possible; it does not validate native SER at 6.6. |
| `dx_ser_both_lower_*` | With profile 6.0, both markers reach DXC but fail library-target validation (`unsupported lib_6_1 or lib_6_2`). This is not evidence of a native instruction being accepted at a lower model. |
| `spv_1_0_*`, `spv_1_3_*`, `spv_1_6_*`: simple shaders | Same-family raw version inputs and profiles combine to the higher version. A 1.0/1.3 profile plus raw 1.6 emits 1.6; a 1.6 profile plus raw 1.0 stays 1.6. |
| Same SPIR-V matrix, `spirv_bundle` shaders | Raw 1.6 alone with a 1.0/1.3 profile does not satisfy the physical-storage-buffer requirement. The public `spirv_1_6` capability alias does, as does an explicit 1.6 profile with no capability inputs. |
| `spv_feature_added`, `spv_feature_removed` | With an explicit 1.6 profile, adding and then omitting the physical-storage-buffer input both pass strict checking and emit 1.6. Removing an input cannot subtract the profile's bundle. These cases simulate the post-override lists; there is no public Slang disable operation. |
| `spv_extension_implied_version` | Profile 1.0 plus only `SPV_EXT_physical_storage_buffer` emits 1.3. A feature input can raise the version without any explicit raw version input. |
| `device_d3d12_all`, `device_d3d12_no_raw_versions` | The real device list with profile 6.6 emits `cs_6_6`, before and after removing all raw version inputs. Simple output alone cannot reveal the higher assumptions retained in the first case. |
| `device_vulkan_all`, `device_vulkan_no_raw_versions` | The real device list with profile 1.3 emits 1.6, even after removing every `_spirv_*` input. Remaining feature dependencies still raise it. |

Source evidence supports the feature dependencies: `C:/projects/slang/source/slang/slang-capabilities.capdef` defines native SER at line 1368 as requiring `_sm_6_9`; physical storage buffers at line 540 require `_spirv_1_3`; cooperative matrices at line 683 and cooperative vectors at line 679 require `_spirv_1_6`. The actual Vulkan list includes those cooperative features. This explains examples, not a claim that these are its only higher-version dependencies.

## Profile names, families, and stages

| Target and profile | Observation | Planned SlangPy treatment |
| --- | --- | --- |
| DXIL with `cs_6_6` | Compute shader emits `cs_6_6`. | Accept matching DX profiles; prefer stage-neutral `sm_*` for sessions containing multiple stages. |
| DXIL with `ps_6_6`, compute entry point | Warning E36112 and error E36107 during module loading. | Preserve stage meaning and surface Slang diagnostics; do not silently strip the stage. |
| DXIL with `spirv_1_6` | Strict simple shader rejects a missing SM capability. | Reject unsupported profile families earlier. |
| DXIL with `glsl_460` | Simple shader compiles as `cs_6_0`. | Successful compilation is not evidence that the requested profile controls DXC; exclude this combination from the supported new contract. |
| SPIR-V with `sm_6_6` | Emits 1.4, even when `_spirv_1_0` is also supplied. | Preserve as an advanced Slang cross-family mapping, with an explicit capability list required. Do not invent a universal SM-to-Vulkan conversion. |
| SPIR-V with `glsl_460` | Emits 1.3. | Same explicit-list rule for advanced cross-family mappings. |
| SPIR-V with `metallib_2_4` | Simple shader emits the default 1.5. | Reject this unsupported combination. |
| PTX with `sm_6_6` or `spirv_1_6`, plus `cuda_sm_8_0` | Both emit `sm_80`; neither supplies CUDA architecture control. | Require `profile=None` for CUDA; use CUDA capabilities. |
| Metal with `metallib_2_4` or `sm_6_6` | Both simple source-generation cases succeed. | Support Metal-family profiles only; no claim about the compiled Metal library version follows from these probes. |
| WGSL or C++ with `sm_6_6` | Simple source generation succeeds. | Require `profile=None`; these backends have no appropriate profile family in the inspected compiler. |
| Misspelled `sm_6_typo`, nonexistent `spirv_9_9`, and `cuda_sm_8_0` as a profile | `findProfile` returns unknown; the probe rejects before session creation. | Report an actionable lookup error. `cuda_sm_8_0` remains a valid capability name. |

The supported-family policy above is a SlangPy API decision based on useful, documented meaning. It is not a claim that Slang rejects every excluded combination. Profile lookup alone does not validate family compatibility. Stage checks may require an entry point and therefore remain compiler diagnostics at module/entry-point compilation time.

## Resolver rules selected from this evidence

Keep `profile=None` automatic. For native-family explicit profiles, preserve the requested profile, remove only higher **inherited version inputs**, and record each removal. Explicit capability-list entries and `True` overrides retain user provenance even when they duplicate a detected name. Reject a known higher explicit version rather than dropping it. Lower version inputs remain compatible because they express available features, not ceilings.

For a retained feature with a known higher-version dependency, fail with the capability, origin, required version, and requested profile. Apply this rule whether the feature was inherited or explicit. Do not silently delete features merely to make a profile fit. For example, an inherited `SPV_KHR_cooperative_matrix` conflicts with an explicit `spirv_1_3`; users can remove that feature and other reported conflicts, provide an authoritative list, or select an appropriate profile. Native SER with `sm_6_6` similarly needs a specific conflict error.

Keep these checks bounded to documented version-family adapters and known dependencies, with regression probes and ledger entries. There is no public full implication query; accepted inputs cannot be advertised as proving a universal capability or emitted-version ceiling. Unknown implications remain Slang's responsibility, and this validation limit must be visible in documentation and resolution notes. Broader dependency metadata belongs upstream.

Allow native DX profiles for D3D12, SPIR-V profiles for Vulkan, and Metal profiles for Metal, subject to compiler lookup and actual device/toolchain validation. For Vulkan's advanced DX/GLSL profiles, require `capabilities` to be explicitly supplied (an empty list is valid), pass that profile through, and report that Slang supplies the cross-family mapping. There is no implicit filtering or guessed version ceiling for that advanced mode. Same-family version inputs in it are additional assumptions under Slang's semantics. This avoids silently mixing a legacy profile mapping with all newer device defaults. The deprecated `shader_model` path remains separate during migration and retains its historical Vulkan behavior.

Boolean removal still means removal from the input list. If the profile supplies a removed feature, report that known fact; do not treat the removal as a request to rewrite the profile or as a contradiction by itself. To avoid a SPIR-V profile bundle, use automatic selection plus an authoritative raw-version/feature list.

These rules are now part of the plan. No resolver has been implemented in this milestone.

## Reproduction and verification

Run from `C:/projects/slangpy` with the previously built compiler libraries and inventory:

    cmake --build --preset windows-msvc-debug
    cmake --build build/compiler-capability-probe
    python tools/compiler_capability_probe/run.py --probe build/compiler-capability-probe/compiler_capability_probe.exe --library build/windows-msvc/Debug/slang-compiler.dll --output build/compiler-capability-probe/results/profiles-release --inventory build/compiler-capability-probe/results/release/device_inventory.json --dxc build/windows-msvc/Debug/dxcompiler.dll --nvrtc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvrtc64_120_0.dll" --filter interaction_
    python tools/compiler_capability_probe/run.py --probe build/compiler-capability-probe/compiler_capability_probe.exe --library build/slang-capabilities-master/Release/bin/slang-compiler.dll --output build/compiler-capability-probe/results/profiles-master --inventory build/compiler-capability-probe/results/release/device_inventory.json --dxc build/windows-msvc/Debug/dxcompiler.dll --nvrtc "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvrtc64_120_0.dll" --filter interaction_
    python tools/compiler_capability_probe/verify.py build/compiler-capability-probe/results/profiles-release build/compiler-capability-probe/results/profiles-master

Expected: 84 observation checks and 84 probes per compiler; all statuses and inspected output properties match. `--suite profiles` is equivalent to selecting the complete `interaction_` group. Without `--filter`, the runner defaults to both the original and interaction suites: 164 cases with this inventory. `--suite baseline` reproduces the original 80 cases. The verifier recognizes either complete suite or both together. Other substring-filtered partial suites are for manual diagnosis, not complete baseline verification.

Final regression verification reran both complete suites into `build/compiler-capability-probe/results/combined-release` and `combined-master`, using the same commands without `--filter` and with those output directories. The verifier reported **150 observation checks passed per compiler, 164 probes each, and matching statuses and inspected properties for all 164 cases**. There were no harness errors. Repository-wide pre-commit passed after formatting, and the new report was checked explicitly. These were probe-tool changes, so no new production Python API tests were needed; the existing shader-suite result in the step-1 report is historical, not a fresh run in milestone 1a.
