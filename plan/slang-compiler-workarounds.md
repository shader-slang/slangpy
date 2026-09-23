# Slang compiler workaround ledger

Updated 2026-09-23 through milestone 4 of [the capability migration](compiler-capabilities.md).

Record every compiler workaround introduced during this migration here, and refer to its ID from implementation comments. Each entry must distinguish production behavior from a probe, name the upstream fix, and state a removal test. Update the affected compiler versions and evidence whenever changing or removing it. Do not treat a passing expected-failure probe as a reason to preserve the compiler defect.

Steps 1 and 1a added investigation tools; milestone 2 implements the opt-in resolver and its required adapters; milestone 3 adds exact CUDA output checks and backend artifact/runtime coverage. Milestone 4 applies the resolver to every session, removes legacy shader-model defaults, and replaces the historical payload profile downgrade with a narrower DXC compatibility option. Probe results are documented in [the initial probe report](compiler-capabilities-probe-results.md) and [the profile interaction report](compiler-profile-probe-results.md). Production regression coverage is in `slangpy/tests/device/test_compiler_capabilities.py` and `test_compiler_target_output.py` against pinned Slang 2026.17.1; master was probed through the standalone session API, not substituted into the production build.

## SLANG-W001: Derive a DXIL profile from selected shader-model capabilities

**Status:** Implemented for opt-in capability sessions in milestone 2; applies to all sessions since milestone 4.

**Affected versions:** Slang 2026.17.1 and local master `b4a57b15cc47d936403cc989a628bfd25d5af5d3`.

**Problem:** `_sm_6_6` or `_sm_6_9` alone produces `lib_6_3` for whole-program DXIL. A capability-selected `WaveMatch` shader is compiled as `cs_6_0` and rejected by DXC. Setting the corresponding profile produces the requested shader model and compiles the operation.

**Local adaptation:** When the new API's `profile` is omitted, `resolve_compiler_target` in `src/sgl/device/compiler_target.cpp` derives a DX profile from known resolved model requirements, with a 6.0 baseline. `SlangSession::create_session` passes it to Slang before computing the descriptor digest. Explicit profiles take precedence after reconciliation. Passing a user-requested profile is ordinary API behavior, not itself a workaround. Probe cases: `dxil_cap_*`, `dxil_profile_*`, `dxil_wave_match_*`. Production tests `test_dx_profile_reconciliation` and `test_functional_api_uses_selected_session` compile and execute WaveMatch under capability-selected SM 6.6.

**Proper Slang implementation:** Derive the effective DXC shader-model profile from the selected capability set, including implied shader-model requirements. Define how explicit conflicting profiles are diagnosed.

**Removal gate:** With no explicit profile, both whole-program probes produce their selected 6.6/6.9 models and the capability-only WaveMatch probe compiles under a suitable model. Verify all supported compiler versions before removing the adapter.

## SLANG-W002: Neutralize the implicit SPIR-V 1.5 profile bundle

**Status:** Implemented for automatic Vulkan selection in milestone 2; applies to all sessions since milestone 4.

**Affected versions:** Both investigated versions.

**Problem:** `_spirv_1_3` with no profile emits SPIR-V 1.5. The default profile also supplies feature assumptions: a shader annotated as requiring `SPV_EXT_physical_storage_buffer` passes strict checking even when only a raw version atom was supplied. Public higher SPIR-V profiles similarly bundle features beyond their version.

**Local adaptation:** Supply `spirv_1_0` as a minimal compatibility profile and add the selected raw version/features. The probes emit the requested 1.3/1.6 versions. Strict checking then rejects the missing physical-storage-buffer requirement, as intended. Cases: `spirv_baseline_*`, `spirv_bundle_*`. The resolver uses this adapter only when `profile` is omitted. Preserve an explicitly selected profile and its feature bundle; do not neutralize it or claim capability removal overrides subtract its requirements. The production regression test promotes warning 41012 to an error to distinguish the raw selection from an explicit profile's retained feature bundle.

**Proper Slang implementation:** Explicit version capability selection should override the implicit backend default. Provide a capability-only target configuration that does not add unrequested higher-profile feature bundles.

**Removal gate:** Without the baseline profile, `_spirv_1_3` emits 1.3, and strict `_spirv_1_6` rejects the missing physical-storage-buffer requirement. This adapter must be tested via `createSession()`; CLI profile-conflict validation is a separate behavior.

**Milestone-1a evidence:** Explicit profile 1.6 supplies the physical-storage-buffer requirement even with an empty input list or after that input is removed. Profile 1.0 plus only the physical-storage-buffer feature emits 1.3. The minimal-profile adapter prevents implicit bundles; it does not prohibit a feature from implying a higher version.

**Production location and coverage:** `resolve_compiler_target` in `src/sgl/device/compiler_target.cpp`, applied in `SlangSession::create_session`. `test_spirv_profiles_and_bundles` distinguishes raw versus bundled inputs, preserves explicit profile requirements after removal, and verifies a missing feature is diagnosed under the minimal automatic profile when its warning is promoted to an error.

## SLANG-W003: Suppress implicit-upgrade warning 41012

**Status:** Broad suppression removed in milestone 4. Warning 41012 is visible by default; validation remains permissive. The SER inference defect below remains unresolved.

**Affected versions:** Both investigated versions produce relevant diagnostics.

**Problem:** The current code comment attributes this to an unset CUDA profile, but CUDA profiles do not exist. More specifically, the HitObject probe infers both native and NVAPI SER requirements. A native-only selection warns that `hlsl_nvapi` is missing; an NVAPI-only selection warns that `ser_hlsl_native` is missing. Restrictive checking turns either into an error. With permissive checking, the respective native/NVAPI HLSL and DXIL compile correctly.

**Local policy:** Keep capability validation permissive for the affected SER operations until the incorrect requirements are fixed. No diagnostic-specific exception is injected. Applications may use existing warning controls to suppress or promote 41012 per session. `test_capability_upgrade_warnings_visible` verifies default visibility and explicit error promotion. Do not add both markers merely to satisfy strict checking when native output is requested: the probe with both markers emits NVAPI's HitObject representation.

**Proper Slang implementation:** Preserve alternative SER requirements correctly during inference/checking. A native path must not require NVAPI; an NVAPI path must not require the native marker. Warning text must accurately describe emitted requirements. CUDA should be configured using capabilities, independently of this fix.

**Removal gate:** `ser_native` and `ser_nvapi` both pass strict checking independently and emit the intended implementations. This remains a prerequisite for universally restrictive defaults, together with fixes to device-detection gaps. The global warning suppression has already been removed; keeping it was not necessary for permissive compilation.

## SLANG-W004: Disable DXC payload qualifiers without lowering the shader model

**Status:** Replaced the legacy 6.7-to-6.6 default downgrade in milestone 4. Implemented in `SlangSession::create_session` in `src/sgl/device/shader.cpp`.

**Affected versions:** Reproduced in the production build with pinned Slang 2026.17.1. This per-entry-point regression has not been rechecked against local master; the earlier whole-program probes passed on both versions and were insufficient to retire this workaround.

**Problem and reproducer:** `slangpy/tests/device/test_pipeline_rt.slang` declares an unannotated `Payload` used by raygeneration, miss, and closesthit shaders. At SM 6.7+, DXC requires `[raypayload]` on the separately emitted miss shader's payload type, but Slang omits it. Creating the ray pipeline fails with "type used as payload requires that it is annotated with the [raypayload] attribute". The default on the available D3D device is now 6.9, so removing the downgrade exposed this existing defect.

**Local adaptation:** For D3D profiles 6.7 and later, add `-disable-payload-qualifiers` to DXC session arguments. Preserve the selected profile and capabilities. Include the generated argument and a note in `SlangTargetInfo`, and include the option in the session digest. Explicit session `-enable-payload-qualifiers` or `-disable-payload-qualifiers` takes precedence. A link-only enable conflicting with the generated disable is rejected with instructions to set the session option. This disables payload-qualifier validation/optimization; it does not correct missing annotations.

**Regression coverage:** `test_ray_payload_compatibility` compiles actual RHI ray pipelines at 6.6/6.7/6.9, checks the unchanged profile, and verifies explicit overrides and conflicting link options. Existing `test_pipeline.py` exercises ray-tracing dispatch/readback under device defaults. Manual comparison passed both compute ray queries and ray pipelines at 6.6 and 6.9.

**Proper Slang implementation:** Generate correct payload annotations/access qualifiers for separately emitted ray-tracing entry points, including payloads introduced through `TraceRay` and hit/miss parameters.

**Removal gate:** With the compatibility argument disabled, compile and execute the existing pipeline shaders at all supported 6.7+ profiles using the minimum supported compiler. Verify emitted payload annotations for separate miss/hit outputs. Then remove the generated argument and conflict guard; retain default profile selection from device inputs.

## SLANG-W005: Validate known profile/capability conflicts using bounded dependency metadata

**Status:** Implemented as bounded validation in milestone 2, with explicit disclosure that unknown implications are not checked.

**Affected versions:** Slang 2026.17.1 and local master `b4a57b15cc47d936403cc989a628bfd25d5af5d3`.

**Problem:** Slang's session API does not consistently diagnose a profile that is lower than selected capabilities. DX strict checking can accept higher requirements while DXC still compiles for the lower model. SPIR-V version/feature inputs can raise the emitted version above the profile. Slang exposes neither a complete public implication query nor profile-family compatibility metadata. Removing detected raw version inputs is insufficient: the real Vulkan list still emits 1.6 under profile 1.3 because retained features imply it.

**Local adaptation:** `resolve_compiler_target` tracks input provenance. For native-family explicit profiles it trims higher inherited version inputs, rejects higher explicit versions, and rejects retained features with known higher requirements from either origin. `required_version` contains a bounded table for native SER, selected SPIR-V extensions, and OptiX cooperative vectors; `canonical_name` normalizes numeric DX/CUDA aliases only on their native backend. Diagnostics identify the input, origin, required version, profile, and remedy. The table is not Slang's full capability graph. Ordinary wrapper policy, such as choosing supported profile families, remains distinct from this missing upstream introspection. Python tests cover explicit versus inherited inputs, reassertion by overrides, alias conflicts/cache equivalence, SER requirements, and detected Vulkan dependent-feature conflicts when present.

**Evidence:** `interaction_dx_lower_wave_*`, `interaction_dx_6_6_with_6_9_*`, `interaction_dx_native_implied_*`, `interaction_spv_extension_implied_version`, and `interaction_device_vulkan_no_raw_versions` in `tools/compiler_capability_probe/run.py`. Source definitions include `ser_hlsl_native`/`ser_dxr`, `SPV_EXT_physical_storage_buffer`, and `SPV_KHR_cooperative_matrix` in `slang-capabilities.capdef`.

**Proper Slang implementation:** Expose versioned capability implication and profile-requirement queries, including applicable target alternatives and profile family/stage information. Provide an explicit validation policy for requested profile/version limits that agrees with downstream output. Preserve intentional additive use, such as the minimal SPIR-V profile plus higher version capabilities; do not fix this by unconditionally rejecting all profile/capability differences.

**Removal gate:** Replace local dependency facts with supported upstream queries and validate the same inputs, origins, and actionable conflict outcomes. Demonstrate the WaveMatch mismatch, native SER dependency, physical-storage-buffer version implication, and real-device Vulkan case are diagnosable without a SlangPy capability graph. Recheck newly introduced capabilities on the supported minimum compiler. The public input-set contract and provenance handling can remain when the workaround metadata is removed.

## SLANG-W006: Supply an explicit backend baseline for empty capability inputs

**Status:** Implemented in milestone 2 in `resolve_compiler_target`; applies to all sessions since milestone 4.

**Affected versions:** Empty-selection bypass observed in both probed compiler versions; production regression tests use 2026.17.1.

**Problem:** The ordinary restrictive capability check can skip an empty profile/capability selection, even though the output target implies a language baseline. See `strict_empty_inferred` versus `strict_baseline_inferred` in the step-1 probes.

**Local adaptation:** Always forward the backend's explicit baseline input: `hlsl`, `spirv`, `cuda`, `metal`, `cpp`, or `wgsl`. Report its origin as `baseline` when synthesized. A removal override cannot remove the fixed backend; report that the baseline remains. `test_defaults_and_empty` and `test_mandatory_baseline_removal` exercise the public contract and real shader execution.

**Proper Slang implementation:** Apply ordinary capability validation against the fixed target baseline even when no optional profile or capability input is supplied. This does not claim to solve entry-point requirement merging or arbitrary semantic subtraction.

**Removal gate:** The standalone empty-selection restrictive probe rejects the missing inferred requirement on every supported compiler without adding a baseline input. SlangPy may continue to report the mandatory backend assumption as API documentation even when explicitly forwarding it is no longer necessary.

## SLANG-W007: Validate explicit CUDA targets using eagerly generated PTX

**Status:** Implemented in milestone 3 in `validate_cuda_program` (`src/sgl/device/compiler_target.cpp`), called by `ShaderProgram::link` before creating the RHI program. Production tests use pinned Slang 2026.17.1 and NVRTC 12.2.

**Problem:** Slang can emit a different architecture from an explicitly selected numeric CUDA capability. Half-using code with tier 5.0 emits `sm_60`; the toolkit minimum raises tier 1.0 to `sm_50` on NVRTC 12.2. Restrictive capability checking is not an exact architecture contract. Runtime specialization, deferred compilation, and persistent-cache reads occur inside RHI, which has no public generated-code validation hook.

**Local adaptation:** If the highest forwarded numeric CUDA tier has explicit/override provenance, require a fully specialized linked component, compile every entry point with that component's actual options/toolchain, parse the PTX `.target` and `.version`, and reject target mismatches. Preserve downstream compiler diagnostics on failure. Execute this check on initial link and hot reload before RHI can use a persistent cache. Inherited highest versions and empty selections retain assumption semantics. No architecture flag is appended, no unrelated NVRTC library is loaded, and no toolkit support table is copied. The actual compile establishes toolkit acceptance; the CUDA driver still checks PTX ISA compatibility on pipeline creation.

**Cost and limitation:** Exact requests compile eagerly even for deferred pipelines, and validation may perform compilation despite an existing RHI persistent cache. Runtime interface specialization is rejected for exact requests, rather than claiming to validate code that does not exist yet. Checking `.target` is not semantic capability subtraction; a requirement annotation alone can leave the PTX target unchanged.

**Cache boundary:** `PersistentCache::expect_entry` registers a digest of validated PTX under the entry-point key. `queryCache` checks registered expectations on every read and returns a miss for different bytes, so RHI falls back to the already validated component. This handles old downstream-toolchain artifacts under the same Slang key, including subsequent stale writes; checking freshly compiled output alone would not validate the artifact RHI actually reads. Expectations are in-memory and scoped to the device/cache lifetime. The native `validated_shader_cache_entries` test covers stale, matching, and subsequently overwritten entries.

**Regression evidence:** `test_cuda_exact_target` covers half-code and toolkit-minimum uplift, matching output, overrides, and semantic-annotation limits; `test_cuda_runtime_specialization` distinguishes exact from inherited policy; the two cache tests include an incompatible pre-existing artifact under the same resolved session digest; `test_cuda_exact_target_failed_reload` verifies failed validation preserves the previous working kernel and a subsequent valid reload succeeds. `test_emitted_version` checks PTX target/version alongside real execution.

**Proper Slang implementation:** Expose an exact CUDA architecture option that validates both generated-code requirements and the selected downstream toolkit and emits one architecture flag. Include that contract in compiler/code-cache identity. RHI should apply any output validation hook after specialization and on both fresh code and cache hits, allowing lazy compilation without a wrapper-side eager pass.

**Removal gate:** Tier 5.0 plus half code and tier 1.0 below the toolkit minimum fail with clear upstream errors; matching tiers compile; unsupported toolkit targets fail without substitution; runtime-specialized, deferred, cached, and hot-reloaded programs honor the same selection. Then remove eager generation/parsing and the runtime-specialization restriction from SlangPy.

## Remaining upstream gaps

**CUDA tiers:** RHI and NVRTC 12.2 support this machine's CC 7.5, but Slang does not recognize `cuda_sm_7_5` or `_cuda_sm_7_5`. The RHI-style recognized-name filter in the probe reduces its device list to CC 7.0, and generated PTX targets `sm_70`. Slang also lacks 8.6 and newer numeric tiers. Fix the capability definitions and downstream mapping upstream. Until then, disclose filtering; do not promise an exact unsupported tier.

**CUDA exact ceilings:** SLANG-W007 rejects mismatched emitted architectures for explicit numeric selections on fully specialized programs. A complete upstream contract is still needed for runtime specialization, lazy/cache-efficient compilation, and semantic capability constraints. Explicit entry-point requirements can bypass the normal restrictive comparison without changing the emitted target; matching PTX is not proof of a semantic ceiling.

**RHI subgroup reporting:** The physical Vulkan device supports wave operations, but its reported compiler list lacks `spvGroupNonUniformArithmetic`; strict wave compilation fails. This belongs primarily in slang-rhi discovery, not in Slang's compiler implementation. Do not conceal it in a generic Slang workaround. The int64-atomic probe does pass strict checking despite the raw-list gap, so the two findings must not be conflated.

## Rejected workaround: Append an NVRTC architecture override

`cuda_sm_8_0` plus downstream `--gpu-architecture=compute_86` fails on NVRTC 12.2 in both compiler versions:

    nvrtc: error : --gpu-architecture (-arch) defined more than once

Even repeating the same `compute_80` architecture fails. Slang emits its own architecture option before forwarding downstream arguments. Thus argument order does not provide a working bridge on this toolchain. Do not implement this proposed fallback. Prefer upstream tier support or an explicit upstream architecture-selection option that resolves to one validated NVRTC argument.

## Tracking discipline

When implementing a later milestone, promote a probe-only entry to implemented only after naming the production file/function, adding regression coverage, and recording the applicable compiler versions. New entries require an upstream owner/fix and a concrete removal criterion. External issue URLs should be added if an issue is actually filed; this task did not file any issues.
