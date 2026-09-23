# Slang compiler workaround ledger

Updated 2026-09-23 through milestone 3 of [the capability migration](compiler-capabilities.md).

Record every compiler workaround introduced during this migration here, and refer to its ID from implementation comments. Each entry must distinguish production behavior from a probe, name the upstream fix, and state a removal test. Update the affected compiler versions and evidence whenever changing or removing it. Do not treat a passing expected-failure probe as a reason to preserve the compiler defect.

Steps 1 and 1a added investigation tools; milestone 2 implements the opt-in resolver and its required adapters; milestone 3 adds exact CUDA output checks and backend artifact/runtime coverage. Existing legacy defaults remain in place for callers who do not opt in. Probe results are documented in [the initial probe report](compiler-capabilities-probe-results.md) and [the profile interaction report](compiler-profile-probe-results.md). Production regression coverage is in `slangpy/tests/device/test_compiler_capabilities.py` and `test_compiler_target_output.py` against pinned Slang 2026.17.1; master was probed through the standalone session API, not substituted into the production build.

## SLANG-W001: Derive a DXIL profile from selected shader-model capabilities

**Status:** Implemented for opt-in capability sessions in milestone 2. Legacy sessions continue setting profiles through `shader_model`.

**Affected versions:** Slang 2026.17.1 and local master `b4a57b15cc47d936403cc989a628bfd25d5af5d3`.

**Problem:** `_sm_6_6` or `_sm_6_9` alone produces `lib_6_3` for whole-program DXIL. A capability-selected `WaveMatch` shader is compiled as `cs_6_0` and rejected by DXC. Setting the corresponding profile produces the requested shader model and compiles the operation.

**Local adaptation:** When the new API's `profile` is omitted, `resolve_compiler_target` in `src/sgl/device/compiler_target.cpp` derives a DX profile from known resolved model requirements, with a 6.0 baseline. `SlangSession::create_session` passes it to Slang before computing the descriptor digest. Explicit profiles take precedence after reconciliation. Passing a user-requested profile is ordinary API behavior, not itself a workaround. Probe cases: `dxil_cap_*`, `dxil_profile_*`, `dxil_wave_match_*`. Production tests `test_dx_profile_reconciliation` and `test_functional_api_uses_selected_session` compile and execute WaveMatch under capability-selected SM 6.6.

**Proper Slang implementation:** Derive the effective DXC shader-model profile from the selected capability set, including implied shader-model requirements. Define how explicit conflicting profiles are diagnosed.

**Removal gate:** With no explicit profile, both whole-program probes produce their selected 6.6/6.9 models and the capability-only WaveMatch probe compiles under a suitable model. Verify all supported compiler versions before removing the adapter.

## SLANG-W002: Neutralize the implicit SPIR-V 1.5 profile bundle

**Status:** Implemented for automatic Vulkan selection on the opt-in path in milestone 2.

**Affected versions:** Both investigated versions.

**Problem:** `_spirv_1_3` with no profile emits SPIR-V 1.5. The default profile also supplies feature assumptions: a shader annotated as requiring `SPV_EXT_physical_storage_buffer` passes strict checking even when only a raw version atom was supplied. Public higher SPIR-V profiles similarly bundle features beyond their version.

**Local adaptation:** Supply `spirv_1_0` as a minimal compatibility profile and add the selected raw version/features. The probes emit the requested 1.3/1.6 versions. Strict checking then rejects the missing physical-storage-buffer requirement, as intended. Cases: `spirv_baseline_*`, `spirv_bundle_*`. The opt-in API uses this adapter only when `profile` is omitted. Preserve an explicitly selected profile and its feature bundle; do not neutralize it or claim capability removal overrides subtract its requirements. The production regression test promotes warning 41012 to an error to distinguish the raw selection from an explicit profile's retained feature bundle.

**Proper Slang implementation:** Explicit version capability selection should override the implicit backend default. Provide a capability-only target configuration that does not add unrequested higher-profile feature bundles.

**Removal gate:** Without the baseline profile, `_spirv_1_3` emits 1.3, and strict `_spirv_1_6` rejects the missing physical-storage-buffer requirement. This adapter must be tested via `createSession()`; CLI profile-conflict validation is a separate behavior.

**Milestone-1a evidence:** Explicit profile 1.6 supplies the physical-storage-buffer requirement even with an empty input list or after that input is removed. Profile 1.0 plus only the physical-storage-buffer feature emits 1.3. The minimal-profile adapter prevents implicit bundles; it does not prohibit a feature from implying a higher version.

**Production location and coverage:** `resolve_compiler_target` in `src/sgl/device/compiler_target.cpp`, applied in `SlangSession::create_session`. `test_spirv_profiles_and_bundles` distinguishes raw versus bundled inputs, preserves explicit profile requirements after removal, and verifies a missing feature is diagnosed under the minimal automatic profile when its warning is promoted to an error.

## SLANG-W003: Suppress implicit-upgrade warning 41012

**Status:** Existing broad suppression in `SlangSession::create_session`; no new suppression added in step 1. Probe compilation deliberately exposes these diagnostics.

**Affected versions:** Both investigated versions produce relevant diagnostics.

**Problem:** The current code comment attributes this to an unset CUDA profile, but CUDA profiles do not exist. More specifically, the HitObject probe infers both native and NVAPI SER requirements. A native-only selection warns that `hlsl_nvapi` is missing; an NVAPI-only selection warns that `ser_hlsl_native` is missing. Restrictive checking turns either into an error. With permissive checking, the respective native/NVAPI HLSL and DXIL compile correctly.

**Local adaptation under investigation:** Keep capability validation permissive for the affected SER operations until the incorrect requirements are fixed. No new production exception has been introduced. Do not add both markers merely to satisfy strict checking when native output is requested: the probe with both markers emits NVAPI's HitObject representation.

**Proper Slang implementation:** Preserve alternative SER requirements correctly during inference/checking. A native path must not require NVAPI; an NVAPI path must not require the native marker. Warning text must accurately describe emitted requirements. CUDA should be configured using capabilities, independently of this fix.

**Removal gate:** `ser_native` and `ser_nvapi` both pass strict checking independently and emit the intended implementations. Then remove/narrow the global 41012 suppression only after checking other workloads, including device-detection gaps. Some warnings are legitimate and should become visible.

## SLANG-W004: Downgrade the default shader model from 6.7 to 6.6

**Status:** Existing production workaround in `src/sgl/device/shader.cpp`; unchanged.

**Problem:** Its comment describes invalid generated HLSL for SM 6.7 ray payloads, but no original reproducer is recorded there.

**New evidence:** Simple `TraceRay` shaders with a payload compile to `lib_6_6` and `lib_6_7` under both investigated compiler versions. The old failure was not reproduced by these cases. This does not establish correctness for every payload annotation or pipeline combination.

**Proper Slang implementation:** Correct HLSL generation for the specific original ray-payload case. The original bug/reproducer must be identified before attributing the current workaround to a still-active compiler defect.

**Removal gate:** Reproduce or locate the original regression, verify it is fixed on the supported minimum compiler, and run affected ray-tracing tests. The new `ray_payload_*` probes are useful smoke coverage but insufficient alone to retire a historical workaround.

## SLANG-W005: Validate known profile/capability conflicts using bounded dependency metadata

**Status:** Implemented as bounded validation in milestone 2, with explicit disclosure that unknown implications are not checked.

**Affected versions:** Slang 2026.17.1 and local master `b4a57b15cc47d936403cc989a628bfd25d5af5d3`.

**Problem:** Slang's session API does not consistently diagnose a profile that is lower than selected capabilities. DX strict checking can accept higher requirements while DXC still compiles for the lower model. SPIR-V version/feature inputs can raise the emitted version above the profile. Slang exposes neither a complete public implication query nor profile-family compatibility metadata. Removing detected raw version inputs is insufficient: the real Vulkan list still emits 1.6 under profile 1.3 because retained features imply it.

**Local adaptation:** `resolve_compiler_target` tracks input provenance. For native-family explicit profiles it trims higher inherited version inputs, rejects higher explicit versions, and rejects retained features with known higher requirements from either origin. `required_version` contains a bounded table for native SER, selected SPIR-V extensions, and OptiX cooperative vectors; `canonical_name` normalizes numeric DX/CUDA aliases only on their native backend. Diagnostics identify the input, origin, required version, profile, and remedy. The table is not Slang's full capability graph. Ordinary wrapper policy, such as choosing supported profile families, remains distinct from this missing upstream introspection. Python tests cover explicit versus inherited inputs, reassertion by overrides, alias conflicts/cache equivalence, SER requirements, and detected Vulkan dependent-feature conflicts when present.

**Evidence:** `interaction_dx_lower_wave_*`, `interaction_dx_6_6_with_6_9_*`, `interaction_dx_native_implied_*`, `interaction_spv_extension_implied_version`, and `interaction_device_vulkan_no_raw_versions` in `tools/compiler_capability_probe/run.py`. Source definitions include `ser_hlsl_native`/`ser_dxr`, `SPV_EXT_physical_storage_buffer`, and `SPV_KHR_cooperative_matrix` in `slang-capabilities.capdef`.

**Proper Slang implementation:** Expose versioned capability implication and profile-requirement queries, including applicable target alternatives and profile family/stage information. Provide an explicit validation policy for requested profile/version limits that agrees with downstream output. Preserve intentional additive use, such as the minimal SPIR-V profile plus higher version capabilities; do not fix this by unconditionally rejecting all profile/capability differences.

**Removal gate:** Replace local dependency facts with supported upstream queries and validate the same inputs, origins, and actionable conflict outcomes. Demonstrate the WaveMatch mismatch, native SER dependency, physical-storage-buffer version implication, and real-device Vulkan case are diagnosable without a SlangPy capability graph. Recheck newly introduced capabilities on the supported minimum compiler. The public input-set contract and provenance handling can remain when the workaround metadata is removed.

## SLANG-W006: Supply an explicit backend baseline for empty capability inputs

**Status:** Implemented on the opt-in path in milestone 2, in `resolve_compiler_target`.

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
