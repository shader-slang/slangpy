# Slang compiler workaround ledger

Updated 2026-09-23 during step 1 of [the capability migration](compiler-capabilities.md).

Record every compiler workaround introduced during this migration here, and refer to its ID from implementation comments. Each entry must distinguish production behavior from a probe, name the upstream fix, and state a removal test. Update the affected compiler versions and evidence whenever changing or removing it. Do not treat a passing expected-failure probe as a reason to preserve the compiler defect.

Step 1 added investigation tools, not production compilation changes. The existing workarounds below predate this task. The proposed adapters have only been applied in isolated probes. Results are documented in [the probe report](compiler-capabilities-probe-results.md).

## SLANG-W001: Derive a DXIL profile from selected shader-model capabilities

**Status:** Existing SlangPy sessions set profiles through legacy `shader_model`; the replacement capability-to-profile adapter is probe-only.

**Affected versions:** Slang 2026.17.1 and local master `b4a57b15cc47d936403cc989a628bfd25d5af5d3`.

**Problem:** `_sm_6_6` or `_sm_6_9` alone produces `lib_6_3` for whole-program DXIL. A capability-selected `WaveMatch` shader is compiled as `cs_6_0` and rejected by DXC. Setting the corresponding profile produces the requested shader model and compiles the operation.

**Local adaptation:** Continue supplying the internal DX profile, deriving it from resolved capability inputs when the new API is implemented. Current production location: `src/sgl/device/shader.cpp`, `SlangSession::create_session`. Probe cases: `dxil_cap_*`, `dxil_profile_*`, `dxil_wave_match_*` in `tools/compiler_capability_probe/run.py`.

**Proper Slang implementation:** Derive the effective DXC shader-model profile from the selected capability set, including implied shader-model requirements. Define how explicit conflicting profiles are diagnosed.

**Removal gate:** With no explicit profile, both whole-program probes produce their selected 6.6/6.9 models and the capability-only WaveMatch probe compiles under a suitable model. Verify all supported compiler versions before removing the adapter.

## SLANG-W002: Neutralize the implicit SPIR-V 1.5 profile bundle

**Status:** Tested in probes only; not implemented in production SlangPy.

**Affected versions:** Both investigated versions.

**Problem:** `_spirv_1_3` with no profile emits SPIR-V 1.5. The default profile also supplies feature assumptions: a shader annotated as requiring `SPV_EXT_physical_storage_buffer` passes strict checking even when only a raw version atom was supplied. Public higher SPIR-V profiles similarly bundle features beyond their version.

**Local adaptation:** In the probes, supply `spirv_1_0` as a minimal compatibility profile and add the selected raw version/features. This emits the requested 1.3/1.6 versions. Strict checking then rejects the missing physical-storage-buffer requirement, as intended. Cases: `spirv_baseline_*`, `spirv_bundle_*`.

**Proper Slang implementation:** Explicit version capability selection should override the implicit backend default. Provide a capability-only target configuration that does not add unrequested higher-profile feature bundles.

**Removal gate:** Without the baseline profile, `_spirv_1_3` emits 1.3, and strict `_spirv_1_6` rejects the missing physical-storage-buffer requirement. This adapter must be tested via `createSession()`; CLI profile-conflict validation is a separate behavior.

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

## Open upstream gaps without an implemented workaround

**CUDA tiers:** RHI and NVRTC 12.2 support this machine's CC 7.5, but Slang does not recognize `cuda_sm_7_5` or `_cuda_sm_7_5`. The RHI-style recognized-name filter in the probe reduces its device list to CC 7.0, and generated PTX targets `sm_70`. Slang also lacks 8.6 and newer numeric tiers. Fix the capability definitions and downstream mapping upstream. Until then, disclose filtering; do not promise an exact unsupported tier.

**CUDA exact ceilings:** With `cuda_sm_5_0`, half-using shader code emits `sm_60`. Strict checking is not a complete architecture ceiling. A complete upstream contract must validate selected architecture against code-generation requirements and toolkit support. Explicit entry-point requirements also bypass the normal restrictive comparison, and an empty selection can skip it entirely.

**RHI subgroup reporting:** The physical Vulkan device supports wave operations, but its reported compiler list lacks `spvGroupNonUniformArithmetic`; strict wave compilation fails. This belongs primarily in slang-rhi discovery, not in Slang's compiler implementation. Do not conceal it in a generic Slang workaround. The int64-atomic probe does pass strict checking despite the raw-list gap, so the two findings must not be conflated.

## Rejected workaround: Append an NVRTC architecture override

`cuda_sm_8_0` plus downstream `--gpu-architecture=compute_86` fails on NVRTC 12.2 in both compiler versions:

    nvrtc: error : --gpu-architecture (-arch) defined more than once

Even repeating the same `compute_80` architecture fails. Slang emits its own architecture option before forwarding downstream arguments. Thus argument order does not provide a working bridge on this toolchain. Do not implement this proposed fallback. Prefer upstream tier support or an explicit upstream architecture-selection option that resolves to one validated NVRTC argument.

## Tracking discipline

When implementing a later milestone, promote a probe-only entry to implemented only after naming the production file/function, adding regression coverage, and recording the applicable compiler versions. New entries require an upstream owner/fix and a concrete removal criterion. External issue URLs should be added if an issue is actually filed; this task did not file any issues.
