# Make compiler target selection capability-based

This ExecPlan follows `.agents/PLANS.md`. It was prepared on 2026-09-23. Both compiler-probe milestones, milestone 2's opt-in public API, and milestone 3's available-backend adapters and bounded CUDA validation are implemented. Additional platform/toolkit coverage, removal of documented upstream workarounds, and default-policy changes remain pending. Keep the Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective sections current as implementation proceeds.

## Purpose / Big Picture


Replace the backend-independent-looking `SlangCompilerOptions.shader_model` with compiler capability selection and an optional explicit Slang profile that describe the chosen backend. Users should be able to use device-derived defaults, supply an explicit capability list, adjust individual supplied capabilities, select CUDA compute capabilities, and choose between native D3D12 and NVAPI shader execution reordering (SER). Capabilities are the primary selection mechanism; `profile` exposes independently useful Slang behavior for advanced control. When omitted, compatibility adapters choose any necessary profile. If Slang eventually makes profiles redundant, the optional field can remain a compatibility shorthand.

The main conclusion is that this is feasible, but device discovery, compiler assumptions, and downstream code-generation settings must remain distinct. An authoritative input list cannot currently be advertised as an absolute prohibition on all other capabilities. Slang expands implications, adds backend defaults, and sometimes increases requirements from shader code.

## Progress


- [x] (2026-09-23) Inspect SlangPy's session construction, bindings, device queries, caches, and shader-model callers.
- [x] (2026-09-23) Delegate independent source investigations of Slang and slang-rhi; reconcile their findings against source.
- [x] (2026-09-23) Specify proposed API semantics, compatibility adapters, migration milestones, and validation gates.
- [x] (2026-09-23) Have both source investigators check the written proposal; run repository-wide pre-commit and checks on the new document.
- [x] (2026-09-23) Build SlangPy and an isolated local-master compiler; run the available-backend probe matrix against both versions, inspect generated outputs, and record limitations in `plan/compiler-capabilities-probe-results.md`.
- [x] (2026-09-23) Start the separate upstream-workaround ledger in `plan/slang-compiler-workarounds.md`, distinguishing existing production workarounds from probe-only adapters and rejected approaches.
- [x] (2026-09-23) Revise the agreed API to include optional `profile`, based on measured DXIL profile selection and SPIR-V profile bundles; update resolver, migration, and validation requirements.
- [x] (2026-09-23) Complete 84 focused profile/capability interaction probes per compiler, verify matching observations, and settle reconciliation rules in `plan/compiler-profile-probe-results.md` and this plan.
- [x] (2026-09-23) Implement opt-in capability resolution, Python bindings, immutable target reports, session/cache/hot-reload wiring, regression tests, and user/API documentation. Validate on D3D12, Vulkan, CUDA, and CPU with pinned Slang 2026.17.1.
- [x] (2026-09-23) Complete milestone 3 on available backends: eager exact CUDA PTX validation for fully specialized programs, validated-cache read guards, emitted DXIL/SPIR-V/PTX checks, native/NVAPI SER execution, and regression coverage.
- [ ] Validate Metal/WebGPU and additional CUDA toolkits/hardware in appropriate CI; remove SLANG-W007's eager compilation/runtime-specialization restrictions when upstream supports exact validation at the specialization/cache boundary. These are explicit follow-up gates, not claims of completed coverage.
- [x] (2026-09-23) Complete milestone 4: migrate all sessions to device-derived defaults, remove the legacy shader-model API, migrate native-profile tests and documentation, expose upgrade warnings, and validate available backends. Replace the reproduced ray-payload downgrade with the documented SLANG-W004 DXC option; retain NVAPI compatibility macros.

## Evidence and Scope


SlangPy was inspected at `a30a5626eaba8dab30979482f7c8a8fe217ad376`. Its slang-rhi submodule is `82c03494bed8e2d42d65555e33b30a18d3d8f071`, dated 2026-09-18. Local Slang master is `b4a57b15cc47d936403cc989a628bfd25d5af5d3`, dated 2026-09-23. Source defaults select Slang `2026.17.1` in `external/CMakeLists.txt:123`; the CUDA investigation also checked that release's capability-to-NVRTC mapping. Do not confuse this with the existing root `build/CMakeCache.txt`, which still records Slang `2026.5.2`. Runtime probes must record the loaded compiler version and library path.

The original investigation below was source inspection. Step 1 subsequently added [measured results](compiler-capabilities-probe-results.md), followed by [explicit-profile interaction results](compiler-profile-probe-results.md) in milestone 1a. These take precedence over unverified hypotheses in the original research. The reason for SlangPy's commented-out device-capability forwarding remains unknown: the available-backend probes did not reproduce it. The comparison's hardware, toolkit, existing Slang submodule revisions, and unavailable runtime backends are explicitly recorded in the probe reports.

Source references use repository-relative paths unless explicitly rooted at `C:/projects/slang`. Line numbers describe these snapshots. Public documentation corroborates the model of aliases, implication, and target-dependent alternatives: [Slang capabilities guide](https://shader-slang.org/slang/user-guide/capabilities). It does not establish a profile-removal schedule. The [capability-system improvement issue](https://github.com/shader-slang/slang/issues/9210) documents continuing evolution, not a completed replacement of profiles.

## Context and Orientation


A target is the output format, such as DXIL, SPIR-V, PTX, Metal library, or WGSL. SlangPy chooses it from the device backend. A profile is Slang's separately represented version and optional shader stage, such as `sm_6_6` or `spirv_1_6`. A capability name may denote an atom, a collection of implied atoms, or alternatives for different targets. A capability set is consequently richer than a flat collection of unrelated booleans.

Runtime features in slang-rhi describe API/device behavior, such as ray tracing or subgroup support. Compiler capabilities describe assumptions given to Slang. They are different namespaces and different bitsets. Neither slang-rhi nor SlangPy's current `has_capability()` expands aliases or checks implication.

SlangPy builds its own Slang sessions in `src/sgl/device/shader.cpp:301`, separately from the session slang-rhi creates for itself. Consequently, adopting capabilities in slang-rhi's default session does not automatically fix SlangPy. The new policy belongs in SlangPy's session assembly first; shared upstream policy can follow.

### What Slang currently implements


Profiles remain live API and implementation concepts. `C:/projects/slang/source/slang/slang-profile.cpp:26` converts a profile into capability requirements. `slang-profile-defs.h` enumerates the accepted profile families. A TODO in `slang-options.cpp:3403` still describes making profiles and capabilities aliases as future work. There is no CUDA profile family in the inspected compiler. Names such as `cuda_sm_9_0` are capabilities, not profiles; SlangPy's comment suggesting a missing CUDA profile is misleading.

`C:/projects/slang/source/slang/slang-target.cpp:66` combines the fixed target, profile-derived capabilities, and repeated `CompilerOptionName::Capability` entries. Incompatible target-specific additions are skipped at lines 229-252. Thus SlangPy's unconditional `hlsl_nvapi` addition is inappropriate policy, but it is not evidence that CUDA/Vulkan capability forwarding necessarily fails due to conflicting target keys.

The capability registry is `C:/projects/slang/source/slang/slang-capabilities.capdef`. `_sm_6_9` is an HLSL-specific atom. `sm_6_9` is a broader alias with alternatives for several backends. This distinction matters: accepting arbitrary Slang aliases must not silently reinterpret them as D3D-only settings when compiling Vulkan. Detected RHI names use the backend-specific atoms.

Higher versions imply lower ones. For example, `_sm_6_10` implies `_sm_6_9`, and `optix_coopvec` implies `_cuda_sm_9_0`. Removing one explicit string does not remove those implications. Alias spellings also need normalization for the well-known version families.

The public API provides `findCapability()` and repeated additive capability options. Capability IDs are explicitly unstable across compiler versions (`include/slang.h:4287`). No public general-purpose capability enumeration, implication-query, subtraction, or negative-capability API was found. Do not copy the complete Slang capability graph into SlangPy.

Ordinary compilation can automatically upgrade capabilities with warning 41012. `RestrictiveCapabilityCheck` promotes relevant diagnostics to errors, including late `__requireCapability` checks. However, an entry point's explicit `[require(...)]` is merged into the allowed set before the ordinary upgrade comparison (`slang-check-shader.cpp:2625`), and backend emitters/downstream baselines can add requirements. Restrictive checks are useful validation, not a demonstrated complete upper-bound contract.

For DXIL, capability-only shader-model selection does not reliably set DXC's profile. The effective profile still follows profile fields and defaults. When no explicit profile is requested, an internal adapter should derive the DX profile from the resolved HLSL version, including known selected capabilities that imply a minimum shader model.

For direct SPIR-V, `slang-target.cpp:114` supplies SPIR-V 1.5 when no version is found in the profile. Merely adding a lower SPIR-V capability cannot undo that baseline. Also, public `spirv_1_5` and `spirv_1_6` aliases include feature bundles absent from the raw `_spirv_*` version atoms (`slang-capabilities.capdef:1655`). Step 1 validated a minimal `spirv_1_0` compatibility profile plus selected raw version/features: this suppresses the hidden 1.5 baseline without importing higher-profile feature bundles. Use that strategy for automatic selection while preserving an explicitly requested profile's semantics. Stop deriving HLSL shader-model profiles to represent Vulkan support.

For CUDA, `slang-code-gen.cpp:609` converts recognized CUDA version atoms into downstream architecture requirements. `source/compiler-core/slang-nvrtc-compiler.cpp:1306` selects the maximum of those requirements, code-generated requirements, and a toolkit-dependent minimum, then emits `-arch=compute_XX`. This path exists in both the pinned 2026.17.1 release and inspected master. The missing connection is therefore partly in SlangPy, not a total lack of upstream architecture selection.

Slang's CUDA tier registry and mapping cover 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.9, and 9.0. They omit, for example, 7.5, 8.6, 8.7, and 10.x/12.x tiers known by RHI. This creates a real difference between the device's architecture, the highest architecture Slang understands, and the architecture NVRTC can compile.

For Metal, version capabilities influence emitted language constructs. Current master also has downstream language-version logic in `slang-code-gen.cpp:769`, including Metal 4 and emitted logging requirements. Otherwise the downstream compiler currently defaults to `-std=metal3.1` (`source/compiler-core/slang-gcc-compiler-util.cpp:971`). A `metallib_2_3` capability therefore does not enforce exact Metal 2.3 output. Audit the pinned compiler separately before removing RHI workarounds.

CPU and WGSL have target-specific capabilities but little device-specific version selection in this stack. A profile abstraction would add little value to their public SlangPy API.

### What slang-rhi currently detects


The shared session constructor in `external/slang-rhi/src/slang-context.h:53` passes detected capability IDs into Slang, silently skips unknown names, and then appends caller options. It separately chooses a profile. Caller options cannot authoritatively replace or subtract the already supplied device capabilities. The public capability enum is manually maintained (`include/slang-rhi/capabilities.h:3`).

D3D12 detects a highest shader model using a table through 6.10 and emits cumulative `_sm_*` capabilities plus `hlsl`. Its default profile is the highest detected shader model. `hlsl_nvapi` is supplied only when NVAPI initialization and extension-UAV configuration permit it. Native SER API availability is reported for appropriate SM 6.9+ ray-tracing devices; the reorder operation may still be a no-op. Both NVAPI and native SER collapse into one runtime Feature. See `src/d3d12/d3d12-device.cpp:45`, `:1024`, `:1188`, and `:1229`.

Vulkan emits `spirv`, cumulative SPIR-V versions, and many extension/instruction capabilities. Vulkan 1.1 maps to SPIR-V 1.3, Vulkan 1.2 to 1.5, and Vulkan 1.3+ to 1.6 (`src/vulkan/vk-device.cpp:1401`). Coverage includes ray tracing/query, NV SER, mesh shaders, clocks, cooperative vectors/matrices, float8/bfloat16, reconvergence, and interlock. It remains incomplete: runtime wave operations and int64 atomics do not have matching granular compiler-capability insertions in the inspected detection code. Atomic-float feature-to-capability mapping also warrants more precise validation. Backend workarounds can deliberately suppress detected capabilities. This is a useful practical list, not a mathematically complete hardware specification.

Vulkan currently detects the NV SER path, not a complete cross-vendor EXT SER path. Its legacy approximate HLSL shader-model Features are separate from the native compiler-capability list and should cease driving compilation.

CUDA queries the driver's compute-capability major/minor and emits cumulative `_cuda_sm_*` names through its table's maximum of 12.1 (`src/cuda/cuda-device.cpp:28`, `:213`). The table explicitly describes hardware tiers, not downstream compiler support. `optix_coopvec` requires suitable OptiX support and CC 9.0+ (`:299`). Dropping unknown names from a modern device can therefore silently select an older NVRTC architecture.

Metal emits `metal` and OS-derived cumulative `metallib_2_3` through `metallib_3_2`. Metal 4 advertisement is explicitly disabled pending downstream compiler handling (`src/metal/metal-device.cpp:254`). Do not override this simply because a Slang capability name exists.

CPU emits `cpp`. WebGPU emits `wgsl`; runtime half/subgroup features do not have equivalent detailed compiler-capability population. D3D11, although not a SlangPy device backend, illustrates the separation: it uses `sm_5_0` as a profile but reports `hlsl` and optional NVAPI without the corresponding shader-model capability. See `src/cpu/cpu-device.cpp:27`, `src/wgpu/wgpu-device.cpp:251`, and `src/d3d11/d3d11-device.cpp:413`.

### What SlangPy currently does


`src/sgl/device/device.cpp:353` searches legacy runtime Features only through SM 6.7 and invents SM 6.0 when none is found. It separately collects RHI capability names and recognized Slang IDs at line 385. `Device.capabilities` exposes the names; `has_capability` is exact membership.

`src/sgl/device/shader.cpp:315` derives a default shader model, including a 6.7-to-6.6 workaround. It suppresses warning 41012 at line 343, unconditionally supplies `hlsl_nvapi` at line 407, and leaves the detected-capability loop commented out at line 412. D3D12 and Vulkan receive `sm_X_Y` profiles at line 444. CUDA, Metal, CPU, and WebGPU do not receive those profiles. NVRTC arguments are forwarded at line 384.

SlangPy also creates/links an NVAPI module based on build/backend status, and emits NVAPI-related definitions and include arguments. Selecting capabilities must be reconciled with that plumbing; removing one compiler flag is not a complete NVAPI policy change.

Session descriptors are hashed with `getSessionDescDigest()` at line 535. Resolved capabilities, internal profiles, and generated downstream options must be installed before computing that digest. Hot reload reconstructs sessions, so resolution must be deterministic and stored with the session. Link-time downstream arguments are a second path that can conflict with a session's architecture selection (`shader.cpp:1632`).

## Proposed Public Contract


Expose three options, using strings rather than fixed enums:

    profile: str | None = None
    capabilities: list[str] | None = None
    capability_overrides: dict[str, bool] = {}

In C++, represent these as `std::optional<std::string>`, `std::optional<std::vector<std::string>>`, and `std::map<std::string, bool>`, respectively. Python annotations above describe fields, not a proposed mutable-default Python function signature.

For `capabilities`, `None` means start from device-derived compiler capability inputs. An explicit list replaces those device-derived inputs. `[]` is distinct from `None`: it requests no optional device-derived inputs. The fixed backend target, mandatory backend baseline, any selected profile's requirements, and Slang's implication rules still exist. An authoritative list replaces device inputs; it does not erase requirements introduced by a separately selected profile. Document the baseline explicitly; an empty list does not mean a compiler with no capabilities or an invalid device.

Apply overrides after choosing that base. `True` adds an input assumption; `False` removes an input assumption. Normalize only known genuinely equivalent spellings for the active backend, remove duplicates, and sort the resulting inputs deterministically. In particular, `_spirv_1_6` and `spirv_1_6` must not be collapsed: the latter adds features. Keep original inputs for diagnostics. Overrides do not mutate `Device.capabilities`, enable Vulkan device extensions, install NVAPI, or create absent hardware features.

The recommended initial contract is input-set editing, not arbitrary semantic negation. Do not silently reinterpret `False` as recursively deleting every capability that implies the named capability. That can remove unrelated requested features and requires a compiler graph Slang does not expose. Provide specific diagnostics for known conflicts, and document that raising a lower version with `True` cannot lower an already selected higher version.

For example, on a device advertising several CUDA tiers, adding `cuda_sm_8_0` to the device defaults will not cap compilation to 8.0. An authoritative list is unambiguous:

    session = device.create_slang_session({
        "capabilities": ["cuda_sm_9_0"],
    })

This assumes a compatible device and toolkit. It selects a recognized CUDA tier and must be verified against emitted PTX. It does not yet promise that an older tier forces Slang below a toolkit minimum.

For device-default D3D12 settings, the proposed native SER selection is:

    session = device.create_slang_session({
        "capability_overrides": {
            "hlsl_nvapi": False,
            "ser_hlsl_native": True,
        },
    })

This requires native SER support and an appropriate internal DX profile. SM 6.9 alone does not imply the separate `ser_hlsl_native` marker. Slang's HLSL emitter prefers NVAPI's HitObject representation when `hlsl_nvapi` is present (`C:/projects/slang/source/slang/slang-emit-hlsl.cpp:1977`), making removal meaningful. Conversely, `hlsl_nvapi=True` may force an assumption but cannot supply an absent runtime NVAPI configuration. Step 1 verified the distinct native/NVAPI HLSL and DXIL paths in permissive mode, but both compiler versions incorrectly require both markers under strict checking. Track this upstream blocker as SLANG-W003; do not enable strict SER by default based on the source-only analysis.

A version-ceiling convenience operation could be useful, but should be explicit and separate from general boolean semantics. It could remove all higher members of a known version family and reject known dependent capabilities such as `optix_coopvec`. Before adding another public field/helper, validate whether authoritative lists, explicit profiles where applicable, and version-filtering examples suffice. A profile is not a universal capability ceiling. If the product requirement is instead that every `False` means an absolute prohibition, obtain upstream implication/negative-capability support before claiming that API is implemented.

Unknown explicitly added user names should fail at session creation unless a documented compatibility adapter recognizes them. A removal may also name an existing detected input even if the loaded compiler does not recognize that input; reject misspelled names that are neither recognized nor present in the base. Automatically detected names unknown to the loaded compiler should be reported in session diagnostics, not silently presented as active. Never silently drop an explicitly requested exact CUDA version. Raw device-list membership must not be used as a universal support check: aliases, incomplete discovery, and implication make that invalid. Validate actual known runtime prerequisites separately, and leave additional explicit assumptions possible for incomplete detection.

`profile=None` requests automatic profile selection, including compatibility adapters where needed; it does not promise that no profile is passed to Slang. An explicit string requests that actual Slang profile and takes precedence over automatic profile selection. Validate it using the loaded compiler's profile lookup and backend compatibility rules. Reject unknown or unsupported selections with an actionable error. Do not invent CUDA profiles: CUDA architecture selection remains capability-based.

Profiles remain independently useful in the measured compiler versions. DXIL capabilities alone leave DXC on a default model, while an explicit profile selects the intended model. SPIR-V profiles introduce feature bundles in addition to a version. Preserve those explicit profile semantics; the minimal SPIR-V compatibility profile is for automatic selection only. For example, `profile="spirv_1_6"` deliberately requests that profile's bundle, whereas automatically resolving raw `_spirv_1_6` should not silently add it.

Reconcile native-family explicit profiles with capability inputs before creating the session, preserving whether each input came from device detection, an authoritative user list, or an override. A request for `profile="sm_6_6"` must not silently retain a device-derived `_sm_6_9` assumption. Remove higher inherited version inputs and report each adjustment. Reject known higher explicit version assumptions rather than silently rewriting them. A `True` override has explicit provenance even if that name was already detected. Lower version inputs remain compatible: they do not express a maximum.

For retained features with known higher-version dependencies, reject the conflict regardless of whether the feature was inherited or explicit. Do not silently remove features to fit a profile. An error must identify the feature, its origin, the required version, the requested profile, and remedies: remove the conflicting inputs, supply an authoritative list, or choose a suitable profile. Include native SER markers/aliases requiring SM 6.9 and tested SPIR-V feature dependencies, such as physical storage buffers requiring 1.3 and cooperative matrices requiring 1.6. Keep bounded adapters with source references and regression evidence (SLANG-W005), not an independent copy of Slang's capability graph. Milestone 1a showed the actual Vulkan device list still emits 1.6 under a 1.3 profile after every raw version input is removed. Unknown implications remain compiler behavior; document the limit of local validation rather than promising a ceiling.

The supported native profile families are DX for D3D12, SPIR-V for Vulkan, and Metal for Metal. Validate names using the loaded compiler and family compatibility using documented adapters. Preserve stage-specific profiles, with Slang reporting incompatibilities when entry points are compiled; recommend stage-neutral profiles for mixed-stage sessions. CUDA, CPU, and WGSL require `profile=None` in the initial contract because no corresponding profile family exists in the inspected compiler. Reject meaningless cross-family selections even when a simple shader happens to compile.

Vulkan's DX and GLSL profiles are a supported advanced exception: require an explicit `capabilities` list (which can be empty), pass the profile through to Slang, and report that its mapping is compiler-defined. Do not automatically mix these profiles with detected capabilities or guess a universal cross-family version ceiling. Additional explicit capabilities retain Slang's additive semantics in this mode. The probe emitted SPIR-V 1.4 for `sm_6_6` and 1.3 for `glsl_460`; these are observations for the tested compiler, not hard-coded conversion rules. The legacy `shader_model` transition path remains separate and preserves its existing Vulkan behavior.

Overrides continue to edit capability inputs only. Removing an input does not subtract requirements supplied by a profile and is not, by itself, a conflict. Report known profile-provided assumptions that remain after removal, and document remaining implication limits. Neither an explicit profile nor restrictive checking guarantees a universal upper bound on shader requirements or emitted versions.

Retain `downstream_args` as an advanced escape hatch, but detect contradictory architecture settings in both session and link options. Define conflict behavior as an error instead of depending on downstream argument order.

Expose a read-only resolution report on the session, with proposed name `target_info`. Include the selected output target, requested profile and whether selection was automatic, resolved profile, supplied capability inputs and their origins, inputs removed during profile reconciliation, compiler capability names actually forwarded, generated downstream arguments, ignored detected names, and compatibility notes. Do not call a list the complete effective capability closure unless Slang can actually provide that information. Keep device-reported capabilities and session-selected capabilities distinct.

## CUDA Compatibility Strategy


First implement the existing upstream path for recognized CUDA capabilities. This is substantially simpler than always manufacturing NVRTC flags and directly addresses CC 8.0/8.9/9.0 selection.

For exact tiers RHI knows but Slang does not, extend Slang's capability registry and architecture mapping upstream. The proposed fallback using a known compiler tier plus an appended exact NVRTC architecture was rejected by step 1: NVRTC 12.2 rejects the duplicate architecture argument, including identical values. Do not implement that bridge. A different mechanism would need its own validation. Architecture-specific variants with suffixes such as `a` or `f` must not be guessed from numeric ordering.

Slang already emits an architecture argument and appends user downstream arguments afterward (`source/compiler-core/slang-nvrtc-compiler.cpp:1382`). Step 1 observed `--gpu-architecture (-arch) defined more than once` on both Slang versions with NVRTC 12.2. Furthermore, half-using code with selected tier 5.0 emitted `sm_60`, demonstrating that code generation can require a newer architecture. Prefer an upstream explicit architecture option or corrected version mapping, with validation against code requirements, over duplicate flags.

Validate three separate facts: the running device supports executing the result; Slang recognizes the semantic features being used; and the actual NVRTC version accepts the requested architecture. A compiler-known maximum is not a toolkit-supported maximum. Inspect the emitted PTX `.target` and `.version`, and reject unsupported or contradictory requests. Automatic defaults may use a documented older compatible architecture when tooling lags hardware, but report that choice. Explicit exact requests must not silently downgrade.

Milestone 3 implements exact numeric selection when the highest forwarded CUDA version originates from an explicit list or True override. `validate_cuda_program` requests each entry point's PTX from the actual linked component/toolchain before handing it to RHI. It rejects any target mismatch, upward or downward, and retains downstream compilation diagnostics. This is link-time validation, not a session-creation toolkit query. It also runs on hot reload and cannot be bypassed by an RHI cache hit. The driver remains responsible for PTX ISA support when loading the pipeline. Device-derived highest versions retain input-assumption semantics, including when a lower explicit version is added.

Validation registers each successful output digest with the device's `PersistentCache`. Cache reads for those entry-point keys must match the validated bytes or return a miss; RHI then uses the already generated component output. This is necessary because the Slang entry-point key does not fully identify the downstream toolchain. Expectations live in memory for the cache/device lifetime, are synchronized for concurrent reads, and do not change the persistent format or unregistered-entry behavior. Test the read boundary directly with stale/matching/overwritten entries, in addition to end-to-end warm-cache dispatch.

The current public RHI API does not expose compiled code after runtime specialization or cached-code retrieval. Exact requests therefore require fully specialized programs at link time; runtime specialization receives an actionable error. The eager check can add compilation work on persistent-cache hits and overrides deferred compilation for explicit requests. Track this bounded workaround as SLANG-W007. Remove these costs/restrictions when upstream can enforce exact architecture at its normal code-generation/cache boundary. Missing tiers still require upstream registry/mapping work; architecture suffixes are not approximated.

## Plan of Work


### Milestone 1: Establish compiler behavior with small probes


Add narrowly scoped probes or tests for session/target options, recording compiler build, backend, detected inputs, forwarded inputs, internal profile, diagnostics, and emitted target metadata. Build SlangPy before running tests. Use both the pinned release and current master in separate build directories if comparison is needed; do not overwrite an existing build's compiler configuration merely to run research.

Probe D3D12 with and without an internally derived profile; Vulkan default versus explicit lower SPIR-V version; CUDA known tiers, missing intermediate tiers, and toolkit minima; and native versus NVAPI SER. Cover inferred requirements, explicit entry-point `[require]`, and late requirements when evaluating restrictive checks. Reproduce the commented-out forwarding failure before assigning it a cause. Verify a simple shader and at least one capability-sensitive operation, not just successful session creation.

Run minimal-profile plus higher raw-capability probes through Slang's `createSession()` API. The `slangc` command-line parser separately rejects certain same-family profile/capability version conflicts (`C:/projects/slang/source/slang/slang-options.cpp:4601`). A CLI rejection alone therefore does not establish that the session API adapter fails.

Acceptance is a recorded backend matrix showing actual behavior and a minimal reproduction for every retained workaround. If complete semantic prohibition cannot be established, keep the input-set contract. If a CUDA bridge cannot meet its exact-request contract, implement/upstream the missing architecture mapping instead of weakening it silently.

### Milestone 1a: Settle explicit profile and capability interactions


Extend `tools/compiler_capability_probe/run.py` and its recorded results with focused session-API cases on both investigated compiler versions before implementing the resolver. Compare lower explicit DX profiles with higher raw shader-model inputs and with capabilities implying a higher model, including native SER. Inspect emitted DXIL and diagnostics under permissive and restrictive checking. Include compatible explicit combinations to ensure validation does not reject ordinary use.

For SPIR-V, compare explicit profiles with lower, equal, and higher raw version inputs and with public capability aliases that introduce feature bundles. Include empty capability inputs and requirements supplied by the profile alone. Confirm that removing a capability input cannot erase a profile-provided requirement. Exercise invalid profile names and relevant cross-family profiles; distinguish documented Slang compatibility, including legacy HLSL profiles used for Vulkan, from genuinely unsupported backend selections. Record CUDA's lack of profiles rather than assigning it a fabricated family.

Use the observations to finalize how explicit profiles filter inherited version inputs and how known dependent features are handled. Define actionable errors for explicit contradictions and unsupported combinations. Record what cannot be checked without upstream implication APIs. Acceptance is an updated interaction matrix, concrete resolver rules and examples, and updated workaround entries where adapters are required.

Completed on 2026-09-23: `plan/compiler-profile-probe-results.md` records 84 probes and 84 observation checks per compiler, with matching statuses and inspected output properties. The Public Contract above now specifies provenance-aware version filtering, rejection of known feature dependencies, profile-bundle removal semantics, stage handling, native families, and the explicit-list exception for Vulkan cross-family profiles. SLANG-W005 tracks the planned bounded validation adapter and its upstream replacement. Production implementation remains milestone 2.

### Milestone 2: Implement deterministic resolution and observability


Completed on 2026-09-23. The implementation and validation results are recorded below; the following requirements describe the delivered opt-in behavior.

Add an internal resolver, proposed files `src/sgl/device/compiler_target.h` and `.cpp`, taking the device backend, detected names, compiler options, and loaded compiler context. Return an owned resolution object containing final target/profile/compiler entries and explanation data. Keep compiler-specific adapters together. Always forward an explicit baseline target capability, even for an empty optional list: otherwise Slang's ordinary restrictive check can skip checking because no specific profile or capability was requested. Add the resolver source to the appropriate `src/sgl` build target if file lists require it.

Milestone-2 transition: activate the resolver when `profile` is set, `capabilities` is non-None, or `capability_overrides` is nonempty. With all three at their default values, preserve the legacy path until milestone 4; `target_info.legacy` reports this distinction. To exercise device-derived inputs now, use a nonempty override, such as the backend's baseline set to True. Explicit None alone is indistinguishable from omission in the C++ descriptor. Do not add a permanent fourth policy field merely for this transition.

Basic DX/SPIR-V profile adapters and capability-selected NVAPI plumbing must be connected in this milestone so the new public options control actual compilation. This brings forward the minimum necessary portion of milestone 3. Comprehensive emitted-version validation, toolkit architecture limits, native SER runtime coverage, and removal of old defaults remain later work. Keep warning 41012 policy unchanged during this milestone. The new report discloses limited implication and exact CUDA validation. Failed session construction must not register a stale session with hot reload.

Add `profile`, `capabilities`, and `capability_overrides` in `src/sgl/device/shader.h`; add dict conversion and field bindings in `src/slangpy_ext/device/shader.cpp`; expose the resolution report through the session binding. Use target-level capability entries because this is target policy, verifying diagnostic behavior for the pinned Slang. Preserve input origins through base selection, overrides, explicit-profile reconciliation, and automatic adapter selection. Store requested options separately from resolved data and resolve identically on hot reload.

Use `findCapability` against the loaded compiler. Permit explicit recognized capabilities newer than RHI's enum. Maintain only small documented adapters for version families and selected runtime prerequisites; avoid an independent compiler capability database. Add Python tests in `slangpy/tests/device/test_compiler_capabilities.py`, with pure resolver coverage in native tests if that improves testing without GPUs.

Acceptance includes None/empty/explicit-list distinctions, automatic versus explicit profiles, profile lookup and compatibility errors, the milestone-1a reconciliation rules, deterministic alias normalization, actionable explicit-unknown errors, disclosed ignored detections, no mutation of device reports, and unchanged cache identity for reordered equivalent input lists. Tests must distinguish an inherited higher version from the same name explicitly reasserted by a `True` override. Test dependent-feature conflicts from both origins, profile-provided features surviving input removal, and Vulkan cross-family profiles requiring an explicit list. Changed target selections must produce distinct session cache identities.

### Milestone 3: Implement backend adapters


Milestone 2 connected the minimum adapters below on the opt-in path. Milestone 3 adds `validate_cuda_program` to `src/sgl/device/compiler_target.cpp` and calls it from `ShaderProgram::link`, implements the bounded exact-request policy described in CUDA Compatibility Strategy, and adds production artifact/runtime tests in `slangpy/tests/device/test_compiler_target_output.py`. Expanded capability tests cover code-driven and toolkit-minimum target changes, overrides, runtime-specialization rejection, warm caches, pre-existing incompatible cache entries, and failed-reload recovery. Ordinary defaults and the legacy path remain unchanged.

In `SlangSession::create_session`, honor a validated explicit profile after reconciling capability inputs. When `profile` is omitted, derive D3D profiles from resolved model assumptions and known SER dependencies. For automatic Vulkan selection, resolve SPIR-V versions independently of HLSL and use the minimal-profile strategy validated in milestone 1 to avoid hidden baseline escalation or unwanted feature bundles. Do not replace an explicit SPIR-V profile with that minimal profile. Feed recognized CUDA capabilities through Slang's existing NVRTC mapping; complete the exact-architecture work described above before advertising missing tiers. Respect Metal compatibility exclusions until validated on supported macOS/toolchain combinations. CPU/WGSL should use native target assumptions without fabricated shader models.

Remove unconditional NVAPI capability injection. Tie compile-time definitions/module plumbing to the selected and available NVAPI path as required by experiments. Do not automatically add native SER markers solely from a shader-model number without checking the relevant runtime conditions and agreeing on default implementation policy. Validate session and link downstream architecture arguments for conflicts.

Compute the session digest only after all resolved options are installed. Ensure hot reload preserves selection and that functional-API kernels inherit the owning session's target settings. Add a functional API test exercising a real capability-sensitive function, not only low-level module loading.

Acceptance requires emitted DXIL/SPIR-V/PTX evidence plus GPU execution on available compatible devices. D3D12 6.9+ SER, macOS Metal, and WebGPU need appropriate CI/hardware coverage; report unavailable coverage rather than treating skipped tests as proof.

Available-machine evidence now includes DXIL shader-model metadata for compute 6.0/6.6, SPIR-V binary headers for 1.0/1.3/1.6, PTX `sm_70` with a version directive, and actual buffer-write/readback execution for each. Explicit Vulkan subgroup and int64-atomic inputs compile with warning 41012 promoted to an error, emit their instruction capabilities, and execute. Native SM 6.9 and NVAPI SM 6.6 HitObject `MakeNop`/`ReorderThread` shaders both create ray pipelines and execute to return 7; generated HLSL contains the selected implementation and excludes the other. This proves API behavior, not a hardware scheduling/reordering performance guarantee. Metal/WebGPU, additional CUDA toolkits, and modern architecture suffixes remain outside measured runtime coverage.

### Milestone 4: Migrate defaults and retire shader models


Implementation decision, 2026-09-23: execute the breaking migration now, as requested, rather than retaining a second compiler policy for a transition release. All sessions resolve device inputs by default; explicit lists/overrides/profiles retain their existing semantics. Remove `ShaderModel`, `shader_model`, `supported_shader_model`, the transitional report flag `legacy`, shader-model conversion/fallback/cap logic. Retain deprecated `__SHADER_TARGET_MAJOR/MINOR` macros for NVAPI headers, describing the selected D3D profile and zero on other backends. Old dictionary keys receive an actionable migration error. Remove global warning 41012 suppression but do not enable universally restrictive checking; known SER and Vulkan reporting gaps still require permissive defaults. The paragraphs below preserve the migration sequencing and acceptance rationale.

Milestones 2 and 3 provided the explicitly selectable path and backend probes. The breaking migration now removes the transition policy instead of retaining a deprecated shader-model translation. Device/default-option copying and explicitly created sessions preserve the documented fresh-descriptor semantics: `create_slang_session()` does not inherit customized options from the default session.

Ordinary sessions use device-derived inputs and expose capability-upgrade warning 41012. Keep compilation permissive and use existing warning controls for explicit stricter checks where tested. Do not enable universally strict checks while SER inference and Vulkan device reporting remain incomplete. Remove the legacy enum/field/device property, fake fallback, and default model cap. Replace the ray-payload downgrade with the narrower DXC option documented as SLANG-W004; preserve the selected model.

Migrate `slangpy/testing/helpers.py` and scalar-type tests in `slangpy/tests/device/slang` to native D3D/SPIR-V profiles and ordinary defaults on other backends. No active sample-submodule caller needs changing; a notebook only contains historical printed device output. Preserve the deprecated shader-model macros for fetched NVAPI headers, using actual D3D profile values and zero elsewhere. Do not invent CUDA/Vulkan shader-model values.

Regenerate Python stubs, binding documentation, and API docs using the repository's build/documentation workflow. Generated `.pyi` files are ignored build outputs and should not be hand-edited as the implementation. Update examples to demonstrate backend-specific names, None versus empty, automatic versus explicit profiles, profile/capability reconciliation and errors, SER selection, unknown names, and the limitations of subtraction. Explain that `profile="sm_6_6"` provides direct D3D profile control, while CUDA uses capabilities and an explicit SPIR-V profile requests its associated feature bundle.

## Concrete Steps


For an implementation, run these commands from `C:/projects/slangpy`, using the project's configured compiler environment:

    cmake --preset windows-msvc
    cmake --build --preset windows-msvc-debug
    pytest slangpy/tests/device/test_compiler_capabilities.py -v
    pytest slangpy/tests/device/slang -v

Inspect `build/windows-msvc/CMakeCache.txt` and the loaded Slang library/version before interpreting results; an existing cache may preserve an older pin. Configure a separate build for master comparisons. After migration, build again before running affected functional-API tests and native tests:

    cmake --build --preset windows-msvc-debug
    pytest slangpy/tests/slangpy_tests -v
    python tools/ci.py unit-test-cpp
    pre-commit run --all-files

Scale test selection to the final edits; broaden only for unresolved coverage. Re-run pre-commit if it changes files. A successful test that only opens a session is insufficient to certify architecture or SER selection.

Milestone 2 validation used the configured Windows debug build and pinned Slang 2026.17.1:

    cmake --build --preset windows-msvc-debug -j 8
    pytest slangpy/tests/device/test_compiler_capabilities.py slangpy/tests/device/test_shader.py slangpy/tests/device/test_shader_cache.py -q
    pytest slangpy/tests/device/test_compiler_capabilities.py --device-types cpu -q
    ./build/windows-msvc/Debug/sgl_tests.exe --test-suite=hot_reload --test-case=check_run_and_verify,change_program_and_recreate,change_program_with_error_and_recreate,change_composed_module_source_and_recreate
    ./build/windows-msvc/Debug/sgl_tests.exe --test-suite=device --test-case=shader
    ./build/windows-msvc/Debug/sgl_tests.exe --test-suite=hot_reload --test-case=change_program_and_auto_detect_changes

Results: 61 Python tests passed across D3D12/Vulkan/CUDA (40 new option/resolution cases, 18 existing shader cases, three cache cases); CPU passed 10 cases with eight skips; six native cases passed 52 assertions. Build/test logs are local artifacts under `build/compiler-target-*.log`. These production API tests were not repeated against master; master coverage remains the separate native probe matrices.

Regenerate binding docstrings with `$env:LIBCLANG_PATH='C:/Program Files/LLVM/bin/libclang.dll'` followed by `cmake --build build/windows-msvc --config Debug --target slangpy_pydoc`; the normal build generates Python stubs. Run `python docs/generate_api.py` for API reference text. Its current full output also changes unrelated stale sections, so this milestone retains only the generated `SlangCompilerOptions`, `SlangSession`, and new `SlangTargetInfo` blocks. The full generated output is preserved locally in `build/compiler-target-generated-api.rst`.

Milestone 3 uses the same compiler/toolchain build, with these additional validation commands after building:

    pytest slangpy/tests/device/test_compiler_capabilities.py slangpy/tests/device/test_compiler_target_output.py slangpy/tests/device/test_shader.py slangpy/tests/device/test_shader_cache.py slangpy/tests/device/test_type_conformance.py -q
    pytest slangpy/tests/device/test_compiler_capabilities.py --device-types cpu -q
    ./build/windows-msvc/Debug/sgl_tests.exe --test-suite=persistent_cache
    ./build/windows-msvc/Debug/sgl_tests.exe --test-suite=hot_reload --test-case=check_run_and_verify,change_program_and_recreate,change_program_with_error_and_recreate,change_composed_module_source_and_recreate
    pre-commit run --all-files
    pre-commit run --files slangpy/tests/device/test_compiler_target_output.py

Results: 85 Python cases pass across D3D12/Vulkan/CUDA, plus 10 CPU cases (13 skips for backend-specific groups and the previously documented CPU functional limitation). Ten native cache cases pass 163 assertions; four native hot-reload cases pass 32 assertions. Logs are under `build/compiler-target-milestone3-*.log`. Artifact tests use pytest temporary directories and inspect the compiled output before those directories are cleaned. Additional manual smoke artifacts are under `build/compiler-target-artifacts/`.

Milestone 4 validation uses the same pinned compiler and available D3D12/Vulkan/CUDA devices, plus CPU for low-level target tests. After rebuilding:

    pytest slangpy/tests/slangpy_tests -q --maxfail=5
    pytest slangpy/tests/device/test_compiler_capabilities.py slangpy/tests/device/test_compiler_target_output.py slangpy/tests/device/slang slangpy/tests/device/test_shader.py slangpy/tests/device/test_shader_cache.py slangpy/tests/device/test_pipeline.py slangpy/tests/device/test_type_conformance.py -q --maxfail=5
    pytest slangpy/tests/device/test_compiler_capabilities.py -k defaults_and_empty -q
    pytest slangpy/tests/device/test_compiler_capabilities.py --device-types cpu -q
    ./build/windows-msvc/Debug/sgl_tests.exe --test-suite=device,hot_reload,persistent_cache
    pre-commit run --all-files

Results: the full functional suite passed 2,664 cases, with 238 skips and seven expected failures. The final combined device suite passed 234 cases with 17 skips and exposed three test-setup errors: the new macro assertion reused a loaded module name with different source. Removing that duplicate load and rerunning all three default-selection cases passed, giving 237 distinct passing focused cases. The actual new SM 6.6/6.7/6.9 pipeline tests, emitted-target tests, shaders, caches, and type conformances all passed in the combined run. CPU passed nine cases with 17 backend/functional skips. Native device/cache/hot-reload suites passed 33 cases and 5,620 assertions. Logs are under `build/compiler-target-milestone4-*.log`. The functional suite required access to existing `.temp` debug dumps; the final link similarly required access to an existing linker artifact. Both completed with the required permissions.

An early broad run reported a one-pixel D3D ray-query mismatch. Standalone comparisons at SM 6.6 and 6.9 passed; the complete pipeline suite then passed twice, including after the final build. No tolerance change or unproven production fix was introduced for that observation. The ray-payload compilation failure was reproducible independently and is tracked separately as SLANG-W004.

Binding docstrings and Python stubs were regenerated. Full API generation was preserved in `build/compiler-target-milestone4-api.rst`; the checked-in reference retains the affected generated option/report blocks and removes the obsolete enum/device property, avoiding unrelated stale-generator changes. Repository-wide pre-commit and `git diff --check` pass. Metal/WebGPU, other CUDA toolkits/hardware, and upstream workaround removal remain explicit follow-up gates.

## Validation and Acceptance


The API must accept device defaults, an authoritative list, overrides, and an optional explicit profile with the exact documented semantics. Explicit unknown names fail early; detected unknown names appear in the resolution report. Empty capability inputs retain documented mandatory baselines and any separately selected profile's requirements. Capability and profile lookup must use strings at the boundary and never persist compiler-specific numeric IDs.

Profile tests must cover automatic selection, an explicit lower D3D profile on a device reporting higher model inputs, explicit contradictory user assumptions, and known features implying higher versions. Check emitted output and the explanation of inherited-input adjustments. SPIR-V tests must distinguish an explicit profile's feature bundle from automatic raw-version selection, including empty inputs and removal overrides. Validate bad names and backend compatibility. CUDA tests must establish that adding the profile field does not replace architecture capabilities or invent CUDA profiles. Do not assert universal ceilings unsupported by the compiler.

CUDA tests must distinguish requested tier from observed PTX target, test lower-tier requests on newer hardware, exercise missing intermediate and newer tiers, reject toolkit-unsupported requests, and cover a shader that genuinely requires a higher tier. Session/link downstream argument conflicts must have deterministic errors.

D3D tests must demonstrate actual DXC profile selection and native/NVAPI HitObject code paths. Vulkan tests must check SPIR-V version and extension declarations, including a lower-version request and existing wave/int64 coverage. Metal tests must certify any removed version workaround on macOS. Tests for explicit `[require]` must prevent overstating restrictive validation as a complete capability ceiling.

Cache tests must show that changed resolved profiles/architectures/options separate cache entries and that input-order changes alone do not. Hot reload must retain the selected policy. Legacy behavior must be exercised during the transition, including conflict errors and deprecation diagnostics.

## Surprises and Discoveries


Milestone 4 reproduced the original ray-payload issue with `slangpy/tests/device/test_pipeline_rt.slang`: separately compiled miss shaders at SM 6.9 lack `[raypayload]`, and DXC rejects them. The standalone whole-program probes did not cover this boundary. DXC's `-disable-payload-qualifiers` fixes this case without lowering the profile. SLANG-W004 now applies that option for automatic/explicit D3D profiles 6.7 and above unless the session explicitly selects payload-qualifier behavior; a conflicting link-only enable is rejected with migration guidance. Regression coverage compiles real ray pipelines at 6.6/6.7/6.9, preserves the reported profile, and exercises caller overrides.

Fetched NVAPI headers actively consume `__SHADER_TARGET_MAJOR/MINOR`; removing them produces undefined-macro diagnostics. Retain these deprecated defines, with D3D profile values and zero for other backends, rather than carrying the obsolete cross-backend ShaderModel API.

Milestone 3 found that the public RHI program API exposes its linked component but no code-validation callback. Code generation can occur at immediate pipeline creation, deferred dispatch, or after runtime specialization, and shader-cache hits bypass Slang entirely. Consequently, a wrapper-side exact architecture contract needs either eager generation before RHI (implemented as SLANG-W007) or an upstream hook; validating only session inputs is insufficient. Tests confirm an inherited tier-5.0 program can populate a cache with `sm_60`, while the same resolved descriptor with explicit provenance is rejected before cache use.

The TITAN RTX/driver combination advertises SM 6.9 and SER API support. Milestone 3 successfully executed both native and NVAPI HitObject/reordering smoke shaders. Native reordering may be a no-op; the test does not establish physical scheduling behavior. PTX output checks independently reproduce code-driven uplift (half data, 5.0 to 6.0) and toolkit baseline uplift (1.0 to 5.0, PTX 8.2).

Slang already has a capability-to-NVRTC architecture path, but its known tiers lag slang-rhi. There are no CUDA profiles to fill in. Evidence: Slang profile definitions, CUDA capability definitions, and `slang-code-gen.cpp:609`.

SlangPy collects the device list but explicitly declines to pass it to its own session. RHI's own default session does pass it. Evidence: SlangPy `shader.cpp:412` versus RHI `slang-context.h:53`.

Neither a list nor restrictive checking currently establishes an absolute hardware/compiler ceiling. Evidence: transitive version implications, default SPIR-V 1.5, toolkit minimum CUDA architectures, and entry-point requirements merged into target assumptions.

Unconditional HLSL NVAPI injection is not sufficient to explain non-HLSL forwarding failures: Slang filters target-incompatible capability additions. Do not turn a plausible initial hypothesis into a reported root cause.

Native D3D SER has its own capability marker; it is not implied merely by SM 6.9 or its profile. Selecting NVAPI can change the emitted HitObject representation. Treat implementation choice and runtime availability separately.

Step 1 confirmed an additional SER bug: each individually selected implementation fails restrictive checking for lack of the other marker, although its permissive output compiles to DXIL. The minimal SPIR-V profile adapter works through the session API and avoids unrequested feature bundles. Device-list forwarding succeeds in the tested small shaders, while strict Vulkan wave compilation fails for the missing RHI-reported subgroup capability. See the separate report for full evidence and limits.

Milestone 1a showed that DX capability validation and DXC model selection can disagree even in strict mode: profile 6.0 plus capability 6.6 accepts an annotated helper but fails downstream for actual WaveMatch. Native SER implications similarly do not raise the DXC profile. For SPIR-V, both higher raw versions and feature dependencies can raise the emitted version above the profile; the real device list still emits 1.6 with a 1.3 profile after stripping raw versions. Explicit profile 1.6 continues to supply physical-storage-buffer assumptions after that input is removed. Several irrelevant profile families compile successfully, so lookup or simple compilation alone cannot establish useful backend compatibility.

## Decision Log

2026-09-23, milestone 4: retire the legacy API and opt-in switch in this breaking change. The user explicitly wanted `shader_model` to go away, and all available-backend validation milestones have passed. Default sessions now start from detected compiler inputs; fresh sessions still use fresh option descriptors. Retain deprecated shader-model macros: fetched NVAPI headers actively consume them. They describe the actual D3D profile and are zero on other backends. A sample notebook contains historical printed `supported_shader_model` output only; no sample source migration or submodule revision change is needed.

2026-09-23, milestone 4: expose capability-upgrade warning 41012 by removing broad suppression, while retaining permissive compilation. Applications can promote the warning with existing `warnings_as_errors` options where suitable. Do not add another public validation-policy field or enable strict SER checks with known upstream false positives. Replace scalar tests' cross-backend shader-model matrix with native D3D/SPIR-V profile cases and ordinary device defaults on other backends. Copy the test dispatch helper's option dictionary before adding helper settings so native profile selection does not leak between calls.

2026-09-23, milestone 3: use actual linked-component PTX for exact numeric CUDA validation instead of discovering/loading a separate NVRTC library or copying a version/support table. This tests the toolchain Slang actually uses. Exact validation is per-program, with downstream diagnostics retained; session reports continue to describe inputs only. Require fully specialized programs for explicit requests until an upstream post-specialization/cache validation hook exists. Document eager compilation and cache cost as SLANG-W007; retain deferred behavior for inherited/empty selections. Guard persistent-cache reads with the validated output digest, because eager compilation alone cannot rule out a different downstream-toolchain artifact under the same Slang key.

2026-09-23, milestone 3: do not infer semantic limits from `.target`. A shader annotated with `[require(_cuda_sm_7_0)]` can emit `sm_50` under a tier-5.0 input. The architecture check correctly accepts matching output; warning/strict policy is still separate. Actual half code raises output to `sm_60` and is rejected. Toolkit minimum behavior is independently reproduced with tier 1.0 producing `sm_50` (PTX 8.2) on NVRTC 12.2.

2026-09-23, milestone 2 implementation: activate the new path for a non-None profile, a non-None capability list (including empty), or nonempty overrides. A neutral backend override can opt into detected inputs. Keep ordinary defaults and warning 41012 policy unchanged. Bring basic backend adapters forward so the public options control actual compilation. Always supply the mandatory backend baseline, tracked as SLANG-W006, including for empty inputs.

2026-09-23, milestone 2 implementation: return copied target reports and session descriptors from Python. This prevents edits to a report or requested-options view from changing later hot reload behavior. Fresh explicit sessions continue to use fresh descriptors rather than inheriting the device default session's options. On the opt-in path, `__SHADER_TARGET_MAJOR/MINOR` describe the resolved D3D profile and are zero on other backends; preserve legacy macro behavior on the legacy path.


2026-09-23, original proposal, superseded after step 1: expose capability strings and optional replacement plus overrides; keep profiles internal. Rationale at the time: accommodate backend diversity and possible future Slang profile removal.

2026-09-23, agreed revision after step 1: expose optional `profile` alongside `capabilities` and `capability_overrides`, retaining device-derived defaults and automatic adapters when it is omitted. Rationale: measured DXIL code generation and SPIR-V feature bundles give profiles independent behavior that advanced callers need to control. Future compiler unification can leave this field as a compatibility shorthand. Explicit profile selection must reconcile inherited version assumptions and diagnose known explicit conflicts; focused interaction probes will settle the detailed rules before resolver implementation.

2026-09-23, decision after milestone 1a: filter higher inherited version inputs for native-family explicit profiles, reject known higher explicit versions, and reject known dependent-feature conflicts from either origin. Rationale: removing raw versions does not remove feature implications; silently dropping features would obscure the requested assumptions. Use bounded validation with disclosed limitations, tracked as SLANG-W005, until upstream exposes dependency and profile-compatibility information.

2026-09-23, decision after milestone 1a: support native profile families plus Vulkan's advanced DX/GLSL mappings with an explicit capability list; reject other unsupported families. Preserve profile bundles and stages. Rationale: measured cross-family acceptance includes both useful mappings and profiles that supply no appropriate version control. Requiring an explicit list for advanced mappings avoids guessing how to reconcile them with detected native versions. Input removal cannot undo a profile's requirements and does not imply a prohibition.

2026-09-23, proposal: define boolean overrides as editing supplied assumptions. Rationale: universal subtraction/forbidding is not supported by the available public Slang API. A stronger contract needs a separate upstream or implementation gate, not undocumented heuristics.

2026-09-23, proposal: preserve both requested and resolved settings and expose resolution evidence. Rationale: the device, compiler registry, toolchain, and backend workarounds can differ; silent filtering makes debugging and reproducibility difficult.

2026-09-23, decision after step 1: implement known CUDA tiers first and prioritize upstream coverage for exact missing tiers. Reject the appended-downstream-architecture bridge. Rationale: the actual NVRTC 12.2 toolchain rejects duplicate architecture flags in both investigated compiler versions.

2026-09-23, decision after step 1: keep strict-default policy separate from capability API introduction. Rationale: measured SER false positives and incomplete Vulkan subgroup reporting would reject valid programs; restrictive checking also does not enforce every selected limit.

2026-09-23, proposal: stage API introduction, default-policy change, and legacy removal. Rationale: the public field change is relatively small; changing compiler assumptions can alter generated code and diagnostics across all backends.

## Idempotence and Recovery


Keep probes and new resolution code separate from the legacy path until validated. Use distinct build directories for compiler-version comparisons. No source or submodule changes are required merely to read this plan. Do not overwrite user compiler caches, commit submodule changes, or update external repositories as part of a speculative probe. If a backend fails, retain its documented compatibility adapter and a minimal reproduction rather than reverting the entire API design.

## Interfaces and Dependencies


No new runtime package dependency is needed. Use the existing Slang API, slang-rhi device reports, standard C++ containers, and nanobind. The proposed resolver produces an immutable per-session target resolution. The public options are `profile`, `capabilities`, and `capability_overrides`. Automatic profile derivation, compiler version translation, generated downstream flags, and compatibility tables remain implementation details visible through diagnostics. Preserve requested profile and input origins separately from resolved compiler settings.

Priority upstream work is broader CUDA tier coverage and exact architecture selection; public capability implication/introspection or disable support; precise device-to-compiler mappings in RHI, particularly Vulkan subgroup/atomic support and native SER; and an explicit capability-set replacement policy in RHI sessions if shared ownership is desired later.

## Outcomes and Retrospective


Milestone 4 applies capability resolution to every session and removes `ShaderModel`, `shader_model`, `supported_shader_model`, and the transition-only `target_info.legacy` flag. The public `profile`, `capabilities`, and `capability_overrides` contract remains unchanged. Documentation now describes device defaults and migration; tests use native profiles instead of cross-backend model numbers. Warning 41012 is visible, with permissive defaults and existing per-session warning controls. NVAPI's shader-model macros remain deprecated compatibility defines. The actual payload regression surfaced only with separate ray entry-point compilation, so SLANG-W004 now disables DXC payload qualifiers instead of lowering profiles; the report and ledger disclose it, and session overrides remain available. Master has not been retested for this specific regression.

The research and completed probe milestones support capabilities as the primary selection mechanism, with an optional explicit profile, and identify concrete work that a simple field replacement would miss. The API distinguishes supplied assumptions from universal prohibition. Known CUDA tiers work through existing Slang mapping; missing exact tiers require upstream work. DX profile synthesis and minimal SPIR-V baseline profiles are implemented adapters on the opt-in path. Milestone 1a settled bounded reconciliation rules for explicit profiles, including feature dependencies, profile bundles, stages, and advanced cross-family use. SER strict checking and RHI subgroup reporting are blockers to strict defaults. Ordinary production sessions retain the legacy policy.

Both delegated source investigators audited the original written proposal. Step 1 built SlangPy and local Slang master, introduced the independent native probe and Python runner/verifier, and ran the existing shader tests (18 passed). The final matrix contains 80 probes per compiler, with 66 selected observation checks per compiler and matching statuses/output properties for all 80 cases. Production buffer-write/readback smoke shaders passed on D3D12, Vulkan, CUDA, and CPU. Metal/WebGPU runtime, newer GPU execution, other NVRTC versions, and the complete functional-API suite are outside this measured coverage. Repository-wide pre-commit and explicit checks on the new files passed after formatting.

Milestone 1a rebuilt SlangPy and the native probe, added 84 interaction cases per compiler, and verified all 84 observations and cross-version output comparisons. A final combined run passed 150 observation checks per compiler, with matching statuses and inspected output properties across all 164 cases. Repository-wide pre-commit passed after formatting. The original 80-case results remain separately reproducible with `--suite baseline`; an unfiltered runner defaults to both suites. See the interaction report for exact commands and limits. The next implementation milestone is deterministic resolution and Python bindings, not a default-policy switch.

Milestone 2 implemented `src/sgl/device/compiler_target.h/.cpp`, the three compiler-option fields and Python bindings, and read-only `SlangSession.target_info` snapshots. Resolution preserves input origins, normalizes native numeric aliases deterministically, validates names/profile families/known dependencies/runtime prerequisites, discloses unknown device inputs, and supplies backend baselines and minimal profile adapters. Resolved settings feed session digests, capability-selected NVAPI plumbing, and fresh reloads. Session/link downstream target flags are rejected when they conflict with this policy. The tests exercise actual compute execution, capability-sensitive DX compilation, SPIR-V profile bundles versus raw inputs, functional API calls and reloads, cache identity, invalid options, and immutable request/report views. Repository-wide pre-commit, explicit checks on new files, and `git diff --check` passed. See `docs/src/compiler_targets.rst` and the updated workaround ledger.

During milestone 2, early resolver errors exposed a constructor hazard: hot reload could retain a pointer to a failed session. Registration now occurs after successful construction, with failure/reload regression coverage. CPU functional dispatch reports zero X groups on this machine under both old and new policies; that case is skipped with an explanation while low-level CPU dispatch passes. At that milestone, Metal/WebGPU/native SER runtime and exact CUDA output checks remained unvalidated; milestone 3's results below supersede those native SER/CUDA gaps.

Milestone 3 resolves the available-backend validation work left by milestone 2. Explicit highest numeric CUDA requests now compile and validate each entry point against its emitted PTX before RHI program creation, including on reload. Unsupported toolkit compilation retains NVRTC diagnostics, and successful compilation with a changed architecture is rejected with the requested/actual target and PTX version. Runtime-specialized exact requests are rejected with a remedy until upstream supplies an appropriate hook; inherited/empty inputs and legacy sessions kept their existing behavior during milestone 3. This is an output architecture check, not a semantic ceiling or a driver/PTX ISA compatibility promise. The new artifact tests confirm native DX/SPIR-V selection and execute both SER implementations; no Metal exclusions or strict/default warning policies were changed. Milestone 4 below completes default migration and legacy API retirement; cross-platform/toolkit CI and the tracked upstream removal gates remain open.

Revision note, 2026-09-23: initial research proposal; reconciled the NVAPI conflict hypothesis with Slang's actual incompatible-option filtering and separated current source defaults from the stale existing build cache.

Revision note, 2026-09-23, step 1: added empirical results and the requested separate workaround ledger; rejected the CUDA duplicate-argument bridge; recorded the SER strict-checking defect, tested profile adapters, unavailable runtime coverage, and the unreproduced historical forwarding failure.

Revision note, 2026-09-23, profile decision: replaced the internal-only profile proposal with optional public `profile`; updated input semantics, observability, migration, backend adapters, and acceptance criteria. Added milestone 1a to settle inherited-versus-explicit capability reconciliation through focused probes. Updated the workaround ledger to restrict automatic profile adapters to cases without an explicit profile. This revision changes the plan only, not production code or measured probe results.

Revision note, 2026-09-23, milestone 1a: completed the 84-case interaction matrix on both compilers and replaced pending reconciliation questions with concrete bounded rules. Added the interaction report and SLANG-W005; documented why dependent features require conflict errors and why Vulkan cross-family profiles need an explicit capability list. No production resolver or default changes were made.

Revision note, 2026-09-23, milestone 2: implemented and tested the opt-in API and resolution report, connected minimum backend adapters, regenerated affected documentation, and promoted implemented workarounds in the ledger. Ordinary defaults, legacy removal, and comprehensive CUDA/backend validation remain later milestones.

Revision note, 2026-09-23, milestone 3: added bounded exact CUDA output validation, explicit runtime-specialization and cache-cost limitations, production backend artifact checks and native/NVAPI execution coverage. Added SLANG-W007 with upstream replacement/removal criteria. The public option fields and ordinary defaults remain unchanged.

Revision note, 2026-09-23, milestone 4: migrated defaults and retired the legacy shader-model API, exposed upgrade warnings, retained NVAPI compatibility defines, and replaced the old profile downgrade with a documented payload-qualifier adapter. Updated migration/API documentation and backend-native scalar tests. Cross-platform/toolkit validation and upstream workaround removal gates remain follow-up work.
