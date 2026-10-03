# Simplify ShaderCursor while preserving typed binding behavior

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy currently has two implementations of shader-object cursor navigation: the low-level `rhi::ShaderCursor` in the `external/slang-rhi` submodule and a separate `sgl::ShaderCursor` in SlangPy. The SGL implementation also contains valuable behavior that the RHI cursor intentionally does not provide, including conversion from C++ and Python values, backend-aware packing of booleans and matrices, SGL resource wrappers, CUDA interoperability, and Python error reporting. Over time, however, fixes in Slang and slang-rhi have made some SGL compatibility code obsolete, while the duplicate navigation implementations have diverged.

After this work, users should observe the same successful binding behavior on D3D12, Vulkan, CUDA, and Metal, with additional rejection of unsafe NumPy inputs and safer cursor lifetime and bounds behavior. For the currently authorized scope, maintainers should see fewer backend workarounds, shared Python conversion, and cheaper bulk writes and repeated shader-object traversal. RHI navigation consolidation is explicitly deferred. The result is demonstrated by focused cursor and parameter-block tests, the native C++ suite, full relevant Python tests, Metal CI, and a dispatch-performance comparison that does not regress the cached functional API path.

The work is divided into independently reviewable stages. Correctness tests and safety fixes land before obsolete code is removed. Python conversion is simplified before native cursor storage is changed. The SGL-to-RHI navigation convergence is last because it has the largest blast radius and must preserve explicit entry-point semantics and the functional API's cached-offset fast path.

## Progress

- [x] (2026-09-21) Replace per-child nanobind keep_alive records with cheaper Python-only ownership. The user rejected the 5-8% traversal regression; the previous ownership implementation is not an accepted final result. Keep native ShaderCursor storage and RHI navigation unchanged. Validate native-to-Python callback cursors, descendant/layout lifetime, ordinary and nested assignment, and the original baseline benchmarks.

- [x] (2026-08-21 09:43Z) Reviewed the SGL, Python binding, functional API, and slang-rhi cursor implementations and recorded the current architectural boundaries.
- [x] (2026-08-21 09:43Z) Established the baseline: the Debug build succeeds; `test_shader_cursor.py` passes 9 tests; `test_parameter_block.py` passes 3 tests; the native suite passes 263 test cases and 20,994 assertions on Windows with D3D12, Vulkan, and CUDA.
- [x] (2026-08-21 09:43Z) Probed current reflection behavior and confirmed that Slang 2026.12.2 reports packed one-byte CUDA `bool1` through `bool4` layouts, while D3D12 and Vulkan retain four-byte uniform bool elements.
- [x] (2026-08-21 10:23Z) Stage 1: added characterization tests and fixed concrete cursor validity, bounds, Python ownership, and nullable BufferView defects. The Debug build, focused D3D12/Vulkan/CUDA tests, native suite, and full Python suite pass. Metal execution remains covered by CI rather than this Windows host.
- [x] (2026-08-21) Stage 2: removed obsolete CUDA bool paths, the SGL Metal relayout, and dead cursor code. Explicit `bool1` works on CUDA, and PR #1122 passed the complete macOS ARM64 Debug and Release C++/Python test jobs, closing the required Metal nested-parameter-block validation.
- [x] (2026-08-21) Stage 3: made NumPy conversion type-safe and centralized Python writer dispatch. Debug and Release builds, focused cursor and marshalling tests, the native Release suite, and the full Release Python suite pass.
- [x] (2026-08-21) Restored reflected scalar, vector, matrix, dictionary, sequence, and NumPy conversion as the direct cursor-write fast path. Registered native writers and `get_this` wrappers now run through one fallback only when the ordinary representation does not match.
- [ ] Deferred by the user on 2026-09-21: Stage 4: introduce an RHI-backed SGL cursor representation while preserving the existing public API and cached functional API behavior.
- [ ] Deferred by the user on 2026-09-21: Stage 5: retire duplicated SGL navigation, complete cross-backend validation, measure performance, and document the final architecture.

- [x] (2026-09-21) Follow-up cleanup: share typed converters, restore supported NumPy layouts and raw-copy semantics, retain owners through Python cursor traversal, and reuse wrappers by RHI object identity.
- [x] (2026-09-21) Debug build, 251 focused Python tests (2 skips), 17 native cursor cases (521 assertions), and pre-commit passed.
- [x] (2026-09-21) Completed alternating Release baseline/final benchmarks on D3D12, Vulkan, and CUDA (three process pairs, nine measured rounds each). Full native Release suite passed 277 cases and 23,123 assertions, with 5 skips.
- [x] (2026-09-21) Full Release Python suite: 4,624 passed, 454 skipped, 7 xfailed, and one pre-existing Vulkan matrix-gradient failure. The same test fails against the saved pre-cleanup binaries with the identical 0.295745313167572 numerical deviation. Pre-commit (including the new benchmark tool) and git diff --check pass. This completed the earlier follow-up; ownership performance was subsequently reopened. RHI stages remain deferred.

## Surprises and Discoveries

- Observation: The CUDA bool reflection workaround is no longer active for the project's pinned compiler. Slang 2026.12.2 reports CUDA `bool`, `bool1`, `bool2`, `bool3`, and `bool4` with one-byte element sizes and one-byte element strides. D3D12 and Vulkan report four-byte uniform bool elements, which the generic layout-aware writer already supports.
  Evidence: A local reflection probe on 2026-08-21 printed CUDA sizes of 1, 1, 2, 3, and 4 bytes respectively. The dedicated ShaderCursor test passed on all three available backends.

- Observation: slang-rhi now installs `MetalArgumentBufferTier2` as the parameter-block shader object's element layout. This is the same layout that SGL reconstructs manually while dereferencing a Metal parameter block.
  Evidence: `external/slang-rhi/src/metal/metal-shader-object-layout.cpp`, in `ShaderObjectLayoutImpl::Builder::setElementTypeLayout`, assigns `m_parameterBlockTypeLayout` to `m_elementTypeLayout` and states that this keeps ShaderCursor and argument-buffer offsets consistent. The relevant slang-rhi change postdates the SGL workaround.

- Observation: Matrix repacking remains necessary. D3D12 reflection reports 16-byte row strides for matrices such as `float2x2` and `float3x3`, while Vulkan and CUDA report tightly packed 8-byte and 12-byte row strides. RHI's raw `setData` operation does not translate a host matrix into those layouts.
  Evidence: A local reflection probe reported `float2x2` stride 32 with two 16-byte rows on D3D12 and stride 16 with two 8-byte rows on Vulkan and CUDA.

- Observation: Scalar and vector NumPy writers can reinterpret bytes as the destination type without checking the source dtype. Equal byte sizes can therefore produce silently incorrect values rather than a conversion error.
  Evidence: `src/slangpy_ext/device/cursor_utils.h` reads NumPy storage through `reinterpret_cast<const ValType*>` in `_write_scalar` and `_write_vector_from_numpy`; only the array path consistently calls `dtype_to_scalar_type`.

- Observation: SGL keeps raw cursor pointers for speed, but dereferencing creates new SGL wrappers and appends each wrapper to the parent `ShaderObject::m_objects`. Repeated navigation can retain duplicate wrappers for the same RHI child object.
  Evidence: `ShaderObject::get_entry_point` and `ShaderObject::get_object` both construct a wrapper and unconditionally append it to `m_objects`.

- Observation: The matrix-result skips were stale independently of the Stage 2 compatibility removal. Once the tests compared shader-produced numeric values on every backend, `float2x2` and `float3x3` passed on D3D12, Vulkan, and CUDA with the existing generic matrix repacking.
  Evidence: `test_shader_cursor.py` passed all nine cases after removing the skips, before the CUDA bool and Metal compatibility paths were deleted.

- Observation: The BufferCursor suite still filtered `bool1` out on CUDA and Metal for slangpy issue 274, even though the current reflected CUDA layout and generic conversion support it.
  Evidence: Removing the filter produced 64 passing BufferCursor tests and two unrelated pointer skips across D3D12, Vulkan, and CUDA, including host writes, device writes, copies, and readback.

- Observation: The complete Python suite contains 4,813 tests and exceeds a 15-minute serial command window on this host. Three pytest workers complete it in under six minutes.
  Evidence: The serial command timed out after 904 seconds without a failure summary; `pytest slangpy/tests -q -n 3 --tb=short` completed with 4,366 passed, 454 skipped, and 7 expected failures in 340.40 seconds.

- Observation: The recursive functional API `unpack_arg` helper cannot be used as the cursor writer's top-level wrapper fallback. A self-returning wrapper nested in a list causes it to rebuild a distinct list on every retry, so identity comparison at the container level cannot terminate recursion.
  Evidence: The existing `SelfReturningWrapper` BufferCursor regression test exposed a stack overflow in the first centralized-dispatch implementation. Resolving only the current object's `get_this` method preserves nested container traversal and terminates when a wrapper returns itself.

## Decision Log

- Decision: Investigate a Python-only owning cursor wrapper with direct owner/layout references and a native ShaderCursor type caster.
  Rationale: Copying references in the Python wrapper should avoid allocating nanobind keep_alive records while preserving the lightweight native dispatch path. Native cursors entering Python callbacks must also become owning snapshots. This supersedes the prior acceptance of the measured traversal overhead.
  Date/Author: 2026-09-21 / Codex.

- Decision: Defer Stages 4 and 5, and implement the conversion, bulk-write, lifetime, and wrapper-reuse cleanup without changing cursor navigation or storage.
  Rationale: User explicitly approved this bounded scope on 2026-09-21. Preserve cached native writers; restore contiguous 1D/2D vector inputs, positive/negative outer row strides, and raw `unchecked_copy` storage. Reuse wrappers by actual RHI object identity while retaining old wrappers so binding replacement does not invalidate existing cursors.
  Date/Author: 2026-09-21 / Codex.

- Decision: Restore the data submodule to the parent repository recorded commit ed8d5e8.
  Rationale: The clean submodule was at an older commit missing PNG fixtures, causing 16 unrelated test failures.
  Date/Author: 2026-09-21 / Codex.


- Decision: Keep typed host-value packing and Python structural conversion in SGL rather than moving them into slang-rhi.
  Rationale: `rhi::ShaderCursor` is intentionally a raw navigation and binding primitive. Matrix packing, Python dict/list/NumPy conversion, SGL resource wrappers, and CUDA interop are higher-level policies and do not belong in the RHI abstraction.
  Date/Author: 2026-08-21 / Codex

- Decision: Land safety and characterization changes before deleting compatibility code or changing cursor storage.
  Rationale: Tests that describe current cross-backend behavior provide a reliable boundary for later subtraction and make each stage independently reviewable.
  Date/Author: 2026-08-21 / Codex

- Decision: Preserve SGL's explicit distinction between root globals and entry-point parameters.
  Rationale: RHI's `getField` searches attached entry points when a root field is not found. SGL deliberately disabled that “do what I mean” fallback because it is ambiguous for programs with multiple entry points. RHI navigation may be reused only through strict struct-field and element operations that cannot trigger that fallback.
  Date/Author: 2026-08-21 / Codex

- Decision: Use composition rather than making `sgl::ShaderCursor` an alias or subclass of `rhi::ShaderCursor`.
  Rationale: Composition lets SGL preserve its public snake_case API, exception behavior, SGL resource conversion, device access, CUDA interop ownership, and stricter semantics while delegating offset calculation to RHI.
  Date/Author: 2026-08-21 / Codex

- Decision: Do not consider the Metal workaround removed until the existing nested parameter-block scenario passes on a Metal CI runner after the removal.
  Rationale: The current host cannot execute Metal. Source inspection strongly indicates that the workaround is redundant, but backend validation is required before the stage is complete.
  Date/Author: 2026-08-21 / Codex

- Decision: Move the nanobind ShaderCursor-to-ShaderObject keep-alive annotation from Stage 4 into Stage 1.
  Rationale: The current raw-pointer representation already has a concrete Python lifetime hazard, and the fix is independent of the future RHI-backed representation. A regression test drops the original ShaderObject Python variable, forces garbage collection, and continues using the cursor.
  Date/Author: 2026-08-21 / Codex

- Decision: Delete the bespoke Python bool-vector conversion path in Stage 2 rather than retaining it temporarily.
  Rationale: `bool1` through `bool4` use the same tightly packed native vector representation as NumPy bool storage. The normal vector path compiled and passed sequence and NumPy ShaderCursor tests on D3D12, Vulkan, and CUDA, while BufferCursor read/write also passed with its old CUDA filter removed.
  Date/Author: 2026-08-21 / Codex

- Decision: Treat numeric matrix cursor indexing as row indexing.
  Rationale: `find_element` returns the matrix element layout, which is one row vector, and advances by the reflected row stride. Python `len()` must therefore report row count rather than scalar cell count so bounds checking matches native navigation.
  Date/Author: 2026-08-21 / Codex

- Decision: Keep public cursor validity and bounds checks enabled in every build, while treating validity as an assertion at the internal typed-write storage handoff.
  Rationale: Public `find_*`, setter, and resource-binding behavior must not become build-dependent. `CursorWriteWrappers` must validate before accessing reflection data, but its subsequent call to `ShaderCursor::_set_data` has already established validity; using `SGL_ASSERT` there removes the duplicate Release check without creating an unchecked public path. Cached navigation remains checked until measurement justifies a dedicated internal path; the user explicitly deferred benchmarking for now.
  Date/Author: 2026-08-21 / Codex

- Decision: NumPy cursor writes accept exact scalar dtypes and the existing same-width signed/unsigned bit-preserving conversions, but reject other numeric dtype changes.
  Rationale: This matches `allow_scalar_conversion` without silently changing the established cursor semantics or allocating conversion buffers. Integer-to-float, float-to-integer, and width-changing inputs now raise deterministic Python exceptions instead of reinterpreting storage.
  Date/Author: 2026-08-21 / Codex

- Decision: Preserve partial dictionary updates and continue ignoring extra dictionary keys.
  Rationale: Both behaviors are established cursor API semantics. Missing fields must remain untouched, and rejecting extra keys would be an unrelated compatibility break in a stage intended to make conversion safer without changing successful writes.
  Date/Author: 2026-08-21 / Codex

- Decision: Keep `unchecked_copy` as a reflected-layout bypass, but always validate NumPy dtype, shape, rank, contiguity, and byte count before reading its storage.
  Rationale: Callers may still opt into raw layout copying, but `unchecked_copy` must not permit out-of-bounds reads or accidental dtype reinterpretation.
  Date/Author: 2026-08-21 / Codex

- Decision: Run ordinary reflected conversion before registered-native and legacy-wrapper fallback.
  Rationale: Plain Python scalars, SGL vectors and matrices, dictionaries, lists, and matching NumPy values are the common public cursor-write path. They must not pay a native-registry lookup or Python `get_this` attribute lookup. A separate handled/not-handled writer table preserves the existing direct functional writer table while retaining one centralized fallback for uncommon values.
  Date/Author: 2026-08-21 / Codex

## Outcomes and Retrospective

### Current ownership follow-up results (2026-09-21)

The rejected keep_alive regression has been removed. Python cursors now hold a direct owner reference; only reinterpretation allocates a shader-object/layout owner pair. Fresh returned values are constructed directly in nanobind storage. Python construction retains the original owner argument without a reverse lookup. When borrowed command-recording objects escape into Python, their wrappers retain the underlying RHI object once. Native ShaderCursor storage, cached native traversal, and RHI navigation remain unchanged. The shared synchronous write helpers borrow their Python inputs by const reference to avoid redundant owning copies.

Final Release comparison: three alternating baseline/final process pairs per backend, nine measured rounds per process (27 samples per case/backend), against the saved pre-follow-up runtime. Negative changes mean less CPU time.

| Case | D3D12 | Vulkan | CUDA |
|---|---:|---:|---:|
| cursor_construction | -14.71% | -16.59% | -12.22% |
| scalar | -2.03% | -3.10% | -2.36% |
| scalar_cached_cursor | -3.40% | -4.44% | -3.61% |
| scalar_assignment | -0.56% | +0.08% | -0.52% |
| nested_assignment | -0.90% | -2.03% | -2.17% |
| nested_attribute_assignment | -0.65% | -1.71% | -2.59% |
| vector | -3.45% | -1.77% | -2.55% |
| numpy_vector | -1.82% | -1.14% | -0.02% |
| bulk_checked_1024 | -51.14% | -51.31% | -47.53% |
| bulk_raw_1024 | -73.87% | -73.96% | -73.62% |
| nested_parameter_block | -16.64% | -17.26% | -14.16% |
| functional_1_args | +0.07% | -0.09% | +0.58% |
| functional_6_args | +0.46% | -0.39% | -0.13% |

No regression above measurement variation remains in these final cases: direct scalar assignment ranges from -0.56% to +0.08%, and cached dispatch from -0.39% to +0.58%. This does not guarantee every workload or platform is regression-free. Metal requires CI. Raw samples: build/cursors-borrowed-owner-{baseline,new}-{d3d12,vulkan,cuda}-{0,1,2}.json; aggregate: build/cursor-borrowed-owner-comparison.json. Earlier benchmark outcomes below are historical and superseded by this comparison.

Final Release validation: 4,630 Python tests passed, 454 skipped, 7 xfailed, and one known baseline failure in test_differentiable_matrix_parameters[DeviceType.vulkan], with the identical 0.295745313167572 deviation reproduced on the saved starting binaries. The native suite passed 277 cases and 23,127 assertions (5 skips). Before the final const-reference-only write adjustment, 152 focused cursor/callback/marshalling tests passed with 2 skips; the final full suite includes those tests. Release builds, final pre-commit checks (including the benchmark script), and git diff --check passed. Logs: build/cursor-owner-python-release.log, build/cursor-owner-native-release.log, build/cursor-owner-precommit.log, and build/cursor-owner-borrow-build.log.

The 2026-09-21 follow-up shares typed converter code using a compile-time fallback flag, while retaining the cached specialized writer tables. Bulk writes construct one metadata view for checked validation and then write directly from row pointers; raw copies validate only row contiguity and destination bounds. Signed outer strides and the earlier one- or two-dimensional contiguous vector representations are supported. Python ShaderCursor traversal retains its parent, and reinterpretation also retains its layout. BufferElementCursor ownership remains native. ShaderObject reuses wrappers by actual RHI child identity and retains replaced children for outstanding native cursors; this bounds repeated traversal to one wrapper per distinct child, not one per lookup. Distinct replacement objects still require retention, and lookup scans the usually small retained-child list. RHI consolidation remains deferred.

Debug validation passed 251 focused Python tests and 17 native cursor tests, including replacement/rebinding and Python reference-count checks. The unrelated data submodule rollback was removed. Release benchmark and broad-suite results are recorded below. The full Python suite found one failure that also reproduces against the saved starting binaries; it is not introduced by this follow-up. All new regression tests pass. Metal execution remains a CI requirement, and zero performance regression is explicitly not claimed because Python child lifetime retention has a small measured cost.

Stages 1 and 2 are implemented on the working branch. Invalid cursors now have fully initialized state and reject direct reads/writes safely; array, vector, matrix, struct-field, and entry-point navigation enforce bounds; Python cursors keep their ShaderObject owner alive; and null BufferView values clear bindings consistently with the other nullable resources. Matrix numeric checks no longer skip Vulkan, CUDA, or Metal by policy.

The old CUDA bool stride overrides and `bool1` rejection are gone from ShaderCursor and BufferCursor. The Python bool-vector converter now uses the normal vector path. The SGL Metal parameter-block relayout constructor is removed, so dereference trusts the child layout supplied by slang-rhi. Unused `_set_array_unsafe`, `is_resource_type`, the late type-check macro, the old basic-type nanobind binder, and its generated doc symbol are removed. The change is a net reduction of roughly 250 source lines before plan bookkeeping.

Windows validation is complete across D3D12, Vulkan, and CUDA: the Debug build succeeds; ShaderCursor is 9/9; parameter blocks are 3/3; BufferCursor is 64 passed and 2 unrelated pointer skips; native tests are 265/265 with 21,031 assertions; and the full Python suite is 4,366 passed, 454 skipped, and 7 expected failures. PR #1122 subsequently passed the macOS ARM64 Debug and Release C++ and Python unit-test jobs, including the Metal coverage required after removing the parameter-block relayout. Stage 2 is complete. No performance measurement was needed because these stages do not change the cached functional dispatch representation.

After reviewing Release overhead, the duplicate always-on validity check in `ShaderCursor::_set_data` was converted to an internal assertion. Public validity and bounds checks remain always enabled, and no unchecked cached-navigation API was introduced without benchmark evidence.

Stage 3 gives ordinary reflected values a direct handled/not-handled fast path, followed by one centralized special-value fallback: registered native writers are tried once, a legacy `get_this` wrapper is resolved once for the current object, and ShaderCursor's CUDA ndarray support is the sole cursor-specific extension. `DescriptorHandle` participates in the native writer registry, while `NativePackedArg` remains explicit because its destination requirements are special. Repeated native-object and wrapper branches were removed from scalar, vector, matrix, pointer, struct, and array handling without adding registry or attribute lookups to successful ordinary writes.

Typed NumPy scalar, vector, matrix, direct-array, and bulk BufferCursor paths validate dtype, rank, shape, inner contiguity, and byte count before accessing storage. Raw unchecked bulk copies preserve prepacked storage and validate row contiguity and destination bounds instead of dtype compatibility. Validated data is sent directly through the existing typed cursor primitives, with no destination-typed `reinterpret_cast` reads. Exact types and same-width signed/unsigned pairs retain their previous bit-preserving behavior; incompatible numeric types reject with a Python `TypeError`, and invalid shape or storage rejects with `ValueError` including the cursor path. Partial dictionaries and ignored extra keys remain compatible and are covered by tests.

## Context and Orientation

A shader object is the host-side storage for a shader's ordinary data, resources, parameter blocks, and entry-point parameters. A shader cursor is a small pointer-like value that identifies one location inside a shader object. Its location consists of a shader object, a reflected Slang type layout, a byte offset for ordinary data, and binding-range and binding-array indices for resources.

The low-level implementation is `rhi::ShaderCursor` in `external/slang-rhi/include/slang-rhi/shader-cursor.h`. It navigates fields, arrays, vectors, matrices, structured container objects, and parameter blocks. It writes raw bytes with `setData`, binds RHI resources with `setBinding`, binds child shader objects, sets descriptor handles, and sets specialization arguments. It stores raw pointers and returns RHI `Result` values or invalid cursors. It does not convert Python or C++ math objects into backend layouts.

The SGL implementation is `sgl::ShaderCursor` in `src/sgl/device/shader_cursor.h` and `src/sgl/device/shader_cursor.cpp`. It currently stores a separate `ShaderObject*`, `slang::TypeLayoutReflection*`, and `sgl::ShaderOffset`, then reproduces much of the RHI navigation math. It additionally checks reflected types, accepts SGL buffers, textures, samplers, acceleration structures and descriptor handles, supports CUDA tensor views and device pointers, and inherits typed scalar/vector/matrix writes from `src/sgl/device/cursor_access_wrappers.h`.

`src/sgl/device/shader_object.h` and `src/sgl/device/shader_object.cpp` wrap `rhi::IShaderObject`. They translate SGL offsets and resources into RHI calls and retain CUDA interop buffers. They currently create and retain SGL wrappers for child and entry-point shader objects because SGL cursors hold weak pointers.

`src/slangpy_ext/device/cursor_utils.h` is the Python conversion layer shared by ShaderCursor and BufferCursor. `WriteConverterTable` examines the destination's reflected Slang type and accepts Python scalars, SGL math values, NumPy arrays, dictionaries, sequences, registered native objects, and legacy `get_this` wrappers. `src/slangpy_ext/device/shader_cursor.cpp` adds CUDA ndarray binding. This layer is distinct from the functional API's Slang type resolution: it writes a concrete Python value into an already-reflected destination.

The functional API uses ShaderCursor during dispatch in `src/slangpy_ext/utils/slangpy.cpp` and specialized marshalls such as `src/slangpy_ext/utils/slangpytensor.cpp` and `src/slangpy_ext/utils/slangpyvalue.cpp`. On the first call it navigates reflected fields and caches their offsets. Later calls reserve an ordinary-data block and write through cached offsets. This is the performance-sensitive path that must not be forced back through repeated name lookup.

The principal Python tests are `slangpy/tests/device/test_shader_cursor.py`, `slangpy/tests/device/test_parameter_block.py`, `slangpy/tests/device/test_buffer_cursor.py`, and relevant functional API tests under `slangpy/tests/slangpy_tests`. Native cursor tests are in `tests/sgl/device/test_cursors.cpp`. New Python APIs or behaviors require tests under `slangpy/tests/`.

## Plan of Work

### Stage 1: Characterize behavior and fix immediate safety defects

This stage changes no cursor representation. It adds tests around behavior that later stages must preserve and fixes defects whose remedies do not depend on the architectural refactor.

Extend `slangpy/tests/device/test_shader_cursor.py` and its Slang fixture to cover `bool1`, all currently skipped matrix shapes, invalid positive indices, invalid entry-point indices, and attempts to use a cursor after dropping the Python variable that originally held the shader object while the owning pass or root object remains valid. Expected matrix results must be expressed as numeric values produced by the shader rather than as backend-specific raw padding. Do not skip `float2x2` or `float3x3` on Vulkan, CUDA, or Metal.

Extend `tests/sgl/device/test_cursors.cpp` with native checks for an invalid default cursor, out-of-range struct/array/vector indices, out-of-range entry points, and nullable resource references. Cover at least `Buffer`, `BufferView`, `Texture`, `TextureView`, `Sampler`, and `AccelerationStructure` where the reflected destination supports the resource.

Initialize `ShaderCursor::m_type_layout` to `nullptr` in `src/sgl/device/shader_cursor.h`. Define `is_valid()` to require a non-null shader object, a non-null type layout, and a valid offset. Make `is_reference`, `dereference`, `find_entry_point`, and any direct setter either reject an invalid cursor with an SGL exception or return an invalid cursor according to the existing `find_` convention. Add explicit entry-point and element bounds checks before constructing child cursors.

Fix `ShaderObject::set_buffer_view` in `src/sgl/device/shader_object.cpp` so a null `ref<const BufferView>` clears the binding in the same way that null Buffer, Texture, and TextureView values do. Inspect the other nullable resource setters and make their behavior consistent.

Acceptance for this stage is that new tests fail against at least the relevant old behavior, pass after the fixes, and existing cursor, parameter-block, and native tests remain green on D3D12, Vulkan, and CUDA. Metal CI must at minimum run the expanded matrix and parameter-block tests.

### Stage 2: Remove obsolete compatibility paths and dead code

With characterization tests in place, delete workarounds whose triggering compiler or RHI behavior is no longer present.

In `src/sgl/device/cursor_access_wrappers.h`, remove the CUDA-specific bool-vector stride override from both write and read paths. Keep the generic conversion that compares CPU element size, reflected element size, and reflected element stride; that generic path is what supports one-byte CUDA bools and four-byte D3D12/Vulkan uniform bools.

In `src/sgl/device/shader_cursor.cpp` and `src/sgl/device/buffer_cursor.cpp`, replace the special CUDA-rejecting `bool1` implementations with the normal vector specialization. In `src/slangpy_ext/device/cursor_utils.h`, initially keep the Python bool-vector entry point unless tests prove it is now identical to the normal vector path; if it is identical, delete `bool_vector_case`, `_write_bool_vector`, and `_write_bool_vector_from_numpy` in the same stage.

Remove the SGL `ShaderCursor(ShaderObject*, bool need_dereference, TypeLayoutReflection*)` constructor and the manual `MetalArgumentBufferTier2` relayout. `dereference()` should trust the child RHI shader object's `getElementTypeLayout()`, which slang-rhi now sets to its argument-buffer layout on Metal. Run the nested parameter-block test on Metal before marking this subtraction complete.

Delete `_set_array_unsafe` if a repository-wide search still shows no callers. Delete the unused local `is_resource_type` helper, the ineffective late `SGL_ENABLE_CURSOR_TYPE_CHECKS` definition in `shader_cursor.cpp`, and the unused `bind_writable_cursor_basic_types` template if no generated or external code depends on it. Do not delete the shared matrix and bool size-conversion machinery that remains exercised by tests.

Acceptance is zero backend-specific bool stride branches, no SGL Metal relayout, successful explicit `bool1` writes on CUDA, and successful nested parameter blocks on Metal. The source tree must have no references to Slang issue 7441 unless retained in a historical changelog.

### Stage 3: Make Python conversion type-safe and centralize dispatch

This stage addresses Python writer complexity without changing native cursor storage.

Add tests to `slangpy/tests/device/test_shader_cursor.py` or a focused new test module under `slangpy/tests/device/` for NumPy scalar, vector, matrix, and array inputs. Cover matching dtypes, signed/unsigned same-width handling according to the existing `allow_scalar_conversion` contract, mismatched integer-to-float and float-to-integer dtypes, empty arrays, non-contiguous arrays, incorrect ranks, and incorrect shapes. A mismatched dtype must never be silently reinterpreted. Either perform a documented numeric conversion or raise a `TypeError`/`ValueError`; prefer rejection unless an existing non-NumPy path clearly promises numeric conversion.

Refactor `WriteConverterTable` in `src/slangpy_ext/device/cursor_utils.h` so every NumPy path obtains a source scalar type through `dtype_to_scalar_type`, validates contiguity, rank, shape, and byte count, and then invokes the shared typed cursor write. Remove direct `reinterpret_cast<ValType*>` reads from unvalidated storage. If copying into an aligned local C++ scalar or math value is necessary, use `memcpy` only after validation.

At the start of `write_internal`, try the ordinary representation selected by the reflected Slang kind. If the value does not match that representation, try the registered native cursor writer once, invoke the legacy `get_this` unpacking once and recurse only if it produced a different object, then try cursor-specific extensions. Remove repeated `write_registered_native_object` and `try_unpack_and_retry` calls from scalar, vector, matrix, pointer, struct, and array branches. Register `DescriptorHandle` in `cursor_utils::register_cursor_writers` so the backend-dependent DescriptorHandle special case can use the same native writer mechanism. Keep `NativePackedArg` explicit if registration would obscure its requirement that the destination be a shader-object field. This ordering keeps plain Python and SGL values off the registry and Python-attribute fallback paths.

Preserve partial dictionary updates: missing struct keys remain untouched. Decide explicitly whether extra dictionary keys remain ignored or become errors, record that choice in the Decision Log, and add a test.

Acceptance is that matching NumPy writes produce unchanged values on every backend, mismatched dtype cases reject deterministically, and the writer has one native-object lookup and one wrapper-unpack location. Run both ShaderCursor and BufferCursor tests because they share the table.

### Stage 3 follow-up: complete cleanup without RHI consolidation

In `src/slangpy_ext/device/cursor_utils.h`, share conversion helpers between generic dispatch and specialized cached writers, and validate bulk NumPy rows once before walking their pointers. Preserve the outer signed stride, reject non-contiguous inner storage, accept the previously supported contiguous vector shapes, and keep unchecked copies as bounded raw storage writes without dtype conversion. Add compatibility tests to both cursor test modules.

Use nanobind owner retention for ShaderCursor traversal, dereference, entry-point selection, field-by-index selection, and reinterpretation (including its layout owner). BufferElementCursor already owns its buffer and layout natively and should not acquire redundant Python retention. Add lifetime tests that drop intermediate Python cursors and verify ownership and usable writes.

In `src/sgl/device/shader_object.cpp`, look up the actual RHI child on each call and reuse a wrapper by that child identity. Keep replaced child wrappers alive for outstanding non-owning native cursors. Native tests must exercise repeated entry-point and parameter-block traversal, replacement, and rebinding the earlier object.

Measure before and after using `python -m tools.benchmark_cursors --device d3d12 --output build/cursors-before-d3d12.json` (and corresponding final output), after a Release build. Include direct Python and NumPy values, checked and raw bulk rows, nested parameter blocks, and warmed functional command recording with one and six arguments. Record that this baseline is the reviewed local diff before the follow-up cleanup, not the original HEAD. Keep RHI-specific stages below only as deferred design notes.

### Stage 4 (deferred): Introduce an RHI-backed SGL cursor representation

This stage changes internal storage while preserving all public SGL and Python method names. Implement it additively where practical so old and new navigation results can be compared during development.

Include `slang-rhi/shader-cursor.h` from `src/sgl/device/shader_cursor.h` and make `sgl::ShaderCursor` contain an `rhi::ShaderCursor` for the current low-level location. Retain a raw pointer or other lightweight anchor to the owning SGL `ShaderObject` only for device access, SGL exception translation, CUDA interop lifetime retention, and Python ownership. Do not maintain a second independent type-layout or offset as authoritative state. Convert the RHI `ShaderOffset` to the public `sgl::ShaderOffset` only at the API boundary.

Before removing `shader_object()`, update functional API code that assumes every cursor location has a corresponding SGL child wrapper. Add cursor-level internal operations for reserving data, setting data at a cached offset, and binding an SGL resource at a cached offset. Update `src/slangpy_ext/utils/slangpy.cpp`, `src/slangpy_ext/utils/slangpytensor.cpp`, `src/slangpy_ext/utils/slangpytorchtensor.cpp`, and `src/slangpy_ext/utils/slangpyvalue.cpp` to use those operations or to derive a cursor from the current RHI base object plus a cached layout and offset. The cached path must continue to avoid field-name lookup after its first call.

Move CUDA interop retention into an owner-facing helper that can retain an interop buffer while the actual binding is applied to the RHI base object held by the cursor. Do not recreate an SGL `ShaderObject` wrapper for every dereference. Once all callers use the RHI base object, remove the unconditional child-wrapper accumulation in `ShaderObject::get_object` and `ShaderObject::get_entry_point`, or replace it with a deduplicated cache only if another public API still requires wrappers.

Give the Python `ShaderCursor(ShaderObject)` constructor an explicit nanobind keep-alive relationship so the Python shader object cannot be collected while the cursor exists. Keep the native C++ cursor lightweight; measure any ref-counted ownership alternative before adopting it on the functional dispatch path.

During development, add a native test helper that compares the old reflected name/offset result with the new `rhi::ShaderCursor` result for structs, arrays of structs containing resources, vectors, matrices, parameter blocks, nested parameter blocks, root globals, and entry-point parameters. Remove the comparison helper with the old implementation at the end of Stage 5.

Acceptance is that public SGL and Python cursor APIs are unchanged, cached functional calls still use cached offsets, repeated parameter-block traversal does not grow a wrapper list, and all Stage 1 through Stage 3 tests pass.

### Stage 5 (deferred): Delegate navigation to RHI and finish convergence

Replace the hand-written offset arithmetic in `sgl::ShaderCursor::find_field`, `get_field_by_index`, and `find_element` with strict operations on the contained `rhi::ShaderCursor`.

For field names, preserve SGL's strict behavior by checking that the current reflected kind is a struct, finding the field index by name, and then using RHI's struct element-by-index traversal. For constant buffers and parameter blocks, explicitly dereference first. Do not call RHI `getField` on a root object because that method may search entry points. Continue to expose `find_entry_point` as the only way to enter entry-point parameters.

Use RHI `getElement` for arrays, vectors, matrices, structs, and shader-object containers after SGL performs the desired bounds check. This restores container and struct indexing that the RHI cursor supports and removes SGL's disabled container block. Decide whether Python numeric indexing of a struct should be public; if enabled, document and test it, and if not, keep `get_field_by_index` as the explicit public operation while still using RHI internally.

Delegate raw `setData`, `reserveData`, `setObject`, `setBinding`, `setDescriptorHandle`, and pointer writes to the contained RHI cursor, translating failed `Result` values through the existing SGL error mechanism. Retain SGL's reflected type checks where they provide useful public diagnostics, resource wrapper conversion, CUDA sampler compatibility, and CUDA tensor-view behavior. Do not duplicate RHI offset conversion in `ShaderObject` after callers have migrated.

After parity tests pass, delete obsolete SGL navigation state, offset arithmetic, dereference wrapper creation, and disabled code. Update comments in `shader_cursor.h` to describe the new ownership and delegation model accurately.

Measure a representative cached functional call before and after this stage using an existing benchmark if one is present. If no suitable benchmark exists, add a focused test utility that warms a simple tensor call, performs enough repeated dispatch recordings to measure CPU overhead, and reports median time. Treat a repeatable regression greater than five percent in the cached call-setup path as a stop condition requiring profiling before landing. GPU execution time is not the relevant measurement.

Acceptance is that one implementation—RHI—owns cursor offset navigation, SGL owns typed policy, no public behavior regresses, and the cached functional dispatch overhead remains within the agreed threshold.

## Concrete Steps

Work from `C:\src\slangpy-side`. At the start of each stage, inspect `git status --short` and preserve unrelated user changes. Search with `rg` before deleting any symbol. Apply source edits with the repository patch workflow.

Always build outside the sandbox before running tests, as required by `AGENTS.md`:

    cd C:\src\slangpy-side
    cmake --build --preset windows-msvc-debug

Run focused Python tests outside the sandbox after every relevant edit:

    pytest slangpy/tests/device/test_shader_cursor.py -v
    pytest slangpy/tests/device/test_parameter_block.py -v
    pytest slangpy/tests/device/test_buffer_cursor.py -v

Run focused functional tests chosen by searching for the marshalls modified in Stage 4. At minimum, include tensor, NumPy, value, resource, and torch-integration tests when those implementations change. Record the exact selected node IDs in Progress when the stage begins.

Run native tests outside the sandbox after native cursor or shader-object edits:

    python tools/ci.py unit-test-cpp

Before completing each stage, run the broader Python suite outside the sandbox:

    pytest slangpy/tests -v

For any stage that changes Metal parameter-block or matrix behavior, run the focused test commands on a macOS Metal runner. Record the runner, device, and pass counts in Artifacts and Notes.

After all source changes in each stage, run pre-commit and rerun it if it modifies files:

    pre-commit run --all-files

Inspect the final diff for the stage:

    git diff --check
    git diff --stat
    git status --short

Do not commit automatically unless the active workflow or user request asks for commits. Keep each stage in a state that can be committed or submitted as a separate review.

## Validation and Acceptance

The currently authorized work is accepted when Stages 1 through 3 and the Stage 3 follow-up are complete. Stages 4 and 5 and the RHI convergence criteria below remain deferred. The following observable behavior applies where it does not require those deferred stages.

Python code can assign matching Python scalars, SGL vectors and matrices, nested dictionaries and lists, matching NumPy values, resources, descriptor handles, CUDA arrays, and parameter blocks through `ShaderCursor` exactly as before. `bool1` through `bool4` work on CUDA rather than using an old blanket rejection. D3D12 and Vulkan uniform bools continue to receive correct four-byte values.

For typed writes, incorrect NumPy dtype, shape, rank, size, or inner contiguity produces a deterministic Python exception and never a bit reinterpretation. Error messages identify the cursor path and expected type where practical.

The numeric values read by shaders for every tested matrix shape match the host input on D3D12, Vulkan, CUDA, and Metal. There are no backend skips for `float2x2` or `float3x3` solely because their padding differs.

Nested parameter blocks containing both ordinary fields and resources pass on Metal after the SGL relayout is removed. The RHI shader object's element layout is the sole layout used for dereferenced Metal parameter blocks.

Invalid cursors, invalid entry-point indices, invalid element indices, and null supported resource bindings fail safely or clear the binding according to the documented API; none dereferences an uninitialized or null pointer.

Repeated navigation into the same parameter block does not retain an ever-growing collection of SGL child wrappers. A Python cursor keeps the necessary owner alive for the cursor's usable lifetime.

Repository search shows no duplicated SGL offset arithmetic for field, array, vector, matrix, or container traversal. The SGL cursor stores an RHI cursor as its authoritative low-level location and layers typed behavior over it.

The Debug build, focused Python tests, relevant functional tests, full `slangpy/tests`, native C++ suite, Metal CI, and pre-commit all pass. Cached functional call setup has no repeatable CPU overhead regression greater than five percent; any accepted exception must be recorded with profiling evidence and user approval in the Decision Log.

## Idempotence and Recovery

All build and test commands are safe to rerun. Each stage is designed to leave the repository buildable and testable, so a failed later stage can be abandoned without reverting earlier correctness improvements.

If removal of the CUDA bool workaround exposes a backend discrepancy, restore only the smallest compatibility branch and record the exact reflected size, stride, Slang version, and failing test in Surprises and Discoveries. Do not restore the old blanket `bool1` rejection without evidence that current CUDA code generation is incorrect.

If Metal parameter-block CI fails after removing the SGL relayout, compare the child RHI shader object's `getElementTypeLayout()` with `MetalArgumentBufferTier2` reflection and record field offsets for the first mismatch. Restore the SGL workaround temporarily only if the RHI object layout is demonstrably wrong, and open or prepare a slang-rhi fix rather than maintaining two permanent layout authorities.

If the RHI-backed cursor causes semantic or performance regressions, keep the Stage 1 through Stage 3 changes and revert only Stage 4 or Stage 5. The characterization tests define the behavior that any revised representation must satisfy.

Do not use destructive git commands to recover. Use file-scoped patches or ordinary version-control reverts only when explicitly authorized. Preserve unrelated changes in a dirty worktree.

## Artifacts and Notes

Baseline captured on Windows on 2026-08-21 with the repository at SlangPy commit `1f1d2a54a22f82844a64caeb7cd35c26b7aa1c61`, slang-rhi commit `20cae56bc618df7e5a980f4686782bc64baec2c1`, and fetched Slang version 2026.12.2:

    test_shader_cursor.py: 9 passed
    test_parameter_block.py: 3 passed
    native sgl_tests: 263 passed, 7 skipped, 20,994 assertions passed

The native skips were unrelated NVTT backend availability. D3D12, Vulkan, and CUDA were exercised. Metal was not available on the baseline host and remains an explicit validation requirement.

Stage 1 and 2 Windows validation on 2026-08-21:

    cmake --build --preset windows-msvc-debug: passed
    test_shader_cursor.py: 9 passed
    test_parameter_block.py: 3 passed
    test_buffer_cursor.py: 64 passed, 2 skipped
    native sgl_tests: 265 passed, 7 skipped, 21,031 assertions passed
    full slangpy/tests: 4,366 passed, 454 skipped, 7 xfailed
    pre-commit run --all-files: passed

The first serial full-suite attempt reached the 904-second tool timeout. The complete rerun used three pytest workers and finished in 340.40 seconds.

PR #1122 Metal validation on 2026-08-21:

    https://github.com/shader-slang/slangpy/pull/1122
    checks workflow: passed
    ci workflow: passed
    macOS ARM64 clang Debug: build, C++ unit tests, and Python unit tests passed
    macOS ARM64 clang Release: build, C++ unit tests, Python unit tests, and examples passed

These macOS jobs exercise the expanded ShaderCursor and parameter-block suite after removal of the SGL Metal relayout, satisfying the remaining Stage 2 acceptance requirement.

Stage 3 Windows validation on 2026-08-21:

    cmake --build --preset windows-msvc-debug: passed
    cmake --build --preset windows-msvc-release: passed
    focused ShaderCursor, BufferCursor, and parameter-block tests: 82 passed, 2 skipped
    focused cursor and functional marshalling tests: 101 passed, 2 skipped, 4 xfailed
    post-fast-path focused cursor and functional marshalling tests: 104 passed, 2 skipped, 4 xfailed
    native Release sgl_tests: 266 passed, 7 skipped, 21,036 assertions passed
    full Release slangpy/tests: 4,372 passed, 454 skipped, 7 xfailed
    post-fast-path full Release slangpy/tests: 4,375 passed, 454 skipped, 7 xfailed
    pre-commit run --all-files: passed

The user explicitly deferred benchmarking for this stage. The new validation runs only for direct NumPy cursor conversion, while the dispatch refactor removes repeated native-writer and wrapper checks and does not change the functional API's cached cursor-offset path.

Update this section with short before/after performance measurements, Metal results, and any representative error messages introduced by NumPy validation.


### Stage 3 follow-up validation, 2026-09-21

Builds use Slang 2026.17.1, as specified in external/CMakeLists.txt. Debug validation passed 251 focused Python tests (2 skips) and 17 native cursor cases (521 assertions). Release native validation passed 277 cases (23,123 assertions, 5 skips). The full Release Python suite finished in 174 seconds with 4,624 passed, 454 skipped, 7 xfailed, and one failure in test_differentiable_matrix_parameters[DeviceType.vulkan]. An isolated rerun and a rerun with the saved pre-cleanup binaries both fail with exactly the same gradient deviation (0.295745313167572), confirming this failure predates the follow-up. The baseline comparison log is build/cursor-cleanup-baseline-gradient.log. Final pre-commit and git diff --check pass.

Performance was compared to the reviewed local diff before this follow-up, not to original HEAD. The starting Release sgl.dll and Python extension were saved before edits, then run in an isolated package/runtime under build/cursor-baseline. Each backend used three alternating baseline/final process pairs, each containing nine measured rounds after two warmups. Results below are medians of all 27 samples, in nanoseconds per operation. Functional workloads record 2,000 warmed calls per round with an explicit result tensor; submission and GPU synchronization are outside timing. Bulk operations write 1,024 rows. The scalar/vector/NumPy-vector cases include creating and destroying a child Python cursor; scalar_cached_cursor and scalar_assignment do not.

d3d12:

    scalar: 965.0 -> 1019.8 ns (+5.7%)
    scalar_cached_cursor: 428.5 -> 418.8 ns (-2.3%)
    scalar_assignment: 701.7 -> 696.3 ns (-0.8%)
    vector: 1233.5 -> 1300.9 ns (+5.5%)
    numpy_vector: 1774.4 -> 1867.6 ns (+5.3%)
    bulk_checked_1024: 338311.0 -> 165583.0 ns (-51.1%)
    bulk_raw_1024: 68066.0 -> 17346.0 ns (-74.5%)
    nested_parameter_block: 2486.1 -> 2317.8 ns (-6.8%)
    functional_1_args: 7551.2 -> 7144.2 ns (-5.4%)
    functional_6_args: 8038.6 -> 8095.6 ns (+0.7%)

vulkan:

    scalar: 946.3 -> 1024.3 ns (+8.2%)
    scalar_cached_cursor: 422.4 -> 420.1 ns (-0.5%)
    scalar_assignment: 693.7 -> 694.5 ns (+0.1%)
    vector: 1234.9 -> 1297.8 ns (+5.1%)
    numpy_vector: 1756.8 -> 1883.5 ns (+7.2%)
    bulk_checked_1024: 336971.0 -> 165455.0 ns (-50.9%)
    bulk_raw_1024: 68261.0 -> 17317.0 ns (-74.6%)
    nested_parameter_block: 2428.4 -> 2282.1 ns (-6.0%)
    functional_1_args: 6651.4 -> 6775.2 ns (+1.9%)
    functional_6_args: 7802.9 -> 7859.9 ns (+0.7%)

cuda:

    scalar: 946.5 -> 1018.8 ns (+7.6%)
    scalar_cached_cursor: 423.7 -> 417.9 ns (-1.4%)
    scalar_assignment: 693.0 -> 692.1 ns (-0.1%)
    vector: 1239.4 -> 1311.1 ns (+5.8%)
    numpy_vector: 1764.3 -> 1901.2 ns (+7.8%)
    bulk_checked_1024: 339263.0 -> 166239.0 ns (-51.0%)
    bulk_raw_1024: 69253.0 -> 17611.0 ns (-74.6%)
    nested_parameter_block: 2408.0 -> 2291.9 ns (-4.8%)
    functional_1_args: 5579.9 -> 5216.5 ns (-6.5%)
    functional_6_args: 5806.9 -> 6078.9 ns (+4.7%)

Checked bulk writes take about half the CPU time and raw bulk writes about one quarter. Nested parameter-block traversal improves roughly 5-7%. Held-cursor writes and ordinary scalar assignment remain approximately unchanged. Temporary Python child-cursor writes cost an extra 55-140 ns (5-8%), reflecting newly required owner retention. This earlier tradeoff was rejected by the user and is superseded by the current ownership follow-up results above. Cached functional medians range from improvements to a maximum 4.7% increase, with substantial variation among individual runs; these measurements did not identify a repeatable regression above the plan's five-percent threshold. Metal cannot be executed on this Windows host and needs CI coverage for the new changes.

Raw samples are build/cursors-{baseline,final}-{d3d12,vulkan,cuda}-{0,1,2}.json and the aggregate is build/cursor-cleanup-performance.json. Reproduce with tools/benchmark_cursors.py using Release builds. Full-suite logs are build/cursor-cleanup-python-release.log and build/cursor-cleanup-native-release.log. Baseline runtime artifacts are ignored build outputs and do not alter tracked sources.

## Interfaces and Dependencies

Do not add external dependencies. Use the existing Slang reflection API, `rhi::ShaderCursor`, SGL reference and resource types, nanobind, NumPy ndarray bindings, doctest, and pytest.

At the end of Stage 5, `sgl::ShaderCursor` must remain the public C++ and Python-facing type. Its public operations must continue to include construction from `ShaderObject`, `reinterpret`, `dereference`, `find_field`, `find_element`, `find_entry_point`, `find_field_index`, `get_field_by_index`, `has_field`, `has_element`, `set_data`, typed `set`, resource setters, descriptor-handle binding, CUDA tensor-view binding, pointer binding, and the Python traversal and write operators.

Internally, `sgl::ShaderCursor` must contain an `rhi::ShaderCursor` as the authoritative base object, type layout, container type, and offset. SGL may retain an owner or device anchor, but it must not maintain independent navigation offsets that can diverge from RHI. Cached functional API code may construct a cursor at a known layout and offset without name traversal, provided that constructor uses the current RHI base object and is covered by parity tests.

`CursorWriteWrappers` remains responsible for packing a typed CPU scalar, vector, array, or matrix into reflected ordinary-data storage. Its implementation must use reflected element sizes and strides and must not include compiler-version-specific CUDA bool assumptions after Stage 2.

`WriteConverterTable` remains responsible for turning Python objects into typed cursor writes. Its NumPy paths must validate metadata before reading storage, and its native writer and `get_this` fallback must each have one clear dispatch point.

Revision note, 2026-08-21: Initial ExecPlan created from the completed ShaderCursor architecture review. The stages deliberately separate safety, obsolete compatibility removal, Python conversion cleanup, and RHI convergence so every stage can be reviewed and validated independently.

Revision note, 2026-08-21: Implemented Stage 1 and the source changes for Stage 2. Recorded Windows D3D12/Vulkan/CUDA and full-suite results, moved Python owner keep-alive into the immediate safety stage, removed the now-proven-redundant Python bool-vector path, and left Stage 2 open solely for Metal CI validation.

Revision note, 2026-08-21: Marked Stage 2 complete after PR #1122 passed both macOS ARM64 Debug and Release C++/Python test jobs, providing the required Metal validation for the removed parameter-block relayout.

Revision note, 2026-08-21: Implemented Stage 3. Centralized native and wrapper dispatch, registered `DescriptorHandle`, replaced unvalidated NumPy reads with validated typed cursor writes, preserved dictionary compatibility, and recorded Debug/Release, focused, native, and full-suite validation.

Revision note, 2026-08-21: Adjusted Stage 3 dispatch ordering after reviewing direct cursor-write overhead. Successful ordinary reflected values now bypass native-registry and `get_this` fallback work, with regression coverage on D3D12, Vulkan, and CUDA.

Revision note, 2026-09-21: User deferred RHI consolidation and approved the Stage 3 follow-up. Benchmarking is now in scope; prior deferral is historical. Existing review validation passed 92 focused Python tests and 16 native cursor tests using Slang 2026.17.1.

Revision note, 2026-09-21: Completed the authorized follow-up, added reproducible Release benchmarks, documented the measured Python ownership cost, and confirmed that the sole full-suite failure also occurs in the starting binaries. No RHI consolidation was performed.

Revision note, 2026-09-21 (ownership follow-up in progress): Replaced keep_alive parent chains with a Python-only ShaderCursor subclass holding direct Python references to the shader object and optional reinterpreted layout. The native cursor remains unchanged. A native-cursor type caster preserves public Python ShaderCursor identity and owner retention at callback boundaries. Initial Release build and 136 focused tests passed (2 skips). Alternating comparisons show that direct ownership removes most of the previous regression but leaves roughly 2-3% on temporary scalar traversal; this is not yet accepted as the final result. Investigating direct construction in nanobind instance storage to avoid the general return-value pointer lookup and move callback. Added ordinary nested assignment benchmarks and tests for parent release, owner release, and native callback descendants.

Ownership validation discovery: The new callback-retention test reproduced an access violation after recording ended. CommandEncoder::_get_root_object creates ShaderObject with retain=false, so keeping only its SGL wrapper alive does not retain the underlying RHI object. Python cursor construction now calls an idempotent retain_rhi_shader_object() on its owner. This is only done when a cursor crosses into Python; native-only dispatch still borrows its object as before. The callback test remains in the suite to verify this boundary. Ordinary Python child traversal copies a single owner reference; reinterpretation stores the shader-object/layout pair in a tuple, and subsequent traversal copies that tuple reference. Direct construction of returned wrapper values uses nanobind instance APIs and preserves the public ShaderCursor type.

Ownership follow-up validation update: Release build passed. Focused cursor, parameter-block, callback, and native-marshalling tests passed 149 tests with 2 skips, including the callback-retention reproduction and repeated reinterpretation. Full native Release suite passed 277 cases and 23,127 assertions with 5 skips. Full Python suite and final single-reference ownership timings are pending. Repository pre-commit and benchmark-script checks pass.

Ownership follow-up results before the last constructor optimization: Full Python suite passed 4,627 tests, skipped 454 and xfailed 7; the only failure is the previously reproduced Vulkan matrix-gradient error of 0.295745313167572. Isolated alternating benchmarks show temporary scalar writes 1-2% faster, vector writes 0-2% faster, nested parameter-block traversal 14-16% faster, checked bulk roughly 50% faster and raw bulk 73-75% faster. Other writes/dispatch vary by approximately 0-2.5%. The added root-construction benchmark exposed a 14-15% cost from converting the constructor's native ShaderObject pointer back to its Python owner. Replaced that redundant lookup with nanobind pointer_and_handle<ShaderObject>, retaining the original Python argument directly while preserving type checking and the one-time RHI retain. Constructor invalid-input tests were added; rebuild and final checks pending.

Final constructor validation: 152 focused tests passed (2 skips), including invalid owner rejection. Root construction is now 14-15% faster than baseline, temporary scalar writes 1-2% faster, vector writes 0.5-1.2% faster, NumPy writes effectively unchanged. However, direct scalar assignment remains about 2% slower across backends, and cached dispatch samples show increases up to 3.5%; do not classify those as proven noise. Removed two redundant owning Python-object copies from WriteConverterTable::write and write_internal by taking const references. Their callers keep the inputs alive throughout the synchronous conversion, including recursively accessed dictionary/sequence values. Validate this final adjustment and measure again before performance signoff.

Revision note, 2026-09-21: Completed the cheaper ownership implementation and validation. The previous temporary-child traversal regression is removed, direct assignment is effectively unchanged, and cached functional dispatch remains within 0.6% of the starting runtime in the final alternating comparison. All local validation is complete except the independently confirmed pre-existing Vulkan gradient failure; Metal execution remains a CI requirement. RHI consolidation stays deferred.
