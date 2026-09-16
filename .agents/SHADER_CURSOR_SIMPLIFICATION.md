# Simplify ShaderCursor while preserving typed binding behavior

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy currently has two implementations of shader-object cursor navigation: the low-level `rhi::ShaderCursor` in the `external/slang-rhi` submodule and a separate `sgl::ShaderCursor` in SlangPy. The SGL implementation also contains valuable behavior that the RHI cursor intentionally does not provide, including conversion from C++ and Python values, backend-aware packing of booleans and matrices, SGL resource wrappers, CUDA interoperability, and Python error reporting. Over time, however, fixes in Slang and slang-rhi have made some SGL compatibility code obsolete, while the duplicate navigation implementations have diverged.

After this work, users should observe the same successful binding behavior on D3D12, Vulkan, CUDA, and Metal, with additional rejection of unsafe NumPy inputs and safer cursor lifetime and bounds behavior. Maintainers should see one source of truth for shader-object navigation and offset calculation, fewer backend workarounds, and a smaller Python writer dispatch. The result is demonstrated by focused cursor and parameter-block tests, the native C++ suite, full relevant Python tests, Metal CI, and a dispatch-performance comparison that does not regress the cached functional API path.

The work is divided into independently reviewable stages. Correctness tests and safety fixes land before obsolete code is removed. Python conversion is simplified before native cursor storage is changed. The SGL-to-RHI navigation convergence is last because it has the largest blast radius and must preserve explicit entry-point semantics and the functional API's cached-offset fast path.

## Progress

- [x] (2026-08-21 09:43Z) Reviewed the SGL, Python binding, functional API, and slang-rhi cursor implementations and recorded the current architectural boundaries.
- [x] (2026-08-21 09:43Z) Established the baseline: the Debug build succeeds; `test_shader_cursor.py` passes 9 tests; `test_parameter_block.py` passes 3 tests; the native suite passes 263 test cases and 20,994 assertions on Windows with D3D12, Vulkan, and CUDA.
- [x] (2026-08-21 09:43Z) Probed current reflection behavior and confirmed that Slang 2026.12.2 reports packed one-byte CUDA `bool1` through `bool4` layouts, while D3D12 and Vulkan retain four-byte uniform bool elements.
- [x] (2026-08-21 10:23Z) Stage 1: added characterization tests and fixed concrete cursor validity, bounds, Python ownership, and nullable BufferView defects. The Debug build, focused D3D12/Vulkan/CUDA tests, native suite, and full Python suite pass. Metal execution remains covered by CI rather than this Windows host.
- [ ] Stage 2: implementation and Windows validation are complete. Obsolete CUDA bool paths, the SGL Metal relayout, and dead cursor code are removed; explicit `bool1` works on CUDA. Keep this stage open until the nested parameter-block scenario passes on a Metal runner, as required by the Decision Log.
- [ ] Stage 3: make NumPy conversion type-safe and simplify Python writer dispatch.
- [ ] Stage 4: introduce an RHI-backed SGL cursor representation while preserving the existing public API and cached functional API behavior.
- [ ] Stage 5: retire duplicated SGL navigation, complete cross-backend validation, measure performance, and document the final architecture.

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

## Decision Log

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

## Outcomes and Retrospective

Stages 1 and 2 are implemented on the working branch. Invalid cursors now have fully initialized state and reject direct reads/writes safely; array, vector, matrix, struct-field, and entry-point navigation enforce bounds; Python cursors keep their ShaderObject owner alive; and null BufferView values clear bindings consistently with the other nullable resources. Matrix numeric checks no longer skip Vulkan, CUDA, or Metal by policy.

The old CUDA bool stride overrides and `bool1` rejection are gone from ShaderCursor and BufferCursor. The Python bool-vector converter now uses the normal vector path. The SGL Metal parameter-block relayout constructor is removed, so dereference trusts the child layout supplied by slang-rhi. Unused `_set_array_unsafe`, `is_resource_type`, the late type-check macro, the old basic-type nanobind binder, and its generated doc symbol are removed. The change is a net reduction of roughly 250 source lines before plan bookkeeping.

Windows validation is complete across D3D12, Vulkan, and CUDA: the Debug build succeeds; ShaderCursor is 9/9; parameter blocks are 3/3; BufferCursor is 64 passed and 2 unrelated pointer skips; native tests are 265/265 with 21,031 assertions; and the full Python suite is 4,366 passed, 454 skipped, and 7 expected failures. Stage 2 remains formally open only for the required Metal nested-parameter-block run. No performance measurement was needed because these stages do not change the cached functional dispatch representation.

After reviewing Release overhead, the duplicate always-on validity check in `ShaderCursor::_set_data` was converted to an internal assertion. Public validity and bounds checks remain always enabled, and no unchecked cached-navigation API was introduced without benchmark evidence.

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

At the start of `write_internal`, try the registered native cursor writer once. If none applies, invoke the legacy `get_this` unpacking once and recurse only if it produced a different object. Then switch on the reflected Slang kind. Remove repeated `write_registered_native_object` and `try_unpack_and_retry` calls from scalar, vector, matrix, pointer, struct, and array branches. Register `DescriptorHandle` in `cursor_utils::register_cursor_writers` so the backend-dependent DescriptorHandle special case can use the same native writer mechanism. Keep `NativePackedArg` explicit if registration would obscure its requirement that the destination be a shader-object field.

Preserve partial dictionary updates: missing struct keys remain untouched. Decide explicitly whether extra dictionary keys remain ignored or become errors, record that choice in the Decision Log, and add a test.

Acceptance is that matching NumPy writes produce unchanged values on every backend, mismatched dtype cases reject deterministically, and the writer has one native-object lookup and one wrapper-unpack location. Run both ShaderCursor and BufferCursor tests because they share the table.

### Stage 4: Introduce an RHI-backed SGL cursor representation

This stage changes internal storage while preserving all public SGL and Python method names. Implement it additively where practical so old and new navigation results can be compared during development.

Include `slang-rhi/shader-cursor.h` from `src/sgl/device/shader_cursor.h` and make `sgl::ShaderCursor` contain an `rhi::ShaderCursor` for the current low-level location. Retain a raw pointer or other lightweight anchor to the owning SGL `ShaderObject` only for device access, SGL exception translation, CUDA interop lifetime retention, and Python ownership. Do not maintain a second independent type-layout or offset as authoritative state. Convert the RHI `ShaderOffset` to the public `sgl::ShaderOffset` only at the API boundary.

Before removing `shader_object()`, update functional API code that assumes every cursor location has a corresponding SGL child wrapper. Add cursor-level internal operations for reserving data, setting data at a cached offset, and binding an SGL resource at a cached offset. Update `src/slangpy_ext/utils/slangpy.cpp`, `src/slangpy_ext/utils/slangpytensor.cpp`, `src/slangpy_ext/utils/slangpytorchtensor.cpp`, and `src/slangpy_ext/utils/slangpyvalue.cpp` to use those operations or to derive a cursor from the current RHI base object plus a cached layout and offset. The cached path must continue to avoid field-name lookup after its first call.

Move CUDA interop retention into an owner-facing helper that can retain an interop buffer while the actual binding is applied to the RHI base object held by the cursor. Do not recreate an SGL `ShaderObject` wrapper for every dereference. Once all callers use the RHI base object, remove the unconditional child-wrapper accumulation in `ShaderObject::get_object` and `ShaderObject::get_entry_point`, or replace it with a deduplicated cache only if another public API still requires wrappers.

Give the Python `ShaderCursor(ShaderObject)` constructor an explicit nanobind keep-alive relationship so the Python shader object cannot be collected while the cursor exists. Keep the native C++ cursor lightweight; measure any ref-counted ownership alternative before adopting it on the functional dispatch path.

During development, add a native test helper that compares the old reflected name/offset result with the new `rhi::ShaderCursor` result for structs, arrays of structs containing resources, vectors, matrices, parameter blocks, nested parameter blocks, root globals, and entry-point parameters. Remove the comparison helper with the old implementation at the end of Stage 5.

Acceptance is that public SGL and Python cursor APIs are unchanged, cached functional calls still use cached offsets, repeated parameter-block traversal does not grow a wrapper list, and all Stage 1 through Stage 3 tests pass.

### Stage 5: Delegate navigation to RHI and finish convergence

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

The whole plan is accepted when all five stages are complete and the following observable behavior holds.

Python code can assign matching Python scalars, SGL vectors and matrices, nested dictionaries and lists, matching NumPy values, resources, descriptor handles, CUDA arrays, and parameter blocks through `ShaderCursor` exactly as before. `bool1` through `bool4` work on CUDA rather than using an old blanket rejection. D3D12 and Vulkan uniform bools continue to receive correct four-byte values.

Incorrect NumPy dtype, shape, rank, size, or contiguity produces a deterministic Python exception and never a bit reinterpretation. Error messages identify the cursor path and expected type where practical.

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

The first serial full-suite attempt reached the 904-second tool timeout. The complete rerun used three pytest workers and finished in 340.40 seconds. Metal is unavailable on this host; `test_parameter_block.py` and the expanded ShaderCursor matrix/bool coverage must still run on a macOS Metal CI worker before Stage 2 is checked complete.

Update this section with short before/after performance measurements, Metal results, and any representative error messages introduced by NumPy validation.

## Interfaces and Dependencies

Do not add external dependencies. Use the existing Slang reflection API, `rhi::ShaderCursor`, SGL reference and resource types, nanobind, NumPy ndarray bindings, doctest, and pytest.

At the end of Stage 5, `sgl::ShaderCursor` must remain the public C++ and Python-facing type. Its public operations must continue to include construction from `ShaderObject`, `reinterpret`, `dereference`, `find_field`, `find_element`, `find_entry_point`, `find_field_index`, `get_field_by_index`, `has_field`, `has_element`, `set_data`, typed `set`, resource setters, descriptor-handle binding, CUDA tensor-view binding, pointer binding, and the Python traversal and write operators.

Internally, `sgl::ShaderCursor` must contain an `rhi::ShaderCursor` as the authoritative base object, type layout, container type, and offset. SGL may retain an owner or device anchor, but it must not maintain independent navigation offsets that can diverge from RHI. Cached functional API code may construct a cursor at a known layout and offset without name traversal, provided that constructor uses the current RHI base object and is covered by parity tests.

`CursorWriteWrappers` remains responsible for packing a typed CPU scalar, vector, array, or matrix into reflected ordinary-data storage. Its implementation must use reflected element sizes and strides and must not include compiler-version-specific CUDA bool assumptions after Stage 2.

`WriteConverterTable` remains responsible for turning Python objects into typed cursor writes. Its NumPy paths must validate metadata before reading storage, and its native writer and `get_this` fallback must each have one clear dispatch point.

Revision note, 2026-08-21: Initial ExecPlan created from the completed ShaderCursor architecture review. The stages deliberately separate safety, obsolete compatibility removal, Python conversion cleanup, and RHI convergence so every stage can be reviewed and validated independently.

Revision note, 2026-08-21: Implemented Stage 1 and the source changes for Stage 2. Recorded Windows D3D12/Vulkan/CUDA and full-suite results, moved Python owner keep-alive into the immediate safety stage, removed the now-proven-redundant Python bool-vector path, and left Stage 2 open solely for Metal CI validation.
