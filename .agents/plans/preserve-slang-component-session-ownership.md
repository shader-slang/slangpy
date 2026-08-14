# Preserve Slang component and session ownership in SGL shader objects

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy users can pack Python values into standalone shader objects and pass those values to GPU functions. A Slang type layout is meaningful only in the compiler session and component program that created it. SlangPy currently discards that component identity when it creates a standalone shader object, causing ordinary packed arguments to be treated as if they belong to slang-rhi's private compiler session. With slang-rhi commit `0969630b`, the unsafe mismatch is rejected instead of allowing cross-session compiler operations.

After this change, standalone shader objects created from module, entry-point, linked-program, or existing-shader-object reflection retain their originating Slang component. Objects from the same custom SlangPy session bind successfully. Objects retained across hot reload or created by a genuinely different session fail before mutation with an error that identifies a stale or cross-session shader object. The behavior is demonstrated by native SGL tests and the existing packed-argument Python suite.

## Progress

- [x] (2026-08-13 12:11Z) Read the cross-session handoff, relevant SGL reflection/shader-object code, local slang-rhi API change, and repository planning instructions.
- [x] (2026-08-13 12:14Z) Freshly configured and built the unmodified checkout against `C:/src/slang-rhi-side`; all 353 debug build steps completed successfully.
- [x] (2026-08-13 12:24Z) Added a narrow internal way to resolve and retain the `slang::IComponentType` that owns SGL reflection.
- [x] (2026-08-13 12:24Z) Propagated the component through root, standalone, child, entry-point, and pass-encoder SGL `ShaderObject` wrappers.
- [x] (2026-08-13 12:24Z) Switched program-derived reflection to slang-rhi's owner-aware `createShaderObjectFromTypeLayout` overload.
- [x] (2026-08-13 12:24Z) Added deterministic, contextual cross-session and stale-after-reload failure handling before the RHI call.
- [x] (2026-08-13 12:31Z) Added and passed native and Python coverage for owner mapping, custom sessions, component lifetime, derived layouts, cross-session rejection, and hot reload.
- [x] (2026-08-13 12:33Z) Rebuilt and passed focused native tests, the expanded packed-argument suite, and the full native suite outside the sandbox.
- [x] (2026-08-13 12:38Z) Ran `pre-commit run --all-files`; clang-format modified C++ files on the first pass and every hook passed on the second pass.

## Surprises and Discoveries

- Observation: The current checkout already pins `external/slang-rhi` to `0969630b` in commit `1308075f`, although the handoff describes using `SGL_LOCAL_RHI` without updating a submodule.
  Evidence: `git submodule status external/slang-rhi` reports `0969630b0a3dfe8f5c10c64129506b8bc5e2a8e8`, and `git show 1308075f` changes only that gitlink.

- Observation: Reflection descendants already carry the original high-level SGL owner rather than the immediate reflection wrapper.
  Evidence: methods in `src/sgl/device/reflection.h` call `detail::from_slang(m_owner, ...)`, while module, entry-point, and program layout creation supplies the corresponding `SlangModule`, `SlangEntryPoint`, or `ShaderProgram` as `m_owner`.

- Observation: The unmodified packed-argument suite fails all 12 cases on D3D12, Vulkan, and CUDA at the same RHI session check.
  Evidence: `pytest slangpy/tests/slangpy_tests/test_packed_arg.py -v` reports 12 failures from `ShaderObject::set_object`, each with `SLANG_E_INVALID_ARG`.

- Observation: Root shader-object wrappers are also created when a command pass binds a pipeline, not only through `Device::create_root_shader_object`.
  Evidence: `RenderPassEncoder::bind_pipeline`, `ComputePassEncoder::bind_pipeline`, and `RayTracingPassEncoder::bind_pipeline` all call `CommandEncoder::_get_root_object`. These roots need the pipeline program's component so their derived reflection preserves ownership.

- Observation: Slang's C compatibility declarations introduce global `SlangSession` and `SlangEntryPoint` names that are ambiguous with the SGL classes in a test using `using namespace sgl`.
  Evidence: The first regression-test build failed with MSVC C2872 for both names; explicitly using `sgl::SlangSession` and `sgl::SlangEntryPoint` resolves the intended types.

- Observation: A module-only `ProgramLayout` does not expose the module's entry points through `get_entry_point_by_name`, while an explicitly created `SlangEntryPoint` and a linked program do.
  Evidence: The first focused native run passed lifetime, cross-session, and hot-reload cases but the module-owner subcase found no `compute_main` entry-point layout. The module-owner test now uses a global `ParameterBlock` reflected from the module layout.

## Decision Log

- Decision: Work in the active checkout `C:/src/slangpy-side` and use the requested local RHI directory `C:/src/slang-rhi-side` for a fresh configuration.
  Rationale: The handoff's `C:/src/slangpy` path refers to a different checkout, while the active checkout contains the handoff branch and expected RHI revision. Using `SGL_LOCAL_RHI` verifies the exact neighboring implementation and avoids relying on an old configured binary.
  Date/Author: 2026-08-13 / Codex

- Decision: Keep component resolution internal to native SGL and do not expose it through Python bindings.
  Rationale: The component is lifetime and compiler-session metadata needed by SGL and slang-rhi, not a new public user-facing API.
  Date/Author: 2026-08-13 / Codex

- Decision: Require every SGL `ShaderObject` wrapper to receive a non-null component, including temporary roots returned by pass encoders.
  Rationale: Making the constructor invariant explicit prevents future ownerless wrappers and ensures `element_type_layout()` can always recover the correct component. Pass encoders can obtain it from the bound pipeline's program.
  Date/Author: 2026-08-13 / Codex

## Outcomes and Retrospective

The implementation retains the originating Slang component on every SGL shader object, uses the owner-aware RHI creation path, and rejects stale or cross-session bindings with actionable context before mutation. All construction sites now provide a component, and the only SlangPy call to `createShaderObjectFromTypeLayout` is the three-argument owner-aware overload. Focused native coverage passes 48 assertions, the Python packed-argument suite passes 15 cases across D3D12, Vulkan, and CUDA, and the complete native suite passes 250 test cases with 21,437 assertions. The exact post-format source was rebuilt and both the complete native suite and packed-argument suite passed again. Pre-commit passes all hooks.

## Context and Orientation

Slang is the shader compiler used by SlangPy. A `slang::ISession` is a compiler context; reflection and specialized types from different sessions cannot safely be mixed. A `slang::IComponentType` is a module, entry point, composite, or linked program that owns program-derived reflection and can report its session.

SlangPy's native SGL layer creates its own Slang sessions in `src/sgl/device/shader.cpp`. slang-rhi also owns an internal Slang session. `src/sgl/device/device.cpp`, in `Device::create_shader_object(const TypeLayoutReflection*)`, currently calls the two-argument `rhi::IDevice::createShaderObjectFromTypeLayout` overload. That legacy overload associates the layout with slang-rhi's internal session and does not retain the program component that owns the layout.

The functional API pack path creates standalone shader objects from reflection in `slangpy/builtin/value.py`, `slangpy/builtin/array.py`, `slangpy/builtin/tensor.py`, and `slangpy/builtin/descriptor.py`. `src/slangpy_ext/device/cursor_utils.h` later binds a packed shader object into a root object with `ShaderCursor::set_object`. A root object comes from a linked `ShaderProgram`, so its layout belongs to SlangPy's session. slang-rhi commit `0969630b` checks the two objects' sessions before changing the destination and returns `SLANG_E_INVALID_ARG` if they differ.

`src/sgl/device/reflection.h` defines `BaseReflectionObject::owner()`. Reflection wrappers preserve a strong reference to one of four known high-level owners: `SlangModule`, `SlangEntryPoint`, `ShaderProgram`, or `ShaderObject`. `SlangModule::slang_component_type()` and `SlangEntryPoint::slang_entry_point()` already expose their native components internally. `ShaderProgramData::linked_program` retains the linked component but `ShaderProgram` lacks a narrow raw accessor. `ShaderObject`, defined in `src/sgl/device/shader_object.h`, currently retains only the device and the slang-rhi shader object, so a layout obtained from `ShaderObject::element_type_layout()` cannot recover its original component.

Hot reload replaces the `slang::ISession` and swaps the current module, entry-point, and program data. Existing reflection wrappers are invalidated and ordinary caches are cleared. A user-held packed argument can nevertheless retain an old shader object. It must remain memory-safe, and binding it into a new root object must report that it is stale or cross-session. Repacking after reload creates an object from the new component and must succeed.

## Plan of Work

First, run a fresh Windows MSVC configure against `C:/src/slang-rhi-side`, followed by the debug build, outside the sandbox. Record whether the unmodified code compiles against the new virtual interface. This separates environmental or ABI problems from implementation changes.

Next, add an internal helper near SGL reflection ownership code that accepts a `BaseReflectionObject` or its `owner()` and returns the owning `slang::IComponentType`. It will handle exactly `SlangModule`, `SlangEntryPoint`, `ShaderProgram`, and `ShaderObject`, using explicit failure for null or unknown owners. Add a raw internal accessor on `ShaderProgram` for `ShaderProgramData::linked_program`. Do not compare a component session to `Device::slang_session()`, because a device supports multiple valid custom SlangPy sessions.

Extend `ShaderObject` construction to accept and strongly retain a `Slang::ComPtr<slang::IComponentType>`. `Device::create_root_shader_object` obtains it from the linked `ShaderProgram`. `Device::create_shader_object(const TypeLayoutReflection*)` resolves it from reflection, calls the three-argument slang-rhi overload, and gives it to the new wrapper. `ShaderObject::get_object()` and `get_entry_point()` inherit the parent's component. Consequently, `ShaderObject::element_type_layout()` can continue using the SGL shader object as the reflection owner while the internal helper recovers the original component from that wrapper.

Before `ShaderObject::set_object()` invokes slang-rhi, compare the stored components' sessions when both objects are non-null. Throw an SGL error that explains the objects belong to different Slang sessions and that a packed value held across hot reload must be packed again. This produces useful context and proves the destination cannot be mutated. Keep the slang-rhi check as a defense-in-depth invariant.

Add focused native tests under `tests/sgl/device/`, reusing the GPU test context and custom-session patterns already present in `test_cursors.cpp` and `test_hot_reload.cpp`. Tests will create layouts owned by a module, entry point, linked program, and existing shader object; verify same-session binding; release original high-level wrappers where practical; and verify cross-session rejection. A hot-reload test will retain an old standalone object, recreate sessions, verify the old object fails clearly, then create and bind a new object successfully. Extend `slangpy/tests/slangpy_tests/test_packed_arg.py` only where Python-level coverage is needed beyond the native cases, keeping all Python arguments type-annotated.

Finally, rebuild before every test phase. Run relevant native tests and `slangpy/tests/slangpy_tests/test_packed_arg.py` outside the sandbox. Run `pre-commit run --all-files`; if it changes files, inspect the edits, rerun the build and affected tests as needed, and run pre-commit again until clean.

## Concrete Steps

From `C:\src\slangpy-side`, configure and build outside the sandbox:

    cmake --preset windows-msvc --fresh -DSGL_LOCAL_RHI=ON -DSGL_LOCAL_RHI_DIR=C:/src/slang-rhi-side
    cmake --build --preset windows-msvc-debug

After implementation, repeat the build, then run focused tests outside the sandbox:

    python tools/ci.py unit-test-cpp
    pytest slangpy/tests/slangpy_tests/test_packed_arg.py -v

Run any narrower native test filter supported by `tools/ci.py` during iteration, but the full native unit-test command is required before completion. Finish with:

    pre-commit run --all-files

## Validation and Acceptance

The fresh baseline and final debug builds must complete without ABI or linker errors. The native tests must show that module-, entry-point-, linked-program-, and shader-object-derived layouts retain the correct component. A standalone shader object from a custom session must bind into a root object from a program in that same session. Releasing module or program wrappers must not invalidate a standalone shader object whose retained component is still needed.

Binding objects from different sessions must throw before changing the destination, and the message must contain enough context to identify a cross-session or stale-after-reload shader object. After hot reload, an old packed or standalone object must fail this way, while a newly packed or created object must bind and dispatch successfully. The complete existing packed-argument Python test file must pass for all available backends and optional Torch paths on the machine.

Acceptance also requires no program-derived `TypeLayoutReflection` path in SlangPy to call the legacy ownerless overload, successful native and Python focused tests after a clean rebuild, and a clean pre-commit run.

## Idempotence and Recovery

The configure and build commands are safe to repeat. `--fresh` intentionally regenerates CMake state so no binary compiled against the previous `rhi::IDevice` virtual interface remains. Tests create their data in test-managed temporary directories. If a build or test fails, preserve the output in this plan, fix the smallest relevant issue, rebuild, and rerun the failed command before expanding validation.

Do not reset or overwrite unrelated worktree changes. Do not update or edit the neighboring `C:/src/slang-rhi-side` repository as part of this plan; it is an input fixed at commit `38f18c94`.

## Artifacts and Notes

The investigation handoff is `slangpy-cross-session-handoff.md`. The current branch is `dev/skallweit/fix-cross-slang-session`. Before implementation, `Device::create_shader_object` contains this unsafe call:

    createShaderObjectFromTypeLayout(type_layout->get_slang_type_layout(), ...)

The unmodified ABI baseline completed with:

    cmake --preset windows-msvc --fresh -DSGL_LOCAL_RHI=ON -DSGL_LOCAL_RHI_DIR=C:/src/slang-rhi-side
    cmake --build --preset windows-msvc-debug
    [352/353] Generating C:/src/slangpy-side/slangpy/__init__.pyi
    Exit code: 0

After implementation, focused and full validation produced:

    build/windows-msvc/Debug/sgl_tests.exe -tc=shader_object*
    [doctest] test cases: 4 | 4 passed | 0 failed
    [doctest] assertions: 48 | 48 passed | 0 failed

    pytest slangpy/tests/slangpy_tests/test_packed_arg.py -v
    15 passed in 22.52s

    python tools/ci.py unit-test-cpp
    [doctest] test cases: 250 | 250 passed | 0 failed | 8 skipped
    [doctest] assertions: 21437 | 21437 passed | 0 failed

After slang-rhi hardened the owner-aware interfaces in commit `38f18c94`, SlangPy advanced its submodule gitlink, changed `ShaderProgram::slang_component_type()` to query the authoritative `IShaderProgram::getSlangProgram()`, and repeated a fresh configure and rebuild. Validation against the hardened interface produced:

    build/windows-msvc/Debug/sgl_tests.exe -tc=shader_object*
    [doctest] test cases: 4 | 4 passed | 0 failed
    [doctest] assertions: 48 | 48 passed | 0 failed

    pytest slangpy/tests/slangpy_tests/test_packed_arg.py -v
    15 passed in 26.92s

    python tools/ci.py unit-test-cpp
    [doctest] test cases: 250 | 250 passed | 0 failed | 8 skipped
    [doctest] assertions: 21491 | 21491 passed | 0 failed

    pre-commit run --all-files
    all hooks passed

The required slang-rhi interface is:

    createShaderObjectFromTypeLayout(slang::IComponentType* slangOwner,
                                     slang::TypeLayoutReflection* typeLayout,
                                     rhi::IShaderObject** outObject)

## Interfaces and Dependencies

The implementation uses existing Slang COM ownership through `Slang::ComPtr<slang::IComponentType>` and does not introduce dependencies. `ShaderProgram` provides a narrow native accessor returning the exact linked component reported by `IShaderProgram::getSlangProgram()`. `ShaderObject` must retain an owning component and provide an internal raw accessor so reflection-owner resolution and session validation can use it. The SGL reflection helper must return the component for each of the four known owner types and throw an explicit SGL error for unsupported owners.

Revision note: Initial plan created from `slangpy-cross-session-handoff.md` and direct inspection of the active SlangPy and slang-rhi checkouts. It records the existing submodule discrepancy and the decision to baseline with `SGL_LOCAL_RHI`. Updated through implementation, pre-commit formatting, final acceptance searches, successful post-format native and Python validation, and the follow-up migration to slang-rhi commit `38f18c94`.
