# Unified Cursor Writer Registry For Python-Keyed Values

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy already has a native cursor-writer registry that lets C++ value types describe how they are written into a `ShaderCursor` or `BufferElementCursor`. The next step is to let pure Python object types participate in that same cursor-writing system without creating a second registry in `src/slangpy_ext/` and without making the pure native `src/sgl/` code include Python headers.

After this change, a value type can be registered from either native C++ code or Python code, and the rest of the cursor-writing system can use one lookup result and one erased writer callback shape. A user should be able to register a Python class with a Python writer callback, pass an instance to `ShaderCursor.write(...)` or a buffer cursor write path, and observe the same behavior as if the value had been unpacked into a dictionary. Existing native values such as `Buffer`, `Texture`, `Sampler`, `ShaderObject`, and `Tensor` must continue to work through the same registry.

This plan is intentionally focused on cursor writing and value signatures. Full functional-API marshall creation for Python-keyed types, such as making all NumPy array binding native-side, is a later extension and is not required for the first working slice.

## Progress

- [x] (2026-05-29 11:21Z) Wrote this ExecPlan from the current design discussion and repository inspection.
- [x] (2026-05-29) Replaced the native registry key with `CursorWriterTypeKeyKind`, an opaque `type_key`, `debug_name`, and a mutation generation counter.
- [x] (2026-05-29) Updated the extension resolver so native nanobind-backed objects and pure Python objects both resolve through the same core registry.
- [x] (2026-05-29) Added Python-backed cursor-writer registration in `src/slangpy_ext/` without adding a second registry.
- [x] (2026-05-29) Added cleanup for Python-backed registry entries during `sgl::static_shutdown()`.
- [x] (2026-05-29) Added C++ and Python tests that demonstrate both native and pure Python cursor writers use the unified registry.
- [x] (2026-05-29) Built `slangpy_ext` and `sgl_tests`, ran focused C++/Python tests, and ran `pre-commit run --all-files`.

## Surprises and Discoveries

- Observation: The current core registry is already process-wide and native-owned in `src/sgl/device/cursor_utils.cpp`, but its key is `const std::type_info*`.
  Evidence: `cursor_writer_type_info_registry()` returns a static `std::vector<CursorWriterTypeInfo>`, and `find_cursor_writer_type_info(const std::type_info& type)` scans `info.type`.

- Observation: Core SGL already has a central shutdown function that is called from the Python extension module free hook.
  Evidence: `src/slangpy_ext/slangpy_ext.cpp` assigns `nanobind_slangpy_ext_module.m_free = [](void*) { sgl::static_shutdown(); };`.

- Observation: Python-backed callbacks may hold `nb::object` references inside `std::function` lambdas, so they must be removed before Python finalization.
  Evidence: The design stores Python callbacks as captured nanobind objects in registry entries, and captured `nb::object` destructors release Python references.

## Decision Log

- Decision: Use one core registry, not one native registry plus one extension registry.
  Rationale: Cursor writing has the same erased operation after lookup: a function writes a value pointer into a cursor. Splitting registries would duplicate lookup and dispatch logic while producing the same final operation.
  Date/Author: 2026-05-29 / Codex

- Decision: Core SGL will store an opaque `const void* type_key` paired with a native-defined enum `CursorWriterTypeKeyKind`.
  Rationale: Core must not include Python headers or mention `PyTypeObject`, but it can safely store an opaque pointer and an enum saying whether the pointer represents `std::type_info` or a Python type. This keeps the registry unified while keeping Python-specific interpretation inside `src/slangpy_ext/`.
  Date/Author: 2026-05-29 / Codex

- Decision: Use an enum rather than an open-ended domain pointer.
  Rationale: The key kinds are expected to be closed for this repository: native C++ type info and Python type. An enum is clearer and easier to validate than a domain pointer.
  Date/Author: 2026-05-29 / Codex

- Decision: Python writer callbacks are invoked without acquiring the GIL inside each callback.
  Rationale: The supported call paths are Python-entered and single-threaded, so the GIL is already held when cursor writing reaches these callbacks. If a future non-Python or background-thread entry point invokes Python callbacks, that boundary should acquire the GIL rather than adding overhead to every writer callback now.
  Date/Author: 2026-05-29 / Codex

- Decision: Python-backed entries are cleared during `sgl::static_shutdown()`.
  Rationale: `static_shutdown()` is already called by the SlangPy module free hook. Clearing Python-backed entries there destroys captured `nb::object` callback references while the Python runtime is still alive.
  Date/Author: 2026-05-29 / Codex

- Decision: Marshall creation for Python-keyed values is out of scope for the first implementation.
  Rationale: The immediate problem is unifying cursor-writer registration and dispatch. The same registry can later grow marshall factory metadata, but this slice should not combine that with the key/lifetime refactor.
  Date/Author: 2026-05-29 / Codex

## Outcomes and Retrospective

Implementation is complete. Core SGL now owns a single cursor-writer registry that supports native C++ keys and opaque Python type keys while keeping all Python-specific interpretation in `src/slangpy_ext/`. Native values still use the nanobind extraction path, pure Python values pass their original `PyObject*` through the erased writer callback, and Python-backed entries are removed from the registry during `sgl::static_shutdown()`.

Focused validation so far:

    cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
    build\windows-msvc\Debug\sgl_tests.exe --test-case="*cursor_writer*"
    build\windows-msvc\Debug\sgl_tests.exe --test-case="*write_to_cursor*"
    python -m pytest slangpy/tests/slangpy_tests/test_write_to_cursor_marshall.py -v -p no:cacheprovider
    python -m pytest slangpy/tests/slangpy_tests/test_python_cursor_writer.py -v -p no:cacheprovider

One early run of `sgl_tests.exe --test-case="*write_to_cursor*"` failed while it was running in parallel with another `sgl_tests.exe` process. Rerunning the same command by itself passed.

## Context and Orientation

A cursor is an object that points at a value inside shader-visible data. `ShaderCursor` writes data into shader objects and root parameters. `BufferElementCursor` writes data into an element of a CPU-side buffer cursor. Both cursor types expose `set(...)` and `write(...)` behavior that eventually writes scalars, vectors, matrices, structs, arrays, resources, or registered custom objects.

The current native cursor-writer registry lives in `src/sgl/device/cursor_utils.h` and `src/sgl/device/cursor_utils.cpp`. The descriptor is named `CursorWriterTypeInfo`. It currently stores `const std::type_info* type`, erased writer functions for shader and buffer cursors, optional Slang type metadata, a signature writer, and imports. Built-in native registrations are added by `cursor_utils::register_cursor_writers()` during `sgl::static_init()` in `src/sgl/sgl.cpp`.

The current Python extension lookup lives in `src/slangpy_ext/device/cursor_writer.h` and `src/slangpy_ext/device/cursor_writer.cpp`. It takes a `nanobind` Python object, asks nanobind whether the object is backed by one of the registered C++ types, and returns a `NativeCursorWriterValue` containing the registry entry and the extracted C++ pointer. The hot-path lookup caches by `PyTypeObject*`, but only for native nanobind-backed types.

The current Python cursor write tables live mainly in `src/slangpy_ext/device/cursor_utils.h`. `WriteConverterTable<CursorType>::write_registered_native_object(...)` calls `sgl::slangpy::find_native_cursor_writer(...)`. The shader cursor binding in `src/slangpy_ext/device/shader_cursor.cpp` uses `WriteConverterTable<ShaderCursor>` to implement `write_shader_cursor(...)`.

`std::type_info` is the C++ runtime type descriptor used by `typeid(T)`. Nanobind uses it to test and extract C++ objects from Python wrappers. `PyTypeObject` is the CPython runtime type descriptor for Python classes, but this plan never exposes that type name to core SGL. Core stores Python type pointers only as opaque `const void*` values and lets the extension interpret them.

`SignatureBuffer` is the native cache-signature builder in `src/sgl/core/signature_buffer.h`. A cursor-writer entry must be able to add a stable signature for a value so SlangPy can cache generated call data. Native values do this through C++ callbacks today; Python-backed entries may use a static signature string or call a Python signature callback.

## Plan of Work

Milestone 1 changes the core registry key without adding Python support. In `src/sgl/device/cursor_utils.h`, introduce:

    enum class CursorWriterTypeKeyKind {
        native_type_info,
        python_type,
    };

Replace `CursorWriterTypeInfo::type` with:

    CursorWriterTypeKeyKind key_kind{CursorWriterTypeKeyKind::native_type_info};
    const void* type_key{nullptr};
    std::string debug_name;

Keep the existing erased writer fields, `slang_type_name`, `write_signature`, and `imports`. Native registration through `register_cursor_writer<T>()` should set `key_kind` to `native_type_info`, `type_key` to `&typeid(T)`, and `debug_name` to `detail::type_name<T>()` or `typeid(T).name()`.

In `src/sgl/device/cursor_utils.cpp`, update `register_cursor_writer_type(...)` to validate `type_key`, use `debug_name` in error messages, and reject duplicates by the pair `(key_kind, type_key)`. Keep `find_cursor_writer_type_info(const std::type_info& type)`, but make it search only entries whose `key_kind` is `native_type_info` and compare `*static_cast<const std::type_info*>(info.type_key)` against `type`. Add:

    SGL_API void unregister_cursor_writer_types(CursorWriterTypeKeyKind key_kind);
    SGL_API uint64_t cursor_writer_registry_generation();

`unregister_cursor_writer_types(...)` removes all entries of that key kind. `cursor_writer_registry_generation()` returns a monotonically increasing counter that changes whenever entries are added or removed. The generation lets extension caches drop stale pointers after vector reallocation or cleanup.

Milestone 2 updates the existing extension resolver to understand the new native key representation. In `src/slangpy_ext/device/cursor_writer.cpp`, replace all `info.type` uses with native-only helper code:

    const std::type_info& native_type_info(const CursorWriterTypeInfo& info)

This helper asserts that `info.key_kind == CursorWriterTypeKeyKind::native_type_info` and casts `info.type_key` back to `const std::type_info*`. `native_cursor_writer_pointer(...)` continues to take `const std::type_info&` because nanobind needs that exact type descriptor.

The uncached native lookup should skip non-native entries while doing nanobind `nb_type_isinstance(...)` scans. The exact-type fast path should keep using `cursor_utils::find_cursor_writer_type_info(...)`. The cache should store the registry generation, not only registry size. On each lookup, if `cursor_utils::cursor_writer_registry_generation()` has changed, clear the cache before reading a cached pointer.

Milestone 3 adds Python-keyed registration in the extension while still writing into the same core registry. In `src/slangpy_ext/device/cursor_writer.h`, add an extension-only registration API. The exact shape can be adjusted to match nanobind conventions, but it must accept a Python type and optional callbacks:

    void register_python_cursor_writer_type(
        nb::type_object python_type,
        nb::object write_shader_cursor,
        nb::object write_buffer_cursor,
        nb::object write_signature,
        std::string slang_type_name,
        std::vector<std::string> imports
    );

If a callback is `None`, leave the corresponding erased function empty. At least one cursor writer and one signature writer must exist, matching the core validation contract. A static signature string may be converted into a C++ lambda before registration so callers do not need to provide a dynamic Python signature callback.

The Python registration function builds a `CursorWriterTypeInfo` with `key_kind = CursorWriterTypeKeyKind::python_type`, `type_key = python_type.ptr()`, and a readable `debug_name` such as `module.qualname`. The writer lambdas capture `nb::object` callbacks. They receive `const void* value`, interpret it as the original `PyObject*`, create a borrowed `nb::object`, and call the Python callback:

    callback(cursor, nb::borrow<nb::object>(reinterpret_cast<PyObject*>(const_cast<void*>(value))));

Do not acquire the GIL inside these lambdas. Add a comment stating that Python cursor-writer callbacks are invoked only from Python-entered paths where the GIL is already held.

Milestone 4 unifies lookup results for native and Python-keyed entries. Rename the extension result type from `NativeCursorWriterValue` to a neutral name such as `CursorWriterValue`, and rename `find_native_cursor_writer(...)` to `find_cursor_writer(...)`. If this churn is too broad, keep the old names as wrappers during the transition, but new code should use the neutral names.

The unified resolver in `src/slangpy_ext/device/cursor_writer.cpp` should handle both key kinds. For native entries, it keeps the current nanobind behavior and returns the extracted C++ pointer. For Python entries, it should first compare exact Python type pointer:

    Py_TYPE(obj.ptr()) == reinterpret_cast<PyTypeObject*>(const_cast<void*>(info.type_key))

If exact comparison fails, it may use `PyObject_IsInstance(...)` or an explicit MRO scan so registrations on Python base classes also work. Return the original Python object pointer as `value` for Python entries. Cache both hits and misses by `PyTypeObject*` and registry generation.

Update `src/slangpy_ext/device/cursor_utils.h` so `WriteConverterTable<CursorType>::write_registered_native_object(...)` becomes `write_registered_object(...)` or similarly neutral and calls the unified resolver. Its call site should not know whether the value pointer is a C++ object or a Python object; it only invokes the erased writer stored in the registry entry.

Milestone 5 wires cleanup into shutdown. In `src/sgl/sgl.cpp`, call:

    cursor_utils::unregister_cursor_writer_types(CursorWriterTypeKeyKind::python_type);

near the start of `sgl::static_shutdown()` after the reference-count early return and before waiting for tasks or tearing down other subsystems. This preserves process-lifetime native registrations while destroying Python-backed callbacks before interpreter finalization. If later testing shows `static_shutdown()` can run after Python finalization in some embedding scenario, add an explicit call from the existing `atexit` hook in `src/slangpy_ext/slangpy_ext.cpp` while the GIL is held, but keep the core cleanup API as the owner of registry removal.

Milestone 6 exposes a Python-level registration function only if needed for tests and user-facing use. The existing `slangpy/bindings/cursor.py::register_cursor_writer_marshal(...)` currently registers functional fallback metadata in Python dictionaries; it does not register runtime cursor writer callbacks into the core registry. Add a separate public helper only if the implementation needs pure Python users to register cursor writers directly. A possible name is `register_cursor_writer_type(...)`, but avoid overloading it with the existing functional marshall helper unless both operations are intentionally combined. The helper should call the native extension function from Milestone 3.

Milestone 7 adds tests and keeps existing behavior stable. Native C++ tests under `tests/sgl/device/test_cursors.cpp` should cover key-kind duplicate rejection, native lookup after the opaque key migration, and unregistering only Python-kind entries. Python tests under `slangpy/tests/slangpy_tests/` should register a small Python class with a cursor writer callback, write it into a struct-shaped cursor, and verify the resulting data matches the callback's fields. Existing tests for `WriteToCursorMarshall`, native resource cursor writing, tensors, and descriptor handles should still pass.

## Concrete Steps

Work from the repository root:

    cd C:\sw\slangpy

Before editing, inspect the current relevant code:

    rg -n "CursorWriterTypeInfo|find_native_cursor_writer|register_cursor_writer_type|static_shutdown" src/sgl src/slangpy_ext slangpy tests

Implement Milestone 1 in `src/sgl/device/cursor_utils.h` and `src/sgl/device/cursor_utils.cpp`. Keep the public API source-compatible where possible by preserving `find_cursor_writer_type_info(const std::type_info&)`.

Implement Milestones 2 through 4 in `src/slangpy_ext/device/cursor_writer.h`, `src/slangpy_ext/device/cursor_writer.cpp`, `src/slangpy_ext/device/cursor_utils.h`, and any direct callers found by `rg "find_native_cursor_writer|NativeCursorWriterValue|get_native_cursor_writer_type_info"`.

Implement Milestone 5 in `src/sgl/sgl.cpp`. If the extension needs a test-only cleanup function, put it in `src/slangpy_ext/device/cursor_writer.cpp` and expose it only as a private `_...` function through the nanobind module.

Add or update tests. Prefer focused test names that state the behavior:

    register_cursor_writer_rejects_duplicate_key_kind_and_key
    unregister_cursor_writer_types_removes_python_keyed_entries
    test_python_cursor_writer_callback_writes_struct

Build before running tests, as required by this repository:

    cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests

Run focused C++ tests. Adjust the exact doctest filters to match the final test names:

    build\windows-msvc\Debug\sgl_tests.exe --test-case="*cursor_writer*"
    build\windows-msvc\Debug\sgl_tests.exe --test-case="*write_to_cursor*"

Run focused Python tests:

    python -m pytest slangpy/tests/slangpy_tests/test_write_to_cursor_marshall.py -v -p no:cacheprovider
    python -m pytest slangpy/tests/slangpy_tests/test_python_cursor_writer.py -v -p no:cacheprovider

Run pre-commit after implementation:

    pre-commit run --all-files

If pre-commit modifies files, inspect the diff, then rerun `pre-commit run --all-files` until it passes.

## Validation and Acceptance

The implementation is accepted when all of these behaviors are true.

Existing native cursor writers still work. Passing a native `Buffer`, `Texture`, `Sampler`, `ShaderObject`, `AccelerationStructure`, or `Tensor` to the existing cursor-writing paths succeeds exactly as before. The focused C++ cursor tests and existing Python write-to-cursor marshall tests pass.

A pure Python type can be registered without adding another registry. The registration stores a `CursorWriterTypeInfo` in the core `cursor_utils` registry with `key_kind == CursorWriterTypeKeyKind::python_type` and an opaque Python type pointer. The core registry does not include Python headers and does not mention `PyTypeObject`.

The unified resolver returns the same result shape for native and Python values. For native values, `value` is the extracted C++ pointer. For Python values, `value` is the original Python object pointer. The writer call site invokes `info.write_shader_cursor(cursor, value)` or `info.write_buffer_cursor(cursor, value)` without branching on native versus Python.

Python-backed callbacks are destroyed during shutdown. A test or debug assertion should show that calling `cursor_utils::unregister_cursor_writer_types(CursorWriterTypeKeyKind::python_type)` removes Python-keyed entries and advances the registry generation. Static shutdown calls this cleanup path. Native entries are not removed by this cleanup.

The extension lookup cache cannot return stale pointers after registration or cleanup. Any lookup cache in `src/slangpy_ext/device/cursor_writer.cpp` compares a stored generation against `cursor_utils::cursor_writer_registry_generation()` and clears itself when the generation changes.

Expected successful command summary:

    cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
    # build completes with no errors

    build\windows-msvc\Debug\sgl_tests.exe --test-case="*cursor_writer*"
    # all selected doctest cases pass

    python -m pytest slangpy/tests/slangpy_tests/test_python_cursor_writer.py -v -p no:cacheprovider
    # the pure Python cursor writer test passes

    pre-commit run --all-files
    # all hooks pass

## Idempotence and Recovery

The implementation should be safe to apply incrementally. The opaque key migration is mechanical and can be verified before adding Python entries. If a later milestone fails, keep Milestone 1 compiling and temporarily leave Python registration unexposed.

`register_cursor_writer_type(...)` must continue to reject duplicate registrations for the same `(key_kind, type_key)` pair. Re-running `sgl::static_init()` without a matching process restart should not silently add duplicate native entries. This plan does not require changing existing native-entry lifetime, but it must not make duplicate behavior worse.

`unregister_cursor_writer_types(CursorWriterTypeKeyKind::python_type)` must be safe to call more than once. Calling it when no Python entries exist should leave the registry unchanged except that it may advance the generation only if entries were actually removed. Prefer advancing the generation only on mutation so caches are not cleared unnecessarily.

If a Python callback raises, preserve the existing cursor-write error wrapping style in `WriteConverterTable`: include the Python value type in the error message and keep the cursor path stack context. Do not swallow the exception.

If a test creates Python cursor-writer registrations, make it unregister them in a `finally` block or through a test fixture so later tests do not inherit registrations.

## Artifacts and Notes

The central native type shape should end up close to this. The exact field order may change to match local style:

    enum class CursorWriterTypeKeyKind {
        native_type_info,
        python_type,
    };

    struct CursorWriterTypeInfo {
        CursorWriterTypeKeyKind key_kind{CursorWriterTypeKeyKind::native_type_info};
        const void* type_key{nullptr};
        std::string debug_name;

        ShaderCursorObjectWriteFunc write_shader_cursor;
        BufferElementCursorObjectWriteFunc write_buffer_cursor;

        std::string slang_type_name;
        std::function<void(SignatureBuffer&, const void*)> write_signature;
        std::vector<std::string> imports;
    };

Native registration should remain concise:

    template<typename T>
    void register_cursor_writer()
        requires(CanRegisterCursorWriter<T>)
    {
        CursorWriterTypeInfo info;
        info.key_kind = CursorWriterTypeKeyKind::native_type_info;
        info.type_key = &typeid(T);
        info.debug_name = std::string(detail::type_name<T>());
        ...
        register_cursor_writer_type(std::move(info));
    }

Python registration should be visibly the same registry operation:

    CursorWriterTypeInfo info;
    info.key_kind = CursorWriterTypeKeyKind::python_type;
    info.type_key = python_type.ptr();
    info.debug_name = python_debug_name(python_type);
    info.write_shader_cursor = [callback = nb::object(write_shader_cursor)](ShaderCursor& cursor, const void* value)
    {
        callback(cursor, nb::borrow<nb::object>(reinterpret_cast<PyObject*>(const_cast<void*>(value))));
        return true;
    };
    cursor_utils::register_cursor_writer_type(std::move(info));

Do not place this Python lambda in core SGL. The lambda is constructed in `src/slangpy_ext/`, where `nanobind.h` and Python types are already available. Core SGL only stores and later destroys the erased `std::function`.

## Interfaces and Dependencies

The core native API in `src/sgl/device/cursor_utils.h` must provide:

    enum class CursorWriterTypeKeyKind;
    struct CursorWriterTypeInfo;
    void register_cursor_writer_type(CursorWriterTypeInfo info);
    std::span<const CursorWriterTypeInfo> cursor_writer_type_infos();
    const CursorWriterTypeInfo* find_cursor_writer_type_info(const std::type_info& type);
    void unregister_cursor_writer_types(CursorWriterTypeKeyKind key_kind);
    uint64_t cursor_writer_registry_generation();
    template<typename T> void register_cursor_writer();

Core SGL must not include `nanobind.h`, `Python.h`, or mention `PyTypeObject`.

The extension API in `src/slangpy_ext/device/cursor_writer.h` should provide neutral lookup names:

    struct CursorWriterValue {
        const cursor_utils::CursorWriterTypeInfo* info;
        const void* value;
    };

    std::optional<CursorWriterValue> find_cursor_writer(nb::handle obj);
    nb::object get_cursor_writer_type_info(nb::handle obj);
    void register_python_cursor_writer_type(...);

If existing Python or C++ code still expects `_get_native_cursor_writer_type_info` or `find_native_cursor_writer`, keep compatibility wrappers temporarily, but route them through the new neutral functions.

The Python-facing helper, if added, belongs near `slangpy/bindings/cursor.py` because that file already owns `WriteToCursorMarshall` registration concepts. Keep the naming clear so users can tell whether they are registering runtime cursor callbacks, functional marshall metadata, or both.

## Revision Notes

2026-05-29: Initial ExecPlan created. It captures the agreed design: one core registry, an enum key kind plus opaque key, Python callbacks stored as erased lambdas constructed in the extension, no GIL acquisition inside hot callbacks, and cleanup of Python-backed entries during `sgl::static_shutdown()`.
