# Cursor Writer Binding Plan

## Goal

Replace the current split system with one native registration path for cursor-writable value types, with optional simple functional-API fallback metadata.

The desired user model is:

1. A native value type implements `write_to_cursor` for both `ShaderCursor` and `BufferElementCursor`.
2. If the type wants the simple `WriteToCursorMarshall` functional fallback, it provides its static Slang type name directly on the class.
3. Functional fallback types may provide a static or dynamic signature directly on the class; if they do not, the signature defaults to the C++ class name.
4. The project calls one registration function:

```cpp
sgl::cursor_utils::register_cursor_writer<T>();
```

That single call always registers direct cursor writing. If class-owned Slang metadata is present, it also registers functional-API fallback marshalling through `WriteToCursorMarshall`.

The new policy should be:

- `get_this` is the legacy wrapper path.
- `write_to_cursor` plus `register_cursor_writer<T>()` is the new bindable-value path.
- A single type must not use both paths.
- Native registered cursor writers should be identified from the native registry, not from Python-side compatibility predicates.
- The old cursor-writer caches and `has_registered_type_or_signature` compatibility path should be removed as part of this migration, not preserved as fallback behavior.
- The registered class is the only metadata surface. Do not introduce a separate traits structure.
- `slang_type_name` is required only for the simple functional fallback path; when present it is always static and class-owned.
- Signature metadata is optional for functional fallback types, class-owned, and may be static or value-aware.
- Imports are optional for functional fallback types, static, class-owned, and copied once during registration.
- Public `register_cursor_writer<T>()` is all-or-nothing for direct cursor writing: the type must provide both cursor-write paths. Functional metadata is an optional bundle so resource types can register direct cursor writers while keeping bespoke functional marshalls.

## User Class Contract

Minimal static-signature form for simple functional fallback:

```cpp
struct MyHandle {
    static constexpr std::string_view slang_type_name = "MyHandle";
    static constexpr std::string_view slangpy_signature = "MyHandle";

    uint32_t id = 0;
    uint32_t flags = 0;

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["id"] = id;
        cursor["flags"] = flags;
    }

    static std::vector<std::string_view> slangpy_imports()
    {
        return {"my_project/my_handle.slang"};
    }
};
```

Dynamic-signature form:

```cpp
struct TypedHandle {
    static constexpr std::string_view slang_type_name = "TypedHandle";

    uint32_t id = 0;
    uint32_t kind = 0;

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["id"] = id;
        cursor["kind"] = kind;
    }

    void write_slangpy_signature(SignatureBuffer& sig) const
    {
        sig.add("TypedHandle:");
        sig.add(kind);
    }
};
```

Writer-only form for types that have a bespoke functional API marshall:

```cpp
struct MyResourceHandle {
    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["id"] = id;
    }
};
```

The `write_to_cursor` implementation may be a templated cursor overload or two explicit overloads, but it must compile for both `ShaderCursor&` and `BufferElementCursor&`. If functional metadata is present and neither `slangpy_signature` nor `write_slangpy_signature(...)` exists, registration should use the C++ class name as the default signature. This signature is only a cache key; it is not used as a visible Slang type name.

## Correction Work Completed

The implementation has been corrected to this contract:

- Removed `CursorWriterTraits<T>` and all trait-specific detection.
- Removed value-aware `slang_type_name()` support; `T::slang_type_name` is the only Slang type name source.
- `register_cursor_writer<T>()` now requires `write_to_cursor(...)` support for both `ShaderCursor&` and `BufferElementCursor&`.
- `T::slang_type_name` is optional and enables simple functional fallback metadata when present.
- `write_signature(SignatureBuffer&, const void*)` remains in the registry so static and dynamic signatures share one call path.
- Signature sources are detected in this order:
  1. `value.write_slangpy_signature(SignatureBuffer&) const` for dynamic signatures.
  2. `T::write_slangpy_signature(SignatureBuffer&)` for static function signatures.
  3. `T::slangpy_signature` for static string signatures.
  4. Default C++ class-name signature.
- The registry stores `slang_type_name` as copied registration data for functional fallback entries, not as a `const void*` callback.
- `slangpy_imports()` is static and class-owned only; it is called once at registration and its strings are copied for functional fallback entries.
- Tests now put `slang_type_name`, signatures, and imports on the registered class itself.
- Tests cover missing `T::slang_type_name` as writer-only registration and one-cursor-only writers as compile-time rejections via `CanRegisterCursorWriter`.
- Removed the legacy low-level writer wrapper APIs; the native registry now accepts only complete `register_cursor_writer<T>()` entries.

## Progress So Far

Latest progress update:

- The public cursor-writer registration contract has been tightened to require both cursor paths while keeping functional metadata optional. `CanRegisterCursorWriter<T>` now requires `write_to_cursor(ShaderCursor&)` and `write_to_cursor(BufferElementCursor&)`; `CanRegisterFunctionalCursorWriter<T>` additionally requires `T::slang_type_name`.
- One-cursor-only types remain useful for cursor `set()` contract tests, but they are compile-time rejected for public `register_cursor_writer<T>()` registration.
- Native tests now cover writer-only registrations, default class-name signatures, static string signatures, static function signatures, dynamic value-aware signatures, copied imports, duplicate registration rejection, partial functional metadata rejection, and one-cursor-only compile-time rejection under the stricter contract.
- Final verification for this checkpoint passed after clang-format updates and a full `pre-commit run --all-files`.

Completed in the current implementation slice:

- `SignatureBuffer` moved to `src/sgl/core/signature_buffer.h` in namespace `sgl`.
- `SignatureBuilder` remains the Python-facing wrapper and now includes the native `SignatureBuffer` header.
- `src/sgl/CMakeLists.txt` includes the new header.
- Added native C++ coverage in `tests/sgl/core/test_signature_buffer.cpp`.
- `HasWriteToCursor<T, TCursor>` now checks the actual `obj.write_to_cursor(cursor)` call expression.
- `ShaderCursor::set()` now checks `HasWriteToCursor<T, ShaderCursor>`.
- `ShaderCursor::set()` and `operator=` are non-const, so `write_to_cursor(*this)` writes through the original cursor without copying it.
- Added native C++ coverage in `tests/sgl/device/test_cursors.cpp` for cursor-specific `set()` behavior, both-overload registration, and negative concept checks.
- Introduced a combined native `CursorWriterTypeInfo` registry in `src/sgl/device/cursor_utils.h/.cpp`.
- Added `cursor_utils::register_cursor_writer<T>()`.
- Removed legacy low-level cursor-writer registration functions; the combined registry stores only entries that provide both cursor writers, with optional functional metadata.
- `src/slangpy_ext/device/cursor_utils.h` direct cursor writes now consult the combined native registry and no longer keep `WriteConverterTable::m_native_object_writer_cache`.
- Static metadata and imports are captured into the native descriptor at registration time from class-owned metadata only when the type opts into functional fallback.
- Added native C++ coverage for registry lookup, duplicate rejection, one-cursor registration rejection, and static metadata/signature/imports.
- Renamed the Python marshall API to `WriteToCursorMarshall`, `WriteToCursorMarshallInfo`, and `register_write_to_cursor_type`.
- `WriteToCursorMarshall` now derives from `NativeValueMarshall`, so dispatch uses the native cursor-write fast path rather than Python `create_calldata()`.
- `slangpy/bindings/typeregistry.py` now falls back to native cursor-writer metadata when Python type registration has no hit.
- `NativeCallDataCache::get_value_signature()` now uses the native cursor-writer registry only for entries with functional metadata; writer-only entries fall through to bespoke native/Python signature paths.
- Removed the old `has_registered_type_or_signature` Python predicate, native callback, and native predicate cache.
- `get_this` remains the legacy wrapper path for objects not owned by functional cursor-writer metadata; registered functional cursor writers with `get_this` fail with a clear conflict error.

Verification run for this slice:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
build\windows-msvc\Debug\sgl_tests.exe --test-suite=signature_buffer,cursors
python -m pytest slangpy/tests/slangpy_tests/test_write_to_cursor_marshall.py -v -p no:cacheprovider
python -m pytest slangpy/tests/slangpy_tests/test_type_resolution.py -v -p no:cacheprovider
python -m pytest slangpy/tests/slangpy_tests/test_instances.py::test_packed_vs_unpacked_cache -v -p no:cacheprovider --device-types d3d12
pre-commit run --files .plans/bindings.md src/sgl/device/cursor_utils.cpp src/sgl/device/cursor_utils.h src/slangpy_ext/utils/slangpy.cpp tests/sgl/device/test_cursors.cpp
pre-commit run --all-files
```

## Phase 1: Move SignatureBuffer To Native

Status: complete.

Move `SignatureBuffer` out of `src/slangpy_ext/utils/slangpy.h` and into a pure native header:

```text
src/sgl/core/signature_buffer.h
```

Use namespace `sgl` for the native type:

```cpp
namespace sgl {
class SignatureBuffer;
}
```

Keep `SignatureBuilder` in `sgl::slangpy` as the Python-facing object wrapper around `sgl::SignatureBuffer`.

Expected edits:

- Add `src/sgl/core/signature_buffer.h`.
- Move the existing implementation unchanged in behavior:
  - stack-owned object
  - non-copyable
  - `short_vector<uint8_t, 1024>` inline storage
  - `std::string_view view() const`
  - existing uint32/uint64 lower-case hex formatting
- Add the new header to `src/sgl/CMakeLists.txt`.
- Include the new header from `src/slangpy_ext/utils/slangpy.h`.
- Remove the local `SignatureBuffer` definition from the extension header.
- Update extension code to refer to `sgl::SignatureBuffer` where qualification is needed.
- Keep existing `SignatureBuilder` behavior unchanged.

Tests to change:

- Existing signature tests should continue to call the Python-visible `SignatureBuilder` and `get_value_signature` APIs without behavior changes.
- Any C++ files including `utils/slangpy.h` only for `SignatureBuffer` should include `sgl/core/signature_buffer.h` instead.

Tests to add:

- Add a C++ unit test for `SignatureBuffer` through the native C++ test system:
  - appends strings and integer values
  - preserves byte order for existing integer hex formatting
  - returns the expected `std::string_view`

## Phase 2: Fix The Cursor Contract

Status: complete.

Make each cursor check the cursor type it actually passes. The existing `write_to_cursor` implementation is already the contract; this phase should be a narrow correctness fix, not a redesign.

Expected edits:

- In `src/sgl/device/cursor_utils.h`, keep the existing `HasWriteToCursor<T, TCursor>` style unless a small compile fix is required.
- In `src/sgl/device/shader_cursor.h`, change `ShaderCursor::set()` to check `HasWriteToCursor<T, ShaderCursor>`, not `BufferElementCursor`.
- Keep `BufferElementCursor::set()` checking `HasWriteToCursor<T, BufferElementCursor>`.
- `ShaderCursor::set()` should not be `const`. This avoids copying the cursor just to pass a mutable `ShaderCursor&` into `write_to_cursor`.

```cpp
value.write_to_cursor(*this);
```

Tests to change:

- Update any existing tests or examples that relied on the incorrect `ShaderCursor` concept check.

Tests to add:

- C++ coverage for a type that only implements `write_to_cursor(ShaderCursor&)`.
- C++ coverage for a type that only implements `write_to_cursor(BufferElementCursor&)`.
- C++ coverage for a type that implements both cursor overloads.
- A negative compile-time test with `static_assert(!HasWriteToCursor<...>)` for a type without `write_to_cursor`.

## Phase 3: Introduce The Combined Native Registry

Status: complete for native direct cursor-write registration. Functional-API consumption is implemented in Phase 7.

Replace the current split native writer registries with one combined registry that owns cursor writer functions and optional functional-API metadata.

The registry should be native-first. It should not rely on Python callbacks to determine whether a type is registered.

Expected edits:

- Introduce a descriptor in native code, likely near `src/sgl/device/cursor_utils.h`:

```cpp
struct CursorWriterTypeInfo {
    const std::type_info* type;

    ShaderCursorObjectWriteFunc write_shader_cursor;
    BufferElementCursorObjectWriteFunc write_buffer_cursor;

    // Functional API metadata from T::slang_type_name and class-owned signature hooks.
    std::string slang_type_name;
    std::function<void(SignatureBuffer&, const void*)> write_signature;
    std::vector<std::string> imports;
};
```

- Store one registry of `CursorWriterTypeInfo`.
- Provide exact-type lookup by `std::type_info` / `std::type_index`.
- Expose read-only access and lookup helpers for the extension.
- Remove the separate `shader_cursor_object_writer_registry()` and `buffer_element_cursor_object_writer_registry()` storage.
- Remove `WriteConverterTable::m_native_object_writer_cache`; registered native object lookup should consult the combined registry directly.
- Delete the low-level `register_shader_cursor_object_writer*` / `register_buffer_element_cursor_object_writer*` APIs so they do not remain as a second registration model.
- Duplicate registration of a registered type should fail with a clear error.
- Define whether base-type lookup is supported for nanobind-exposed derived types. If supported, document lookup precedence: exact native type first, then registered base type.

Tests to change:

- Existing native-object cursor writer tests should move to the combined registration API.

Tests to add:

- Registering one type creates one registry entry with the expected cursor writers and metadata state.
- Duplicate registration of the same type is rejected with a clear error.
- Public `register_cursor_writer<T>()` does not accept one-cursor-only types.
- Registry lookup by exact native type succeeds.
- If base lookup is supported, derived nanobind/Python-exposed types resolve to the registered base writer.

## Phase 4: Add `register_cursor_writer<T>()`

Status: complete.

Make `register_cursor_writer<T>()` the single public native API.

Expected edits:

- Add:

```cpp
template<typename T>
void register_cursor_writer();
```

- Require `T` to satisfy both `HasWriteToCursor<T, ShaderCursor>` and `HasWriteToCursor<T, BufferElementCursor>`.
- Register both cursor writer functions.
- Optionally capture `T::slang_type_name` when it is static and convertible to `std::string_view`.
- Do not support value-aware `slang_type_name()` and do not infer a Slang type from the native C++ type name.
- Register functional metadata for `register_cursor_writer<T>()` only when `T::slang_type_name` is present.
- Direct cursor writing without SlangPy functional metadata is valid for types that keep bespoke functional marshalls.

Functional metadata sources:

- For functional fallback, the class must provide:

```cpp
static constexpr std::string_view slang_type_name = "MySlangType";
```

- The class may provide any one of these signature forms:

```cpp
static constexpr std::string_view slangpy_signature = "MyTypeSignature";
static void write_slangpy_signature(SignatureBuffer& sig);
void write_slangpy_signature(SignatureBuffer& sig) const;
```

- The class may provide static imports:

```cpp
static std::vector<std::string_view> slangpy_imports();
```

Signature policy:

- If no class-owned signature is provided, default the signature to the C++ class name.
- Imports are constant for a registered type. Do not support value-aware imports.
- Dynamic signatures are allowed through `value.write_slangpy_signature(SignatureBuffer&) const`.
- The static Slang type name must not be used as the default signature unless it is also the desired class-name signature.
- Signatures should be compact and cheap to produce; do not stringify large values or import lists on the hot cache-key path.

Tests to change:

- Replace direct public calls to `register_shader_cursor_object_writer<T>()` and `register_buffer_element_cursor_object_writer<T>()` in tests with `register_cursor_writer<T>()`.

Tests to add:

- Minimal type with `write_to_cursor` support for both cursor kinds and static `slang_type_name` can register.
- Minimal type missing static `slang_type_name` registers as writer-only with no functional metadata.
- Minimal type with only one cursor overload is rejected clearly.
- Minimal type without explicit signature defaults to the C++ class-name signature.
- Type with static metadata gets a stable signature.
- Type with dynamic signature gets a value-dependent signature.
- Type with static metadata gets its declared imports.
- Registration writes correctly through both `ShaderCursor` and `BufferElementCursor` when both overloads exist.

## Phase 5: Capture Constant Imports At Registration

Status: complete.

Imports are a constant list of constant strings for a registered type. Keep this mechanism simple.

Expected shape:

```cpp
static std::vector<std::string_view> slangpy_imports();
```

Expected edits:

- If a type provides a static `slangpy_imports()` function, call it once during registration.
- Copy the returned strings into the native registry descriptor as owned `std::vector<std::string>`.
- Do not call imports callbacks during signature generation or dispatch.
- Do not support value-aware imports.
- Imports should be string literals or other stable strings. The returned container itself does not need stable storage because registration copies it immediately.

Tests to add:

- Static imports are added during code generation.
- The imports function is called once at registration, not per signature generation or dispatch.
- The registry owns copied import strings after registration.

## Phase 6: Rename CursorMarshall To WriteToCursorMarshall And Require Native Fast Path

Status: complete.

Rename the marshall to match the contract it represents, and require the native fast dispatch path.

Expected edits:

- Rename `slangpy/bindings/cursor.py` classes and APIs:
  - `CursorMarshall` -> `WriteToCursorMarshall`
  - `CursorMarshallInfo` -> `WriteToCursorMarshallInfo`
  - `register_cursor_type` -> `register_write_to_cursor_type`, if keeping a Python-side registration API for Python-only future work
- Update exports in `slangpy/bindings/__init__.py`.
- Update error messages and tests to use the new name.
- `WriteToCursorMarshall` must derive from `NativeValueMarshall` or use an equivalent native cached write path.
- Repeated dispatches must not call Python `create_calldata()` to write the value.
- Read-only scalar direct binding should write the original native object through the native registered `write_shader_cursor` function.
- Writable/inout use is not supported unless a separate readback design is added.

Tests to change:

- Rename `slangpy/tests/slangpy_tests/test_cursor_marshall.py` to `test_write_to_cursor_marshall.py`.
- Update import names and expected error text.

Tests to add:

- `WriteToCursorMarshall` can resolve an exact Slang type name.
- `WriteToCursorMarshall` can add declared imports during code generation.
- `WriteToCursorMarshall` direct-binds a read-only scalar value.
- `WriteToCursorMarshall` rejects writable/inout use with a clear error.
- Repeated calls use the native dispatch write path and do not invoke Python `create_calldata()` for the cursor value.

## Phase 7: Make The Extension Consume The Native Registry

Status: complete for direct cursor writes, signature generation, and Python fallback construction. A full nanobind-exposed functional API smoke test remains as follow-up coverage.

The extension should use the combined native registry for both direct cursor writes and functional-API marshalling.

Expected edits:

- In `src/slangpy_ext/device/cursor_utils.h`, replace `write_registered_native_object()` lookup with combined-registry lookup.
- Remove the old per-`PyTypeObject*` writer cache instead of preserving it.
- Add a native lookup path that identifies whether a nanobind object is backed by a registered native cursor writer type.
- Add a Python-visible or extension-internal helper that can create `WriteToCursorMarshall` from native registry metadata when Python `PYTHON_TYPES` has no hit.
- Have that helper read the copied `CursorWriterTypeInfo::slang_type_name` string directly instead of invoking a value-aware type-name callback.
- Update `slangpy/bindings/typeregistry.py::get_or_create_type()` or the `BoundVariable` construction path so native registered cursor writer values fall back to the native registry.
- Update `NativeCallDataCache::get_value_signature()` so native registered cursor writer metadata participates in signature generation only when functional metadata is present.
- Define precedence:
  - registered functional cursor-writer metadata owns simple fallback marshalling for registered cursor writer objects
  - existing built-in native signatures such as `Texture` and `Buffer` remain unchanged for types without functional cursor-writer metadata
  - Python-only registered `PYTHON_TYPES` / `PYTHON_SIGNATURES` continue to work for non-native values
  - `get_this` is only blocked for objects with functional cursor-writer metadata; writer-only registrations may still use bespoke marshalling paths
- Conflict handling should occur when creating a marshall/signature for a Python-visible object. If a functional cursor writer object also exposes `get_this`, fail clearly.

Tests to change:

- Existing tests that manually register a Python marshall for a native object should switch to one native registration call where possible.
- Tests expecting registered types with `get_this` to bypass unpacking should be removed or rewritten to expect a conflict error.

Tests to add:

- A nanobind-exposed native test type registered with `register_cursor_writer<T>()` can be passed directly to a SlangPy function.
- The same native type can be written directly to `ShaderCursor`.
- The same native type can be written to `BufferElementCursor`.
- A registered cursor writer type without `get_this` does not unpack.
- A type that tries to use both `get_this` and cursor writer registration fails with a clear error.
- Python-only `PYTHON_TYPES`, `PYTHON_SIGNATURES`, and `slangpy_signature` behavior still works for non-cursor-writer values.

## Phase 8: Remove `has_registered_type_or_signature`

Status: complete.

Remove the brittle compatibility path that calls from native C++ back into Python to decide whether `get_this` should run.

Expected edits:

- Delete `has_registered_type_or_signature()` from `slangpy/bindings/typeregistry.py`.
- Delete `_has_registered_type_or_signature()` from `slangpy/core/calldata.py`.
- Delete the native `has_registered_type_or_signature()` helper from `src/slangpy_ext/utils/slangpy.cpp`.
- Remove its native static type cache.
- Restore simple unpack policy:
  - if an object has `get_this`, legacy unpacking owns it
  - if an object is registered as a cursor writer, direct marshalling owns it
  - using both paths on one Python-visible type is an error
- Ensure native signature generation no longer imports `slangpy.bindings.typeregistry` for this predicate.

Tests to add:

- A legacy `get_this` wrapper still unpacks as before.
- A registered cursor writer type bypasses legacy unpacking by native registry detection, not by Python predicate.
- Native signature generation no longer calls back into Python for the registered-type predicate.

## Phase 9: Keep Python-Only Types On The Same Contract

Python-only support is a follow-up, but it should mirror the native model rather than introduce a new conceptual path.

Expected future shape:

```python
class TextureHandle:
    slang_type_name = "TextureHandle"

    def write_to_cursor(self, cursor: object) -> None:
        cursor["id"] = self.id
        cursor["type"] = self.type

    def write_slangpy_signature(self, sig: object) -> None:
        sig.add(f"TextureHandle:{self.type}")

    @staticmethod
    def slangpy_imports() -> tuple[str, ...]:
        return ("my_project/texture_handle.slang",)
```

Registration can remain Python-side for Python-only values:

```python
register_write_to_cursor_type(TextureHandle)
```

Do not require this branch to implement Python-only `write_to_cursor` dispatch. If implemented, add an explicit Python-method dispatch path; the current native cursor writer path only handles registered native objects and normal cursor-recursive values.

Tests to add when implemented:

- Python-only type with `write_to_cursor` can register.
- Python-only type requires explicit static class-owned `slang_type_name` for functional API use.
- Python-only type with value-aware metadata changes call signatures correctly.
- Python-only type with `get_this` conflict is rejected.

## Phase 10: Public API Cleanup

After the native registry and marshall rename are in place, clean up exported names.

Expected edits:

- Export the new API names:
  - `WriteToCursorMarshall`
  - `WriteToCursorMarshallInfo`
  - `register_write_to_cursor_type`
- Remove old public cursor-writer registration names unless they are still needed internally during the PR.
- Keep compatibility aliases only if needed while the PR is draft.
- Update documentation comments and examples to show `register_cursor_writer<T>()`.

Tests to change:

- Update import tests and any docs snippets that mention `CursorMarshall` or `register_cursor_type`.

Tests to add:

- Import smoke test for the new Python names.
- Optional deprecation/compatibility test if old aliases are temporarily retained.

## Final Verification

Required by repository policy:

1. Build before running tests.
2. Run focused C++ and Python tests for the changed paths.
3. Run broader SlangPy Python tests if build time allows.
4. Run pre-commit and rerun if it modifies files.

Suggested commands on Windows:

```powershell
cmake --build --preset windows-msvc-debug --target slangpy_ext sgl_tests
python tools/ci.py unit-test-cpp
python -m pytest slangpy/tests/slangpy_tests/test_write_to_cursor_marshall.py -v
python -m pytest slangpy/tests/slangpy_tests/test_type_resolution.py -v
python -m pytest slangpy/tests/slangpy_tests/test_instances.py -v
pre-commit run --all-files
```
