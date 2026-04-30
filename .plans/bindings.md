# Cursor Writer Binding Plan

## Goal

Replace the current split system with one native registration path for simple bindable value types.

The desired user model is:

1. A native value type implements `write_to_cursor`.
2. The type provides explicit SlangPy metadata when it should be usable through the functional API.
3. The project calls one registration function:

```cpp
sgl::cursor_utils::register_cursor_writer<T>();
```

That single call registers direct cursor writing and, when metadata is available, functional-API marshalling through `WriteToCursorMarshall`.

The new policy should be:

- `get_this` is the legacy wrapper path.
- `write_to_cursor` plus `register_cursor_writer<T>()` is the new bindable-value path.
- A single type must not use both paths.
- Native registered cursor writers should be identified from the native registry, not from Python-side compatibility predicates.
- The old cursor-writer caches and `has_registered_type_or_signature` compatibility path should be removed as part of this migration, not preserved as fallback behavior.

## Phase 1: Move SignatureBuffer To Native

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

Make each cursor check the cursor type it actually passes. The existing `write_to_cursor` implementation is already the contract; this phase should be a narrow correctness fix, not a redesign.

Expected edits:

- In `src/sgl/device/cursor_utils.h`, keep the existing `HasWriteToCursor<T, TCursor>` style unless a small compile fix is required.
- In `src/sgl/device/shader_cursor.h`, change `ShaderCursor::set()` to check `HasWriteToCursor<T, ShaderCursor>`, not `BufferElementCursor`.
- Keep `BufferElementCursor::set()` checking `HasWriteToCursor<T, BufferElementCursor>`.
- Account for `ShaderCursor::set()` being a `const` member only if needed. If calling `write_to_cursor(*this)` deduces a const cursor and causes friction, create a local mutable cursor copy before calling the writer:

```cpp
ShaderCursor cursor = *this;
value.write_to_cursor(cursor);
```

Tests to change:

- Update any existing tests or examples that relied on the incorrect `ShaderCursor` concept check.

Tests to add:

- C++ coverage for a type that only implements `write_to_cursor(ShaderCursor&)`.
- C++ coverage for a type that only implements `write_to_cursor(BufferElementCursor&)`.
- C++ coverage for a type that implements both cursor overloads.
- A negative compile-time test with `static_assert(!HasWriteToCursor<...>)` for a type without `write_to_cursor`.

## Phase 3: Introduce The Combined Native Registry

Replace the current split native writer registries with one combined registry that owns cursor writer functions and optional functional-API metadata.

The registry should be native-first. It should not rely on Python callbacks to determine whether a type is registered.

Expected edits:

- Introduce a descriptor in native code, likely near `src/sgl/device/cursor_utils.h`:

```cpp
struct CursorWriterTypeInfo {
    const std::type_info* type;

    ShaderCursorObjectWriteFunc write_shader_cursor;
    BufferElementCursorObjectWriteFunc write_buffer_cursor;

    // Functional API metadata. slang_type_name has no default.
    bool has_functional_metadata;
    std::function<std::string_view(const void*)> slang_type_name;
    std::function<void(SignatureBuffer&, const void*)> write_signature;
    std::vector<std::string> imports;
};
```

- Store one registry of `CursorWriterTypeInfo`.
- Provide exact-type lookup by `std::type_info` / `std::type_index`.
- Expose read-only access and lookup helpers for the extension.
- Remove the separate `shader_cursor_object_writer_registry()` and `buffer_element_cursor_object_writer_registry()` storage.
- Remove `WriteConverterTable::m_native_object_writer_cache`; registered native object lookup should consult the combined registry directly.
- Either delete the low-level `register_shader_cursor_object_writer*` / `register_buffer_element_cursor_object_writer*` APIs or make them internal wrappers that merge into the combined registry. They should not remain a second public registration model.
- Duplicate registration of a fully registered type should fail with a clear error. If internal wrappers are retained during migration, they must have explicit merge semantics and tests.
- Define whether base-type lookup is supported for nanobind-exposed derived types. If supported, document lookup precedence: exact native type first, then registered base type.

Tests to change:

- Existing native-object cursor writer tests should move to the combined registration API.

Tests to add:

- Registering one type creates one registry entry with the expected cursor writers and metadata state.
- Duplicate registration of the same type is rejected with a clear error.
- A type with only one cursor overload registers successfully, with only that writer populated.
- Registry lookup by exact native type succeeds.
- If base lookup is supported, derived nanobind/Python-exposed types resolve to the registered base writer.

## Phase 4: Add `register_cursor_writer<T>()`

Make `register_cursor_writer<T>()` the single public native API.

Expected edits:

- Add:

```cpp
template<typename T>
void register_cursor_writer();
```

- Require `T` to satisfy `HasWriteToCursor<T, ShaderCursor>` or `HasWriteToCursor<T, BufferElementCursor>`.
- Register the available cursor writer functions.
- Register functional metadata only when the type provides explicit Slang metadata.
- Do not invent a default `slang_type_name`. A native C++ type name is not a reliable visible Slang type.
- A no-metadata type may still register for direct cursor writing. If it is passed to the functional API, the error should say that registered cursor writer type `T` has no Slang type metadata.

Functional metadata sources:

- Prefer a `CursorWriterTraits<T>` specialization for static metadata when the described class should not store extra data.
- Also support value-aware member methods when needed.
- Static metadata may provide:

```cpp
static constexpr std::string_view slang_type_name = "MySlangType";
static void write_slangpy_signature(SignatureBuffer& sig);
static std::vector<std::string_view> slangpy_imports();
```

- Value-aware metadata may provide:

```cpp
std::string_view slang_type_name() const;
void write_slangpy_signature(SignatureBuffer& sig) const;
```

Signature policy:

- If functional metadata is present, a signature must be available.
- For static metadata, defaulting the signature to the explicit static Slang type name is acceptable.
- Imports are constant for a registered type. Do not support value-aware imports.
- For value-aware `slang_type_name()`, `write_slangpy_signature()` is required and must vary for the same cases.
- Signatures should be compact and cheap to produce; do not stringify large values or import lists on the hot cache-key path.

Tests to change:

- Replace direct public calls to `register_shader_cursor_object_writer<T>()` and `register_buffer_element_cursor_object_writer<T>()` in tests with `register_cursor_writer<T>()`.

Tests to add:

- Minimal type with only `write_to_cursor` and no metadata can register.
- Minimal type without metadata writes through supported cursors.
- Minimal type without metadata fails through the functional API with a clear missing-Slang-metadata error.
- Type with static metadata gets a stable signature.
- Type with static metadata gets its declared imports.
- Registration writes correctly through both `ShaderCursor` and `BufferElementCursor` when both overloads exist.

## Phase 5: Capture Constant Imports At Registration

Imports are a constant list of constant strings for a registered type. Keep this mechanism simple.

Expected shape:

```cpp
static std::vector<std::string_view> slangpy_imports();
```

Expected edits:

- If a type or `CursorWriterTraits<T>` provides an imports function, call it once during registration.
- Copy the returned strings into the native registry descriptor as owned `std::vector<std::string>`.
- Do not call imports callbacks during signature generation or dispatch.
- Do not support value-aware imports.
- Imports should be string literals or other stable strings. The returned container itself does not need stable storage because registration copies it immediately.

Tests to add:

- Static imports are added during code generation.
- The imports function is called once at registration, not per signature generation or dispatch.
- The registry owns copied import strings after registration.

## Phase 6: Rename CursorMarshall To WriteToCursorMarshall And Require Native Fast Path

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

The extension should use the combined native registry for both direct cursor writes and functional-API marshalling.

Expected edits:

- In `src/slangpy_ext/device/cursor_utils.h`, replace `write_registered_native_object()` lookup with combined-registry lookup.
- Remove the old per-`PyTypeObject*` writer cache instead of preserving it.
- Add a native lookup path that identifies whether a nanobind object is backed by a registered native cursor writer type.
- Add a Python-visible or extension-internal helper that can create `WriteToCursorMarshall` from native registry metadata when Python `PYTHON_TYPES` has no hit.
- Update `slangpy/bindings/typeregistry.py::get_or_create_type()` or the `BoundVariable` construction path so native registered cursor writer values fall back to the native registry.
- Update `NativeCallDataCache::get_value_signature()` so native registered cursor writer metadata participates in signature generation.
- Define precedence:
  - registered cursor writer metadata owns registered cursor writer objects
  - existing built-in native signatures such as `Texture` and `Buffer` remain unchanged for types not registered as cursor writers
  - Python-only registered `PYTHON_TYPES` / `PYTHON_SIGNATURES` continue to work for non-native values
  - `get_this` is only considered for objects not registered as cursor writers
- Conflict handling should occur when creating a marshall/signature for a Python-visible object. If a registered cursor writer object also exposes `get_this`, fail clearly.

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
    def write_to_cursor(self, cursor: object) -> None:
        cursor["id"] = self.id
        cursor["type"] = self.type

    def slang_type_name(self) -> str:
        return "TextureHandle"

    def write_slangpy_signature(self, sig: object) -> None:
        sig.add(f"TextureHandle:{self.type}")

    def slangpy_imports(self) -> tuple[str, ...]:
        return ("my_project/texture_handle.slang",)
```

Registration can remain Python-side for Python-only values:

```python
register_write_to_cursor_type(TextureHandle)
```

Do not require this branch to implement Python-only `write_to_cursor` dispatch. If implemented, add an explicit Python-method dispatch path; the current native cursor writer path only handles registered native objects and normal cursor-recursive values.

Tests to add when implemented:

- Python-only type with `write_to_cursor` can register.
- Python-only type requires explicit `slang_type_name` for functional API use.
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
