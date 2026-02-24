---
description: Instructions for optimizing NativeValueMarshall dispatch performance by caching shader offsets and type-specialized writer function pointers
---

# NativeValueMarshall Optimization Plan

## Problem

Every call to `NativeValueMarshall::write_shader_cursor_pre_dispatch` (in `src/slangpy_ext/utils/slangpyvalue.cpp`) does:

1. **Two string-based name lookups** through Slang reflection: `cursor[binding->variable_name()]` then `["value"]`. Each calls `find_field()` which does `findFieldIndexByName()` through the Slang reflection API.
2. **Full `write_shader_cursor` dispatch** via `WriteConverterTable::write_internal()` — a switch on `TypeReflection::Kind`, reflection calls for `getScalarType()`/`getColumnCount()`/`getRowCount()`, isinstance checks for DescriptorHandle/PackedArg/StridedBufferView, then finally the actual `cursor.set()` call.

For a plain `float` scalar, this is ~6 layers of indirection to do a 4-byte memcpy. `NativeTensorMarshall` already proves this overhead can be nearly eliminated via caching.

## Solution: Cache Offset + Reuse WriteConverterTable Function Pointers

The `WriteConverterTable` (in `src/slangpy_ext/device/cursor_utils.h`) already maintains pre-indexed function-pointer arrays fully specialized for concrete C++ types:

```cpp
std::function<void(CursorType&, nb::object)> m_write_scalar[(int)ScalarType::COUNT];
std::function<void(CursorType&, nb::object)> m_write_vector[(int)ScalarType::COUNT][5];
std::function<void(CursorType&, nb::object)> m_write_matrix[(int)ScalarType::COUNT][5][5];
```

Each entry (e.g., `_write_scalar<float>`, `_write_vector<float3>`) does `nb::cast<T>(nbval)` → `cursor.set(val)` with no further type dispatch. By caching both the resolved `ShaderOffset` and the appropriate function pointer from these tables, we eliminate all per-call overhead except the final `nb::cast` + `cursor.set()`.

## Implementation Steps

### Step 1: Add a Public Writer-Lookup Method to WriteConverterTable

**File:** `src/slangpy_ext/device/cursor_utils.h`

Add a public method to `WriteConverterTable` that resolves the correct writer function pointer from a `slang::TypeLayoutReflection*`, without running the full `write_internal` dispatch. This should return a `std::function<void(CursorType&, nb::object)>` (or null/empty for unsupported types like structs/arrays/resources).

```cpp
/// Resolve a type-specialized writer function for the given type layout.
/// Returns an empty function for types that require recursive dispatch (struct, array, resource).
/// For scalar/vector/matrix types, returns the pre-specialized writer that handles
/// nb::cast + cursor.set() with the correct C++ type.
std::function<void(CursorType&, nb::object)> resolve_writer(slang::TypeLayoutReflection* type_layout) const
{
    auto kind = (TypeReflection::Kind)type_layout->getKind();
    auto type = type_layout->getType();
    if (!type)
        return {};
    switch (kind) {
    case TypeReflection::Kind::scalar:
        return m_write_scalar[(int)type->getScalarType()];
    case TypeReflection::Kind::vector:
        return m_write_vector[(int)type->getScalarType()][type->getColumnCount()];
    case TypeReflection::Kind::matrix:
        return m_write_matrix[(int)type->getScalarType()][type->getRowCount()][type->getColumnCount()];
    default:
        return {};
    }
}
```

The `resolve_writer` method must be `public` (the tables `m_write_scalar` etc. are currently `private`). Either make `resolve_writer` public and keep the tables private, or make this a friend. The cleanest approach is a single public method.

The `WriteConverterTable` for `ShaderCursor` is instantiated as a static singleton `detail::_writeconv` of type `ShaderCursorWriteConverterTable` in `src/slangpy_ext/device/shader_cursor.cpp`. This singleton needs to be accessible from `NativeValueMarshall`. Either:
- Expose a free function like `resolve_shader_cursor_writer(slang::TypeLayoutReflection*)` declared in a header, or
- Make the singleton externally accessible.

### Step 2: Add Cached Offset Struct to NativeValueMarshall

**File:** `src/slangpy_ext/utils/slangpyvalue.h`

Add a mutable cache struct to `NativeValueMarshall`:

```cpp
struct CachedValueWrite {
    ShaderOffset value_offset;                                    // Offset to the "value" sub-field
    slang::TypeLayoutReflection* value_type_layout = nullptr;     // Type layout for "value" field
    std::function<void(ShaderCursor&, nb::object)> writer;        // Pre-resolved writer fn
    bool is_valid = false;
};

mutable CachedValueWrite m_cached;
```

Add a private helper:

```cpp
void ensure_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const;
```

### Step 3: Implement Caching and Optimized Dispatch

**File:** `src/slangpy_ext/utils/slangpyvalue.cpp`

The `ensure_cached` method runs once on first dispatch:

```cpp
void NativeValueMarshall::ensure_cached(ShaderCursor cursor, NativeBoundVariableRuntime* binding) const
{
    if (m_cached.is_valid)
        return;

    // Resolve the cursor path once
    ShaderCursor field = cursor[binding->variable_name()]["value"];
    m_cached.value_offset = field.offset();
    m_cached.value_type_layout = field.slang_type_layout();

    // Look up the pre-specialized writer from the WriteConverterTable
    m_cached.writer = resolve_shader_cursor_writer(m_cached.value_type_layout);

    m_cached.is_valid = true;
}
```

The optimized `write_shader_cursor_pre_dispatch`:

```cpp
void NativeValueMarshall::write_shader_cursor_pre_dispatch(
    CallContext* context,
    NativeBoundVariableRuntime* binding,
    ShaderCursor cursor,
    nb::object value,
    nb::list read_back
) const
{
    SGL_UNUSED(context);
    SGL_UNUSED(read_back);
    AccessType primal_access = binding->access().first;
    if (!value.is_none() && (primal_access == AccessType::read || primal_access == AccessType::readwrite)) {
        ensure_cached(cursor, binding);
        if (m_cached.writer) {
            // Fast path: construct cursor directly from cached offset, call pre-resolved writer
            ShaderCursor value_cursor(cursor.shader_object(), m_cached.value_type_layout, m_cached.value_offset);
            m_cached.writer(value_cursor, value);
        } else {
            // Fallback for complex types (struct, array, resource): full dispatch
            ShaderCursor field = cursor[binding->variable_name()]["value"];
            write_shader_cursor(field, value);
        }
    }
}
```

### Step 4: Verify ShaderCursor Can Be Constructed from Cached Components

Check that `ShaderCursor` has (or can be given) a constructor that accepts `(ShaderObject*, slang::TypeLayoutReflection*, ShaderOffset)`. Looking at `src/sgl/device/shader_cursor.h`, the class stores exactly these three fields:

```cpp
slang::TypeLayoutReflection* m_type_layout;
ShaderObject* m_shader_object;
ShaderOffset m_offset;
```

If no suitable constructor exists, add one. This is the key enabler — it lets us skip cursor navigation entirely and jump directly to the target field.

### Step 5: Testing

Run existing tests to verify correctness:

```bash
pytest slangpy/tests/slangpy_tests/ -v -k "value"
pytest slangpy/tests/ -v
```

No new tests are needed for this optimization — it's a pure performance improvement that preserves identical behavior. If existing tests pass, the optimization is correct.

### Step 6: Benchmarking

Create or use an existing benchmark that exercises value dispatch in a tight loop to measure the improvement. The benchmark should call a Slang function with scalar/vector value arguments many times to isolate the dispatch overhead.

## What This Eliminates Per Call

| Cost | Before | After |
|------|--------|-------|
| `find_field(variable_name)` | Every call | Once (cached) |
| `find_field("value")` | Every call | Once (cached) |
| `write_internal()` Kind switch | Every call | Once (cached fn ptr) |
| `getScalarType()`/`getColumnCount()` reflection | Every call | Once (cached fn ptr) |
| `isinstance` checks (DescriptorHandle, PackedArg, etc.) | Every call | Never |
| `_write_scalar<T>` → `nb::cast<T>` → `cursor.set()` | Every call | Every call (this is the irreducible cost) |

## What This Preserves

- All edge-case handling in `CursorWriteWrappers` (`_set_scalar`, `_set_vector`, `_set_matrix`) — bool 1B→4B conversion, float3 stride padding on Metal, etc. These are handled by `cursor.set()` which still runs.
- Struct/array/resource value fallback — these go through the full `write_shader_cursor` path as before.
- Thread safety — the cache is `mutable` on a per-marshall instance (same pattern as `NativeTensorMarshall`). Each `CallData` has its own marshall instances, so no contention.

## Future: Full reserve_data Bypass (Optional Phase 2)

If profiling shows the remaining `cursor.set()` → `_set_data()` → RHI `setData()` path is still significant, the cached writer function pointer can be replaced with a bespoke writer that uses `shader_object->reserve_data()` to get a raw pointer and writes directly via memcpy. This would bypass `CursorWriteWrappers` entirely but requires reimplementing bool conversion and stride-mismatch handling. The caching structure from this plan stays the same — only the function pointer target changes.

## Files Changed

| File | Change |
|------|--------|
| `src/slangpy_ext/device/cursor_utils.h` | Add `resolve_writer()` public method to `WriteConverterTable` |
| `src/slangpy_ext/device/shader_cursor.cpp` (or new header) | Expose `resolve_shader_cursor_writer()` free function wrapping the singleton |
| `src/sgl/device/shader_cursor.h` | Add constructor from `(ShaderObject*, TypeLayoutReflection*, ShaderOffset)` if not present |
| `src/slangpy_ext/utils/slangpyvalue.h` | Add `CachedValueWrite` struct, `m_cached` member, `ensure_cached()` method |
| `src/slangpy_ext/utils/slangpyvalue.cpp` | Implement `ensure_cached()`, rewrite `write_shader_cursor_pre_dispatch` to use cache |
