## Phase 1: Direct Type Marshalling

**Goal**: For dim-0, non-composite arguments, emit the raw Slang type in CallData and use direct assignment in the trampoline — eliminating `ValueType<T>` wrappers, `__slangpy_load`/`__slangpy_store` indirection, mapping constants, and `Context.map()` calls.

**Parent plan**: [plan-simplifyKernelGen.prompt.md](plan-simplifyKernelGen.prompt.md)

---

### Step 1.1: Define eligibility predicate

Add a **global Python function** `is_direct_bind_eligible(binding: BoundVariable) -> bool` (e.g., in [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py) or a small utility module). This is intentionally NOT a method on `NativeMarshall` — placing it on the C++ side would require nanobind trampoline plumbing for a function that is only consumed during Python-side codegen. A simple Python function avoids all C++/nanobind complexity.

The conditions for leaf types are:
- `binding.call_dimensionality is not None and binding.call_dimensionality == 0` (note: `call_dimensionality` is initialized to `None`, so a `None` check is required)
- `not binding.children` (not composite/dict)
- The marshall's Python type is one of the known direct-eligible types (`ValueMarshall`, `ScalarMarshall`, `VectorMarshall`, `MatrixMarshall`, `ArrayMarshall`, `ValueRefMarshall`). Types like `WangHashArgMarshall` are excluded.

Individual marshalls call this function inside their `gen_calldata`, `gen_trampoline_load`, `gen_trampoline_store`, and `create_calldata` methods to decide which codegen path to take.

Add a companion function `is_direct_bind_recursive(binding: BoundVariable) -> bool` that handles composite types:
- If `binding.children is None`: delegates to `is_direct_bind_eligible(binding)`
- If `binding.children is not None`: returns `True` only if `binding.call_dimensionality is not None and binding.call_dimensionality == 0` AND every child's `is_direct_bind_recursive()` returns `True`. This handles dicts bound to Slang structs where all fields are dim-0 leaves (or recursively dim-0 structs).
- Additionally, the `binding.vector_type` must be a concrete Slang struct type (not `UnknownType`). Dicts without `_type` may resolve to `UnknownType` and are ineligible.

Both functions are consulted by the marshalls and by `gen_call_data_code` (for struct/dict bindings).

---

### Step 1.2: Implement for `ValueMarshall` (scalars/matrices)

In [slangpy/builtin/value.py](slangpy/builtin/value.py):

- Modify `gen_calldata`: call `is_direct_bind_eligible(binding)`. When eligible, emit `typealias _t_{name} = {raw_slang_type};` instead of `ValueType<{type}>`
- Add `gen_trampoline_load`: when direct-eligible, emit `{name} = {data_name};` and return `True`
- Add `gen_trampoline_store`: when direct-eligible (read-only scalars), return `True` (suppress default store, no-op)
- Modify `create_calldata`: when direct-eligible, return raw value instead of `{"value": data}`. The cursor write system in [cursor_utils.h](src/slangpy_ext/device/cursor_utils.h) already handles writing scalars/vectors/matrices directly — the `write_internal` method dispatches on `TypeReflection::Kind::scalar/vector/matrix`.

#### Step 1.2a: Critical C++ change — `NativeValueMarshall` fast path

`NativeValueMarshall::write_shader_cursor_pre_dispatch` in [slangpyvalue.cpp](src/slangpy_ext/utils/slangpyvalue.cpp) has a cached fast path that navigates `cursor[variable_name]["value"]` on first call:

```cpp
ShaderCursor field = cursor[binding->variable_name()]["value"];
m_cached.value_offset = field.offset();
```

If the Slang type changes from `ValueType<int>` (which has a `value` sub-field) to raw `int` (a scalar with no sub-fields), the `["value"]` navigation will crash. This affects **all** `NativeValueMarshall` subclasses: `ValueMarshall`, `VectorMarshall`, `MatrixMarshall`, `StructMarshall`, `ArrayMarshall`.

**Required fix**: Add a `direct_bind` flag to `NativeValueMarshall` (set from the Python side when `can_direct_bind` returns `True`). In `ensure_cached`, branch on this flag:
- **`direct_bind == false`** (current path): navigate `cursor[variable_name]["value"]`
- **`direct_bind == true`** (new path): navigate `cursor[variable_name]` only (no `"value"` sub-field), and cache the resulting offset/layout/writer directly

The flag can be set during `CallData` construction when the binding is finalized, or passed via a `NativeBoundVariableRuntime` property. Alternatively, detect the absence of the `"value"` field by checking the type layout — but an explicit flag is safer and clearer.

---

### Step 1.3: Implement for `VectorMarshall`, `MatrixMarshall`, and `ArrayMarshall`

In [slangpy/builtin/value.py](slangpy/builtin/value.py):
- `VectorMarshall`: same pattern as `ValueMarshall`. `gen_calldata` emits `typealias _t_{name} = {vector_type};` instead of `VectorValueType<{et},{n}>`.
- `MatrixMarshall`: same pattern. Note that `MatrixMarshall` does **not** override `gen_calldata` — it inherits `ValueMarshall.gen_calldata` which emits `ValueType<{matrix_type}>` (not `MatrixValueType`). The `MatrixValueType<...>` name only appears in `resolve_types` for the experimental vectorization path. The direct-bind override goes on `gen_calldata` and emits the raw matrix type (e.g., `float4x4`) instead of `ValueType<float4x4>`.

In [slangpy/builtin/array.py](slangpy/builtin/array.py):
- `ArrayMarshall`: at dim-0, it already falls through to `super().gen_calldata()` (i.e. `ValueMarshall`) which uses `ValueType<T>`. The same direct-bind pattern applies — emit the raw array type instead of wrapping in `ValueType`.

---

### Step 1.4: Implement for `StructMarshall` (dict → struct)

In [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py):

When a Python dict is bound to a Slang struct and `is_direct_bind_recursive(binding)` returns `True` (all children are dim-0 and direct-eligible recursively), the **Slang-side** struct can bypass the inline `__slangpy_load`/`__slangpy_store` struct generation. The **Python/C++ side** keeps the existing tree of marshalls unchanged — they continue to recurse through children and cache offsets for efficient cursor writes.

This is a Slang-code-gen-only simplification:
- **Current behavior (children path in `gen_call_data_code`)**: generates an inline struct `_t_{name}` with field declarations, `__slangpy_load`/`__slangpy_store` methods, and mapping constants — delegates each child's code gen recursively
- **Direct-eligible behavior**: emit `typealias _t_{name} = {vector_type.full_name};` (the raw Slang struct type). Skip generating the inline struct, its load/store methods, child type aliases, and child mapping constants entirely.

In the trampoline:
- `gen_trampoline_load`: emit `{name} = {data_name};` (direct struct assignment) and return `True`
- `gen_trampoline_store`: return `True` (suppress default store for read-only structs)

**Python/C++ dispatch — keep the child tree, fix the per-child cursor path**:

The Python-side tree of marshalls is kept for dispatch. When a `BoundVariable` has children (dict case), the C++ dispatch in [slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp) `NativeBoundVariableRuntime::write_shader_cursor_pre_dispatch` still takes the children branch:

```cpp
ShaderCursor child_field = cursor[m_variable_name.c_str()];
for (const auto& [name, child_ref] : *m_children) {
    child_ref->write_shader_cursor_pre_dispatch(context, child_field, child_value, read_back);
}
```

Each child leaf calls `NativeValueMarshall::write_shader_cursor_pre_dispatch`, which navigates `cursor[variable_name]["value"]`. If the Slang struct type changes from the inline struct (where each child field is `ValueType<float>` with a `value` sub-field) to the raw Slang struct (where each child field is `float` directly), the `["value"]` navigation will crash.

**Solution**: Set the `direct_bind` flag from Step 1.2a on each child's `NativeValueMarshall`. The per-child flag causes each leaf's `ensure_cached` to navigate `cursor[variable_name]` only (no `["value"]` sub-field). This is the same fix as Step 1.2a applied to each child — no changes to the children dispatch path itself are needed.

`StructMarshall.create_calldata` is dead code for the children path: when `m_children` is set on the C++ `NativeBoundVariableRuntime`, the children dispatch branch runs instead of calling the marshall's `create_calldata`. The current `ValueMarshall.create_calldata` (which `StructMarshall` inherits) is never called for dict bindings. It can be removed from `StructMarshall` if desired, but is harmless.

**Complexity considerations:**
- The recursive eligibility check must traverse all children. Nested dicts (struct-of-struct) work if all leaves are direct-eligible.
- The `vector_type` on the `BoundVariable` must be a concrete Slang struct type (not `UnknownType`). If the dict has `_type` specified, the struct type is resolved; if not, it may be `UnknownType` and ineligible.
- Writable struct fields (inout/out parameters) need the same treatment as writable scalars — the struct in CallData stays as the raw type, but the trampoline does direct assignment both ways.
- This optimization can be deferred if it proves too complex for the initial Phase 1 implementation — the fallback (current inline struct with load/store) always works. Priority should be leaf types first.

---

### Step 1.5: Implement for `ValueRefMarshall`

In [slangpy/builtin/valueref.py](slangpy/builtin/valueref.py):

Note: There is only **one** `ValueRefMarshall` class (not separate `ValueRef`/`RWValueRef` classes). It inherits from `Marshall` (not `NativeValueMarshall`). Read vs. write behavior is determined by `binding.access` at codegen time — the same class emits `ValueRef<T>` or `RWValueRef<T>` depending on access mode.

- In `gen_calldata`, call `is_direct_bind_eligible(binding)`. When eligible:
- Read-only path (`access[0] == AccessType.read`): `gen_calldata` emits raw type, `gen_trampoline_load` does direct assignment, `create_calldata` returns raw value
- Writable path (`access[0] != AccessType.read`): `gen_calldata` emits `RWStructuredBuffer<{type}>`, `gen_trampoline_load` emits `{name} = {data_name}[0];`, `gen_trampoline_store` emits `{data_name}[0] = {name};`, `create_calldata` returns the buffer directly (no `{"value": buffer}` wrapper). Note: `RWStructuredBuffer<T>` is a **resource type** in Slang — the cursor write system handles it via the resource binding mechanism. Buffer objects are written to resource-typed cursor fields via the `write_value` virtual path in [cursor_utils.h](src/slangpy_ext/device/cursor_utils.h), not the struct/scalar dispatch in `write_internal`.

Since `ValueRefMarshall` extends `Marshall` (not `NativeValueMarshall`), the C++ fast path issue from Step 1.2a does not apply — `NativeMarshall::write_shader_cursor_pre_dispatch` calls `create_calldata` and then passes the result to the generic `write_shader_cursor(cursor, cd_val)`, which dispatches based on the Slang type layout. There is no cached `["value"]` navigation.

---

### Step 1.6: Implement for tensor marshalls

In [slangpy/builtin/tensorcommon.py](slangpy/builtin/tensorcommon.py): The `TensorView`/`DiffTensorView` case already works via direct assignment.

For `Tensor<T,D>` (the slangpy Tensor type): this is a **complex struct** containing `_data` (a `StructuredBuffer`/pointer), `_shape[D]`, `_strides[D]`, and `_offset`. It is NOT a simple assignable type like a scalar — it always requires its buffer resource descriptor and metadata to be bound. However, since it is already a well-defined Slang struct, it can still use direct assignment (`name = call_data.name;`) in the trampoline when dim-0. The `gen_calldata` already emits the correct tensor type name. Add `gen_trampoline_load` to handle `ITensorType` dim-0 with direct assignment (same as TensorView pattern — the struct is copied as a whole).

Note: `Tensor<T,D>` cannot be simplified to a raw value the way scalars can — it stays as a struct in CallData. The simplification here is only at the trampoline level (bypassing `__slangpy_load`/`__slangpy_store`).

---

### Step 1.7: Eliminate unused boilerplate in code generation

In [slangpy/core/callsignature.py](slangpy/core/callsignature.py):

- **Mapping constants**: In `BoundVariable.gen_call_data_code()` ([slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py)), skip emitting `static const int _m_{name} = 0;` when `is_direct_bind_eligible(self)` (for leaves) or `is_direct_bind_recursive(self)` (for composites) returns `True`. These constants are only consumed by `__slangpy_context__.map(_m_{name})` calls, which direct-bound variables skip.
- **`import "slangpy"`**: Keep this import. Attempting to detect and eliminate it provides negligible benefit for significant complexity. The slangpy Slang module is always available and the link-time constants are always emitted. The focus of this phase is eliminating wrapper types and `__slangpy_load`/`__slangpy_store` indirection, not the import.

---

### Step 1.8: Handle autodiff (bwds mode)

For differentiable types in bwds mode:
- Primal reads are still direct-eligible (just a direct assignment)
- Derivative writes need writable backing — use `RWStructuredBuffer<T>` for derivative fields (similar to `RWValueRef` pattern)
- The trampoline must remain `[Differentiable]`, but direct assignment `a = call_data.a;` is trivially differentiable in Slang
- The `gen_trampoline_load`/`gen_trampoline_store` implementations need to account for `access[1]` (derivative access) and emit derivative load/store code when needed
- This is the most complex part of Phase 1; consider implementing prim-mode direct binding first, then extending to bwds

---

### Step 1.9: Tests

Extend [slangpy/tests/slangpy_tests/test_kernel_gen.py](slangpy/tests/slangpy_tests/test_kernel_gen.py). All tests use `generate_code()` which calls `func.debug_build_call_data(*args, **kwargs)` and returns `cd.code`. Tests are parametrized across `helpers.DEFAULT_DEVICE_TYPES`.

**Assertion helpers** (added to test file):
- `assert_contains(code, *patterns)` — assert all patterns appear in generated code
- `assert_not_contains(code, *patterns)` — assert none appear

**Gating tests** — assert CURRENT behavior so they break when each step is implemented:

| Test | Slang Source | Args | Asserts (current behavior) | Breaks when |
|------|-------------|------|---------------------------|-------------|
| `test_gate_scalar_uses_valuetype` | `int add(int a, int b) { return a + b; }` | `(1, 2)` | `ValueType<int>` present, `__slangpy_load` for `a`/`b`, `__slangpy_store` for `_result` | Step 1.2 |
| `test_gate_float_scalar_uses_valuetype` | `float mul(float x, float y) { return x * y; }` | `(1.0, 2.0)` | `ValueType<float>` present, `__slangpy_load` present | Step 1.2 |
| `test_gate_vector_uses_vectorvaluetype` | `float3 scale(float3 v, float s) { return v * s; }` | `(spy.math.float3(1,2,3), 1.0)` | `VectorValueType<float,3>` for `v` (no space after comma), `ValueType<float>` for `s` | Step 1.3 |
| `test_gate_matrix_uses_valuetype` | `float4x4 ident(float4x4 m) { return m; }` | `(spy.math.float4x4.identity(),)` | `ValueType<float4x4>` present | Step 1.3 |
| `test_gate_valueref_read_uses_wrapper` | `float read_val(float v) { return v; }` | `(spy.ValueRef(1.0),)` | `ValueRef<float>` present, `__slangpy_load` present | Step 1.5 |
| `test_gate_valueref_write_uses_wrapper` | `int add(int a, int b) { return a + b; }` | `(1, 2)` (auto `_result`) | `RWValueRef<int>` for `_result`, `__slangpy_store` present | Step 1.5 |
| `test_gate_array_dim0_uses_valuetype` | `void process(float a[4]) { }` | `([1.0, 2.0, 3.0, 4.0],)` | `ValueType<` present for array binding | Step 1.3 |
| `test_gate_mapping_constants_present` | `int add(int a, int b) { return a + b; }` | `(1, 2)` | `static const int _m_a = 0` and `_m_b` and `_m__result` present | Step 1.7 |
| `test_gate_context_map_in_trampoline` | `int add(int a, int b) { return a + b; }` | `(1, 2)` | `__slangpy_context__.map(_m_a)` in trampoline | Step 1.7 |
| `test_gate_struct_uses_slangpy_load` | `struct S { float x; float y; }; float sum(S s) { return s.x + s.y; }` | `({"x": 1.0, "y": 2.0},)` | inline struct `_t_s` with `__slangpy_load`, child mapping constants `_m_x`, `_m_y` | Step 1.4 |

**Negative gates** — should REMAIN passing after Phase 1 (these types are NOT direct-bind eligible):

| Test | Slang Source | Args | Asserts (must stay) |
|------|-------------|------|--------------------|
| `test_gate_wanghasharg_uses_wrapper` | `int rng(WangHashArg rng) { return 0; }` | `(spy.WangHashArg(1),)` | `WangHashArg<` in type alias, `__slangpy_load` present |
| `test_gate_vectorized_scalar_keeps_wrapper` | `float square(float x) { return x * x; }` | `(Tensor.numpy(np.array([1,2,3], dtype=np.float32)),)` | `ValueType<float>` present (dim > 0, not direct-eligible) |
| `test_gate_vectorized_dict_keeps_struct_load` | `struct S { float x; float y; }; void apply(S s, float scale) {}` | `({"x": Tensor(...), "y": Tensor(...)}, 1.0)` | inline struct with `__slangpy_load` (children are vectorized, dim > 0) |

**Autodiff gating tests:**

| Test | Slang Source | Args | Asserts |
|------|-------------|------|---------|
| `test_gate_bwds_scalar_uses_valuetype` | `[Differentiable] float square(float x) { return x * x; }` | `func.bwds.debug_build_call_data(diffPair(2.0), diffPair(d=1.0))` | `ValueType<float>` present, `[Differentiable]` on trampoline, `bwd_diff(_trampoline)` in kernel |
| `test_gate_bwds_trampoline_is_differentiable` | same as above | same | `[Differentiable]` appears before `void _trampoline` |

**Post-implementation tests** — should pass AFTER Phase 1 is complete:

- `test_phase1_scalar_direct_bind`: verify NO `ValueType` or `__slangpy_load` for scalar args
- `test_phase1_vector_direct_bind`: verify NO `VectorValueType` for vector args
- `test_phase1_valueref_direct_bind`: verify `RWStructuredBuffer` appears directly for writable result
- `test_phase1_struct_direct_bind`: verify NO inline struct with `__slangpy_load` for dim-0 dict-to-struct
- `test_phase1_no_mapping_constants`: verify NO `_m_a`, `_m_b` for direct-bound args
- `test_phase1_functional_scalar_add`: dispatch `add(1, 2)` and verify result == 3
- `test_phase1_functional_vector_scale`: dispatch vector scale and verify result
- `test_phase1_functional_struct_sum`: dispatch struct sum via dict and verify result

---

### Implementation Order Within Phase 1

To avoid C++ crashes from the `NativeValueMarshall` fast path (Step 1.2a), the implementation order within Phase 1 must be:

1. **Step 1.2a first**: Update `NativeValueMarshall::ensure_cached` in C++ to handle direct-bind types (no `"value"` sub-field navigation). This is the only C++ change needed — `is_direct_bind_eligible` is pure Python, no nanobind changes required.
2. **Step 1.1**: Add `is_direct_bind_eligible` and `is_direct_bind_recursive` as global Python functions
3. **Steps 1.2–1.7**: Implement Python-side changes for each marshall type
4. **Step 1.4**: For struct children, set the `direct_bind` flag on each child's `NativeValueMarshall` (same per-child flag from Step 1.2a) — no changes to the C++ children dispatch path needed
5. **Step 1.8**: Autodiff support
6. **Step 1.9**: Tests

Never deploy Python-side `gen_calldata` changes that emit raw types without the corresponding C++ fast path fix — the cached `["value"]` navigation will crash at dispatch time.
