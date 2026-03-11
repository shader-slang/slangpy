## Phase 1: Direct Type Marshalling

**Status**: Prim-mode complete (Steps 1.1–1.7, 1.9). Step 1.8 (autodiff derivative fields) deferred.

**Goal**: For dim-0, non-composite arguments, emit the raw Slang type in CallData and use direct assignment in the trampoline — eliminating `ValueType<T>` wrappers, `__slangpy_load`/`__slangpy_store` indirection, mapping constants, and `Context.map()` calls.

**Parent plan**: [plan-simplifyKernelGen.prompt.md](plan-simplifyKernelGen.prompt.md)

---

### Architecture

Direct binding eligibility is determined by a **marshall-driven `can_direct_bind` property** combined with a **single depth-first `calculate_direct_bind` pass** on the `BoundVariable` tree. This follows the same pattern as `calculate_differentiability`.

#### Key components

| Component | Location | Role |
|-----------|----------|------|
| `Marshall.can_direct_bind(binding)` | `slangpy/bindings/marshall.py` | Virtual method (default `False`). Marshalls override to opt in. |
| `can_direct_bind_common(binding)` | `slangpy/bindings/boundvariable.py` | Shared eligibility checks (dim-0, no children, no param block). Marshalls call this then add type-specific logic. |
| `BoundVariable.direct_bind` | `slangpy/bindings/boundvariable.py` | Boolean attribute set by `calculate_direct_bind()`. Consumed by `gen_call_data_code`, `gen_calldata`, `gen_trampoline_load/store`, `create_calldata`. |
| `BoundVariable.calculate_direct_bind()` | `slangpy/bindings/boundvariable.py` | Depth-first tree pass. Leaves delegate to `marshall.can_direct_bind()`. Composites require all children to be direct-bind AND dim-0 with a concrete vector type. If composite is NOT direct-bind, recursively clears children via `_clear_direct_bind()`. |
| `calculate_direct_binding(call)` | `slangpy/core/callsignature.py` | Top-level function iterating `call.args` + `call.kwargs.values()`, calling `arg.calculate_direct_bind()`. |
| `NativeBoundVariableRuntime.direct_bind` | `slangpy.h` / `boundvariableruntime.py` | C++ member + Python propagation. Read by `NativeValueMarshall::ensure_cached` to gate `["value"]` sub-field navigation. |

#### Control flow

```
CallData.build()
  → calculate_differentiability(context, bindings)
  → calculate_direct_binding(bindings)           ← NEW
  → generate_code(...)
      → gen_call_data_code()    — reads binding.direct_bind
      → gen_trampoline()        — reads binding.direct_bind
  → BoundCallRuntime(bindings)  — propagates binding.direct_bind to C++ runtime
```

At dispatch time, `NativeValueMarshall::ensure_cached()` reads `binding->direct_bind()` to decide cursor navigation:
- `direct_bind == false`: `cursor[variable_name]["value"]` (wrapper path)
- `direct_bind == true`: `cursor[variable_name]` (raw type path)

#### Composite (struct/dict) handling

When `calculate_direct_bind()` visits a composite node:
1. Recurse children first (depth-first)
2. If all children have `direct_bind == True` AND the composite is dim-0 with a concrete vector type → set `self.direct_bind = True`
3. Otherwise → call `_clear_direct_bind()` on all children, forcing them to use wrapper types. This is necessary because the parent's generated `__slangpy_load`/`__slangpy_store` expects children to have wrapper types (e.g., `ValueType<float>`). A child emitting raw `float` inside a parent that emits `__slangpy_load` would produce invalid Slang.

---

### Step 1.1: Define eligibility predicate

**Implemented.** A `can_direct_bind(binding)` virtual method on `Marshall` (default `False`) replaces the original `is_direct_bind_eligible` / `is_direct_bind_recursive` global functions. Each marshall subclass overrides `can_direct_bind` to opt in.

A shared helper `can_direct_bind_common(binding)` in `boundvariable.py` provides the common checks:
- `binding.call_dimensionality is not None and binding.call_dimensionality == 0`
- `not binding.children` (not composite/dict)
- `not getattr(binding, "create_param_block", False)` (excludes `PackedArg`)

Marshall subclasses call `can_direct_bind_common(binding)` and optionally add type-specific logic. `StructMarshall` has its own implementation: if it has children, all children must have `direct_bind == True`; otherwise it delegates to `can_direct_bind_common`.

---

### Step 1.2: Implement for `ValueMarshall` (scalars/matrices)

**Implemented.** In [slangpy/builtin/value.py](slangpy/builtin/value.py):

- `can_direct_bind(binding)`: calls `can_direct_bind_common(binding)`
- `gen_calldata`: when `binding.direct_bind`, emits `typealias _t_{name} = {raw_slang_type}` instead of `ValueType<{type}>`
- `gen_trampoline_load`: when `binding.direct_bind`, emits `{name} = {data_name}` and returns `True`
- `gen_trampoline_store`: when `binding.direct_bind` (read-only), returns `True` (suppress default store)
- `create_calldata`: when `binding.direct_bind`, returns raw value instead of `{"value": data}`

#### Step 1.2a: C++ fast path

**Implemented.** `NativeValueMarshall::ensure_cached` in [slangpyvalue.cpp](src/slangpy_ext/utils/slangpyvalue.cpp) reads `binding->direct_bind()` from the `NativeBoundVariableRuntime`:

```cpp
ShaderCursor field = binding->direct_bind()
    ? cursor[binding->variable_name()]
    : cursor[binding->variable_name()]["value"];
```

The `direct_bind` flag is a `bool` member on `NativeBoundVariableRuntime` (declared in [slangpy.h](src/slangpy_ext/utils/slangpy.h)), exposed via nanobind property in [slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp), and propagated from `BoundVariable.direct_bind` via [boundvariableruntime.py](slangpy/bindings/boundvariableruntime.py).

The `m_direct_bind` / `direct_bind` / `set_direct_bind` members were **removed** from `NativeValueMarshall` — the flag lives exclusively on `NativeBoundVariableRuntime`.

---

### Step 1.3: Implement for `VectorMarshall`, `MatrixMarshall`, and `ArrayMarshall`

**Implemented.** All inherit `can_direct_bind` and `gen_trampoline_load`/`gen_trampoline_store` from `ValueMarshall`. `VectorMarshall` overrides `gen_calldata` to emit the raw vector type (e.g., `vector<float,3>`) instead of `VectorValueType<float,3>` when `binding.direct_bind`. `MatrixMarshall` and `ArrayMarshall` (at dim-0) inherit `ValueMarshall.gen_calldata`.

---

### Step 1.4: Implement for `StructMarshall` (dict → struct)

**Implemented.** In [slangpy/builtin/struct.py](slangpy/builtin/struct.py):

- `can_direct_bind(binding)`: if `binding.children is not None`, returns `True` only if all children have `direct_bind == True`. Otherwise delegates to `can_direct_bind_common(binding)`.
- `gen_trampoline_load`: when `binding.direct_bind`, emits `{name} = {data_name}` and returns `True`
- `gen_trampoline_store`: when `binding.direct_bind`, emits `{data_name} = {name}` for writable and returns `True`

In [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py), `gen_call_data_code`:
- When `self.direct_bind`, emits `typealias _t_{name} = {vector_type.full_name}` (raw struct type) — skipping inline struct generation, `__slangpy_load`/`__slangpy_store`, and child type aliases.
- When NOT `self.direct_bind`, uses the standard children path with inline struct.

Children inside non-direct-bind composites have their `direct_bind` cleared by `_clear_direct_bind()` during `calculate_direct_bind`. This ensures children use wrapper types compatible with the parent's `__slangpy_load`/`__slangpy_store`.

---

### Step 1.5: Implement for `ValueRefMarshall`

**Implemented.** In [slangpy/builtin/valueref.py](slangpy/builtin/valueref.py):

- `can_direct_bind(binding)`: calls `can_direct_bind_common(binding)`
- `gen_calldata`: when `binding.direct_bind`, read-only emits raw type, writable emits `RWStructuredBuffer<T>`
- `gen_trampoline_load/store`: when `binding.direct_bind`, read-only does direct assignment, writable does `[0]` indexing
- `create_calldata` / `read_calldata`: when `binding.direct_bind`, skip `{"value": ...}` wrapper

The old `self._direct_bind` attribute was **removed** — all checks now use `binding.direct_bind`.

---

### Step 1.6: Implement for tensor marshalls

**Implemented.** In [slangpy/builtin/tensorcommon.py](slangpy/builtin/tensorcommon.py):

`gen_trampoline_load/store` extended for `ITensorType` at dim-0 (direct struct assignment). Tensor marshalls do NOT implement `can_direct_bind` — tensor dim-0 handling is done via trampoline-level checks on `binding.call_dimensionality` and `binding.vector_type` type, independent of the `direct_bind` flag.

---

### Step 1.7: Eliminate unused boilerplate in code generation

**Implemented.** In [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py), `gen_call_data_code` skips emitting `static const int _m_{name} = 0` mapping constants when `self.direct_bind` is `True`.

---

### Step 1.8: Handle autodiff (bwds mode)

⬜ **Deferred.** Prim-mode direct binding applies to bwds primals (code gen verified), but derivative fields still use the old `ValueType` wrapper path.

---

### Step 1.9: Tests

**Implemented.** 21 tests × 3 device types = 63 cases. All pass on d3d12/vulkan/cuda.

---

### Files Modified

| File | Changes |
|------|---------|
| `src/slangpy_ext/utils/slangpy.h` | `m_direct_bind` member, `direct_bind()` getter, `set_direct_bind()` setter on `NativeBoundVariableRuntime` |
| `src/slangpy_ext/utils/slangpy.cpp` | Nanobind `direct_bind` property on `NativeBoundVariableRuntime` |
| `src/slangpy_ext/utils/slangpyvalue.h` | `m_direct_bind`, `direct_bind()`, `set_direct_bind()` **removed** from `NativeValueMarshall` |
| `src/slangpy_ext/utils/slangpyvalue.cpp` | `ensure_cached` reads `binding->direct_bind()` instead of `m_direct_bind`; nanobind `direct_bind` property **removed** from `NativeValueMarshall` |
| `slangpy/bindings/marshall.py` | `can_direct_bind(binding)` virtual method (default `False`) |
| `slangpy/bindings/boundvariable.py` | `can_direct_bind_common()`, `BoundVariable.direct_bind` attribute, `BoundVariable.calculate_direct_bind()`, `BoundVariable._clear_direct_bind()`. Old functions removed: `is_direct_bind_eligible`, `is_direct_bind_recursive`, `_set_direct_bind_on_children`, `_force_no_direct_bind`, `_DIRECT_BIND_TYPES`. |
| `slangpy/bindings/boundvariableruntime.py` | `self.direct_bind = source.direct_bind` propagation |
| `slangpy/bindings/__init__.py` | Exports `can_direct_bind_common` (removed `is_direct_bind_eligible`, `is_direct_bind_recursive`) |
| `slangpy/core/callsignature.py` | `calculate_direct_binding(call)` function |
| `slangpy/core/calldata.py` | `calculate_direct_binding(bindings)` call after `calculate_differentiability` |
| `slangpy/builtin/value.py` | `can_direct_bind`, `gen_calldata`, `gen_trampoline_load`, `gen_trampoline_store`, `create_calldata` use `binding.direct_bind`. Removed `self.direct_bind` on marshall. |
| `slangpy/builtin/valueref.py` | `can_direct_bind`, `gen_calldata`, `gen_trampoline_load`, `gen_trampoline_store`, `create_calldata`, `read_calldata` use `binding.direct_bind`. Removed `self._direct_bind`. |
| `slangpy/builtin/struct.py` | `can_direct_bind`, `gen_trampoline_load`, `gen_trampoline_store` use `binding.direct_bind` |
| `slangpy/builtin/tensorcommon.py` | `gen_trampoline_load`, `gen_trampoline_store` extended for `ITensorType` dim-0 (unchanged in refactor) |
| `slangpy/tests/slangpy_tests/test_kernel_gen.py` | All Phase 1 tests |

### Test Results

2952 passed / 0 failed in `slangpy/tests/slangpy_tests`. 6 pre-existing failures in `slangpy/tests/device/` (raytracing pipeline, type conformance cache — unrelated).

### Design Decisions

**`direct_bind` lives on `NativeBoundVariableRuntime`, not `NativeValueMarshall`.** The original implementation stored `m_direct_bind` on the marshall itself (`NativeValueMarshall`), but marshalls are shared across calls while bindings are per-call. Moving the flag to the binding makes it immutable per-call and eliminates mutable state on shared marshall instances.

**Marshall-driven `can_direct_bind` replaces hardcoded type list.** The original `is_direct_bind_eligible` used a lazily-populated `_DIRECT_BIND_TYPES` tuple to check marshall type. The new design uses a virtual method — each marshall opts in explicitly. Adding a new direct-bind-eligible type requires only overriding `can_direct_bind` on the new class.

**Single `calculate_direct_bind` pass replaces repeated predicate calls.** The original `is_direct_bind_eligible` / `is_direct_bind_recursive` were called multiple times per variable during code gen. The new design computes `direct_bind` once in a single tree pass after `calculate_differentiability`, and consumers simply read the boolean.

**`_clear_direct_bind` replaces `_force_no_direct_bind`.** When a composite struct is NOT direct-bind-eligible (e.g., has vectorized children), its children must NOT use direct binding either — the parent's generated `__slangpy_load`/`__slangpy_store` expects children to have wrapper types. The old implementation set `_force_no_direct_bind = True` on children during code gen. The new implementation clears `direct_bind` recursively during the `calculate_direct_bind` pass itself, before code gen runs.
