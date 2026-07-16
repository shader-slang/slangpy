## Plan: Simplify Generated SlangPy Kernels

**TL;DR**: A three-phase effort to make generated kernels resemble hand-written GPU code. Phase 1 adds direct type marshalling (bypassing `ValueType<T>` wrappers and `__slangpy_load`/`__slangpy_store`) for dim-0 non-composite types, following the pattern already used by `TensorView`. Phase 2 eliminates the `CallData` struct when all arguments are direct-eligible, passing them as individual uniforms/globals. Phase 3 enables calling pre-written compute kernels directly without generating wrapper shaders.

**Target example** — `add(int a, int b) -> int` with scalar args should go from 40+ lines of boilerplate to approximately:

```slang
import "module";
[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(int3 tid: SV_DispatchThreadID, uniform uint3 _thread_count, uniform int a, uniform int b, uniform RWStructuredBuffer<int> _result)
{
    if (any(tid >= _thread_count)) return;
    _result[0] = add(a, b);
}
```

---

### Phase Plans

- [Phase 1: Direct Type Marshalling](plan-simplifyKernelGen-phase1.prompt.md) — **✅ merged (PR #863)**
- [Phase 2: Eliminate CallData Struct](plan-simplifyKernelGen-phase2.prompt.md) — **not started**
- [Phase 3: Direct Compute Kernel Invocation](plan-simplifyKernelGen-phase3.prompt.md) — **not started**

---

### Phase 1 Summary (Complete — PR #863)

Phase 1 introduced **direct binding**: dim-0 arguments that can be bound using raw Slang types instead of `ValueType<T>` wrappers, eliminating `__slangpy_load`/`__slangpy_store` indirection, `Context.map()` calls, and mapping constants for eligible arguments. PR #863 was merged to `main` on 2026-03-11 (+2,044 / −122 lines, 18 files changed, squash-merged).

#### What Phase 1 Changed

**Architecture**: A marshall-driven `can_direct_bind(binding)` virtual method (default `False`) combined with a single depth-first `calculate_direct_bind()` pass on the `BoundVariable` tree. This follows the same pattern as `calculate_differentiability`. The `direct_bind` boolean is stored on `BoundVariable` (Python) and propagated to `NativeBoundVariableRuntime` (C++).

**Eligibility**: A variable is direct-bind eligible if:
- `call_dimensionality == 0` (not vectorized)
- Not composite with children (unless all children are also direct-bind AND the composite is dim-0 with a concrete Slang struct type and read-only access)
- Not a param block (`PackedArg`)
- The marshall opts in via `can_direct_bind()` override
- For `ValueRefMarshall`: `access[0] == AccessType.read` (writable value refs need buffer logic)

**Code generation effects** — when `binding.direct_bind == True`:
- `gen_calldata` emits `typealias _t_{name} = {raw_slang_type}` instead of `ValueType<T>` / `VectorValueType<T,N>` / `ValueRef<T>`
- `gen_trampoline_load` emits `{value_name} = {data_name};` (direct assignment) instead of `{data_name}.__slangpy_load(context.map(_m_{name}), {name})`
- `gen_trampoline_store` returns `True` (suppresses store for read-only types)
- Mapping constants (`static const int _m_{name} = 0`) are skipped
- `create_calldata` returns the raw value instead of `{"value": data}`

**C++ fast path**: `NativeValueMarshall::ensure_cached` reads `binding->direct_bind()` to decide cursor navigation — `cursor[variable_name]` for direct-bind vs `cursor[variable_name]["value"]` for wrapper path.

**Composite (struct/dict) handling**: When `calculate_direct_bind()` visits a composite, it recurses children first. If all children are direct-bind AND the composite is dim-0 with a concrete vector type and read-only access → the composite itself is direct-bind (emits raw `typealias`). Otherwise the composite is NOT direct-bind, but children **retain** their individual `direct_bind` status — the parent's `__slangpy_load`/`__slangpy_store` body uses `gen_trampoline_load`/`gen_trampoline_store` for each child, so direct-bind children get direct assignment (e.g., `value.y = y;`) while non-direct-bind children use `__slangpy_load(context.map(...))`.

**API changes to `gen_trampoline_load`/`gen_trampoline_store`**: Signature changed from `(cgb, binding, is_entry_point)` → `(cgb, binding, data_name, value_name)`. The caller now computes `data_name` (e.g., `__calldata__.x` or `call_data.x`) and `value_name` (e.g., `x` or `value.x`), allowing these methods to work both at the root trampoline level and inside composite `__slangpy_load`/`__slangpy_store` bodies.

**`read_output` fix** (C++): `NativeBoundVariableRuntime::read_output` was simplified — composites no longer attempt to read output directly (it is handled by their children). The old composite branch had a logic error (checking `res.contains(name)` before insertion).

#### Control Flow (post-Phase 1)

```
CallData.build()
  → calculate_differentiability(context, bindings)
  → calculate_direct_binding(bindings)           ← Phase 1
  → generate_code(...)
      → gen_call_data_code()    — reads binding.direct_bind
      → gen_trampoline()        — reads binding.direct_bind
  → BoundCallRuntime(bindings)  — propagates binding.direct_bind to C++ runtime
```

At dispatch time, `NativeValueMarshall::ensure_cached()` reads `binding->direct_bind()` to decide cursor navigation:
- `direct_bind == false`: `cursor[variable_name]["value"]` (wrapper path)
- `direct_bind == true`: `cursor[variable_name]` (raw type path)

#### Implemented Steps

| Step | Status | Summary |
|------|--------|---------|
| 1.1 | ✅ Done | `Marshall.can_direct_bind(binding)` virtual method. `can_direct_bind_common(binding)` helper. `BoundVariable.calculate_direct_bind()` depth-first tree pass. `calculate_direct_binding(call)` in `callsignature.py`. |
| 1.2 | ✅ Done | `ValueMarshall`: `can_direct_bind`, `gen_calldata`, `gen_trampoline_load/store` read `binding.direct_bind`. |
| 1.2a | ✅ Done | C++ fast path: `NativeValueMarshall::ensure_cached` reads `binding->direct_bind()` from `NativeBoundVariableRuntime`. `m_direct_bind` **removed** from `NativeValueMarshall`. |
| 1.3 | ✅ Done | `VectorMarshall`/`MatrixMarshall`/`ArrayMarshall`: inherit from `ValueMarshall`. `VectorMarshall.gen_calldata` emits raw vector type (e.g., `vector<float,3>`). |
| 1.4 | ✅ Done | `StructMarshall`: `can_direct_bind` checks all children. `BoundVariable.gen_call_data_code` uses `self.direct_bind`. Non-direct-bind composites delegate to children's `gen_trampoline_load/store`. |
| 1.5 | ✅ Done | `ValueRefMarshall`: `can_direct_bind` requires `access[0] == AccessType.read`. Writable value refs (including auto-created `_result`) use `RWValueRef<T>`. |
| 1.6 | ✅ Done | Tensor dim-0: `can_direct_bind` added to `tensorcommon.py`. `gen_trampoline_load/store` extended for dim-0 tensors (`ITensorType`, `TensorViewType`, `DiffTensorViewType`). |
| 1.7 | ✅ Done | Mapping constants (`static const int _m_{name}`) skipped when `self.direct_bind`. |
| 1.8 | ⬜ Deferred | Autodiff derivative fields still use `ValueType` wrappers. Bwds primals use direct bind. |
| 1.9 | ✅ Done | 77 tests (×3 device types = 231 cases). All pass on d3d12/vulkan/cuda. |

#### Files Modified (PR #863)

| File | Changes |
|------|---------|
| `src/slangpy_ext/utils/slangpy.h` | `m_direct_bind` member, `direct_bind()`, `set_direct_bind()` on `NativeBoundVariableRuntime` |
| `src/slangpy_ext/utils/slangpy.cpp` | Nanobind `direct_bind` r/w property on `NativeBoundVariableRuntime`. `read_output` composite branch simplified. |
| `src/slangpy_ext/utils/slangpyvalue.h` | `CachedValueWrite.direct_bind` field added. `m_direct_bind`/`direct_bind()`/`set_direct_bind()` **removed** from `NativeValueMarshall`. |
| `src/slangpy_ext/utils/slangpyvalue.cpp` | `ensure_cached` reads `binding->direct_bind()` for cursor path; caches `direct_bind` value. |
| `slangpy/bindings/marshall.py` | `can_direct_bind(binding)` virtual method (default `False`). `gen_trampoline_load/store` signature changed to `(cgb, binding, data_name, value_name)`. |
| `slangpy/bindings/boundvariable.py` | `can_direct_bind_common()` helper. `BoundVariable.direct_bind` attribute. `BoundVariable.calculate_direct_bind()` method. `gen_call_data_code` handles direct-bind composites (raw typealias) and delegates to children's `gen_trampoline_load/store`. Mapping constant emission gated on `not self.direct_bind`. |
| `slangpy/bindings/boundvariableruntime.py` | `self.direct_bind = source.direct_bind` propagation to C++ runtime. |
| `slangpy/bindings/__init__.py` | Exports `can_direct_bind_common`. |
| `slangpy/core/callsignature.py` | `calculate_direct_binding(call)` function. Trampoline code gen refactored: `data_name` computed before `gen_trampoline_load` call. Store path moved after `data_name` computation. |
| `slangpy/core/calldata.py` | `calculate_direct_binding(bindings)` call after `calculate_differentiability`. `self.code = code` stored for debugging. |
| `slangpy/builtin/value.py` | `can_direct_bind`, `gen_trampoline_load`, `gen_trampoline_store` added. `gen_calldata` gates on `binding.direct_bind`. |
| `slangpy/builtin/valueref.py` | `can_direct_bind` (read-only gate), `gen_trampoline_load`, `gen_trampoline_store` added. `gen_calldata`, `create_calldata`, `read_calldata` gate on `binding.direct_bind`. `self._direct_bind` removed. |
| `slangpy/builtin/struct.py` | `can_direct_bind` (children check + `AccessType.read` gate). `gen_trampoline_load`, `gen_trampoline_store` delegate to `ValueMarshall` when direct-bind. |
| `slangpy/builtin/tensor.py` | `can_direct_bind` delegates to `tensorcommon`. `gen_trampoline_load/store` signature updated. |
| `slangpy/builtin/tensorcommon.py` | `can_direct_bind()` function added. `gen_trampoline_load/store` signature changed, condition changed from `isinstance(vector_type, TensorViewType)` to `binding.direct_bind`. |
| `slangpy/torchintegration/torchtensormarshall.py` | `can_direct_bind` delegates to `tensorcommon`. `gen_trampoline_load/store` signature updated. |
| `slangpy/benchmarks/test_benchmark_autograd.py` | Removed accidental blank line (1-line whitespace change). |
| `slangpy/tests/slangpy_tests/test_kernel_gen.py` | New file: 77 tests covering all Phase 1 scenarios. |

#### Test Coverage Summary

The test file (`test_kernel_gen.py`) provides 77 test functions × 3 device types = 231 parametrized cases covering:

**Code-gen assertion tests** (`test_gate_*`): Verify generated Slang code patterns — type aliases, trampoline load/store statements, mapping constants, wrapper types, `__slangpy_load`/`__slangpy_store` presence/absence.

**Binding flag tests**: Verify `direct_bind`, `call_dimensionality`, and `vector_type` on `BoundVariable` instances for: scalars, vectors, tensors (dim-0 and vectorized), structs (all-scalar, mixed, nested, deeply nested), writable ValueRef, auto-created `_result`, WangHashArg, bwds primal args.

**Functional GPU dispatch tests** (`test_phase1_functional_*`): End-to-end dispatch verifying correct GPU results for: scalar add/mul, vector scale, struct sum, ValueRef write, mixed scalar+tensor, mixed struct fields, tensor dim-0, 2D/3D tensor→vector, 2D tensor→scalar, 2D tensor→array, nested/deeply-nested structs, struct with matrix/vector/array fields, struct return types, struct with vectorized 2D tensor child.

**Negative gates** (`test_gate_*_keeps_*`): Verify types that are NOT direct-bind eligible remain using wrappers: WangHashArg, vectorized scalar (dim > 0), vectorized dict.

**Helper infrastructure**: `assert_contains`, `assert_not_contains`, `assert_trampoline_has`, `generate_code`, `generate_bwds_code`.

#### Known Issues (from review, not yet addressed)

1. **`set_direct_bind` exposed as read-write nanobind property** — After first dispatch, mutating `direct_bind` would invalidate the cached cursor offset. Consider making it read-only.

2. **C++ cache safety** — `NativeValueMarshall::ensure_cached` caches `direct_bind` but has no debug assertion verifying it matches on subsequent calls.

3. **Dead `binding.direct_bind` checks in writable ValueRef paths** — `create_calldata` and `read_calldata` in `valueref.py` have `assert not binding.direct_bind` in writable code paths (reachable only as assertions, since `can_direct_bind` rejects non-read access).

---

### What Phase 2 Needs to Know

Phase 2 builds on Phase 1's `direct_bind` infrastructure. Key context for implementation:

**Current kernel structure** (post-Phase 1, for `int add(int a, int b)` with args `(1, 2)`):
```slang
import "module";
import "slangpy";
// CallData struct with per-arg type aliases and mapping constants
struct CallData {
    typealias _t_a = int;         // Phase 1: raw type (was ValueType<int>)
    _t_a a;
    typealias _t_b = int;         // Phase 1: raw type (was ValueType<int>)
    _t_b b;
    typealias _t__result = RWValueRef<int>;  // writable _result still wrapped
    _t__result _result;
    static const int _m__result = 0;         // mapping constant only for _result
    uint3 _thread_count;
    // ... shape arrays if call_data_len > 0 ...
};
void _trampoline(CallData call_data /*or __calldata__ on CUDA*/) {
    int a;
    a = call_data.a;              // Phase 1: direct assignment (was __slangpy_load)
    int b;
    b = call_data.b;              // Phase 1: direct assignment
    int _result;
    _result = add(a, b);
    call_data._result.__slangpy_store(__slangpy_context__.map(_m__result), _result);
}
[shader("compute")] [numthreads(32,1,1)]
void compute_main(..., uniform CallData call_data) {
    // thread bounds check, context construction
    _trampoline(call_data);
}
```

**Phase 2 goal**: Eliminate the `CallData` struct entirely when ALL args are direct-bind eligible. Pass args as individual `uniform` parameters on the entry point. Inline the function call into `compute_main` (skip trampoline for prim mode).

**Blocking issue for Phase 2**: Auto-created `_result` is a writable `ValueRef` → NOT direct-bind (needs `RWValueRef<T>` wrapper with buffer). Phase 2 must either:
- Accept that `_result` prevents full CallData elimination for functions with return values, and use a hybrid approach (direct args + `_result` in CallData or as a separate `RWStructuredBuffer` entry point param), OR
- Add a new code path for `_result` that emits `uniform RWStructuredBuffer<T> _result` as an entry point param with `_result[0] = ...` for the store

**Key files for Phase 2**:
- `slangpy/core/callsignature.py` — `generate_code()` builds the trampoline and compute_main
- `slangpy/core/calldata.py` — `CallData.build()` orchestrates the pipeline
- `slangpy/bindings/codegen.py` — `CodeGen` class manages `call_data_structs` block
- `src/slangpy_ext/utils/slangpy.cpp` — `NativeCallData::exec()` dispatches; cursor navigation for uniforms

**`BoundVariable.direct_bind`** is already computed for all args by Phase 1. Phase 2 can check `all(arg.direct_bind for arg in all_args)` to decide whether to use the direct-args path.

**Entry point parameter precedent**: See `slangpy/tests/device/test_pipeline_utils.slang` — manually written compute shaders already use individual `uniform` entry point params on all backends (CUDA, Vulkan, D3D12).

**Design decisions deferred to Phase 2**:
- Whether to support hybrid kernels (some args as entry-point params, some in CallData) or only all-or-nothing
- Handle entry-point parameter size limits (CUDA ~4KB root constants, D3D12 64 DWORD root signature limit)
- Whether to inline the function call directly in compute_main for prim mode, or keep a simplified trampoline

---

### Gating Tests — Pre-Implementation Checklist

Before implementing any phase, add **gating tests** to [slangpy/tests/slangpy_tests/test_kernel_gen.py](slangpy/tests/slangpy_tests/test_kernel_gen.py) that assert the CURRENT generated kernel patterns. These tests document the baseline and will intentionally break as each simplification step is implemented.

**Design principles:**
- All gating tests are code-generation-only (no GPU dispatch) — fast and deterministic
- All tests use the existing `generate_code()` helper → `func.debug_build_call_data()` → `cd.code`
- Tests are parametrized across `helpers.DEFAULT_DEVICE_TYPES`
- String matching (substring checks) rather than regex or golden files
- Named `test_gate_*` for easy identification
- WangHashArg and dict/composite tests serve as "negative gates" — they remain passing after simplification

**Test infrastructure** (already present in `test_kernel_gen.py`):
```python
def assert_contains(code: str, *patterns: str) -> None
def assert_not_contains(code: str, *patterns: str) -> None
def assert_trampoline_has(code: str, *stmts: str) -> None
def generate_code(device, func_name, module_source, *args, **kwargs) -> str
def generate_bwds_code(device, func_name, module_source, *args, **kwargs) -> str
```

**Phase 2 gating tests to add** (assert CURRENT behavior, will break on implementation):

| Test | Asserts (current behavior) | Breaks when |
|------|---------------------------|-------------|
| `test_gate_calldata_struct_present` | `struct CallData` present | Step 2.1 |
| `test_gate_calldata_uniform_param` | `uniform CallData call_data` in `compute_main` | Step 2.2 |
| `test_gate_thread_count_in_calldata` | `call_data._thread_count` in kernel body | Step 2.4 |
| `test_gate_context_from_calldata` | `Context __slangpy_context__` present | Step 2.4 |
| `test_gate_trampoline_present_for_prim` | `void _trampoline(` present | Step 2.5 |
| `test_gate_trampoline_calls_function` | `_result = add(a, b)` inside trampoline | Step 2.5 |
| `test_gate_kernel_calls_trampoline` | `_trampoline(` inside `compute_main` | Step 2.5 |
| `test_gate_wanghasharg_forces_calldata` (negative) | `struct CallData` present with non-eligible arg | Must stay passing |

---

### Verification (all phases)

```bash
# Build first (required)
cmake --build --preset windows-msvc-debug

# Run kernel gen tests
$env:PRINT_TEST_KERNEL_GEN="1"; pytest slangpy/tests/slangpy_tests/test_kernel_gen.py -v

# Run full test suite
pytest slangpy/tests -v

# Run pre-commit
pre-commit run --all-files
```

### Key Decisions

- Phase 1 changes both `gen_calldata` and trampoline load/store (TensorView-complete pattern, not partial)
- All dim-0 non-composite types are eligible, excluding writable value refs (which need buffer logic)
- Phase 2 targets both `entry_point` (CUDA) and `global_data` (Vulkan/D3D12) modes
- Autograd (bwds mode) is included in simplification, but implemented after prim mode within each phase
- WangHashArg explicitly excluded from direct binding (needs per-thread `thread_id` computation)
