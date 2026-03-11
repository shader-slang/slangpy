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

- [Phase 1: Direct Type Marshalling](plan-simplifyKernelGen-phase1.prompt.md) — **implemented (prim-mode)**
- [Phase 2: Eliminate CallData Struct](plan-simplifyKernelGen-phase2.prompt.md)
- [Phase 3: Direct Compute Kernel Invocation](plan-simplifyKernelGen-phase3.prompt.md)

---

### Phase 1 Progress

Phase 1 prim-mode direct binding is complete. Steps 1.1–1.7, 1.9 are implemented and passing. Step 1.8 (autodiff/bwds) is deferred.

The implementation was refactored from global predicate functions (`is_direct_bind_eligible`, `is_direct_bind_recursive`) with mutable marshall state (`_force_no_direct_bind`, `_set_direct_bind_on_children`) to a marshall-driven `can_direct_bind` property + single depth-first `calculate_direct_bind` pass on the `BoundVariable` tree, following the `calculate_differentiability` pattern.

**What was done:**

| Step | Status | Summary |
|------|--------|---------|
| 1.2a | ✅ Done | C++ fast path: `NativeValueMarshall::ensure_cached` reads `binding->direct_bind()` from `NativeBoundVariableRuntime` to gate `["value"]` sub-field navigation. `m_direct_bind` **removed** from `NativeValueMarshall` — flag lives on `NativeBoundVariableRuntime`. |
| 1.1 | ✅ Done | `Marshall.can_direct_bind(binding)` virtual method (default `False`). Shared `can_direct_bind_common(binding)` helper. `BoundVariable.calculate_direct_bind()` depth-first tree pass. `calculate_direct_binding(call)` in `callsignature.py`. |
| 1.2 | ✅ Done | `ValueMarshall`: `can_direct_bind` overrides. `gen_calldata`, `gen_trampoline_load/store`, `create_calldata` read `binding.direct_bind`. |
| 1.3 | ✅ Done | `VectorMarshall`: `gen_calldata` emits raw `typealias` (e.g., `vector<float,3>`). Inherits trampoline load/store and `can_direct_bind` from `ValueMarshall`. |
| 1.4 | ✅ Done | `StructMarshall`/`BoundVariable`: `can_direct_bind` checks all children. `gen_call_data_code` uses `self.direct_bind`. Non-direct-bind composites let children retain their `direct_bind` status; `gen_call_data_code` delegates to children's `gen_trampoline_load/store`. |
| 1.5 | ✅ Done | `ValueRefMarshall`: `can_direct_bind` requires read-only access. Writable value refs (including auto-created `_result`) use wrapper path (`RWValueRef<T>`). |
| 1.6 | ✅ Done | Tensor dim-0: `gen_trampoline_load/store` extended for `ITensorType` at dim-0 (direct struct assignment). |
| 1.7 | ✅ Done | Mapping constants (`static const int _m_{name}`) skipped when `self.direct_bind`. |
| 1.8 | ⬜ Deferred | Autodiff/bwds mode still uses wrapper types. |
| 1.9 | ✅ Done | 21 tests (×3 device types = 63 cases): 16 code-gen assertion tests + 5 functional GPU dispatch tests. All pass on d3d12/vulkan/cuda. |

**Files modified:**

| File | Changes |
|------|---------|
| `src/slangpy_ext/utils/slangpy.h` | `m_direct_bind` member, getter/setter on `NativeBoundVariableRuntime` |
| `src/slangpy_ext/utils/slangpy.cpp` | Nanobind `direct_bind` property on `NativeBoundVariableRuntime` |
| `src/slangpy_ext/utils/slangpyvalue.h` | `m_direct_bind`, `direct_bind()`, `set_direct_bind()` **removed** from `NativeValueMarshall` |
| `src/slangpy_ext/utils/slangpyvalue.cpp` | `ensure_cached` reads `binding->direct_bind()`; nanobind `direct_bind` property **removed** from `NativeValueMarshall` |
| `slangpy/bindings/marshall.py` | `can_direct_bind(binding)` virtual method (default `False`) |
| `slangpy/bindings/boundvariable.py` | `can_direct_bind_common()`, `BoundVariable.direct_bind`, `calculate_direct_bind()`. Removed: `is_direct_bind_eligible`, `is_direct_bind_recursive`, `_set_direct_bind_on_children`, `_force_no_direct_bind`, `_DIRECT_BIND_TYPES`, `_clear_direct_bind()`. |
| `slangpy/bindings/boundvariableruntime.py` | `self.direct_bind = source.direct_bind` propagation |
| `slangpy/bindings/__init__.py` | Exports `can_direct_bind_common` (removed old predicate exports) |
| `slangpy/core/callsignature.py` | `calculate_direct_binding(call)` function |
| `slangpy/core/calldata.py` | `calculate_direct_binding(bindings)` call after `calculate_differentiability` |
| `slangpy/builtin/value.py` | `can_direct_bind`, `gen_calldata`, `gen_trampoline_load/store`, `create_calldata` use `binding.direct_bind` |
| `slangpy/builtin/valueref.py` | `can_direct_bind` (read-only only), all methods use `binding.direct_bind`. Removed `self._direct_bind`. |
| `slangpy/builtin/struct.py` | `can_direct_bind`, `gen_trampoline_load/store` use `binding.direct_bind` |
| `slangpy/builtin/tensorcommon.py` | `gen_trampoline_load/store` extended for `ITensorType` (unchanged in refactor) |
| `slangpy/tests/slangpy_tests/test_kernel_gen.py` | All Phase 1 tests |

**Test results:** 2952 passed / 0 failed in `slangpy/tests/slangpy_tests`. 6 pre-existing failures in `slangpy/tests/device/` (raytracing pipeline, type conformance cache — unrelated).

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

**Test infrastructure additions:**
```python
def assert_contains(code: str, *patterns: str) -> None:
    for p in patterns:
        assert p in code, f"Expected pattern not found: {p}"

def assert_not_contains(code: str, *patterns: str) -> None:
    for p in patterns:
        assert p not in code, f"Unexpected pattern found: {p}"

def generate_bwds_code(device, func_name, module_source, *args, **kwargs) -> str:
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.bwds.debug_build_call_data(*args, **kwargs)
    if PRINT_TEST_KERNEL_GEN:
        print(cd.code)
    return cd.code
```

**Summary of all gating tests by phase:**

| Phase | Gating Tests (break on implementation) | Negative Gates (must stay passing) |
|-------|---------------------------------------|-----------------------------------|
| 1 | 12 tests: scalar/float/vector/matrix/valueref-read/valueref-write/array/mapping-constants/context-map/struct-slangpy-load/bwds-scalar/bwds-trampoline | 3 tests: wanghasharg/vectorized-scalar/vectorized-dict |
| 2 | 7 tests: calldata-struct/calldata-uniform/thread-count/context-from-calldata/trampoline-present/trampoline-calls/kernel-calls-trampoline | 1 test: wanghasharg-forces-calldata |
| 3 | 1 test: compute-shader-generates-wrapper | — |

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
