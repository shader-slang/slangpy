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

**What was done:**

| Step | Status | Summary |
|------|--------|---------|
| 1.2a | ✅ Done | `NativeValueMarshall` C++ fast path: `m_direct_bind` flag gates `["value"]` sub-field navigation in `ensure_cached`. Exposed via nanobind `direct_bind` property. |
| 1.1 | ✅ Done | `is_direct_bind_eligible()` and `is_direct_bind_recursive()` in `boundvariable.py`. Excludes `PackedArg` bindings and children inside non-direct-bind structs (via `_force_no_direct_bind` flag). |
| 1.2 | ✅ Done | `ValueMarshall`: `gen_calldata` emits raw `typealias`; `gen_trampoline_load/store` do direct assignment; `create_calldata` returns raw value. `ScalarMarshall`/`MatrixMarshall` inherit. |
| 1.3 | ✅ Done | `VectorMarshall`: `gen_calldata` emits raw `typealias` (e.g., `vector<float,3>`). Inherits trampoline load/store from `ValueMarshall`. |
| 1.4 | ✅ Done | `StructMarshall`/`BoundVariable`: `gen_call_data_code` children path emits `typealias _t_{name} = {struct_type}` when `is_direct_bind_recursive`. Sets `direct_bind` on child marshalls via `_set_direct_bind_on_children`. Non-direct-bind structs set `_force_no_direct_bind` on children to prevent incorrect leaf optimization. `gen_trampoline_load/store` added. |
| 1.5 | ✅ Done | `ValueRefMarshall`: read-only emits raw type + direct assignment; writable emits `RWStructuredBuffer<T>` + `[0]` load/store. `create_calldata`/`read_calldata` skip `{"value": ...}` wrapper when direct-eligible. |
| 1.6 | ✅ Done | Tensor dim-0: `gen_trampoline_load/store` extended for `ITensorType` at dim-0 (direct struct assignment). |
| 1.7 | ✅ Done | Mapping constants (`static const int _m_{name}`) skipped for direct-bind-eligible variables. |
| 1.8 | ⬜ Deferred | Autodiff/bwds mode still uses wrapper types. Prim-mode direct binding does apply to bwds primals (code gen verified), but derivative fields still use the old path. |
| 1.9 | ✅ Done | 21 tests (×3 device types = 63 cases): 16 code-gen assertion tests + 5 functional GPU dispatch tests. All pass on d3d12/vulkan/cuda. |

**Files modified:**

| File | Changes |
|------|---------|
| `src/slangpy_ext/utils/slangpyvalue.h` | `m_direct_bind` flag, getter/setter |
| `src/slangpy_ext/utils/slangpyvalue.cpp` | `ensure_cached` direct-bind branch; nanobind export |
| `slangpy/bindings/boundvariable.py` | `is_direct_bind_eligible`, `is_direct_bind_recursive`, `_set_direct_bind_on_children`, `_force_no_direct_bind`, mapping-constant skip in `gen_call_data_code` |
| `slangpy/bindings/__init__.py` | Exports for predicates |
| `slangpy/builtin/value.py` | `gen_calldata`, `gen_trampoline_load`, `gen_trampoline_store`, `create_calldata` |
| `slangpy/builtin/valueref.py` | `gen_calldata`, `gen_trampoline_load`, `gen_trampoline_store`, `create_calldata`, `read_calldata` |
| `slangpy/builtin/struct.py` | `gen_trampoline_load`, `gen_trampoline_store` |
| `slangpy/builtin/tensorcommon.py` | `gen_trampoline_load`, `gen_trampoline_store` extended for `ITensorType` |
| `slangpy/tests/slangpy_tests/test_kernel_gen.py` | All Phase 1 tests |

**Test results:** 2952 passed / 0 failed in `slangpy/tests/slangpy_tests`. 6 pre-existing failures in `slangpy/tests/device/` (raytracing pipeline, type conformance cache — unrelated).

**Implementation note — `_force_no_direct_bind`:** The plan did not anticipate that children inside non-direct-bind composite structs (mixed dim-0/dim-N children) would incorrectly inherit direct binding from their leaf predicates. A `_force_no_direct_bind` flag was added: when `gen_call_data_code` takes the non-direct-bind struct path, it marks all children so their `is_direct_bind_eligible`/`is_direct_bind_recursive` return `False`. This prevents generating e.g. `typealias _t_velocity = vector<float,2>` for a child inside a struct that still uses `__slangpy_load`. Similarly, `PackedArg` bindings (`create_param_block = True`) are excluded since `ParameterBlock<int>` is invalid in Slang.

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
- All dim-0 non-composite types are eligible, including tensors and value refs
- Phase 2 targets both `entry_point` (CUDA) and `global_data` (Vulkan/D3D12) modes
- Autograd (bwds mode) is included in simplification, but implemented after prim mode within each phase
- WangHashArg explicitly excluded from direct binding (needs per-thread `thread_id` computation)
