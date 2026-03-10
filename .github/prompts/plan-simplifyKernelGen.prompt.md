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

- [Phase 1: Direct Type Marshalling](plan-simplifyKernelGen-phase1.prompt.md)
- [Phase 2: Eliminate CallData Struct](plan-simplifyKernelGen-phase2.prompt.md)
- [Phase 3: Direct Compute Kernel Invocation](plan-simplifyKernelGen-phase3.prompt.md)

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
