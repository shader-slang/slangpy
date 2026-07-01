## Phase 2: Eliminate CallData Struct

**Goal**: Move kernel uniforms out of the `CallData` struct into individual entry-point parameters. Eliminate the trampoline in forward (prim) mode. Fall back to `ParameterBlock<CallData>` when total inline-uniform size exceeds a runtime per-device threshold.

**Parent plan**: [plan-simplifyKernelGen.prompt.md](plan-simplifyKernelGen.prompt.md)

**Status**: Steps 2.0–2.2 and 2.4 complete. Step 2.3 (trampoline elimination for prim mode) not started. Code generation logic has been extracted from `callsignature.py` into [generator.py](slangpy/core/generator.py) (see [plan-extractCodegenToGenerator.prompt.md](plan-extractCodegenToGenerator.prompt.md)).

---

### Key Architectural Decisions

These decisions correct several assumptions in the original plan:

1. **Entry-point param placement is orthogonal to `direct_bind`.** Any type — wrapped or raw — can be an entry-point parameter (e.g., `uniform ValueType<int> a` or `uniform int a` or `uniform Tensor<float,2> t`). `direct_bind` governs whether `__slangpy_load`/`__slangpy_store` is needed inside the kernel; entry-point placement governs where the uniform lives in the shader layout.

2. **Trampoline elimination is independent of `direct_bind`.** The current trampoline body is: declare locals → load (direct assignment or `__slangpy_load`) → call function → store (`__slangpy_store`). All of that can appear directly in `compute_main`. The trampoline only exists because bwds mode needs a `[Differentiable]` wrapper for `bwd_diff()`. In prim mode, it is eliminated regardless of whether args use wrappers.

3. **All-or-nothing fallback.** When total inline-uniform size exceeds the platform threshold, ALL args go back into `ParameterBlock<CallData>` (the current path). No hybrid mixing of entry-point params and CallData.

4. **Shape arrays and `_thread_count` obey the same rules** as user args — they become entry-point params by default, and go into `CallData` on fallback. Phase 2 is NOT scoped only to `call_data_len == 0`.

5. **Two code paths based on where data lives:**
   - **Fast path** (entry-point params): In Slang, uniforms are entry-point parameters and can be used directly (in forward) or passed directly to the trampoline (in backward).
   - **Fallback path** (`ParameterBlock<CallData>`): In Slang, uniforms live in a `CallData` struct. They must be read into local variables before being used (in forward) or passed to the trampoline (in backward). This is the current behavior.

6. **C++ dispatch changes are isolated to `NativeCallData::exec`.** Marshalls receive a `ShaderCursor` pointing to wherever their data lives — they don't care whether it's inside a `CallData` struct or an entry-point param. In the fast path, `m_runtime->write_shader_cursor_pre_dispatch()` receives the entry-point cursor directly. No marshall code changes needed.

7. **`CallDataMode` is eliminated.** The `global_data` vs `entry_point` distinction is removed entirely. On the fast path, all backends use entry-point params uniformly. On the fallback path, all backends use `ParameterBlock<CallData>` — CUDA supports `ParameterBlock` and in practice will never hit the fallback (CUDA's inline-uniform limit is ~4KB). This removes the `CallDataMode` enum, the CUDA-specific `is_entry_point` codegen branch in `callsignature.py`/`generator.py`, and the corresponding C++ branch in `slangpy.cpp`.

8. **`PackedArg` / param-block types are unchanged.** They stay as `ParameterBlock<T>` at module scope, orthogonal to Phase 2.

---

### Code Organization (post-extraction)

All code generation logic now lives in [generator.py](slangpy/core/generator.py). [callsignature.py](slangpy/core/callsignature.py) retains binding-pipeline functions (`specialize`, `bind`, `calculate_*`, `estimate_entrypoint_arguments_size`, etc.) and re-exports `generate_code`, `generate_constants`, `KernelGenException` from `generator.py` for backward compatibility.

| File | Role |
|------|------|
| [generator.py](slangpy/core/generator.py) | All code emission: `generate_code()`, `_emit_trampoline()`, `_emit_entry_point_signature()`, `_emit_kernel_body()`, `_emit_shape_and_metadata_params()`, `_emit_link_time_constants()`, `_emit_call_data_definitions()`, `_emit_trampoline_loads/stores()`, `_data_name()`, `_validate_and_compute_group_shape()`, `gen_call_data_code()`, `gen_calldata_type_name()`, `KernelGenException` |
| [callsignature.py](slangpy/core/callsignature.py) | Binding pipeline: `specialize()`, `bind()`, `apply_explicit_vectorization()`, `apply_implicit_vectorization()`, `finalize_mappings()`, `calculate_differentiability()`, `calculate_direct_binding()`, `estimate_entrypoint_arguments_size()`, `calculate_call_dimensionality()`, `create_return_value_binding()` |
| [calldata.py](slangpy/core/calldata.py) | `CallData` class orchestrating build pipeline; wildcard-imports from `callsignature.py` |
| [codegen.py](slangpy/bindings/codegen.py) | `CodeGen` class with `skip_call_data`, `entry_point_params` attributes |
| [boundvariable.py](slangpy/bindings/boundvariable.py) | `BoundVariable` methods delegate to `gen_call_data_code()` and `gen_calldata_type_name()` in `generator.py` |

---

### Current Kernel Structure (post-Phase 1)

For `int add(int a, int b)` with scalar args `(1, 2)`:

```slang
import "module";
import "slangpy";

typealias _t_a = int;             // Phase 1: raw type (was ValueType<int>)
typealias _t__result = RWValueRef<int>;  // writable _result still wrapped
static const int _m__result = 0;         // mapping constant only for _result

struct CallData {
    _t_a a;
    _t_a b;
    _t__result _result;
    uint3 _thread_count;
};

void _trampoline(Context __slangpy_context__, CallData __calldata__) {
    int a;
    a = __calldata__.a;            // Phase 1: direct assignment
    int b;
    b = __calldata__.b;            // Phase 1: direct assignment
    int _result;
    _result = add(a, b);
    __calldata__._result.__slangpy_store(__slangpy_context__.map(_m__result), _result);
}

[shader("compute")] [numthreads(32,1,1)]
void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID, ..., uniform CallData call_data) {
    if (any(flat_call_thread_id >= call_data._thread_count)) return;
    Context __slangpy_context__ = {flat_call_thread_id};
    _trampoline(__slangpy_context__, call_data);
}
```

### Target Kernel (Phase 2 fast path, prim mode, all direct-bind)

```slang
import "module";

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(int3 tid: SV_DispatchThreadID,
    uniform uint3 _thread_count,
    uniform int a,
    uniform int b,
    uniform RWStructuredBuffer<int> _result)
{
    if (any(tid >= _thread_count)) return;
    _result[0] = add(a, b);
}
```

### Target Kernel (Phase 2 fast path, prim mode, mixed direct/non-direct-bind)

When some args are not direct-bind (e.g., WangHashArg needs per-thread `thread_id` via `__slangpy_load`), the non-direct-bind args still use their wrapper types as entry-point params. Context is needed:

```slang
import "module";
import "slangpy";

typealias _t_rng = WangHashArgType;  // non-direct-bind wrapper type
static const int _m_rng = 0;

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID,
    uniform uint3 _thread_count,
    uniform _t_rng rng,
    uniform int x,
    uniform RWStructuredBuffer<int> _result)
{
    if (any(flat_call_thread_id >= _thread_count)) return;
    Context __slangpy_context__ = {flat_call_thread_id};
    int _rng_val;
    rng.__slangpy_load(__slangpy_context__.map(_m_rng), _rng_val);
    int _x_val;
    _x_val = x;
    int _result_val;
    _result_val = func(_rng_val, _x_val);
    _result[0] = _result_val;
}
```

### Target Kernel (Phase 2 fallback path, prim mode)

When entry-point param size exceeds the platform limit, all args go into `ParameterBlock<CallData>`. The trampoline is still eliminated in prim mode — the load/call/store is inlined into `compute_main`, reading from `call_data`:

```slang
import "module";
import "slangpy";

typealias _t_a = int;
typealias _t__result = RWValueRef<int>;
static const int _m__result = 0;

struct CallData {
    _t_a a;
    _t_a b;
    _t__result _result;
    uint3 _thread_count;
};
ParameterBlock<CallData> call_data;

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID, ...) {
    if (any(flat_call_thread_id >= call_data._thread_count)) return;
    Context __slangpy_context__ = {flat_call_thread_id};
    int a;
    a = call_data.a;
    int b;
    b = call_data.b;
    int _result;
    _result = add(a, b);
    call_data._result.__slangpy_store(__slangpy_context__.map(_m__result), _result);
}
```

---

### Step 2.0: Gating tests ✅

**Status: DONE**

Tests added to [slangpy/tests/slangpy_tests/test_kernel_gen.py](slangpy/tests/slangpy_tests/test_kernel_gen.py). All 21 parametrized cases (7 tests × 3 device types) pass.

| Test | Source | Args | Original assertion | Status |
|------|--------|------|--------------------|--------|
| `test_gate_p2_calldata_struct_present` | `int add(int a, int b)` | `(1, 2)` | `struct CallData` in code | ✅ Flipped — now asserts `struct CallData` ABSENT (Step 2.2 done) |
| `test_gate_p2_calldata_uniform_param` | same | same | `uniform CallData call_data` or `ParameterBlock<CallData>` | ✅ Flipped — now asserts both ABSENT (Step 2.2 done) |
| `test_gate_p2_thread_count_in_calldata` | same | same | `call_data._thread_count` | ✅ Flipped — now asserts ABSENT (Step 2.2 done) |
| `test_gate_p2_trampoline_present_for_prim` | same | same | `void _trampoline(` present | Still asserts present (Step 2.3 pending) |
| `test_gate_p2_kernel_calls_trampoline` | same | same | `_trampoline(` in `compute_main` body | Still asserts present (Step 2.3 pending) |
| `test_gate_p2_sv_group_id_present` | same | same | `SV_GroupID` in `compute_main` signature | ✅ Flipped — now asserts ABSENT for dim-0 calls (Step 2.2 done) |

Negative gates (must stay passing after Phase 2):

| Test | Asserts |
|------|---------|
| `test_gate_p2_wanghasharg_keeps_load` | Non-direct-bind arg still uses `__slangpy_load` |

Bwds gates:

| Test | Status |
|------|--------|
| `test_gate_scalar_uses_valuetype` | ✅ Passing — asserts fast-path trampoline with `__in_` prefix params |
| `test_gate_bwds_scalar_uses_valuetype` | ✅ Passing — bwds trampoline has `no_diff` on all params (Step 2.4 done) |

---

### Step 2.1: Determine fast vs fallback path ✅

**Status: DONE**

In [slangpy/core/calldata.py](slangpy/core/calldata.py), after `calculate_direct_binding(bindings)`:

1. **Query a runtime per-device threshold** for max entry-point parameter inline-uniform size. This is a property of the device/backend — large for D3D12/CUDA (thousands of bytes), potentially as low as 128–256 bytes on Vulkan.
2. **Accumulate inline-uniform byte size** of each bound variable's `calldata_type_name`, plus `_thread_count` (12 bytes) and shape arrays (`call_data_len * 3 * sizeof(int)` for `_grid_stride`, `_grid_dim`, `_call_dim`). **Resource types** (`RWStructuredBuffer`, `Texture2D`, `TensorView`, etc.) don't count — they are bound as descriptors, not inline data.
3. **Decision**: If total size ≤ threshold → `self.use_entrypoint_args = True` (fast path). Otherwise → `self.use_entrypoint_args = False` (fallback path — current behavior).
4. **Store** `use_entrypoint_args` on the `CallData` instance and propagate to C++ `NativeCallData`.

`PackedArg` / param-block types are excluded from this accounting — they stay as `ParameterBlock<T>` regardless.

**Implementation details:**

- `DeviceLimits.max_entry_point_uniform_size` added to C++ struct ([device.h](src/sgl/device/device.h)) with per-backend defaults: Vulkan=128, D3D12=256, CUDA=4096 bytes ([device.cpp](src/sgl/device/device.cpp)).
- `estimate_entrypoint_arguments_size()` in [callsignature.py](slangpy/core/callsignature.py) — sums `vector_type.uniform_layout.size` for each depth-0 bound variable (skipping `PackedArg`), plus 12 bytes for `_thread_count` and `call_dimensionality * 4 * 3` for shape arrays.
- `use_entrypoint_args` property added to `NativeCallData` C++ class ([slangpy.h](src/slangpy_ext/utils/slangpy.h)) with Python binding.
- `CallData.__init__()` in [calldata.py](slangpy/core/calldata.py) sets `self.use_entrypoint_args = inline_size <= threshold` after `calculate_direct_binding()`.

**Tests** (7 tests × 3 device types = 21 parametrized cases, all pass):

| Test | Asserts |
|------|---------|
| `test_step21_scalar_uses_entrypoint_args` | Simple `int add(int,int)` with `(1,2)` → `use_entrypoint_args=True` |
| `test_step21_threshold_property_positive` | `device.info.limits.max_entry_point_uniform_size > 0` |
| `test_step21_vector_uses_entrypoint_args` | `float3` args → `use_entrypoint_args=True` |
| `test_step21_struct_uses_entrypoint_args` | All-scalar struct dict → `use_entrypoint_args=True` |
| `test_step21_tensor_uses_entrypoint_args` | Tensor (descriptor-only, 0 inline bytes) → `use_entrypoint_args=True` |
| `test_step21_many_float4x4_may_exceed_vulkan` | 8×float4x4 (524 bytes) exceeds Vulkan/D3D12 thresholds, not CUDA |
| `test_step21_wanghasharg_uses_entrypoint_args` | Non-direct-bind WangHashArg with small inline size → `use_entrypoint_args=True` |

---

### Step 2.2: Code generation — entry-point params (fast path) ✅

**Status: DONE**

In [generator.py](slangpy/core/generator.py) `generate_code()` (line 778), when `use_entrypoint_args == True`:

**CodeGen changes** in [slangpy/bindings/codegen.py](slangpy/bindings/codegen.py):
- `self.skip_call_data: bool = False` — when `True`, don't emit `struct CallData` / `begin_block()` and gate `end_block()` in `finish()`.
- `self.entry_point_params: list[str] = []` — collects individual uniform param declarations.
- `finish()` ignores the `call_data` block and `use_param_block_for_call_data` when `skip_call_data` is set.

**CallData struct elimination**: `generate_code()` sets `cg.skip_call_data = True` when `use_entrypoint_args`. No `struct CallData` emitted.

**`_emit_call_data_code`** in [generator.py](slangpy/core/generator.py#L345): At `depth == 0`, when `use_entrypoint_args`, appends to `cg.entry_point_params` instead of `cg.call_data.declare(...)`. The `call_data_structs` block (type aliases, wrapper structs, mapping constants) still gets emitted at module scope.

**`_thread_count` and shape arrays**: `_emit_shape_and_metadata_params()` ([generator.py](slangpy/core/generator.py#L466)) appends to `cg.entry_point_params` instead of `cg.call_data`. Same for `_grid_stride`, `_grid_dim`, `_call_dim` when `call_data_len > 0`.

**Entry-point signature**: `_emit_entry_point_signature()` ([generator.py](slangpy/core/generator.py#L652)) emits `compute_main` signature as:
```slang
void compute_main(
    int3 flat_call_thread_id: SV_DispatchThreadID,
    [int3 flat_call_group_id: SV_GroupID,]          // only when call_data_len > 0
    [int flat_call_group_thread_id: SV_GroupIndex,]  // only when call_data_len > 0
    uniform uint3 _thread_count,
    [uniform int[N] _grid_stride, ...]               // only when call_data_len > 0
    uniform _t_a a,
    uniform _t_b b,
    uniform _t__result _result
)
```

Drop `SV_GroupID` and `SV_GroupIndex` when `call_data_len == 0` — they feed `init_thread_local_call_shape_info` which isn't called when there are no shape arrays.

**Bounds check**: Changes from `call_data._thread_count` to just `_thread_count`.

**Shape info init**: Changes from `call_data._grid_stride` etc. to just `_grid_stride`, `_grid_dim`, `_call_dim`.

**Fallback path** (`use_entrypoint_args == False`): `struct CallData` is emitted with `ParameterBlock<CallData> call_data` at module scope on ALL backends (including CUDA). The old `CallDataMode` distinction between `entry_point` (CUDA) and `global_data` (non-CUDA) is removed — `ParameterBlock` works on CUDA, and in practice CUDA will never hit the fallback due to its large (~4KB) inline-uniform limit.

See [slangpy/tests/device/test_pipeline_utils.slang](slangpy/tests/device/test_pipeline_utils.slang) for examples of manually-written compute shaders that use entry point parameters on all backends (CUDA, Vulkan, D3D12).

---

### Step 2.3: Trampoline elimination for prim mode

**Status: NOT STARTED** — Trampoline is still generated for prim mode on both paths. The load/call/store sequence needs to be inlined into `compute_main`.

When `call_mode == prim` — on **both** fast and fallback paths:

- Don't generate the `_trampoline` function.
- Inline the load/call/store sequence directly into `compute_main` after the bounds check and (if needed) Context construction.
- The load/call/store codegen reuses the same logic currently in `_emit_trampoline_loads()` ([generator.py](slangpy/core/generator.py#L528)) and `_emit_trampoline_stores()` ([generator.py](slangpy/core/generator.py#L551)), but emitted into `cg.kernel` instead of `cg.trampoline` with adjusted `data_name` from `_data_name()` ([generator.py](slangpy/core/generator.py#L513)):

| Path | `data_name` for non-param-block args |
|------|-------------------------------------|
| Fast | `x.variable_name` (entry-point param name directly) |
| Fallback | `call_data.{x.variable_name}` (global `ParameterBlock<CallData>`, all backends) |
| Param blocks | `_param_{x.variable_name}` (unchanged) |

**Context construction**: Needed only when any arg is non-direct-bind (i.e., calls `__slangpy_load`/`__slangpy_store`). When all args satisfy `direct_bind == True`, skip Context construction entirely — no `Context __slangpy_context__` declaration, no `import "slangpy"`.

**Key functions to modify in [generator.py](slangpy/core/generator.py)**:
- `_emit_trampoline()` (line 578): gate on `call_mode != prim` — only emit for bwds mode.
- `_emit_kernel_body()` (line 708): when prim mode, inline the load/call/store sequence directly instead of calling `_trampoline()`.
- `generate_code()` (line 778): skip `_emit_trampoline()` call when prim mode.

**Note**: The trampoline elimination does NOT depend on `direct_bind`. Even non-direct-bind args with `__slangpy_load` work inline in `compute_main` — the `__slangpy_load` call just needs the data reference and a `Context` value, both available in `compute_main`.

---

### Step 2.4: Trampoline with individual params for bwds mode ✅

**Status: DONE** — Fast-path trampoline takes individual params with `no_diff` on all params. All 3 device types pass.

When `call_mode == bwds`:

- Still generate a `[Differentiable]` trampoline function via `_emit_trampoline()` ([generator.py](slangpy/core/generator.py#L578)).
- **Fast path**: Trampoline takes individual params instead of a struct. All params get `no_diff` — entry-point uniforms are never differentiable. Differentiation happens through local variable assignments inside the trampoline body, matching the struct-based approach where `CallData` was implicitly non-differentiable. No `in`/`out`/`inout` modifiers are added — `compute_main` passes its uniforms straight through:
  ```slang
  [Differentiable]
  void _trampoline(Context __slangpy_context__, no_diff float __in_a, no_diff float __in_b, no_diff NoneType __in__result)
  ```
  `compute_main` calls `bwd_diff(_trampoline)(__slangpy_context__, a, b, _result)` passing entry-point param names directly.
- **Fallback path**: Trampoline reads from global `ParameterBlock<CallData> call_data` as it does today (on all backends). `compute_main` calls `bwd_diff(_trampoline)(__slangpy_context__, call_data)`.
- `_gen_trampoline_argument()` in `boundvariable.py` remains unused dead code — the inline generation in `_emit_trampoline()` ([generator.py](slangpy/core/generator.py#L578)) is simpler and avoids the `in`/`out`/`inout` modifiers that caused Slang autodiff errors.

**Key insight**: Adding `in`/`out`/`inout` modifiers to trampoline params caused Slang autodiff issues (e.g., `out` params get reversed to `in` by `bwd_diff`, changing arity). The trampoline params are just pass-through uniforms — all data flow logic (loads, stores, differentiation) is handled internally via local variables.

---

### Step 2.5: C++ dispatch changes ✅

**Status: DONE** — `CallDataMode` enum fully removed. Fast path uses `find_entry_point(0)` on all backends. Fallback path uses global `ParameterBlock<CallData>` on all backends.

In [src/slangpy_ext/utils/slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp), store `m_use_entrypoint_args` on `NativeCallData` (received from Python `CallData`). Also add to [slangpy.h](src/slangpy_ext/utils/slangpy.h).

Modify `bind_call_data` lambda in `exec()`:

**Fast path** (`m_use_entrypoint_args == true`):
- All backends: Navigate via `cursor.find_entry_point(0)`. This is the entry-point cursor.
- Write `_thread_count` as an entry-point param: `entry_point_cursor["_thread_count"]`.
- Write shape arrays as entry-point params: `entry_point_cursor["_grid_stride"]`, etc.
- Pass `entry_point_cursor` as the `call_data_cursor` argument to `m_runtime->write_shader_cursor_pre_dispatch()`. Each `NativeBoundVariableRuntime` already navigates `cursor[m_variable_name]`, so it finds the entry-point param by name automatically. **No marshall code changes needed.**
- Cache entry-point param field indices on first call (analogous to existing `m_cached_call_data_offsets`).
- The `reserve_data` + raw-pointer optimization for `_thread_count` and shape arrays may not work for individual entry-point params at disjoint offsets. Use cursor-based writes for these metadata fields (they're small, performance impact minimal), or check if `reserve_data` still works across the entry-point shader object.

**Fallback path** (`m_use_entrypoint_args == false`):
- All backends: Navigate to global `call_data` field via `cursor.find_field("call_data")`, dereference (it's a `ParameterBlock`), write struct data. The old `CallDataMode` branch (CUDA using `find_entry_point(0)` for call_data) is removed. Remove `m_call_data_mode`, `CallDataMode` enum, and all associated branches from `slangpy.h`, `slangpy.cpp`, `calldata.py`, and `callsignature.py`.

---

### Step 2.6: `_result` handling

**Status: NOT STARTED**

Auto-created `_result` is a writable `ValueRef`, currently NOT direct-bind eligible (needs `RWValueRef<T>` wrapper with buffer logic). Phase 2 handles this differently on the two paths:

**Fast path**: `_result` is emitted as `uniform RWValueRef<int> _result` on the entry point. In prim mode, the inlined code stores via `_result.__slangpy_store(...)`. In the all-direct-bind case where Context is omitted, add a new code path: emit `uniform RWStructuredBuffer<T> _result` with `_result[0] = value` for the store. This requires `ValueRefMarshall` to support writable direct-bind for the entry-point-param case specifically, using `RWStructuredBuffer<T>` instead of `RWValueRef<T>`.

**Fallback path**: `_result` stays as `RWValueRef<T>` inside `CallData`, same as current behavior.

**Implementation note**: The `RWStructuredBuffer<T>` approach for `_result` is only used when `use_entrypoint_args == True` AND all other args are direct-bind (so Context can be omitted). When non-direct-bind args are present, Context exists and `_result` can continue to use `RWValueRef<T>.__slangpy_store(context, value)`.

---

### Step 2.7: Tests

**Status: NOT STARTED**

**Post-implementation tests** — should pass AFTER Phase 2 is complete:

| Test | Verifies |
|------|----------|
| `test_phase2_no_calldata_struct` | `struct CallData` absent for eligible call |
| `test_phase2_uniform_params_on_entry` | Individual `uniform` params on `compute_main` |
| `test_phase2_no_trampoline_prim` | No `void _trampoline(` for prim-mode calls |
| `test_phase2_inline_call` | Function call inlined directly in `compute_main` |
| `test_phase2_thread_count_as_uniform` | `uniform uint3 _thread_count` as entry-point param |
| `test_phase2_no_context_all_direct` | No `Context __slangpy_context__` when all args direct-bind |
| `test_phase2_context_kept_non_direct` | `Context` present when some args use `__slangpy_load` |
| `test_phase2_bwds_trampoline_individual` | Bwds trampoline has individual params with `no_diff` |
| `test_phase2_bwds_bwd_diff_call` | `bwd_diff(_trampoline)(ctx, a, b, ...)` in kernel |
| `test_phase2_no_sv_group_when_dim0` | No `SV_GroupID`/`SV_GroupIndex` when `call_data_len == 0` |
| `test_phase2_sv_group_when_vectorized` | `SV_GroupID`/`SV_GroupIndex` present when `call_data_len > 0` |
| `test_phase2_fallback_keeps_calldata` | Force fallback → `struct CallData` still emitted |
| `test_phase2_fallback_no_trampoline_prim` | Even fallback path eliminates trampoline in prim mode |
| `test_phase2_functional_scalar_add` | `add(1, 2) == 3` end-to-end dispatch |
| `test_phase2_functional_bwds` | Backward pass correct gradients |
| `test_phase2_functional_vectorized` | Vectorized call (shapes) with entry-point params |
| `test_phase2_functional_mixed_direct` | Mix of direct-bind + non-direct-bind args |

---

### Implementation Order

1. **Step 2.0** ✅ — Gating tests (baseline documentation)
2. **Step 2.1** ✅ — Fast/fallback determination + size query
3. **Step 2.2 + 2.5** ✅ — Code gen + C++ dispatch for entry-point params + `CallDataMode` removal (landed together)
4. **Step 2.4** ✅ — Bwds trampoline with individual params (fast path) — `no_diff` on all params
5. **Step 2.3** — Trampoline elimination for prim mode (both paths)
6. **Step 2.6** — `_result` as `RWStructuredBuffer<T>` for all-direct-bind case
7. **Step 2.7** — Post-implementation tests + functional tests

**Note:** Implementation order deviated from original plan — Steps 2.2 + 2.5 were done before 2.3 (trampoline elimination), combined with `CallDataMode` removal. Step 2.4 done — all trampoline params use `no_diff` without IO modifiers.

---

### Key Files

| File | Changes |
|------|---------|
| [slangpy/core/generator.py](slangpy/core/generator.py) | ✅ All code generation logic extracted here from `callsignature.py`. `generate_code()` orchestrator (line 778), `_emit_trampoline()` (line 578), `_emit_entry_point_signature()` (line 652), `_emit_kernel_body()` (line 708), `_emit_shape_and_metadata_params()` (line 466), `_emit_link_time_constants()` (line 424), `_emit_call_data_definitions()` (line 503), `_emit_trampoline_loads/stores()`, `_data_name()`, `_validate_and_compute_group_shape()`, `gen_call_data_code()`, `gen_calldata_type_name()`, `KernelGenException`. Entry-point params fast/fallback code paths. Bwds `no_diff` on all trampoline params (Step 2.4). Trampoline still generated for prim mode (Step 2.3 pending). |
| [slangpy/core/calldata.py](slangpy/core/calldata.py) | ✅ `use_entrypoint_args` flag, size threshold check, `CallDataMode` removed |
| [slangpy/core/callsignature.py](slangpy/core/callsignature.py) | ✅ Binding-pipeline functions only (`specialize`, `bind`, `calculate_*`, `estimate_entrypoint_arguments_size`). Re-exports `generate_code`, `generate_constants`, `KernelGenException` from `generator.py`. |
| [slangpy/bindings/codegen.py](slangpy/bindings/codegen.py) | ✅ `skip_call_data` flag, `entry_point_params` list |
| [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py) | ✅ `gen_call_data_code` and `gen_calldata_type_name` delegate to `generator.py`. `_gen_trampoline_argument()` unused dead code. |
| [slangpy/bindings/marshall.py](slangpy/bindings/marshall.py) | ✅ `use_entrypoint_args` field on `BindContext`, `CallDataMode` removed |
| [src/slangpy_ext/utils/slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp) | ✅ `use_entrypoint_args` binding; `bind_call_data` fast path via `find_entry_point(0)`, `CallDataMode` branches removed |
| [src/slangpy_ext/utils/slangpy.h](src/slangpy_ext/utils/slangpy.h) | ✅ `m_use_entrypoint_args` on `NativeCallData`; `m_call_data_mode` removed |
| [src/sgl/device/device.h](src/sgl/device/device.h) | ✅ `max_entry_point_uniform_size` on `DeviceLimits` |
| [src/sgl/device/device.cpp](src/sgl/device/device.cpp) | ✅ Per-backend defaults for `max_entry_point_uniform_size` |
| [src/slangpy_ext/device/device.cpp](src/slangpy_ext/device/device.cpp) | ✅ Python binding for `max_entry_point_uniform_size` |
| [src/sgl/utils/slangpy.h](src/sgl/utils/slangpy.h) | ✅ `CallDataMode` enum removed |
| [slangpy/core/dispatchdata.py](slangpy/core/dispatchdata.py) | ✅ `CallDataMode` removed; imports `generate_constants` from `generator.py` |
| [slangpy/core/packedarg.py](slangpy/core/packedarg.py) | ✅ `CallDataMode` removed |
| [slangpy/core/function.py](slangpy/core/function.py) | ✅ `CallDataMode` removed from imports |
| [slangpy/slangpy/__init__.pyi](slangpy/slangpy/__init__.pyi) | ✅ `CallDataMode` class and `call_data_mode` property removed |
| [slangpy/tests/slangpy_tests/test_type_resolution.py](slangpy/tests/slangpy_tests/test_type_resolution.py) | ✅ `CallDataMode` removed from `BindContext` creation |
| [slangpy/tests/slangpy_tests/test_kernel_gen.py](slangpy/tests/slangpy_tests/test_kernel_gen.py) | ✅ Gating tests + Step 2.1 tests updated for new behavior; post-implementation tests (Step 2.7) pending |

---

### Verification

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
