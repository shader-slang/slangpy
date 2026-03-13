## Phase 2: Eliminate CallData Struct

**Goal**: Move kernel uniforms out of the `CallData` struct into individual entry-point parameters. Eliminate the trampoline in forward (prim) mode. Fall back to `ParameterBlock<CallData>` when total inline-uniform size exceeds a runtime per-device threshold.

**Parent plan**: [plan-simplifyKernelGen.prompt.md](plan-simplifyKernelGen.prompt.md)

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

7. **`CallDataMode` is eliminated.** The `global_data` vs `entry_point` distinction is removed entirely. On the fast path, all backends use entry-point params uniformly. On the fallback path, all backends use `ParameterBlock<CallData>` — CUDA supports `ParameterBlock` and in practice will never hit the fallback (CUDA's inline-uniform limit is ~4KB). This removes the `CallDataMode` enum, the CUDA-specific `is_entry_point` codegen branch in `callsignature.py`, and the corresponding C++ branch in `slangpy.cpp`.

8. **`PackedArg` / param-block types are unchanged.** They stay as `ParameterBlock<T>` at module scope, orthogonal to Phase 2.

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
3. **Decision**: If total size ≤ threshold → `self.use_direct_args = True` (fast path). Otherwise → `self.use_direct_args = False` (fallback path — current behavior).
4. **Store** `use_direct_args` on the `CallData` instance and propagate to C++ `NativeCallData`.

`PackedArg` / param-block types are excluded from this accounting — they stay as `ParameterBlock<T>` regardless.

**Implementation details:**

- `DeviceLimits.max_entry_point_uniform_size` added to C++ struct ([device.h](src/sgl/device/device.h)) with per-backend defaults: Vulkan=128, D3D12=256, CUDA=4096 bytes ([device.cpp](src/sgl/device/device.cpp)).
- `calculate_inline_uniform_size()` added to [callsignature.py](slangpy/core/callsignature.py) — sums `vector_type.uniform_layout.size` for each depth-0 bound variable (skipping `PackedArg`), plus 12 bytes for `_thread_count` and `call_dimensionality * 4 * 3` for shape arrays.
- `use_direct_args` property added to `NativeCallData` C++ class ([slangpy.h](src/slangpy_ext/utils/slangpy.h)) with Python binding.
- `CallData.__init__()` in [calldata.py](slangpy/core/calldata.py) sets `self.use_direct_args = inline_size <= threshold` after `calculate_direct_binding()`.

**Tests** (7 tests × 3 device types = 21 parametrized cases, all pass):

| Test | Asserts |
|------|---------|
| `test_step21_scalar_uses_direct_args` | Simple `int add(int,int)` with `(1,2)` → `use_direct_args=True` |
| `test_step21_threshold_property_positive` | `device.info.limits.max_entry_point_uniform_size > 0` |
| `test_step21_vector_uses_direct_args` | `float3` args → `use_direct_args=True` |
| `test_step21_struct_uses_direct_args` | All-scalar struct dict → `use_direct_args=True` |
| `test_step21_tensor_uses_direct_args` | Tensor (descriptor-only, 0 inline bytes) → `use_direct_args=True` |
| `test_step21_many_float4x4_may_exceed_vulkan` | 8×float4x4 (524 bytes) exceeds Vulkan/D3D12 thresholds, not CUDA |
| `test_step21_wanghasharg_uses_direct_args` | Non-direct-bind WangHashArg with small inline size → `use_direct_args=True` |

---

### Step 2.2: Code generation — entry-point params (fast path) ✅

**Status: DONE**

In [slangpy/core/callsignature.py](slangpy/core/callsignature.py) `generate_code()`, when `use_direct_args == True`:

**CodeGen changes** in [slangpy/bindings/codegen.py](slangpy/bindings/codegen.py):
- Add a `skip_call_data` flag to `CodeGen.__init__`. When `True`, don't emit `struct CallData` / `begin_block()` and gate `end_block()` in `finish()`.
- Add `self.entry_point_params: list[str] = []` to collect individual uniform param declarations.
- `finish()` ignores the `call_data` block and `use_param_block_for_call_data` when `skip_call_data` is set.

**CallData struct elimination**: Set `cg.skip_call_data = True` when `use_direct_args`. No `struct CallData` emitted.

**`gen_call_data_code` change** in [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py): At `depth == 0`, when `use_direct_args`, append to `cg.entry_point_params` instead of `cg.call_data.declare(...)`. The `call_data_structs` block (type aliases, wrapper structs, mapping constants) still gets emitted at module scope.

**`_thread_count` and shape arrays**: Instead of `cg.call_data.append_statement("uint3 _thread_count")`, append to `cg.entry_point_params`. Same for `_grid_stride`, `_grid_dim`, `_call_dim` when `call_data_len > 0`.

**Entry-point signature**: `compute_main` signature becomes:
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

**Fallback path** (`use_direct_args == False`): `struct CallData` is emitted with `ParameterBlock<CallData> call_data` at module scope on ALL backends (including CUDA). The old `CallDataMode` distinction between `entry_point` (CUDA) and `global_data` (non-CUDA) is removed — `ParameterBlock` works on CUDA, and in practice CUDA will never hit the fallback due to its large (~4KB) inline-uniform limit.

See [slangpy/tests/device/test_pipeline_utils.slang](slangpy/tests/device/test_pipeline_utils.slang) for examples of manually-written compute shaders that use entry point parameters on all backends (CUDA, Vulkan, D3D12).

---

### Step 2.3: Trampoline elimination for prim mode

**Status: NOT STARTED** — Trampoline is still generated for prim mode on both paths. The load/call/store sequence needs to be inlined into `compute_main`.

When `call_mode == prim` — on **both** fast and fallback paths:

- Don't generate the `_trampoline` function.
- Inline the load/call/store sequence directly into `compute_main` after the bounds check and (if needed) Context construction.
- The load/call/store codegen reuses the same logic currently in [callsignature.py lines 378–449](slangpy/core/callsignature.py#L378-L449), but emitted into `cg.kernel` instead of `cg.trampoline` with adjusted `data_name`:

| Path | `data_name` for non-param-block args |
|------|-------------------------------------|
| Fast | `x.variable_name` (entry-point param name directly) |
| Fallback | `call_data.{x.variable_name}` (global `ParameterBlock<CallData>`, all backends) |
| Param blocks | `_param_{x.variable_name}` (unchanged) |

**Context construction**: Needed only when any arg is non-direct-bind (i.e., calls `__slangpy_load`/`__slangpy_store`). When all args satisfy `direct_bind == True`, skip Context construction entirely — no `Context __slangpy_context__` declaration, no `import "slangpy"`.

**Note**: The trampoline elimination does NOT depend on `direct_bind`. Even non-direct-bind args with `__slangpy_load` work inline in `compute_main` — the `__slangpy_load` call just needs the data reference and a `Context` value, both available in `compute_main`.

---

### Step 2.4: Trampoline with individual params for bwds mode ✅

**Status: DONE** — Fast-path trampoline takes individual params with `no_diff` on all params. All 3 device types pass.

When `call_mode == bwds`:

- Still generate a `[Differentiable]` trampoline function.
- **Fast path**: Trampoline takes individual params instead of a struct. All params get `no_diff` — entry-point uniforms are never differentiable. Differentiation happens through local variable assignments inside the trampoline body, matching the struct-based approach where `CallData` was implicitly non-differentiable. No `in`/`out`/`inout` modifiers are added — `compute_main` passes its uniforms straight through:
  ```slang
  [Differentiable]
  void _trampoline(Context __slangpy_context__, no_diff float __in_a, no_diff float __in_b, no_diff NoneType __in__result)
  ```
  `compute_main` calls `bwd_diff(_trampoline)(__slangpy_context__, a, b, _result)` passing entry-point param names directly.
- **Fallback path**: Trampoline reads from global `ParameterBlock<CallData> call_data` as it does today (on all backends). `compute_main` calls `bwd_diff(_trampoline)(__slangpy_context__, call_data)`.
- `_gen_trampoline_argument()` in `boundvariable.py` remains unused dead code — the inline generation in `callsignature.py` is simpler and avoids the `in`/`out`/`inout` modifiers that caused Slang autodiff errors.

**Key insight**: Adding `in`/`out`/`inout` modifiers to trampoline params caused Slang autodiff issues (e.g., `out` params get reversed to `in` by `bwd_diff`, changing arity). The trampoline params are just pass-through uniforms — all data flow logic (loads, stores, differentiation) is handled internally via local variables.

---

### Step 2.5: C++ dispatch changes ✅

**Status: DONE** — `CallDataMode` enum fully removed. Fast path uses `find_entry_point(0)` on all backends. Fallback path uses global `ParameterBlock<CallData>` on all backends.

In [src/slangpy_ext/utils/slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp), store `m_use_direct_args` on `NativeCallData` (received from Python `CallData`). Also add to [slangpy.h](src/slangpy_ext/utils/slangpy.h).

Modify `bind_call_data` lambda in `exec()`:

**Fast path** (`m_use_direct_args == true`):
- All backends: Navigate via `cursor.find_entry_point(0)`. This is the entry-point cursor.
- Write `_thread_count` as an entry-point param: `entry_point_cursor["_thread_count"]`.
- Write shape arrays as entry-point params: `entry_point_cursor["_grid_stride"]`, etc.
- Pass `entry_point_cursor` as the `call_data_cursor` argument to `m_runtime->write_shader_cursor_pre_dispatch()`. Each `NativeBoundVariableRuntime` already navigates `cursor[m_variable_name]`, so it finds the entry-point param by name automatically. **No marshall code changes needed.**
- Cache entry-point param field indices on first call (analogous to existing `m_cached_call_data_offsets`).
- The `reserve_data` + raw-pointer optimization for `_thread_count` and shape arrays may not work for individual entry-point params at disjoint offsets. Use cursor-based writes for these metadata fields (they're small, performance impact minimal), or check if `reserve_data` still works across the entry-point shader object.

**Fallback path** (`m_use_direct_args == false`):
- All backends: Navigate to global `call_data` field via `cursor.find_field("call_data")`, dereference (it's a `ParameterBlock`), write struct data. The old `CallDataMode` branch (CUDA using `find_entry_point(0)` for call_data) is removed. Remove `m_call_data_mode`, `CallDataMode` enum, and all associated branches from `slangpy.h`, `slangpy.cpp`, `calldata.py`, and `callsignature.py`.

---

### Step 2.6: `_result` handling

**Status: NOT STARTED**

Auto-created `_result` is a writable `ValueRef`, currently NOT direct-bind eligible (needs `RWValueRef<T>` wrapper with buffer logic). Phase 2 handles this differently on the two paths:

**Fast path**: `_result` is emitted as `uniform RWValueRef<int> _result` on the entry point. In prim mode, the inlined code stores via `_result.__slangpy_store(...)`. In the all-direct-bind case where Context is omitted, add a new code path: emit `uniform RWStructuredBuffer<T> _result` with `_result[0] = value` for the store. This requires `ValueRefMarshall` to support writable direct-bind for the entry-point-param case specifically, using `RWStructuredBuffer<T>` instead of `RWValueRef<T>`.

**Fallback path**: `_result` stays as `RWValueRef<T>` inside `CallData`, same as current behavior.

**Implementation note**: The `RWStructuredBuffer<T>` approach for `_result` is only used when `use_direct_args == True` AND all other args are direct-bind (so Context can be omitted). When non-direct-bind args are present, Context exists and `_result` can continue to use `RWValueRef<T>.__slangpy_store(context, value)`.

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
| [slangpy/core/calldata.py](slangpy/core/calldata.py) | ✅ `use_direct_args` flag, size threshold check, `CallDataMode` removed |
| [slangpy/core/callsignature.py](slangpy/core/callsignature.py) | ✅ Entry-point params, fast/fallback code paths, `is_entry_point` branch removed. Trampoline still generated (Step 2.3 pending). Bwds `no_diff` on all trampoline params (Step 2.4 done). |
| [slangpy/bindings/codegen.py](slangpy/bindings/codegen.py) | ✅ `skip_call_data` flag, `entry_point_params` list |
| [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py) | ✅ `gen_call_data_code` depth-0 entry-point path. `_gen_trampoline_argument()` unused — inline generation in `callsignature.py` used instead. |
| [slangpy/bindings/marshall.py](slangpy/bindings/marshall.py) | ✅ `use_direct_args` field on `BindContext`, `CallDataMode` removed |
| [src/slangpy_ext/utils/slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp) | ✅ `use_direct_args` binding; `bind_call_data` fast path via `find_entry_point(0)`, `CallDataMode` branches removed |
| [src/slangpy_ext/utils/slangpy.h](src/slangpy_ext/utils/slangpy.h) | ✅ `m_use_direct_args` on `NativeCallData`; `m_call_data_mode` removed |
| [src/sgl/device/device.h](src/sgl/device/device.h) | ✅ `max_entry_point_uniform_size` on `DeviceLimits` |
| [src/sgl/device/device.cpp](src/sgl/device/device.cpp) | ✅ Per-backend defaults for `max_entry_point_uniform_size` |
| [src/slangpy_ext/device/device.cpp](src/slangpy_ext/device/device.cpp) | ✅ Python binding for `max_entry_point_uniform_size` |
| [src/sgl/utils/slangpy.h](src/sgl/utils/slangpy.h) | ✅ `CallDataMode` enum removed |
| [slangpy/core/dispatchdata.py](slangpy/core/dispatchdata.py) | ✅ `CallDataMode` removed |
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

---

### PR #862 Code Review — Proposed Improvements

#### High Severity

**1. Potential correctness bug — fast-path shape offset caching guarded by runtime data**

In [slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp) `bind_call_data`, the fast-path caching block guards shape offset caching with `call_shape.size() > 0`. If the *first* call to a multi-dimensional `NativeCallData` uses `has_thread_count=true` (which returns empty `call_shape`), shape offsets won't be cached. A subsequent normal call would find `is_valid == true` but shape offsets would be uninitialized, leading to writes at garbage offsets. The fallback path is more robust, using `call_dim.is_valid()` instead.

**DO NOT FIX**: Reason: The '_thread_count' is written to the call signature, so by definition a given call data would never be used in both situations.

**2. Benchmark changes are debugging artifacts**

[test_benchmark_autograd.py](slangpy/benchmarks/test_benchmark_autograd.py) changes `ITERATIONS` 10→100, `WARMUPS` 10→1000, `RUN_SLANGTORCH_BENCHMARK` False→True. This will make CI benchmarks 10–100× slower. Revert to original values.

**FIXED**: Restored `ITERATIONS=10`, `WARMUPS=10`, `RUN_SLANGTORCH_BENCHMARK=False`.

**3. Overly broad `except Exception` in calldata.py fallback**

[calldata.py](slangpy/core/calldata.py): The fallback from fast path to `ParameterBlock<CallData>` catches `except Exception`, which swallows `TypeError`, `KeyError`, `AttributeError`, etc. The caught exception `e` is never logged.

**FIXED**: Narrowed to `except RuntimeError as e` and included `str(e)` in the debug message.

---

#### Medium Severity — Structural

**4. `generate_code()` in callsignature.py is too long (~334 lines)**

Extract into sub-functions:

| Lines | Extract to | Purpose |
|-------|-----------|---------|
| ~L294–L339 | `_validate_and_compute_group_shape()` | Group shape validation & stride computation |
| ~L341–L388 | `_generate_link_time_constants()` | Link-time constants (group shape/stride arrays) |
| ~L390–L409 | `_generate_shape_params()` | Shape array & `_thread_count` param gen (fast/fallback) |
| ~L415–L517 | `_generate_trampoline()` | Trampoline function (signature, loads, call, stores) |
| ~L520–L565 | `_generate_entry_point_signature()` | Compute/ray-tracing entry-point signature |
| ~L567–L604 | `_generate_kernel_body()` | Kernel body (bounds check, shape init, dispatch) |

Additionally, the duplicated `data_name` computation at ~L449 and ~L497 should be extracted:
```python
def _data_name(x: BoundVariable, use_direct_args: bool) -> str:
    if x.create_param_block:
        return f"_param_{x.variable_name}"
    return f"__in_{x.variable_name}" if use_direct_args else f"call_data.{x.variable_name}"
```

**DO NOT FIX** Reason: This is a complex change and will be deferred to a later step.

**5. `bind_call_data` in slangpy.cpp has ~70 lines of duplicated write logic**

The `reserve_data` + `write_strided_array_helper` ×3 + `write_value_helper` + `write_shader_cursor_pre_dispatch` sequence is identical between fast and fallback paths. Extract a helper that takes a `ShaderCursor`:

```cpp
auto write_uniforms = [&](ShaderCursor target) {
    ShaderObject* so = target.shader_object();
    void* base = so->reserve_data(offsets.field_offset, offsets.field_size);
    // ... write shape arrays, thread_count ...
    m_runtime->write_shader_cursor_pre_dispatch(context, cursor, target, ...);
};
```

Fast path → `write_uniforms(ep)`, fallback → `write_uniforms(call_data_cursor)`.

**FIXED**: Extracted `write_uniforms` lambda taking `(ShaderCursor target, ShaderCursor root_cursor)`. Fast path calls `write_uniforms(ep, cursor)`, fallback calls `write_uniforms(call_data_cursor, cursor)`.

**6. `_try_build_shader` parameter pattern in calldata.py**

Takes `use_direct_args` parameter then immediately sets `self.use_direct_args` and `context.use_direct_args`. The method never reads the flag except to store it.

**FIXED**: Caller sets `self.use_direct_args` before calling; `_try_build_shader` reads `self.use_direct_args` and sets `context.use_direct_args`. Parameter removed.

---

#### Low Severity

**7. Unconditional `print(code)` in test_kernel_gen.py L107** — should be guarded by `PRINT_TEST_KERNEL_GEN` env var.

**FIXED**: Guarded with `if PRINT_TEST_KERNEL_GEN:` (existing module-level flag).

**8. Test duplication** — ~30 tests near-identical between test_kernel_gen.py and test_code_gen.py. The merged tests in test_code_gen.py should replace the originals.

**DO NOT FIX**: Reason: The kernel gen tests are temporary, designed for gating, and will be deleted once phases are complete.

**9. Unused `nodes` variable** — [callsignature.py L278](slangpy/core/callsignature.py): `nodes: list[BoundVariable] = []` declared but never used.

**FIXED**: Deleted unused variable.

**10. Stale docstring** — [callsignature.py L275](slangpy/core/callsignature.py): Says "Generate a list of call data nodes" — doesn't match what the function does.

**FIXED**: Updated to "Generate Slang kernel code for the given function call signature."

**11. Missing return type annotations** — `generate_code()`, `generate_constants()`, `CallData.build()` all need `-> None`.

**FIXED**: Added `-> None` to `generate_code()`, `generate_constants()`, `CallData.build()`, and `_try_build_shader()`.

**12. `type_conformances: Any`** — [calldata.py](slangpy/core/calldata.py) should be `list[TypeConformance]`.

**FIXED**: Changed to `list["TypeConformance"]` and added `TypeConformance` to the `from slangpy import (...)` block.

**13. Bare `except:`** — [callsignature.py L59](slangpy/core/callsignature.py): `is_generic_vector` catches all exceptions including `SystemExit`. Use `except Exception:`.

**FIXED**: Changed to `except Exception:`.

**14. Typo: `santized_module`** — [calldata.py](slangpy/core/calldata.py): Missing 'i'. Pre-existing.

**DO NOT FIX**: Reason: Cosmetic typo in a variable name that's used in multiple places. Fixing would require renaming across the file, which is low value and risks introducing bugs.

**15. D3D12 `max_entry_point_uniform_size = 256` may be optimistic** — root descriptors consume some of the 64-DWORD root signature budget. Comment should note shared budget; consider smaller default.

**DO NOT FIX**: Reason: More complex logic is actually needed and can be addressed later.

**16. Fallback path always includes `SV_GroupID`/`SV_GroupIndex`** — even when `call_data_len == 0`. Asymmetric with fast path.

**DO NOT FIX**: Reason: Can be addressed later.

**17. Hash salt `"[CallData]\n"`** — emitted even when CallData struct is absent. Cosmetic.

**FIXED**: Removed `"[CallData]\n"` prefix from hash salt.

**18. `Tuple` import in test_code_gen.py** — should use lowercase `tuple[...]` for consistency.

**FIXED**: Changed to `tuple[...]` and removed `Tuple` from typing import.

---

#### Additional Findings (subagent review, March 2026)

**19. Latent correctness bug — `can_direct_bind_common()` missing write-access guard**

[boundvariable.py](slangpy/bindings/boundvariable.py) `can_direct_bind_common()` does not check whether the binding has write access. This creates an inconsistency:

- `ValueRefMarshall.can_direct_bind()` explicitly rejects writable bindings — correct
- `StructMarshall.can_direct_bind()` with children checks `access[0] == AccessType.read` — correct
- `StructMarshall.can_direct_bind()` without children falls through to `can_direct_bind_common()` — **missing access check**
- `ValueMarshall.can_direct_bind()` delegates entirely to `can_direct_bind_common()` — safe in practice (`ValueMarshall.is_writable = False`) but fragile

If a writable dim-0 leaf binding gets `direct_bind=True`, `ValueMarshall.gen_trampoline_store()` returns `True` without emitting store code, silently dropping writes.

**DO NOT FIX**: Reasion: This logic is subtle but correct, based on the desired behaviour.

**20. Dead `_gen_trampoline_argument()` method**

[boundvariable.py](slangpy/bindings/boundvariable.py) `_gen_trampoline_argument()` is never called anywhere in the codebase. The inline generation in [callsignature.py](slangpy/core/callsignature.py) replaced it.

**FIXED**: Deleted the method.

**21. Redundant `hasattr` guard in `calculate_direct_bind()`**

[boundvariable.py](slangpy/bindings/boundvariable.py) `calculate_direct_bind()` uses `hasattr(self.python, "can_direct_bind")`, which is always `True` because `Marshall` base class defines `can_direct_bind()`. Simplify to `if self.python is not None:`.

**DO NOT FIX**: Reason: For marshalls that inherit directly from NativeMarshall, this is not necessarily true.

**22. Unnecessary `getattr` in `can_direct_bind_common()`**

[boundvariable.py](slangpy/bindings/boundvariable.py) `can_direct_bind_common()` uses `getattr(binding, "create_param_block", False)`. `BoundVariable.__init__()` always sets `create_param_block`, so `binding.create_param_block` suffices.

**FIXED**: Replaced `getattr(binding, "create_param_block", False)` with `binding.create_param_block`.

**23. Wasteful `CodeGen.call_data` initialization when `skip_call_data=True`**

[codegen.py](slangpy/bindings/codegen.py) `__init__` unconditionally calls `self.call_data.append_line("struct CallData")` and `begin_block()`, even when `skip_call_data=True`. The block is never serialized so there's no output impact, but it allocates a dangling block object.

**DO NOT FIX**: Reason: Harmless — the block is never emitted. Restructuring `__init__` to conditionally skip initialization adds complexity for no functional benefit.

**24. `entry_point_params` ownership pattern undocumented**

[codegen.py](slangpy/bindings/codegen.py) collects `entry_point_params` via `boundvariable.py`, but [callsignature.py](slangpy/core/callsignature.py) reads and emits them. This cross-module ownership pattern is unconventional and lacks a comment explaining the flow.

**DO NOT FIX**: Reason: `CodeGen` is already a shared state bag consumed by multiple modules. Adding a comment is fine but not blocking.

**25. `direct_bind` and `use_direct_args` exposed as read-write in `.pyi` stubs**

[__init__.pyi](slangpy/slangpy/__init__.pyi) exposes `direct_bind` on `NativeBoundVariableRuntime` and `use_direct_args` on `NativeCallData` with setters. Mutating these after first dispatch could invalidate cached cursor offsets in `NativeValueMarshall::ensure_cached`.

**DO NOT FIX**: Reason: These are set during `CallData` construction before first dispatch. The cached `NativeCallData` is per-signature, so a new signature gets a fresh instance. Post-construction mutation would require going through `debug_build_call_data` which rebuilds everything. Not a practical concern.

**26. No fallback-path codegen test in `test_code_gen.py`**

[test_code_gen.py](slangpy/tests/slangpy_tests/test_code_gen.py) has no test that forces `use_direct_args=False` (e.g., by exceeding `max_entry_point_uniform_size`) and asserts the `ParameterBlock<CallData>` codegen. The `test_step21_many_float4x4_may_exceed_vulkan` in `test_kernel_gen.py` checks the flag but not the generated code.

**DO NOT FIX**: Reason: Step 2.7 will add comprehensive post-implementation tests including `test_phase2_fallback_keeps_calldata` and `test_phase2_fallback_no_trampoline_prim`.

**27. No test for writable `inout` struct at dim-0**

No test verifies the behavior of a writable (inout) dim-0 struct with all-scalar fields. This is the scenario where Fix 19 would prevent silent write loss.

**Fix**: Add after Fix 19 is applied —test a writable dim-0 struct dict to confirm `direct_bind=False`.

**Status: NOT FIXED** — blocked on Fix 19.
