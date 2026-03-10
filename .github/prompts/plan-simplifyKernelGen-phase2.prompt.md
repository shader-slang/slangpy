## Phase 2: Eliminate CallData Struct

**Goal**: When ALL arguments are direct-eligible, bypass the `CallData` struct entirely and pass arguments as individual parameters on the entry point (or individual globals).

**Parent plan**: [plan-simplifyKernelGen.prompt.md](plan-simplifyKernelGen.prompt.md)

---

### Step 2.1: Determine eligibility

Add a check in [slangpy/core/calldata.py](slangpy/core/calldata.py) after all bindings are resolved: if every `BoundVariable` satisfies `is_direct_bind_eligible` (for leaves) or `is_direct_bind_recursive` (for composites) AND `call_data_len == 0` (no N-dimensional shape arrays needed), set a new flag `self.use_direct_args = True`.

---

### Step 2.2: New code generation path

In [slangpy/core/callsignature.py](slangpy/core/callsignature.py), when `use_direct_args`:

- **Skip CallData struct generation** entirely. Note: `CodeGen.__init__` in [codegen.py](slangpy/bindings/codegen.py) unconditionally emits `struct CallData { ... }` — the constructor creates the `self.call_data` block and `finish()` calls `self.call_data.end_block()`. To eliminate CallData, either:
  - Add a `skip_call_data` flag to `CodeGen.__init__` that conditionally initializes the block, and condition the `end_block()` in `finish()` on the same flag, OR
  - Clear `self.call_data` contents before `finish()` when `use_direct_args` is true
- **Generate compute_main** with individual `uniform` parameters. The current compute_main signature has three semantic params:
  ```
  void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID, int3 flat_call_group_id: SV_GroupID, int flat_call_group_thread_id: SV_GroupIndex, uniform CallData call_data)
  ```
  When `use_direct_args` and `call_data_len == 0`, the `SV_GroupID` and `SV_GroupIndex` params are unused (they feed `init_thread_local_call_shape_info` which reads `call_data._grid_stride`/`_grid_dim`/`_call_dim`). They can be dropped, simplifying to:
  ```
  void compute_main(int3 flat_call_thread_id: SV_DispatchThreadID, uniform uint3 _thread_count, uniform int a, uniform int b, uniform RWStructuredBuffer<int> _result)
  ```
- **Inline the function call** into compute_main (skip trampoline for prim mode): `_result[0] = add(a, b);`
- **Keep trampoline** for bwds mode (needed for `bwd_diff()`). The trampoline wraps the call with `[Differentiable]` and allows `bwd_diff(_trampoline)` from compute_main. In this case, generate a trampoline that takes individual params instead of a struct. Direct assignment `a = param_a;` is trivially differentiable in Slang for floating-point types. For non-differentiable types (int, etc.), autodiff is irrelevant.

---

### Step 2.3: Entry point parameters for all backends

Currently, CUDA (entry_point mode) already passes a `CallData` struct as a `uniform` entry point parameter. The simplification extends this: instead of a single struct, pass individual `uniform` parameters on the entry point — for ALL backends, not just CUDA.

See [slangpy/tests/device/test_pipeline_utils.slang](slangpy/tests/device/test_pipeline_utils.slang) for examples of manually-written compute shaders that use entry point parameters on all backends:
```slang
[shader("compute")]
[numthreads(16, 16, 1)]
void setcolor(
    uint3 tid: SV_DispatchThreadID,
    RWTexture2D<float4> render_texture,
    uniform int2 pos,
    uniform int2 size,
    uniform float4 color
)
```

Entry point parameters work on all backends (CUDA, Vulkan, D3D12). For `global_data` mode, the C++ side currently navigates `cursor["call_data"]` to write into a `ParameterBlock<CallData>` global. With direct args, it would instead navigate `cursor.find_entry_point(0)` and write each parameter by index — the same mechanism CUDA already uses, but now applied universally.

The `CallData` struct can be omitted entirely when all args are direct-eligible. If some args still need the struct (e.g., shape arrays for `call_data_len > 0`, or non-direct-eligible types), emit a hybrid: direct-eligible args as individual entry point params, and the remaining data in a `CallData` struct that is also an entry point param.

**Entry point size limits**: Some platforms impose limits on the total size of entry point parameter data (e.g., CUDA root constants are limited to ~4KB, D3D12 root signature has a 64 DWORD limit). To handle this:
- Define a per-backend threshold for maximum entry point parameter data size (queryable from device/backend info)
- During code generation, accumulate the uniform byte size of each direct-eligible argument. Resource types (`RWStructuredBuffer`, `Texture2D`, etc.) don't count toward the limit — they are bound as descriptors, not inline data
- If a single argument exceeds the threshold, force it back to `CallData`
- If the cumulative total exceeds the threshold, force remaining arguments (in declaration order) back to `CallData`
- The result may be a hybrid kernel: some args as entry point params, the rest in a `CallData` struct entry point param
- The C++ dispatch side must know which args are direct vs CallData-bound (store a per-argument flag or a bitmask on `NativeCallData`)

---

### Step 2.4: C++ dispatch changes

In [src/slangpy_ext/utils/slangpy.cpp](src/slangpy_ext/utils/slangpy.cpp):

- **Store `use_direct_args` flag** on `NativeCallData` (receive from Python `CallData`)
- **Both modes**: In `bind_call_data`, navigate via `cursor.find_entry_point(0)` and write each argument directly to its own entry point parameter by index. This is the same cursor API already used for CUDA entry_point mode — it just needs to write individual params instead of navigating into a single `CallData` struct field.
- **Thread count**: Write `_thread_count` as a separate entry point parameter instead of a struct field
- **Context construction**: The current kernel code constructs a `Context __slangpy_context__` from `call_data` fields (e.g., `flat_call_thread_id, CallShapeInfo::get_call_id().shape`). When `use_direct_args` and `call_data_len == 0`, the Context is simplified to just `{flat_call_thread_id}` and `CallShapeInfo` / `init_thread_local_call_shape_info` can be skipped. If Context is eliminated entirely (Phase 2 with inlined function calls), this becomes moot.
- **Skip shape array writing** (`_grid_stride`, `_grid_dim`, `_call_dim`) since `call_data_len == 0`
- **Cache parameter offsets**: Cache the entry point parameter indices at first dispatch (similar to existing `m_cached_call_data_offsets`)

---

### Step 2.5: Trampoline elimination for prim mode

When `use_direct_args` and `call_mode == prim`:
- Don't generate a trampoline function
- Emit the function call directly in `compute_main` using the uniform parameter names
- For output variables, emit the store directly (e.g., `_result[0] = add(a, b);`)

When `call_mode == bwds`:
- Still generate a trampoline (needed for `bwd_diff()`)
- Pass individual params to the trampoline instead of a struct

---

### Step 2.6: Tests

**Gating tests** — assert CURRENT behavior so they break when Phase 2 is implemented:

| Test | Slang Source | Args | Asserts (current behavior) | Breaks when |
|------|-------------|------|---------------------------|-------------|
| `test_gate_calldata_struct_present` | `int add(int a, int b) { return a + b; }` | `(1, 2)` | `struct CallData` present in generated code | Step 2.1 |
| `test_gate_calldata_uniform_param` | same | same | `uniform CallData call_data` in `compute_main` signature (note: actual signature also includes `SV_GroupID` and `SV_GroupIndex` params) | Step 2.2 |
| `test_gate_thread_count_in_calldata` | same | same | `call_data._thread_count` in kernel body | Step 2.4 |
| `test_gate_context_from_calldata` | same | same | `Context __slangpy_context__` construction present in kernel body | Step 2.4 |
| `test_gate_trampoline_present_for_prim` | same | same | `void _trampoline(` present | Step 2.5 |
| `test_gate_trampoline_calls_function` | same | same | `_result = add(a, b)` inside trampoline | Step 2.5 |
| `test_gate_kernel_calls_trampoline` | same | same | `_trampoline(` inside `compute_main` body | Step 2.5 |

**Negative gates** — should REMAIN passing after Phase 2:

| Test | Slang Source | Args | Asserts (must stay) |
|------|-------------|------|--------------------|
| `test_gate_wanghasharg_forces_calldata` | `int rng(WangHashArg rng, int x) { return x; }` | `(spy.WangHashArg(1), 1)` | `struct CallData` present (non-eligible arg forces fallback) |

**Post-implementation tests** — should pass AFTER Phase 2 is complete:

- `test_phase2_no_calldata_struct`: verify `struct CallData` absent for all-eligible scalar call
- `test_phase2_uniform_params_on_entry`: verify individual `uniform int a`, `uniform int b` on `compute_main`
- `test_phase2_no_trampoline_prim`: verify no `_trampoline(` for prim-mode eligible calls
- `test_phase2_thread_count_as_uniform`: verify `uniform uint3 _thread_count` as entry point param
- `test_phase2_inline_function_call`: verify `_result[0] = add(a, b)` directly in kernel
- `test_phase2_bwds_keeps_trampoline`: verify bwds mode still has `_trampoline` and `bwd_diff`
- `test_phase2_mixed_args_hybrid`: mix direct-eligible + WangHashArg → hybrid kernel
- `test_phase2_functional_all_backends`: dispatch scalar add on each backend, verify result
