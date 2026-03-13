## Extract codegen into generator.py

**Goal**: Extract the code-emission logic from [callsignature.py](slangpy/core/callsignature.py) (`generate_code`, `generate_constants`, `KernelGenException`, helpers) and `BoundVariable.gen_call_data_code` from [boundvariable.py](slangpy/bindings/boundvariable.py) into a new [generator.py](slangpy/core/generator.py) file. The new file decomposes the monolithic `generate_code` (332 lines) into clearly-named sub-functions with doc comments showing what Slang code each one emits. `callsignature.py` retains the binding-pipeline functions (`specialize`, `bind`, `calculate_*`, etc.). Each step is a pure move/rename with no behavioral changes, verifiable by the existing test suites.

**Parent plan**: [plan-simplifyKernelGenPhase2-cleanup.prompt.md](plan-simplifyKernelGenPhase2-cleanup.prompt.md)

---

### Step 1: Create `slangpy/core/generator.py` with `generate_constants` and `KernelGenException`

Move these small, self-contained pieces first:

- **Move** `KernelGenException` (lines 40–43) from [callsignature.py](slangpy/core/callsignature.py#L40-L43).
- **Move** `is_slangpy_vector` (lines 240–247) from [callsignature.py](slangpy/core/callsignature.py#L240-L247) — private helper, prefix with `_`.
- **Move** `generate_constants` (lines 250–268) from [callsignature.py](slangpy/core/callsignature.py#L250-L268).
- **In [callsignature.py](slangpy/core/callsignature.py)**: Add `from slangpy.core.generator import KernelGenException, generate_constants` and delete the moved code. Keep a re-export of `KernelGenException` so any external consumer of the wildcard import from [calldata.py](slangpy/core/calldata.py#L8) continues to work.
- **In [dispatchdata.py](slangpy/core/dispatchdata.py#L7)**: Change `from slangpy.core.callsignature import generate_constants` → `from slangpy.core.generator import generate_constants`.

**Verify**: `pytest slangpy/tests -v` — all tests pass, no import errors.

---

### Step 2: Extract `gen_call_data_code` as a free function

Move `BoundVariable.gen_call_data_code` (lines 604–693 of [boundvariable.py](slangpy/bindings/boundvariable.py#L604-L693)) into `generator.py` as a free function, along with the related `gen_calldata_type_name` helper (lines 258–272 of [boundvariable.py](slangpy/bindings/boundvariable.py#L258-L272)).

- **In `generator.py`**: Create two free functions:
  - `gen_calldata_type_name(binding: BoundVariable, cgb: CodeGenBlock, type_name: str) -> None` — same logic, takes `binding` as first arg instead of `self`.
  - `gen_call_data_code(binding: BoundVariable, cg: CodeGen, context: BindContext, depth: int = 0) -> None` — same logic, recursive calls use the free function. References to `self` become `binding`. Internal calls to `self.gen_calldata_type_name(...)` become `gen_calldata_type_name(binding, ...)`. Recursive calls on children become `gen_call_data_code(child, cg, context, depth + 1)`.
- **In [boundvariable.py](slangpy/bindings/boundvariable.py)**: Replace the method bodies with thin delegations:
  ```python
  def gen_calldata_type_name(self, cgb, type_name):
      from slangpy.core.generator import gen_calldata_type_name
      gen_calldata_type_name(self, cgb, type_name)

  def gen_call_data_code(self, cg, context, depth=0):
      from slangpy.core.generator import gen_call_data_code
      gen_call_data_code(self, cg, context, depth)
  ```
  This preserves the existing call interface (`node.gen_call_data_code(cg, context)` in [callsignature.py line 406](slangpy/core/callsignature.py#L406)) and any marshall subclass code that calls `self.gen_calldata_type_name`. The `MAX_INLINE_TYPE_LEN` constant moves to `generator.py`.
- **Move** the import of `CodeGen` and `CodeGenBlock` into `generator.py` (already needed for Step 1).

**Verify**: `pytest slangpy/tests -v` — all tests pass.

---

### Step 3: Decompose `generate_code` into sub-functions and move to `generator.py`

This is the main step. Move `generate_code` (lines 271–603 of [callsignature.py](slangpy/core/callsignature.py#L271-L603)) into `generator.py` and split it into clearly-named sub-functions. Each function has a docstring describing what Slang code it emits.

The decomposition:

| New function | Source lines | What it emits |
|---|---|---|
| `_validate_and_compute_group_shape(build_info, call_data_len) -> tuple[int, list[int], list[int]]` | [293–340](slangpy/core/callsignature.py#L293-L340) | Nothing — pure validation. Returns `(call_group_size, call_group_strides, call_group_shape_vector)`. |
| `_emit_link_time_constants(cg, build_info, call_data_len, call_group_size, call_group_strides, call_group_shape_vector)` | [342–371](slangpy/core/callsignature.py#L342-L371) | `export static const int call_data_len = ...`, group stride/shape arrays. Also calls `generate_constants()`. |
| `_emit_shape_and_metadata_params(cg, call_data_len, use_entrypoint_args)` | [373–403](slangpy/core/callsignature.py#L373-L403) | `_grid_stride`, `_grid_dim`, `_call_dim`, `_thread_count` — as entry-point params (fast) or `CallData` fields (fallback). |
| `_emit_call_data_definitions(cg, context, signature)` | [405–406](slangpy/core/callsignature.py#L405-L406) | Per-variable call data (wrapper structs, type aliases, mapping constants). Calls `gen_call_data_code` on each node. |
| `_data_name(x, use_entrypoint_args) -> str` | Duplicated at [449](slangpy/core/callsignature.py#L449) and [497](slangpy/core/callsignature.py#L497) | Helper: returns `__in_{name}`, `call_data.{name}`, or `_param_{name}`. |
| `_emit_trampoline(cg, context, build_info, signature, root_params, use_entrypoint_args)` | [408–500](slangpy/core/callsignature.py#L408-L500) | `[Differentiable] void _trampoline(...)` — param declarations, loads, function call, stores. |
| `_emit_entry_point_signature(cg, build_info, call_data_len, call_group_size, use_entrypoint_args)` | [503–541](slangpy/core/callsignature.py#L503-L541) | `[shader("compute")] [numthreads(...)] void compute_main(...)` or `[shader("raygen")] void raygen_main(...)`. |
| `_emit_kernel_body(cg, context, build_info, root_params, call_data_len, use_entrypoint_args)` | [543–603](slangpy/core/callsignature.py#L543-L603) | Bounds check, `init_thread_local_call_shape_info`, Context construction, trampoline call. |

The top-level `generate_code` becomes a ~30-line orchestrator that calls these in order:

```python
def generate_code(context, build_info, signature, cg):
    use_entrypoint_args = context.use_entrypoint_args
    cg.add_import("slangpy")
    call_data_len = context.call_dimensionality

    call_group_size, strides, shape = _validate_and_compute_group_shape(build_info, call_data_len)

    cg.add_import(build_info.module.name)
    if use_entrypoint_args:
        cg.skip_call_data = True

    _emit_link_time_constants(cg, build_info, call_data_len, call_group_size, strides, shape)
    _emit_shape_and_metadata_params(cg, call_data_len, use_entrypoint_args)
    _emit_call_data_definitions(cg, context, signature)

    root_params = sorted(signature.values(), key=lambda x: x.param_index)

    _emit_trampoline(cg, context, build_info, root_params, use_entrypoint_args)
    _emit_entry_point_signature(cg, build_info, call_data_len, call_group_size, use_entrypoint_args)
    cg.kernel.begin_block()
    _emit_kernel_body(cg, context, build_info, root_params, call_data_len, use_entrypoint_args)
    cg.kernel.end_block()
```

- **In [callsignature.py](slangpy/core/callsignature.py)**: Delete `generate_code` and add `from slangpy.core.generator import generate_code` (or let the existing wildcard import consumer in [calldata.py](slangpy/core/calldata.py) point to `generator` instead).
- **Update [calldata.py](slangpy/core/calldata.py#L8)**: Change `from slangpy.core.callsignature import *` to explicit imports: binding-pipeline functions from `callsignature`, and `generate_code`, `KernelGenException` from `generator`. This eliminates the wildcard import, making dependencies explicit.

**Verify**: `pytest slangpy/tests -v` — all tests pass. `$env:SLANGPY_PRINT_GENERATED_SHADERS="1"; pytest slangpy/tests/slangpy_tests/test_code_gen.py -v` — generated code unchanged.

---

### Step 4: Clean up `callsignature.py`

After Step 3, `callsignature.py` no longer has any codegen functions. Clean up:

- Remove unused imports that were only needed by codegen (`CodeGen`, `PipelineType`, `AccessType`, `NoneMarshall`, `BoundVariableException` if no longer referenced).
- Remove re-exports of moved symbols once [calldata.py](slangpy/core/calldata.py) uses direct imports from `generator`.
- Add `from slangpy.core.generator import KernelGenException, ResolveException` re-exports **only if** external consumers import them from `callsignature` (check via grep). If only `calldata.py` uses them, the explicit import is sufficient.

**Verify**: `pytest slangpy/tests -v`. `pre-commit run --all-files`.

---

### Step 5: Add comments to `generator.py` sub-functions

Enrich each sub-function's docstring with an example of the Slang code it generates, for both the fast path and fallback path. For example:

```python
def _emit_shape_and_metadata_params(
    cg: CodeGen,
    call_data_len: int,
    use_entrypoint_args: bool,
) -> None:
    """Emit shape arrays and _thread_count.

    Fast path (entry-point params)::

        uniform int[2] _grid_stride
        uniform int[2] _grid_dim
        uniform int[2] _call_dim
        uniform uint3 _thread_count

    Fallback (CallData struct fields)::

        int[2] _grid_stride;
        int[2] _grid_dim;
        int[2] _call_dim;
        uint3 _thread_count;
    """
```

This is documentation-only, no functional changes.

**Verify**: `pre-commit run --all-files` (formatting check).

---

### Verification

At each step:
```bash
cmake --build --preset windows-msvc-debug
pytest slangpy/tests -v
pre-commit run --all-files
```

After Step 3 specifically, also verify generated shader output is unchanged:
```powershell
$env:SLANGPY_PRINT_GENERATED_SHADERS="1"; pytest slangpy/tests/slangpy_tests/test_code_gen.py -v
```

Compare output before/after to confirm byte-identical generated Slang code.

---

### Decisions

- `gen_call_data_code` extracted as free function in `generator.py`; thin delegation stub kept on `BoundVariable` to preserve the method-call interface (`node.gen_call_data_code(cg, context)`) used in `generate_code` and potentially in external/user code.
- `generator.py` lives at `slangpy/core/generator.py` alongside `callsignature.py` and `calldata.py`.
- Wildcard import `from slangpy.core.callsignature import *` in `calldata.py` replaced with explicit imports to make dependencies clear.
- Sub-function names prefixed with `_` (private to the module); only `generate_code`, `generate_constants`, `gen_call_data_code`, `gen_calldata_type_name`, `KernelGenException` are public.

---

### Key Files

| File | Changes |
|------|---------|
| [slangpy/core/generator.py](slangpy/core/generator.py) | **NEW** — `generate_code`, `generate_constants`, `gen_call_data_code`, `gen_calldata_type_name`, `KernelGenException`, private helpers |
| [slangpy/core/callsignature.py](slangpy/core/callsignature.py) | Remove `generate_code`, `generate_constants`, `KernelGenException`, `is_slangpy_vector`; add re-exports from `generator` |
| [slangpy/bindings/boundvariable.py](slangpy/bindings/boundvariable.py) | `gen_call_data_code` and `gen_calldata_type_name` become thin delegation stubs; `MAX_INLINE_TYPE_LEN` moves out |
| [slangpy/core/calldata.py](slangpy/core/calldata.py) | Replace `from slangpy.core.callsignature import *` with explicit imports from `callsignature` and `generator` |
| [slangpy/core/dispatchdata.py](slangpy/core/dispatchdata.py) | Import `generate_constants` from `generator` instead of `callsignature` |
