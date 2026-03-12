## Plan: Consolidate test_kernel_gen.py → test_code_gen.py

**TL;DR**: Reduce 80 test functions to ~34 by: (1) merging codegen-pattern + binding-flag tests that generate the same kernel into single combined tests, (2) dropping functional GPU dispatch tests that duplicate coverage in existing test files, (3) consolidating tests that use identical Slang source strings, and (4) subsuming shallow struct nesting tests into deeper ones.

Current: **80 tests × 3 device types = 240 parametrized cases** (~1841 lines)
Proposed: **~34 tests × 3 device types = ~102 parametrized cases** (~700-800 lines)

---

### Consolidation Strategies

#### Strategy A: Merge same-source codegen tests

Five tests use `int add(int a, int b)` with args `(1, 2)` and generate the exact same kernel:
- `test_gate_scalar_uses_valuetype`
- `test_gate_valueref_write_uses_wrapper`
- `test_gate_mapping_constants_present`
- `test_gate_context_map_in_trampoline`
- `test_result_binding_not_direct_bind`

→ **1 combined test** `test_scalar_direct_bind`: generates kernel once, asserts all codegen patterns (no `ValueType`, no `typealias`, direct assignment, `RWValueRef<int>` for `_result`, no mapping constants for args, `_m__result` present, no `context.map` for args) **and** binding flags (`args[0].direct_bind=True`, `kwargs["_result"].direct_bind=False`).

Also fold `test_gate_float_scalar_uses_valuetype` into this as a sub-assertion (generate a second kernel for `float mymul` and check same patterns), or drop entirely since `int` and `float` exercise the same `ValueMarshall` code path.

#### Strategy B: Merge codegen + binding flag pairs

Each of these pairs generates the same kernel twice — merge into one test using a new `generate_code_and_bindings` helper that returns `(code, bindings)`:

| Merged test | From (codegen) | From (binding flags) |
|---|---|---|
| `test_vector_direct_bind` | `test_gate_vector_uses_vectorvaluetype` | (no binding-flag test exists; add flags check) |
| `test_struct_all_scalar_direct_bind` | `test_gate_struct_uses_slangpy_load` | `test_struct_all_scalars_binding_flag` |
| `test_mixed_scalar_tensor` | `test_gate_mixed_args_scalar_and_tensor` | `test_gate_mixed_args_direct_bind_flags` |
| `test_struct_mixed_fields` | `test_gate_struct_mixed_fields_codegen` + `test_mixed_children_direct_bind_codegen` | `test_gate_struct_mixed_fields_binding_flags` |
| `test_tensor_dim0_direct_bind` | `test_gate_tensor_dim0_codegen` | `test_gate_tensor_dim0_binding_flags` |
| `test_2d_tensor_to_vector` | `test_gate_2d_tensor_to_vector_codegen` | `test_gate_2d_tensor_to_vector_binding_flags` |
| `test_3d_tensor_to_vector` | `test_gate_3d_tensor_to_vector_codegen` | `test_gate_3d_tensor_to_vector_binding_flags` |
| `test_2d_tensor_to_scalar` | `test_gate_2d_tensor_to_scalar_codegen` | `test_gate_2d_tensor_to_scalar_binding_flags` |
| `test_2d_tensor_to_array` | `test_gate_2d_tensor_to_1d_array_codegen` | `test_gate_2d_tensor_to_1d_array_binding_flags` |
| `test_mixed_vectorized_dim0_tensor` | `test_gate_mixed_vectorized_and_dim0_tensor_codegen` | `test_gate_mixed_vectorized_and_dim0_tensor_binding_flags` |
| `test_struct_return_not_direct_bind` | `test_gate_struct_return_codegen` | `test_gate_struct_return_binding_flags` |
| `test_nested_struct_with_tensor_child` | `test_gate_nested_struct_with_tensor_child_codegen` | `test_gate_nested_struct_with_tensor_child_binding_flags` |

#### Strategy C: Subsume shallow struct tests into deeper ones

The 3-level-deep all-scalar struct test (`test_gate_deeply_nested_struct_codegen` + `test_gate_deeply_nested_struct_binding_flags`) covers the same pattern as the 2-level nested struct (`test_gate_nested_struct_codegen` + `test_gate_nested_struct_binding_flags`). Keep only **`test_deeply_nested_struct_direct_bind`** combining both. Drop tests for 2-level nesting.

#### Strategy D: Consolidate all-scalar struct composite field variants

Four test groups cover "struct with {vector/matrix/array/struct-array} fields, all dim-0 → direct-bind". These all exercise the same property (recursive `can_direct_bind` returning `True`):
- struct with vector field (`test_gate_struct_with_vector_fields_codegen` + binding)
- struct with matrix field (`test_gate_struct_with_matrix_field_codegen`)
- struct with array field (`test_gate_struct_with_array_field_codegen` + binding)
- struct with struct-array field (`test_gate_struct_with_struct_array_field_codegen`)

→ Merge into **one parametrized test** `test_struct_composite_fields_direct_bind` with parameters for the variant (vector field, array field). Drop matrix and struct-array codegen-only tests — if vector+array pass, the mechanism works for all composite field types.

#### Strategy E: Consolidate negative gates

Merge `test_gate_wanghasharg_uses_wrapper`, `test_wanghasharg_binding_flag`, and `test_struct_with_wanghash_child_not_direct_bind` into **one test** `test_wanghasharg_not_direct_bind` that covers standalone + struct-child cases.

Keep `test_gate_vectorized_scalar_keeps_wrapper` and `test_gate_vectorized_dict_keeps_struct_load` as separate small tests (they're already minimal).

#### Strategy F: Consolidate long-name tests

Merge `test_gate_long_struct_name_gets_typealias`, `test_gate_short_struct_name_inlined`, `test_gate_long_scalar_type_name_gets_typealias` into **one test** `test_long_type_name_typealias`.

#### Strategy G: Drop functional dispatch tests with existing coverage

| Dropped test | Covered by |
|---|---|
| `test_phase1_functional_scalar_add` | `test_simple_function_call.py::test_returnvalue` |
| `test_phase1_functional_float_mul` | Same mechanism as scalar_add, float type tested elsewhere |
| `test_phase1_functional_valueref_write` | `test_simple_function_call.py::test_scalar_outparam` |
| `test_phase1_functional_struct_return` | `test_return_types.py::test_return_struct_as_dict` |
| `test_phase1_functional_struct_sum` | Similar to struct_return, struct dispatch tested elsewhere |
| `test_phase1_functional_nested_struct` | Subsumed by deeply_nested + nested_with_tensor tests |
| `test_phase1_functional_struct_with_vector_fields` | Covered by composite field parametrized test pattern |
| `test_phase1_functional_struct_with_matrix_field` | Same mechanism as vector_fields test |
| `test_phase1_functional_struct_with_array_field` | Same mechanism, array dispatch tested in `test_simple_function_call.py` |
| `test_phase1_functional_deeply_nested_struct` | 3-level dispatch validates same mechanism as 2-level |
| `test_phase1_functional_vector_scale` | `test_vector_function_call.py` covers vector dispatch |
| `test_phase1_functional_3d_tensor_to_vector` | If 2D→vector works, 3D exercises same path with extra dim |
| `test_phase1_functional_2d_tensor_to_scalar` | Element-wise tensor dispatch covered by `test_tensor.py` |
| `test_phase1_functional_valueref_read_input` | Read ValueRef → scalar is tested indirectly; codegen test verifies the binding |
| `test_phase1_functional_long_struct_name` | Long name is a codegen-only concern; dispatch is identical to short-name struct |

---

### Proposed Final Test List (~34 tests)

**New helper:**
```python
def generate_code_and_bindings(device, func_name, module_source, *args, **kwargs):
    """Generate code and return (code_str, bindings) from a single debug_build_call_data call."""
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.debug_build_call_data(*args, **kwargs)
    return cd.code, cd.debug_only_bindings
```

**Codegen + binding flag tests (21):**

| # | Test name | Scenario | Merges from |
|---|---|---|---|
| 1 | `test_scalar_direct_bind` | int/float scalar dim-0; _result writable | 5 codegen tests + 1 binding test + float variant |
| 2 | `test_vector_direct_bind` | float3 dim-0 | codegen test + new binding assertions |
| 3 | `test_matrix_direct_bind` | float4x4 dim-0 | standalone |
| 4 | `test_array_direct_bind` | float[4] dim-0 | standalone |
| 5 | `test_valueref_read_direct_bind` | read-only ValueRef | standalone |
| 6 | `test_writable_valueref_not_direct_bind` | inout ValueRef (RWValueRef) | standalone |
| 7 | `test_struct_all_scalar_direct_bind` | S{float x, y} via dict | codegen + binding pair |
| 8 | `test_struct_composite_fields_direct_bind` | parametrized: struct with vector / array field | 4 codegen + 2 binding tests |
| 9 | `test_deeply_nested_struct_direct_bind` | 3-level Top{Mid{Bot}} | subsumes 2-level; codegen + binding pair |
| 10 | `test_struct_mixed_fields` | S{x(tensor), y(scalar)} | 2 codegen + 1 binding test |
| 11 | `test_nested_struct_with_tensor_child` | Outer{Inner{x(tensor),y},s} | codegen + binding pair |
| 12 | `test_struct_return_not_direct_bind` | function returning struct | codegen + binding pair |
| 13 | `test_struct_vectorized_2d_child` | S{float3 v (2D tensor), float s} | standalone |
| 14 | `test_mixed_scalar_and_tensor` | scalar + tensor args | codegen + binding pair |
| 15 | `test_tensor_dim0_direct_bind` | Tensor<float,1> at dim-0 | codegen + binding pair |
| 16 | `test_2d_tensor_to_vector` | 2D(10,3) → float3 | codegen + binding pair |
| 17 | `test_3d_tensor_to_vector` | 3D(2,5,3) → float3 | codegen + binding pair |
| 18 | `test_2d_tensor_to_scalar` | 2D(4,5) → float | codegen + binding pair |
| 19 | `test_2d_tensor_to_array` | 2D(4,8) → half[8] | codegen + binding pair |
| 20 | `test_mixed_vectorized_dim0_tensor` | vectorized + dim-0 tensor | codegen + binding pair |
| 21 | `test_long_type_name_typealias` | long/short struct name, wrapper name | 3 tests merged |

**Negative gates (3):**

| # | Test name | Scenario | Merges from |
|---|---|---|---|
| 22 | `test_wanghasharg_not_direct_bind` | standalone + struct child | 3 tests merged |
| 23 | `test_vectorized_scalar_keeps_wrapper` | 1D tensor → float | standalone |
| 24 | `test_vectorized_dict_keeps_wrapper` | dict with tensor children | standalone |

**Autodiff (1):**

| # | Test name | Scenario | Merges from |
|---|---|---|---|
| 25 | `test_bwds_direct_bind` | codegen + binding flags for bwds polynomial | 3 tests merged |

**Functional GPU dispatch — novel scenarios only (9):**

| # | Test name | Scenario | Why novel |
|---|---|---|---|
| 26 | `test_dispatch_mixed_scalar_tensor` | scalar + 1D tensor | Not tested elsewhere |
| 27 | `test_dispatch_struct_mixed_fields` | struct{tensor+scalar} | Unique dispatch scenario |
| 28 | `test_dispatch_tensor_dim0` | Tensor<float,1> at dim-0 | Specific dim-0 behavior |
| 29 | `test_dispatch_2d_tensor_to_vector` | 2D→float3 | Novel param mapping |
| 30 | `test_dispatch_2d_tensor_to_array` | 2D→half[8] generic | Unique test |
| 31 | `test_dispatch_mixed_vectorized_dim0_tensor` | vectorized + dim-0 tensor | Unique |
| 32 | `test_dispatch_nested_struct_with_tensor` | nested struct with tensor leaf | Unique |
| 33 | `test_dispatch_struct_vectorized_2d_child` | struct with 2D tensor→float3 child | Unique |
| 34 | `test_dispatch_struct_array_of_structs` | struct with `Inner items[4]` | Unique |

---

### Old → New Mapping

| Old test (test_kernel_gen.py) | New test (test_code_gen.py) | Action |
|---|---|---|
| `test_kernel_gen_basic` | — | **Dropped** (subset of `test_scalar_direct_bind`) |
| `test_gate_scalar_uses_valuetype` | `test_scalar_direct_bind` | **Merged** |
| `test_gate_float_scalar_uses_valuetype` | `test_scalar_direct_bind` | **Merged** (or dropped) |
| `test_gate_vector_uses_vectorvaluetype` | `test_vector_direct_bind` | **Merged** |
| `test_gate_matrix_uses_valuetype` | `test_matrix_direct_bind` | **Kept** (standalone) |
| `test_gate_array_dim0_uses_valuetype` | `test_array_direct_bind` | **Kept** (standalone) |
| `test_gate_valueref_read_uses_wrapper` | `test_valueref_read_direct_bind` | **Kept** (standalone) |
| `test_gate_valueref_write_uses_wrapper` | `test_scalar_direct_bind` | **Merged** |
| `test_gate_mapping_constants_present` | `test_scalar_direct_bind` | **Merged** |
| `test_gate_context_map_in_trampoline` | `test_scalar_direct_bind` | **Merged** |
| `test_gate_struct_uses_slangpy_load` | `test_struct_all_scalar_direct_bind` | **Merged** |
| `test_gate_bwds_scalar_uses_valuetype` | `test_bwds_direct_bind` | **Merged** |
| `test_gate_bwds_trampoline_is_differentiable` | `test_bwds_direct_bind` | **Merged** |
| `test_gate_wanghasharg_uses_wrapper` | `test_wanghasharg_not_direct_bind` | **Merged** |
| `test_gate_vectorized_scalar_keeps_wrapper` | `test_vectorized_scalar_keeps_wrapper` | **Kept** |
| `test_gate_vectorized_dict_keeps_struct_load` | `test_vectorized_dict_keeps_wrapper` | **Kept** |
| `test_phase1_functional_scalar_add` | — | **Dropped** (covered by `test_simple_function_call.py`) |
| `test_phase1_functional_float_mul` | — | **Dropped** |
| `test_phase1_functional_vector_scale` | — | **Dropped** (covered by `test_vector_function_call.py`) |
| `test_phase1_functional_struct_sum` | — | **Dropped** |
| `test_phase1_functional_valueref_write` | — | **Dropped** (covered by `test_simple_function_call.py`) |
| `test_gate_mixed_args_scalar_and_tensor` | `test_mixed_scalar_and_tensor` | **Merged** |
| `test_gate_mixed_args_direct_bind_flags` | `test_mixed_scalar_and_tensor` | **Merged** |
| `test_phase1_functional_mixed_scalar_tensor` | `test_dispatch_mixed_scalar_tensor` | **Kept** |
| `test_gate_struct_mixed_fields_codegen` | `test_struct_mixed_fields` | **Merged** |
| `test_gate_struct_mixed_fields_binding_flags` | `test_struct_mixed_fields` | **Merged** |
| `test_phase1_functional_struct_mixed_fields` | `test_dispatch_struct_mixed_fields` | **Kept** |
| `test_gate_tensor_dim0_codegen` | `test_tensor_dim0_direct_bind` | **Merged** |
| `test_gate_tensor_dim0_binding_flags` | `test_tensor_dim0_direct_bind` | **Merged** |
| `test_phase1_functional_tensor_dim0` | `test_dispatch_tensor_dim0` | **Kept** |
| `test_mixed_children_direct_bind_codegen` | `test_struct_mixed_fields` | **Merged** (overlap with struct_mixed_fields) |
| `test_writable_valueref_not_direct_bind` | `test_writable_valueref_not_direct_bind` | **Kept** |
| `test_result_binding_not_direct_bind` | `test_scalar_direct_bind` | **Merged** |
| `test_struct_all_scalars_binding_flag` | `test_struct_all_scalar_direct_bind` | **Merged** |
| `test_struct_with_wanghash_child_not_direct_bind` | `test_wanghasharg_not_direct_bind` | **Merged** |
| `test_wanghasharg_binding_flag` | `test_wanghasharg_not_direct_bind` | **Merged** |
| `test_bwds_primal_binding_flags` | `test_bwds_direct_bind` | **Merged** |
| `test_gate_2d_tensor_to_vector_codegen` | `test_2d_tensor_to_vector` | **Merged** |
| `test_gate_2d_tensor_to_vector_binding_flags` | `test_2d_tensor_to_vector` | **Merged** |
| `test_phase1_functional_2d_tensor_to_vector` | `test_dispatch_2d_tensor_to_vector` | **Kept** |
| `test_gate_3d_tensor_to_vector_codegen` | `test_3d_tensor_to_vector` | **Merged** |
| `test_gate_3d_tensor_to_vector_binding_flags` | `test_3d_tensor_to_vector` | **Merged** |
| `test_phase1_functional_3d_tensor_to_vector` | — | **Dropped** (2D→vector is sufficient) |
| `test_gate_2d_tensor_to_scalar_codegen` | `test_2d_tensor_to_scalar` | **Merged** |
| `test_gate_2d_tensor_to_scalar_binding_flags` | `test_2d_tensor_to_scalar` | **Merged** |
| `test_phase1_functional_2d_tensor_to_scalar` | — | **Dropped** (covered by `test_tensor.py`) |
| `test_gate_2d_tensor_to_1d_array_codegen` | `test_2d_tensor_to_array` | **Merged** |
| `test_gate_2d_tensor_to_1d_array_binding_flags` | `test_2d_tensor_to_array` | **Merged** |
| `test_phase1_functional_2d_tensor_to_1d_array` | `test_dispatch_2d_tensor_to_array` | **Kept** |
| `test_gate_mixed_vectorized_and_dim0_tensor_codegen` | `test_mixed_vectorized_dim0_tensor` | **Merged** |
| `test_gate_mixed_vectorized_and_dim0_tensor_binding_flags` | `test_mixed_vectorized_dim0_tensor` | **Merged** |
| `test_phase1_functional_mixed_vectorized_and_dim0_tensor` | `test_dispatch_mixed_vectorized_dim0_tensor` | **Kept** |
| `test_gate_nested_struct_codegen` | — | **Dropped** (subsumed by deeply_nested) |
| `test_gate_nested_struct_binding_flags` | — | **Dropped** (subsumed by deeply_nested) |
| `test_phase1_functional_nested_struct` | — | **Dropped** |
| `test_gate_struct_with_vector_fields_codegen` | `test_struct_composite_fields_direct_bind` | **Merged** (parametrized) |
| `test_gate_struct_with_vector_fields_binding_flags` | `test_struct_composite_fields_direct_bind` | **Merged** |
| `test_phase1_functional_struct_with_vector_fields` | — | **Dropped** |
| `test_gate_struct_with_matrix_field_codegen` | — | **Dropped** (covered by vector+array variants) |
| `test_phase1_functional_struct_with_matrix_field` | — | **Dropped** |
| `test_gate_struct_with_array_field_codegen` | `test_struct_composite_fields_direct_bind` | **Merged** (parametrized) |
| `test_gate_struct_with_array_field_binding_flags` | `test_struct_composite_fields_direct_bind` | **Merged** |
| `test_phase1_functional_struct_with_array_field` | — | **Dropped** |
| `test_gate_deeply_nested_struct_codegen` | `test_deeply_nested_struct_direct_bind` | **Merged** |
| `test_gate_deeply_nested_struct_binding_flags` | `test_deeply_nested_struct_direct_bind` | **Merged** |
| `test_phase1_functional_deeply_nested_struct` | — | **Dropped** |
| `test_gate_nested_struct_with_tensor_child_codegen` | `test_nested_struct_with_tensor_child` | **Merged** |
| `test_gate_nested_struct_with_tensor_child_binding_flags` | `test_nested_struct_with_tensor_child` | **Merged** |
| `test_phase1_functional_nested_struct_with_tensor` | `test_dispatch_nested_struct_with_tensor` | **Kept** |
| `test_gate_struct_with_struct_array_field_codegen` | — | **Dropped** (covered by array field variant) |
| `test_phase1_functional_struct_with_struct_array_field` | `test_dispatch_struct_array_of_structs` | **Kept** |
| `test_gate_struct_return_codegen` | `test_struct_return_not_direct_bind` | **Merged** |
| `test_gate_struct_return_binding_flags` | `test_struct_return_not_direct_bind` | **Merged** |
| `test_phase1_functional_struct_return` | — | **Dropped** (covered by `test_return_types.py`) |
| `test_gate_struct_with_vectorized_2d_tensor_child_codegen` | `test_struct_vectorized_2d_child` | **Kept** |
| `test_phase1_functional_struct_with_vectorized_2d_tensor` | `test_dispatch_struct_vectorized_2d_child` | **Kept** |
| `test_gate_long_struct_name_gets_typealias` | `test_long_type_name_typealias` | **Merged** |
| `test_gate_short_struct_name_inlined` | `test_long_type_name_typealias` | **Merged** |
| `test_gate_long_scalar_type_name_gets_typealias` | `test_long_type_name_typealias` | **Merged** |
| `test_phase1_functional_long_struct_name` | — | **Dropped** |
| `test_phase1_functional_valueref_read_input` | — | **Dropped** |

---

### Verification

```bash
# Build first (required)
cmake --build --preset windows-msvc-debug

# Run new test file
pytest slangpy/tests/slangpy_tests/test_code_gen.py -v

# Confirm full suite still passes (existing tests in other files cover dropped dispatch tests)
pytest slangpy/tests -v

# Run pre-commit
pre-commit run --all-files
```

### Key Decisions

- Combined codegen+binding tests: one `debug_build_call_data` call yields both `.code` and `.debug_only_bindings` — no redundant kernel generation
- Dropped `test_kernel_gen_basic`: its sole assertion (`"add" in code`) is a strict subset of `test_scalar_direct_bind`
- Dropped matrix/struct-array field variants: if vector field and array field pass, the `can_direct_bind` recursion works for all composite types
- Dropped 2-level nested struct: the 3-level test covers the same recursion with deeper nesting
- Dropped 15 functional dispatch tests that are covered by existing test files (`test_simple_function_call.py`, `test_return_types.py`, `test_vector_function_call.py`, `test_tensor.py`)
- Kept all negative gates — they deliberately test types NOT eligible for simplification and must remain passing as Phase 2 proceeds
- The old `test_kernel_gen.py` should be deleted once the new `test_code_gen.py` is verified
