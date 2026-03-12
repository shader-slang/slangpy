# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Kernel generation test.

These tests exercise different code paths for kernel generation, to exercise different kernel types, such as:
- passing arguments directly vs via call data
- passing read-only arguments that don't need storing directly rather than via marshalls
- handling the semantic 'dispatch thread id' etc and calling kernels directly

Gating tests (test_gate_*) assert CURRENT generated kernel patterns and will
intentionally break as simplification steps from the kernel-gen simplification
plan are implemented. Negative gates (test_gate_*_keeps_*) must remain
passing after simplification — they cover types that are NOT direct-bind
eligible.
"""

from typing import Any

import numpy as np
import pytest
import os

import slangpy as spy
from slangpy.testing import helpers
from slangpy.types import ValueRef, Tensor, diffPair
from slangpy.types.wanghasharg import WangHashArg

PRINT_TEST_KERNEL_GEN = os.getenv("PRINT_TEST_KERNEL_GEN", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_contains(code: str, *patterns: str) -> None:
    """Assert all patterns appear in generated code."""
    for p in patterns:
        assert p in code, f"Expected pattern not found: {p}"


def assert_not_contains(code: str, *patterns: str) -> None:
    """Assert none of the patterns appear in generated code."""
    for p in patterns:
        assert p not in code, f"Unexpected pattern found: {p}"


def assert_trampoline_has(code: str, *stmts: str) -> None:
    """Assert trampoline contains statements, insensitive to call_data vs __calldata__ prefix."""
    for s in stmts:
        # Replace __calldata__ with both options for matching
        if "__calldata__." in s:
            alt = s.replace("__calldata__.", "call_data.")
            assert (
                s in code or alt in code
            ), f"Expected trampoline statement not found: {s} (or {alt})"
        else:
            assert s in code, f"Expected trampoline statement not found: {s}"


def generate_code(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> str:
    """
    Generate code for the given function and arguments, and return the generated code as a string.
    """
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.debug_build_call_data(*args, **kwargs)
    if PRINT_TEST_KERNEL_GEN:
        print(cd.code)
    return cd.code


def generate_bwds_code(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> str:
    """
    Generate backwards-mode code for the given function and arguments.
    """
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.bwds.debug_build_call_data(*args, **kwargs)
    if PRINT_TEST_KERNEL_GEN:
        print(cd.code)
    return cd.code


# ---------------------------------------------------------------------------
# Basic test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_kernel_gen_basic(device_type: spy.DeviceType):
    """
    Test basic kernel generation with a simple function that adds two numbers.
    """
    src = """
int add(int a, int b) {
    return a + b;
}
"""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", src, 1, 2)
    print(code)
    assert "add" in code


# ===========================================================================
# Phase 1 tests — assert direct-bind behaviour after implementation
# ===========================================================================

# -- Step 1.2: Scalar direct binding --


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_scalar_uses_valuetype(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "add",
        "int add(int a, int b) { return a + b; }",
        1,
        2,
    )
    # Scalars now use direct binding: type used directly in CallData, no ValueType wrapper
    assert_not_contains(code, "ValueType<int>")
    assert_not_contains(code, "typealias _t_a", "typealias _t_b")
    # Trampoline uses direct assignment, no __slangpy_load
    assert_trampoline_has(code, "a = __calldata__.a;", "b = __calldata__.b;")
    # _result is auto-created as writable RWValueRef (not direct-bind)
    assert_contains(code, "RWValueRef<int>")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_float_scalar_uses_valuetype(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "mymul",
        "float mymul(float x, float y) { return x * y; }",
        1.0,
        2.0,
    )
    assert_not_contains(code, "ValueType<float>")
    assert_not_contains(code, "typealias _t_x", "typealias _t_y")


# -- Step 1.3: Vector / Matrix / Array direct binding --


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_vector_uses_vectorvaluetype(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "scale",
        "float3 scale(float3 v, float s) { return v * s; }",
        spy.math.float3(1, 2, 3),
        1.0,
    )
    assert_not_contains(code, "VectorValueType<float,3>")
    assert_not_contains(code, "typealias _t_v")
    assert_contains(code, "vector<float,3> v;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_matrix_uses_valuetype(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "ident",
        "float4x4 ident(float4x4 m) { return m; }",
        spy.math.float4x4.identity(),
    )
    assert_not_contains(code, "ValueType<matrix<float,4,4>>")
    assert_not_contains(code, "typealias _t_m")
    assert_contains(code, "matrix<float,4,4> m;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_array_dim0_uses_valuetype(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "process",
        "void process(float a[4]) { }",
        [1.0, 2.0, 3.0, 4.0],
    )
    assert_not_contains(code, "ValueType<")
    assert_not_contains(code, "typealias _t_a")


# -- Step 1.5: ValueRef direct binding --


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_valueref_read_uses_wrapper(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "read_val",
        "float read_val(float v) { return v; }",
        ValueRef(1.0),
    )
    # Read-only ValueRef uses raw type directly (direct-bind)
    assert_not_contains(code, "typealias _t_v")
    assert_contains(code, "float v;")
    # Direct assignment in trampoline
    assert_trampoline_has(code, "v = __calldata__.v;")
    # _result (writable) still uses RWValueRef wrapper
    assert_contains(code, "RWValueRef<float>")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_valueref_write_uses_wrapper(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "add",
        "int add(int a, int b) { return a + b; }",
        1,
        2,
    )
    # Auto-created _result uses RWValueRef (writable, not direct-bind)
    assert_contains(code, "RWValueRef<int>")
    # Trampoline uses __slangpy_store via wrapper
    assert_contains(code, "__slangpy_store")


# -- Step 1.7: Mapping constants and context.map --


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_mapping_constants_present(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "add",
        "int add(int a, int b) { return a + b; }",
        1,
        2,
    )
    # Direct-bind variables no longer emit mapping constants
    assert_not_contains(
        code,
        "static const int _m_a = 0",
        "static const int _m_b = 0",
    )
    # _result is NOT direct-bind (writable ValueRef), so it keeps mapping constant
    assert_contains(code, "static const int _m__result = 0")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_context_map_in_trampoline(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    code = generate_code(
        device,
        "add",
        "int add(int a, int b) { return a + b; }",
        1,
        2,
    )
    # Direct-bind variables don't use context.map
    assert_not_contains(code, "__slangpy_context__.map(_m_a)")


# -- Step 1.4: Struct / dict direct binding --


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_uses_slangpy_load(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
float sum(S s) { return s.x + s.y; }
"""
    code = generate_code(device, "sum", src, {"_type": "S", "x": 1.0, "y": 2.0})
    # Direct-bind struct: uses raw type directly, no inline struct with __slangpy_load
    assert_not_contains(code, "__slangpy_load")
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "S s;")
    # Direct assignment in trampoline
    assert_trampoline_has(code, "s = __calldata__.s;")


# -- Step 1.8: Autodiff gating --


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_bwds_scalar_uses_valuetype(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    src = """
[Differentiable]
float polynomial(float a, float b) {
    return a * a + b + 1;
}
"""
    code = generate_bwds_code(device, "polynomial", src, 5.0, 10.0, 26.0)
    # bwds still uses direct bind for primals; check differentiable markers remain
    assert_not_contains(code, "ValueType<float>")
    assert_contains(code, "[Differentiable]", "bwd_diff(_trampoline)")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_bwds_trampoline_is_differentiable(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    src = """
[Differentiable]
float polynomial(float a, float b) {
    return a * a + b + 1;
}
"""
    code = generate_bwds_code(device, "polynomial", src, 5.0, 10.0, 26.0)
    # [Differentiable] should appear before the trampoline function
    diff_idx = code.index("[Differentiable]")
    trampoline_idx = code.index("void _trampoline")
    assert diff_idx < trampoline_idx


# ===========================================================================
# Phase 1 negative gates — must REMAIN passing after Phase 1
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_wanghasharg_uses_wrapper(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    src = "uint3 rng(uint3 input) { return input; }"
    code = generate_code(device, "rng", src, WangHashArg(3))
    assert_contains(code, "WangHashArg<")
    # WangHashArg uses wrapper type — field declaration present in CallData
    assert_contains(code, "input")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_vectorized_scalar_keeps_wrapper(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    src = "float square(float x) { return x * x; }"
    tensor = Tensor.from_numpy(
        helpers.get_device(device_type), np.array([1, 2, 3], dtype=np.float32)
    )
    code = generate_code(device, "square", src, tensor)
    # Vectorized (dim > 0) — tensor marshall used, __slangpy_load still present
    assert_contains(code, "__slangpy_load")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_vectorized_dict_keeps_struct_load(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
void apply(S s, float scale) {}
"""
    tensor_x = Tensor.from_numpy(
        helpers.get_device(device_type), np.array([1, 2, 3], dtype=np.float32)
    )
    tensor_y = Tensor.from_numpy(
        helpers.get_device(device_type), np.array([4, 5, 6], dtype=np.float32)
    )
    code = generate_code(device, "apply", src, {"_type": "S", "x": tensor_x, "y": tensor_y}, 1.0)
    # Children are vectorized (dim > 0) — should keep inline struct with __slangpy_load
    assert_contains(code, "__slangpy_load")


# ===========================================================================
# Phase 1 functional dispatch tests — verify GPU results are correct
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_scalar_add(device_type: spy.DeviceType):
    """Dispatch scalar add with direct binding and verify GPU result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "add", "int add(int a, int b) { return a + b; }"
    )
    result = func(3, 7)
    assert result == 10


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_float_mul(device_type: spy.DeviceType):
    """Dispatch float multiply with direct binding."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "mymul", "float mymul(float x, float y) { return x * y; }"
    )
    result = func(3.0, 4.0)
    assert abs(result - 12.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_vector_scale(device_type: spy.DeviceType):
    """Dispatch vector scale with direct binding."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "scale", "float3 scale(float3 v, float s) { return v * s; }"
    )
    result = func(spy.math.float3(1, 2, 3), 2.0)
    assert result.x == 2.0
    assert result.y == 4.0
    assert result.z == 6.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_sum(device_type: spy.DeviceType):
    """Dispatch struct sum via dict with direct binding."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
float sum(S s) { return s.x + s.y; }
"""
    func = helpers.create_function_from_module(device, "sum", src)
    result = func({"_type": "S", "x": 3.0, "y": 7.0})
    assert abs(result - 10.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_valueref_write(device_type: spy.DeviceType):
    """Dispatch with explicit ValueRef output and read back."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "add", "int add(int a, int b) { return a + b; }"
    )
    out = ValueRef(0)
    func(5, 8, _result=out)
    assert out.value == 13


# ===========================================================================
# Mixed direct-bind tests — some args direct, some not
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_mixed_args_scalar_and_tensor(device_type: spy.DeviceType):
    """Scalar arg gets direct-bind; vectorized tensor arg does not."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code = generate_code(
        device,
        "add",
        "float add(float a, float b) { return a + b; }",
        1.0,
        tensor,
    )
    # 'a' is direct-bind (scalar dim-0): type used directly, direct trampoline load
    assert_not_contains(code, "typealias _t_a")
    assert_not_contains(code, "ValueType<float>")
    assert_trampoline_has(code, "a = __calldata__.a;")
    # 'b' is NOT direct-bind (vectorized tensor dim-1): uses Tensor<float, 1>,
    # __slangpy_load, and mapping constant
    assert_contains(code, "Tensor<float, 1>")
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_b")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_mixed_args_direct_bind_flags(device_type: spy.DeviceType):
    """Verify direct_bind flags on bindings for mixed scalar + tensor call."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    func = helpers.create_function_from_module(
        device, "add", "float add(float a, float b) { return a + b; }"
    )
    cd = func.debug_build_call_data(1.0, tensor)
    bindings = cd.debug_only_bindings
    assert bindings.args[0].direct_bind is True, "scalar arg 'a' should be direct_bind"
    assert bindings.args[0].call_dimensionality == 0
    assert bindings.args[1].direct_bind is False, "tensor arg 'b' should NOT be direct_bind"
    assert bindings.args[1].call_dimensionality == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_mixed_scalar_tensor(device_type: spy.DeviceType):
    """Dispatch mixed scalar + tensor and verify GPU result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "add", "float add(float a, float b) { return a + b; }"
    )
    tensor = Tensor.from_numpy(device, np.array([10, 20, 30], dtype=np.float32))
    result = func(5.0, tensor)
    expected = np.array([15, 25, 35], dtype=np.float32)
    np.testing.assert_allclose(result.to_numpy().flatten(), expected, atol=1e-5)


# ===========================================================================
# Struct with mixed direct-bind fields
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_mixed_fields_codegen(device_type: spy.DeviceType):
    """Struct with one tensor field and one scalar field.

    The struct is NOT direct-bind because child x is vectorized (dim-1).
    Child y (scalar) keeps direct_bind=True — gen_call_data_code emits
    direct assignment (value.y = y) instead of y.__slangpy_load(...).
    """
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
void apply(S s, float scale) {}
"""
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code = generate_code(device, "apply", src, {"_type": "S", "x": tensor_x, "y": 1.0}, 2.0)
    # Struct is NOT direct-bind: uses inline struct with __slangpy_load
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "struct _t_s")
    assert_not_contains(code, "typealias _t_s = S;")
    # Child y is direct-bind: type used directly, direct assignment in __slangpy_load
    assert_not_contains(code, "typealias _t_y")
    assert_contains(code, "float y;")
    assert_contains(code, "value.y = y;")
    assert_not_contains(code, "ValueType<float>")
    # Child x should use tensor type
    assert_contains(code, "Tensor<float, 1>")
    # Scalar arg 'scale' is independent — should still be direct-bind
    assert_not_contains(code, "typealias _t_scale")
    assert_contains(code, "float scale;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_mixed_fields_binding_flags(device_type: spy.DeviceType):
    """Verify direct_bind flags on struct children when struct is NOT direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
void apply(S s, float scale) {}
"""
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    func = helpers.create_function_from_module(device, "apply", src)
    cd = func.debug_build_call_data({"_type": "S", "x": tensor_x, "y": 1.0}, 2.0)
    bindings = cd.debug_only_bindings
    s_binding = bindings.args[0]
    assert s_binding.direct_bind is False, "struct 's' should NOT be direct_bind"
    # Child x is a tensor (dim-1), not direct-bind
    assert s_binding.children["x"].direct_bind is False
    # Child y is a scalar (dim-0), keeps its direct_bind status
    assert s_binding.children["y"].direct_bind is True
    # 'scale' is independent scalar — should be direct_bind
    assert bindings.args[1].direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_mixed_fields(device_type: spy.DeviceType):
    """Dispatch struct with mixed tensor+scalar fields and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
float weighted_sum(S s, float scale) { return (s.x + s.y) * scale; }
"""
    func = helpers.create_function_from_module(device, "weighted_sum", src)
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    result = func({"_type": "S", "x": tensor_x, "y": 10.0}, 2.0)
    expected = np.array([22, 24, 26], dtype=np.float32)
    np.testing.assert_allclose(result.to_numpy().flatten(), expected, atol=1e-5)


# ===========================================================================
# Tensor at dim-0 (whole tensor passed to Tensor<T,N> parameter)
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_tensor_dim0_codegen(device_type: spy.DeviceType):
    """1D Tensor passed to Tensor<float,1> param — dim-0, direct assignment."""
    device = helpers.get_device(device_type)
    src = """
float tensor_read(Tensor<float,1> t) {
    return t[0];
}
"""
    tensor = Tensor.from_numpy(device, np.array([42, 2, 3], dtype=np.float32))
    code = generate_code(device, "tensor_read", src, tensor)
    # Type should use Tensor<float, 1> directly (no typealias)
    assert_not_contains(code, "typealias _t_t")
    assert_contains(code, "Tensor<float, 1> t;")
    # Trampoline uses direct assignment (not __slangpy_load)
    assert_trampoline_has(code, "t = __calldata__.t;")
    # No wrapper type for the tensor
    assert_not_contains(code, "ValueType<")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_tensor_dim0_binding_flags(device_type: spy.DeviceType):
    """Tensor at dim-0 has direct_bind=True (consistent with other dim-0 types)."""
    device = helpers.get_device(device_type)
    src = """
float tensor_read(Tensor<float,1> t) {
    return t[0];
}
"""
    tensor = Tensor.from_numpy(device, np.array([42, 2, 3], dtype=np.float32))
    func = helpers.create_function_from_module(device, "tensor_read", src)
    cd = func.debug_build_call_data(tensor)
    bindings = cd.debug_only_bindings
    t_binding = bindings.args[0]
    assert t_binding.direct_bind is True
    assert t_binding.call_dimensionality == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_tensor_dim0(device_type: spy.DeviceType):
    """Dispatch with whole tensor at dim-0 and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
float tensor_read(Tensor<float,1> t) {
    return t[0];
}
"""
    func = helpers.create_function_from_module(device, "tensor_read", src)
    tensor = Tensor.from_numpy(device, np.array([42, 99, 7], dtype=np.float32))
    result = func(tensor)
    assert abs(result - 42.0) < 1e-5


# ===========================================================================
# Mixed direct-bind children in non-direct-bind struct — validates that
# gen_call_data_code correctly uses direct assignment for direct-bind
# children and __slangpy_load for non-direct-bind children.
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mixed_children_direct_bind_codegen(device_type: spy.DeviceType):
    """Validate code gen for struct with mixed direct-bind / non-direct-bind children.

    Scalar child y gets direct assignment (value.y = y) inside __slangpy_load.
    Tensor child x goes through __slangpy_load with context mapping.
    Both patterns coexist in the same generated struct.
    """
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
float weighted_sum(S s, float scale) { return (s.x + s.y) * scale; }
"""
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code = generate_code(device, "weighted_sum", src, {"_type": "S", "x": tensor_x, "y": 1.0}, 2.0)
    # Child y uses raw type and direct assignment
    assert_not_contains(code, "typealias _t_y")
    assert_contains(code, "float y;")
    assert_contains(code, "value.y = y;")
    # No mapping constant for y (direct-bind skips it)
    assert_not_contains(code, "_m_y")
    # Child x uses tensor wrapper with __slangpy_load
    assert_contains(code, "x.__slangpy_load(context.map(_m_x),value.x)")
    # The struct itself is not direct-bind
    assert_contains(code, "struct _t_s")


# ===========================================================================
# Review coverage — binding flag verification tests
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_writable_valueref_not_direct_bind(device_type: spy.DeviceType):
    """Writable ValueRef (inout) must not be direct-bind — needs buffer read/write."""
    device = helpers.get_device(device_type)
    src = "void inc(inout int v) { v += 1; }"
    func = helpers.create_function_from_module(device, "inc", src)
    vr = ValueRef(5)
    cd = func.debug_build_call_data(vr)
    bindings = cd.debug_only_bindings
    v_binding = bindings.args[0]
    assert v_binding.direct_bind is False
    assert v_binding.call_dimensionality == 0
    code = cd.code
    assert_contains(code, "RWValueRef<int>")
    assert_not_contains(code, "typealias _t_v = int;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_result_binding_not_direct_bind(device_type: spy.DeviceType):
    """Auto-created _result (writable ValueRef) must not be direct-bind."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "add", "int add(int a, int b) { return a + b; }"
    )
    cd = func.debug_build_call_data(1, 2)
    result_binding = cd.debug_only_bindings.kwargs["_result"]
    assert result_binding.direct_bind is False
    assert result_binding.call_dimensionality == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_all_scalars_binding_flag(device_type: spy.DeviceType):
    """All-scalar struct at dim-0 should have direct_bind=True (and so should children)."""
    device = helpers.get_device(device_type)
    src = """
struct S { float x; float y; };
float sum(S s) { return s.x + s.y; }
"""
    func = helpers.create_function_from_module(device, "sum", src)
    cd = func.debug_build_call_data({"_type": "S", "x": 1.0, "y": 2.0})
    bindings = cd.debug_only_bindings
    s = bindings.args[0]
    assert s.direct_bind is True
    assert s.children["x"].direct_bind is True
    assert s.children["y"].direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_with_wanghash_child_not_direct_bind(device_type: spy.DeviceType):
    """Struct with a WangHashArg child must NOT be direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct S { uint3 seed; float scale; };
float apply(S s) { return float(s.seed.x) * s.scale; }
"""
    func = helpers.create_function_from_module(device, "apply", src)
    cd = func.debug_build_call_data({"_type": "S", "seed": WangHashArg(3), "scale": 1.0})
    bindings = cd.debug_only_bindings
    s = bindings.args[0]
    assert s.direct_bind is False
    # scale child should still be direct-bind individually
    assert s.children["scale"].direct_bind is True
    code = cd.code
    assert_contains(code, "struct _t_s")
    assert_not_contains(code, "typealias _t_s = S;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_wanghasharg_binding_flag(device_type: spy.DeviceType):
    """WangHashArg (no can_direct_bind override) should have direct_bind=False."""
    device = helpers.get_device(device_type)
    src = "uint3 rng(uint3 input) { return input; }"
    func = helpers.create_function_from_module(device, "rng", src)
    cd = func.debug_build_call_data(WangHashArg(3))
    bindings = cd.debug_only_bindings
    assert bindings.args[0].direct_bind is False
    assert bindings.args[0].call_dimensionality == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_valueref_read_input(device_type: spy.DeviceType):
    """Dispatch with a read-only ValueRef input — verifies direct-bind ValueRef pipeline end-to-end."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "double_it", "float double_it(float v) { return v * 2; }"
    )
    result = func(ValueRef(7.0))
    assert abs(result - 14.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bwds_primal_binding_flags(device_type: spy.DeviceType):
    """In bwds mode, primal args (access[0]=read) should have direct_bind=True."""
    device = helpers.get_device(device_type)
    src = """
[Differentiable]
float polynomial(float a, float b) { return a * a + b + 1; }
"""
    func = helpers.create_function_from_module(device, "polynomial", src)
    cd = func.bwds.debug_build_call_data(5.0, 10.0, 26.0)
    bindings = cd.debug_only_bindings
    # Primal args in bwds mode → access[0]=read → direct_bind should be True
    assert bindings.args[0].direct_bind is True  # 'a'
    assert bindings.args[1].direct_bind is True  # 'b'


# ===========================================================================
# ND tensor → (N-1)D parameter vectorization — kernel source pattern tests
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_2d_tensor_to_vector_codegen(device_type: spy.DeviceType):
    """2D Tensor shape=(10,3) → float3 param: trailing dim consumed by vector, outer dim dispatched."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((10, 3), dtype=np.float32))
    code = generate_code(
        device,
        "scale",
        "float3 scale(float3 v, float s) { return v * s; }",
        tensor,
        2.0,
    )
    # v is vectorized dim-1: tensor wrapping a vector type
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_v")
    # s is scalar dim-0: direct-bind, type used directly
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "float s;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_2d_tensor_to_vector_binding_flags(device_type: spy.DeviceType):
    """2D Tensor shape=(10,3) → float3 param: check binding metadata."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((10, 3), dtype=np.float32))
    func = helpers.create_function_from_module(
        device,
        "scale",
        "float3 scale(float3 v, float s) { return v * s; }",
    )
    cd = func.debug_build_call_data(tensor, 2.0)
    bindings = cd.debug_only_bindings
    v_binding = bindings.args[0]
    # Tensor vectorized over outer dim: call_dimensionality == 1
    assert v_binding.call_dimensionality == 1
    assert v_binding.direct_bind is False
    assert v_binding.vector_type is not None
    assert v_binding.vector_type.full_name == "vector<float,3>"
    # Scalar s: dim-0 direct-bind
    s_binding = bindings.args[1]
    assert s_binding.call_dimensionality == 0
    assert s_binding.direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_2d_tensor_to_vector(device_type: spy.DeviceType):
    """Dispatch 2D tensor → float3 and verify GPU result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device,
        "scale",
        "float3 scale(float3 v, float s) { return v * s; }",
    )
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = Tensor.from_numpy(device, data)
    result = func(tensor, 2.0)
    expected = data * 2.0
    np.testing.assert_allclose(result.to_numpy().reshape(expected.shape), expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_3d_tensor_to_vector_codegen(device_type: spy.DeviceType):
    """3D Tensor shape=(2,5,3) → float3 param: two outer dims dispatched."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((2, 5, 3), dtype=np.float32))
    code = generate_code(
        device,
        "negate",
        "float3 negate(float3 v) { return -v; }",
        tensor,
    )
    # v vectorized dim-2: uses __slangpy_load, mapping constant
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_v")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_3d_tensor_to_vector_binding_flags(device_type: spy.DeviceType):
    """3D Tensor shape=(2,5,3) → float3 param: call_dimensionality == 2."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((2, 5, 3), dtype=np.float32))
    func = helpers.create_function_from_module(
        device,
        "negate",
        "float3 negate(float3 v) { return -v; }",
    )
    cd = func.debug_build_call_data(tensor)
    bindings = cd.debug_only_bindings
    v = bindings.args[0]
    assert v.call_dimensionality == 2
    assert v.direct_bind is False
    assert v.vector_type.full_name == "vector<float,3>"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_3d_tensor_to_vector(device_type: spy.DeviceType):
    """Dispatch 3D tensor → float3 and verify GPU result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device,
        "negate",
        "float3 negate(float3 v) { return -v; }",
    )
    data = np.arange(30, dtype=np.float32).reshape(2, 5, 3)
    tensor = Tensor.from_numpy(device, data)
    result = func(tensor)
    expected = -data
    np.testing.assert_allclose(result.to_numpy().reshape(expected.shape), expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_2d_tensor_to_scalar_codegen(device_type: spy.DeviceType):
    """2D Tensor shape=(4,5) → float scalar: both dims dispatched (call_dim=2)."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((4, 5), dtype=np.float32))
    code = generate_code(
        device,
        "square",
        "float square(float x) { return x * x; }",
        tensor,
    )
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_x")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_2d_tensor_to_scalar_binding_flags(device_type: spy.DeviceType):
    """2D Tensor shape=(4,5) → float scalar: call_dimensionality == 2."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((4, 5), dtype=np.float32))
    func = helpers.create_function_from_module(
        device,
        "square",
        "float square(float x) { return x * x; }",
    )
    cd = func.debug_build_call_data(tensor)
    v = cd.debug_only_bindings.args[0]
    assert v.call_dimensionality == 2
    assert v.direct_bind is False


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_2d_tensor_to_scalar(device_type: spy.DeviceType):
    """Dispatch 2D tensor elementwise to scalar and verify GPU result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device,
        "square",
        "float square(float x) { return x * x; }",
    )
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = Tensor.from_numpy(device, data)
    result = func(tensor)
    expected = data * data
    np.testing.assert_allclose(result.to_numpy().reshape(expected.shape), expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_2d_tensor_to_1d_array_codegen(device_type: spy.DeviceType):
    """2D Tensor shape=(4,8) → half[8] param: trailing dim consumed by array, outer dim dispatched."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((4, 8), dtype=np.float16))
    code = generate_code(
        device,
        "tensor_test_channels<8>",
        r"""
half[NumChannels] tensor_test_channels<let NumChannels : int>(half[NumChannels] data)
{
    [ForceUnroll]
    for (int i = 0; i < NumChannels; ++i)
    {
        data[i] = 2.h * data[i];
    }
    return data;
}
""",
        tensor,
    )
    # data is vectorized (trailing dim consumed by array): __slangpy_load present
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_data")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_2d_tensor_to_1d_array_binding_flags(device_type: spy.DeviceType):
    """2D Tensor shape=(4,8) → half[8] param: call_dimensionality == 1."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((4, 8), dtype=np.float16))
    func = helpers.create_function_from_module(
        device,
        "tensor_test_channels<8>",
        r"""
half[NumChannels] tensor_test_channels<let NumChannels : int>(half[NumChannels] data)
{
    [ForceUnroll]
    for (int i = 0; i < NumChannels; ++i)
    {
        data[i] = 2.h * data[i];
    }
    return data;
}
""",
    )
    cd = func.debug_build_call_data(tensor)
    v = cd.debug_only_bindings.args[0]
    assert v.call_dimensionality == 1
    assert v.direct_bind is False


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_2d_tensor_to_1d_array(device_type: spy.DeviceType):
    """Dispatch 2D tensor → half[8] and verify GPU doubles each element."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device,
        "tensor_test_channels<8>",
        r"""
half[NumChannels] tensor_test_channels<let NumChannels : int>(half[NumChannels] data)
{
    [ForceUnroll]
    for (int i = 0; i < NumChannels; ++i)
    {
        data[i] = 2.h * data[i];
    }
    return data;
}
""",
    ).return_type(Tensor)
    data = np.ones((4, 8), dtype=np.float16)
    tensor = Tensor.from_numpy(device, data)
    result = func(tensor)
    expected = data * 2.0
    np.testing.assert_allclose(
        result.to_numpy().reshape(expected.shape).astype(np.float32),
        expected.astype(np.float32),
        atol=1e-2,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_mixed_vectorized_and_dim0_tensor_codegen(device_type: spy.DeviceType):
    """One tensor vectorized (2D→float3) and another at dim-0 (Tensor<float,1> param)."""
    device = helpers.get_device(device_type)
    src = """
float dot_lookup(float3 v, Tensor<float,1> weights) {
    return v.x * weights[0] + v.y * weights[1] + v.z * weights[2];
}
"""
    vec_tensor = Tensor.from_numpy(device, np.ones((5, 3), dtype=np.float32))
    weight_tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code = generate_code(device, "dot_lookup", src, vec_tensor, weight_tensor)
    # v: vectorized dim-1 (2D→float3), uses __slangpy_load
    assert_contains(code, "_m_v")
    assert_contains(code, "__slangpy_load")
    # weights: dim-0 direct-bind (Tensor<float,1> param), type used directly + direct assignment
    assert_not_contains(code, "typealias _t_weights")
    assert_contains(code, "Tensor<float, 1> weights;")
    assert_trampoline_has(code, "weights = __calldata__.weights;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_mixed_vectorized_and_dim0_tensor_binding_flags(device_type: spy.DeviceType):
    """Binding flags: vectorized tensor has dim>0, dim-0 tensor has direct_bind."""
    device = helpers.get_device(device_type)
    src = """
float dot_lookup(float3 v, Tensor<float,1> weights) {
    return v.x * weights[0] + v.y * weights[1] + v.z * weights[2];
}
"""
    vec_tensor = Tensor.from_numpy(device, np.ones((5, 3), dtype=np.float32))
    weight_tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    func = helpers.create_function_from_module(device, "dot_lookup", src)
    cd = func.debug_build_call_data(vec_tensor, weight_tensor)
    bindings = cd.debug_only_bindings
    v = bindings.args[0]
    assert v.call_dimensionality == 1
    assert v.direct_bind is False
    w = bindings.args[1]
    assert w.call_dimensionality == 0
    assert w.direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_mixed_vectorized_and_dim0_tensor(device_type: spy.DeviceType):
    """Dispatch vectorized float3 + dim-0 Tensor<float,1> and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
float dot_lookup(float3 v, Tensor<float,1> weights) {
    return v.x * weights[0] + v.y * weights[1] + v.z * weights[2];
}
"""
    func = helpers.create_function_from_module(device, "dot_lookup", src)
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    weights = np.array([10, 20, 30], dtype=np.float32)
    result = func(
        Tensor.from_numpy(device, vecs),
        Tensor.from_numpy(device, weights),
    )
    expected = np.array([10, 20, 30], dtype=np.float32)
    np.testing.assert_allclose(result.to_numpy().flatten(), expected, atol=1e-5)


# ===========================================================================
# Composite struct codegen tests — nested structs, vector/matrix/array fields
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_nested_struct_codegen(device_type: spy.DeviceType):
    """Nested struct: Outer{Inner inner, float scale} — all-scalar, direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    float x;
    float y;
};
struct Outer {
    Inner inner;
    float scale;
};
float compute(Outer o) { return (o.inner.x + o.inner.y) * o.scale; }
"""
    code = generate_code(
        device,
        "compute",
        src,
        {"_type": "Outer", "inner": {"_type": "Inner", "x": 1.0, "y": 2.0}, "scale": 3.0},
    )
    # All-scalar nested struct at dim-0: direct-bind → type used directly
    assert_not_contains(code, "typealias _t_o")
    assert_contains(code, "Outer o;")
    assert_not_contains(code, "__slangpy_load")
    assert_not_contains(code, "struct _t_o")
    assert_trampoline_has(code, "o = __calldata__.o;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_nested_struct_binding_flags(device_type: spy.DeviceType):
    """Nested struct: all-scalar → direct_bind=True at every level."""
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    float x;
    float y;
};
struct Outer {
    Inner inner;
    float scale;
};
float compute(Outer o) { return (o.inner.x + o.inner.y) * o.scale; }
"""
    func = helpers.create_function_from_module(device, "compute", src)
    cd = func.debug_build_call_data(
        {"_type": "Outer", "inner": {"_type": "Inner", "x": 1.0, "y": 2.0}, "scale": 3.0}
    )
    bindings = cd.debug_only_bindings
    o = bindings.args[0]
    assert o.direct_bind is True
    assert o.children["inner"].direct_bind is True
    assert o.children["inner"].children["x"].direct_bind is True
    assert o.children["inner"].children["y"].direct_bind is True
    assert o.children["scale"].direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_nested_struct(device_type: spy.DeviceType):
    """Dispatch nested struct and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    float x;
    float y;
};
struct Outer {
    Inner inner;
    float scale;
};
float compute(Outer o) { return (o.inner.x + o.inner.y) * o.scale; }
"""
    func = helpers.create_function_from_module(device, "compute", src)
    result = func({"_type": "Outer", "inner": {"_type": "Inner", "x": 3.0, "y": 7.0}, "scale": 2.0})
    assert abs(result - 20.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_with_vector_fields_codegen(device_type: spy.DeviceType):
    """Struct with vector fields: S{float3 pos, float scale} — all dim-0, direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float3 pos;
    float scale;
};
float3 apply(S s) { return s.pos * s.scale; }
"""
    code = generate_code(
        device,
        "apply",
        src,
        {"_type": "S", "pos": spy.math.float3(1, 2, 3), "scale": 2.0},
    )
    # All-scalar struct with vector field at dim-0: direct-bind → type used directly
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "S s;")
    assert_not_contains(code, "__slangpy_load")
    assert_trampoline_has(code, "s = __calldata__.s;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_with_vector_fields_binding_flags(device_type: spy.DeviceType):
    """Struct with vector field — all children direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float3 pos;
    float scale;
};
float3 apply(S s) { return s.pos * s.scale; }
"""
    func = helpers.create_function_from_module(device, "apply", src)
    cd = func.debug_build_call_data({"_type": "S", "pos": spy.math.float3(1, 2, 3), "scale": 2.0})
    bindings = cd.debug_only_bindings
    s = bindings.args[0]
    assert s.direct_bind is True
    assert s.children["pos"].direct_bind is True
    assert s.children["scale"].direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_with_vector_fields(device_type: spy.DeviceType):
    """Dispatch struct with vector field and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float3 pos;
    float scale;
};
float3 apply(S s) { return s.pos * s.scale; }
"""
    func = helpers.create_function_from_module(device, "apply", src)
    result = func({"_type": "S", "pos": spy.math.float3(1, 2, 3), "scale": 3.0})
    assert result.x == 3.0
    assert result.y == 6.0
    assert result.z == 9.0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_with_matrix_field_codegen(device_type: spy.DeviceType):
    """Struct with matrix field: S{float4x4 m, float scale} — all dim-0, direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float4x4 m;
    float scale;
};
float4x4 apply(S s) { return s.m * s.scale; }
"""
    code = generate_code(
        device,
        "apply",
        src,
        {"_type": "S", "m": spy.math.float4x4.identity(), "scale": 2.0},
    )
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "S s;")
    assert_not_contains(code, "__slangpy_load")
    assert_trampoline_has(code, "s = __calldata__.s;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_with_matrix_field(device_type: spy.DeviceType):
    """Dispatch struct with matrix field and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float4x4 m;
    float scale;
};
float4x4 apply(S s) { return s.m * s.scale; }
"""
    func = helpers.create_function_from_module(device, "apply", src)
    result = func({"_type": "S", "m": spy.math.float4x4.identity(), "scale": 2.0})
    # Identity * 2 → diagonal is 2
    assert abs(result[0][0] - 2.0) < 1e-5
    assert abs(result[1][1] - 2.0) < 1e-5
    assert abs(result[0][1] - 0.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_with_array_field_codegen(device_type: spy.DeviceType):
    """Struct with fixed-size array field: Foo{int vals[4]} — all dim-0, direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct Foo {
    int vals[4];
};
int sum_inner(Foo foo) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += foo.vals[i];
    }
    return s;
}
"""
    code = generate_code(
        device,
        "sum_inner",
        src,
        {"_type": "Foo", "vals": [1, 2, 3, 4]},
    )
    assert_not_contains(code, "typealias _t_foo")
    assert_contains(code, "Foo foo;")
    assert_not_contains(code, "__slangpy_load")
    assert_trampoline_has(code, "foo = __calldata__.foo;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_with_array_field_binding_flags(device_type: spy.DeviceType):
    """Struct with array field: all direct_bind=True."""
    device = helpers.get_device(device_type)
    src = """
struct Foo {
    int vals[4];
};
int sum_inner(Foo foo) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += foo.vals[i];
    }
    return s;
}
"""
    func = helpers.create_function_from_module(device, "sum_inner", src)
    cd = func.debug_build_call_data({"_type": "Foo", "vals": [1, 2, 3, 4]})
    bindings = cd.debug_only_bindings
    foo = bindings.args[0]
    assert foo.direct_bind is True
    assert foo.children["vals"].direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_with_array_field(device_type: spy.DeviceType):
    """Dispatch struct with array field and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct Foo {
    int vals[4];
};
int sum_inner(Foo foo) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += foo.vals[i];
    }
    return s;
}
"""
    func = helpers.create_function_from_module(device, "sum_inner", src)
    result = func({"_type": "Foo", "vals": [10, 20, 30, 40]})
    assert result == 100


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_deeply_nested_struct_codegen(device_type: spy.DeviceType):
    """3-level deep nesting: Top{Mid{Bot{float v}, int c}, float s} — all dim-0, direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct Bot {
    float v;
};
struct Mid {
    Bot bot;
    int c;
};
struct Top {
    Mid mid;
    float s;
};
float compute(Top t) { return t.mid.bot.v * float(t.mid.c) * t.s; }
"""
    code = generate_code(
        device,
        "compute",
        src,
        {
            "_type": "Top",
            "mid": {"_type": "Mid", "bot": {"_type": "Bot", "v": 2.0}, "c": 3},
            "s": 4.0,
        },
    )
    assert_not_contains(code, "typealias _t_t")
    assert_contains(code, "Top t;")
    assert_not_contains(code, "__slangpy_load")
    assert_not_contains(code, "struct _t_t")
    assert_trampoline_has(code, "t = __calldata__.t;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_deeply_nested_struct_binding_flags(device_type: spy.DeviceType):
    """3-level deep: all direct_bind=True at every level."""
    device = helpers.get_device(device_type)
    src = """
struct Bot {
    float v;
};
struct Mid {
    Bot bot;
    int c;
};
struct Top {
    Mid mid;
    float s;
};
float compute(Top t) { return t.mid.bot.v * float(t.mid.c) * t.s; }
"""
    func = helpers.create_function_from_module(device, "compute", src)
    cd = func.debug_build_call_data(
        {
            "_type": "Top",
            "mid": {"_type": "Mid", "bot": {"_type": "Bot", "v": 2.0}, "c": 3},
            "s": 4.0,
        }
    )
    bindings = cd.debug_only_bindings
    t = bindings.args[0]
    assert t.direct_bind is True
    assert t.children["mid"].direct_bind is True
    assert t.children["mid"].children["bot"].direct_bind is True
    assert t.children["mid"].children["bot"].children["v"].direct_bind is True
    assert t.children["mid"].children["c"].direct_bind is True
    assert t.children["s"].direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_deeply_nested_struct(device_type: spy.DeviceType):
    """Dispatch 3-level nested struct and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct Bot {
    float v;
};
struct Mid {
    Bot bot;
    int c;
};
struct Top {
    Mid mid;
    float s;
};
float compute(Top t) { return t.mid.bot.v * float(t.mid.c) * t.s; }
"""
    func = helpers.create_function_from_module(device, "compute", src)
    result = func(
        {
            "_type": "Top",
            "mid": {"_type": "Mid", "bot": {"_type": "Bot", "v": 2.0}, "c": 3},
            "s": 4.0,
        }
    )
    assert abs(result - 24.0) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_nested_struct_with_tensor_child_codegen(device_type: spy.DeviceType):
    """Nested struct where a leaf is a tensor: Outer{Inner{float x (tensor), float y (scalar)}, float s}.

    Outer and Inner are NOT direct-bind (Inner.x is vectorized).
    Inner.y and s retain direct_bind=True inside the non-direct-bind parent.
    """
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    float x;
    float y;
};
struct Outer {
    Inner inner;
    float s;
};
float compute(Outer o) { return (o.inner.x + o.inner.y) * o.s; }
"""
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code = generate_code(
        device,
        "compute",
        src,
        {
            "_type": "Outer",
            "inner": {"_type": "Inner", "x": tensor_x, "y": 10.0},
            "s": 2.0,
        },
    )
    # Outer and Inner are NOT direct-bind: inline structs generated
    assert_contains(code, "struct _t_o")
    assert_contains(code, "__slangpy_load")
    assert_not_contains(code, "typealias _t_o = Outer;")
    # Scalar children retain direct-bind: types used directly
    assert_not_contains(code, "typealias _t_y")
    assert_contains(code, "float y;")
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "float s;")
    # Direct assignment for scalar children within __slangpy_load
    assert_contains(code, "value.y = y;")
    # Tensor child uses standard path
    assert_contains(code, "_m_x")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_nested_struct_with_tensor_child_binding_flags(device_type: spy.DeviceType):
    """Nested struct with tensor: Outer not direct-bind, scalar children retain direct_bind."""
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    float x;
    float y;
};
struct Outer {
    Inner inner;
    float s;
};
float compute(Outer o) { return (o.inner.x + o.inner.y) * o.s; }
"""
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    func = helpers.create_function_from_module(device, "compute", src)
    cd = func.debug_build_call_data(
        {
            "_type": "Outer",
            "inner": {"_type": "Inner", "x": tensor_x, "y": 10.0},
            "s": 2.0,
        }
    )
    bindings = cd.debug_only_bindings
    o = bindings.args[0]
    assert o.direct_bind is False
    assert o.children["inner"].direct_bind is False  # has non-direct child
    assert o.children["inner"].children["x"].direct_bind is False  # tensor dim>0
    assert o.children["inner"].children["y"].direct_bind is True  # scalar dim-0
    assert o.children["s"].direct_bind is True  # scalar dim-0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_nested_struct_with_tensor(device_type: spy.DeviceType):
    """Dispatch nested struct with tensor leaf and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    float x;
    float y;
};
struct Outer {
    Inner inner;
    float s;
};
float compute(Outer o) { return (o.inner.x + o.inner.y) * o.s; }
"""
    func = helpers.create_function_from_module(device, "compute", src)
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    result = func(
        {
            "_type": "Outer",
            "inner": {"_type": "Inner", "x": tensor_x, "y": 10.0},
            "s": 2.0,
        }
    )
    expected = np.array([22, 24, 26], dtype=np.float32)
    np.testing.assert_allclose(result.to_numpy().flatten(), expected, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_with_struct_array_field_codegen(device_type: spy.DeviceType):
    """Struct with array-of-structs field: Outer{Inner items[4]} — all dim-0, direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    int x;
};
struct Outer {
    Inner items[4];
};
int sum_inner(Outer outer) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += outer.items[i].x;
    }
    return s;
}
"""
    code = generate_code(
        device,
        "sum_inner",
        src,
        {
            "_type": "Outer",
            "items": [
                {"_type": "Inner", "x": 10},
                {"_type": "Inner", "x": 20},
                {"_type": "Inner", "x": 30},
                {"_type": "Inner", "x": 40},
            ],
        },
    )
    assert_not_contains(code, "typealias _t_outer")
    assert_contains(code, "Outer outer;")
    assert_not_contains(code, "__slangpy_load")
    assert_trampoline_has(code, "outer = __calldata__.outer;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_with_struct_array_field(device_type: spy.DeviceType):
    """Dispatch struct with array-of-structs field and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct Inner {
    int x;
};
struct Outer {
    Inner items[4];
};
int sum_inner(Outer outer) {
    int s = 0;
    for (int i = 0; i < 4; i++) {
        s += outer.items[i].x;
    }
    return s;
}
"""
    func = helpers.create_function_from_module(device, "sum_inner", src)
    result = func(
        {
            "_type": "Outer",
            "items": [
                {"_type": "Inner", "x": 10},
                {"_type": "Inner", "x": 20},
                {"_type": "Inner", "x": 30},
                {"_type": "Inner", "x": 40},
            ],
        }
    )
    assert result == 100


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_return_codegen(device_type: spy.DeviceType):
    """Function returning a struct: _result uses RWValueRef wrapper, not direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    int x;
    int y;
};
S make_struct(int a, int b) { return { a, b }; }
"""
    code = generate_code(device, "make_struct", src, 4, 5)
    # Scalar inputs are direct-bind, types used directly
    assert_not_contains(code, "typealias _t_a", "typealias _t_b")
    # _result is writable → NOT direct-bind → uses wrapper
    assert_contains(code, "__slangpy_store")
    assert_contains(code, "_m__result")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_return_binding_flags(device_type: spy.DeviceType):
    """Struct return: _result binding is NOT direct-bind (writable)."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    int x;
    int y;
};
S make_struct(int a, int b) { return { a, b }; }
"""
    func = helpers.create_function_from_module(device, "make_struct", src)
    cd = func.debug_build_call_data(4, 5)
    bindings = cd.debug_only_bindings
    result = bindings.kwargs["_result"]
    assert result.direct_bind is False
    # Inputs are direct-bind
    assert bindings.args[0].direct_bind is True
    assert bindings.args[1].direct_bind is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_return(device_type: spy.DeviceType):
    """Dispatch struct return and verify result is dict with correct values."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    int x;
    int y;
};
S make_struct(int a, int b) { return { a, b }; }
"""
    func = helpers.create_function_from_module(device, "make_struct", src)
    result = func(4, 5)
    assert isinstance(result, dict)
    assert result["x"] == 4
    assert result["y"] == 5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_struct_with_vectorized_2d_tensor_child_codegen(device_type: spy.DeviceType):
    """Struct with 2D tensor child vectorized to float3: struct NOT direct-bind.

    S{float3 v (2D tensor→float3), float s (scalar)}.
    The tensor's outer dim becomes dispatch, struct generates inline __slangpy_load.
    """
    device = helpers.get_device(device_type)
    src = """
struct S {
    float3 v;
    float s;
};
float3 apply(S st) { return st.v * st.s; }
"""
    tensor_v = Tensor.from_numpy(device, np.ones((5, 3), dtype=np.float32))
    code = generate_code(
        device,
        "apply",
        src,
        {"_type": "S", "v": tensor_v, "s": 2.0},
    )
    # Struct NOT direct-bind (tensor child is vectorized)
    assert_contains(code, "struct _t_st")
    assert_contains(code, "__slangpy_load")
    assert_not_contains(code, "typealias _t_st = S;")
    # Scalar child s retains direct-bind — type used directly, no alias
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "float s;")
    assert_contains(code, "value.s = s;")
    # Tensor child v uses standard path
    assert_contains(code, "_m_v")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_struct_with_vectorized_2d_tensor(device_type: spy.DeviceType):
    """Dispatch struct with 2D tensor→float3 child and verify GPU result."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float3 v;
    float s;
};
float3 apply(S st) { return st.v * st.s; }
"""
    func = helpers.create_function_from_module(device, "apply", src)
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor_v = Tensor.from_numpy(device, data)
    result = func({"_type": "S", "v": tensor_v, "s": 2.0})
    expected = data * 2.0
    np.testing.assert_allclose(result.to_numpy().reshape(expected.shape), expected, atol=1e-5)


# ===========================================================================
# Long type name heuristic — typealias emitted for names > MAX_INLINE_TYPE_LEN
# ===========================================================================

# Struct name that is deliberately longer than MAX_INLINE_TYPE_LEN (60 chars).
# 70 chars:
_LONG_STRUCT_NAME = "MyVeryLongStructNameThatExceedsSixtyCharactersForTesting12345"
assert len(_LONG_STRUCT_NAME) > 60

_SHORT_STRUCT_NAME = "S"
assert len(_SHORT_STRUCT_NAME) <= 60


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_long_struct_name_gets_typealias(device_type: spy.DeviceType):
    """A direct-bind struct with a name > 60 chars should emit a typealias."""
    device = helpers.get_device(device_type)
    src = f"""
struct {_LONG_STRUCT_NAME} {{
    float x;
    float y;
}};
float sum({_LONG_STRUCT_NAME} s) {{ return s.x + s.y; }}
"""
    code = generate_code(
        device,
        "sum",
        src,
        {"_type": _LONG_STRUCT_NAME, "x": 1.0, "y": 2.0},
    )
    # Long name → typealias _t_s emitted, CallData field declared as _t_s
    assert_contains(code, f"typealias _t_s = {_LONG_STRUCT_NAME};")
    assert_contains(code, "_t_s s;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_short_struct_name_inlined(device_type: spy.DeviceType):
    """A direct-bind struct with a short name should NOT emit a typealias."""
    device = helpers.get_device(device_type)
    src = f"""
struct {_SHORT_STRUCT_NAME} {{
    float x;
    float y;
}};
float sum({_SHORT_STRUCT_NAME} s) {{ return s.x + s.y; }}
"""
    code = generate_code(
        device,
        "sum",
        src,
        {"_type": _SHORT_STRUCT_NAME, "x": 1.0, "y": 2.0},
    )
    # Short name → no typealias, raw type inlined
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, f"{_SHORT_STRUCT_NAME} s;")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_long_scalar_type_name_gets_typealias(device_type: spy.DeviceType):
    """A non-direct-bind arg whose wrapper type name exceeds 60 chars gets a typealias."""
    device = helpers.get_device(device_type)
    src = f"""
struct {_LONG_STRUCT_NAME} {{
    float x;
    float y;
}};
{_LONG_STRUCT_NAME} identity({_LONG_STRUCT_NAME} s) {{ return s; }}
"""
    # Pass as a ValueRef so _result is writable → uses wrapper, and the wrapper
    # type name for _result will include the long struct name.
    code = generate_code(
        device,
        "identity",
        src,
        {"_type": _LONG_STRUCT_NAME, "x": 1.0, "y": 2.0},
    )
    # The _result binding uses RWValueRef<LongName> which exceeds 60 chars
    result_type = f"RWValueRef<{_LONG_STRUCT_NAME}>"
    assert len(result_type) > 60, f"Expected >60 chars, got {len(result_type)}"
    assert_contains(code, f"typealias _t__result = {result_type};")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_phase1_functional_long_struct_name(device_type: spy.DeviceType):
    """End-to-end dispatch with a struct whose name exceeds 60 chars."""
    device = helpers.get_device(device_type)
    src = f"""
struct {_LONG_STRUCT_NAME} {{
    float x;
    float y;
}};
float sum({_LONG_STRUCT_NAME} s) {{ return s.x + s.y; }}
"""
    func = helpers.create_function_from_module(device, "sum", src)
    result = func({"_type": _LONG_STRUCT_NAME, "x": 3.0, "y": 7.0})
    assert abs(result - 10.0) < 1e-5


# ===========================================================================
# Phase 2 gating tests — assert CURRENT behaviour, will break as Phase 2
# steps are implemented. See plan-simplifyKernelGen-phase2.prompt.md
# ===========================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_p2_calldata_struct_present(device_type: spy.DeviceType):
    """struct CallData is emitted for simple scalar call. Breaks at Step 2.2."""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", "int add(int a, int b) { return a + b; }", 1, 2)
    assert_contains(code, "struct CallData")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_p2_calldata_uniform_param(device_type: spy.DeviceType):
    """CallData is passed to kernel via uniform param (CUDA) or ParameterBlock (others). Breaks at Step 2.2."""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", "int add(int a, int b) { return a + b; }", 1, 2)
    # CUDA uses entry-point param; D3D12/Vulkan use ParameterBlock at module scope
    has_uniform = "uniform CallData call_data" in code
    has_param_block = "ParameterBlock<CallData> call_data" in code
    assert (
        has_uniform or has_param_block
    ), "Expected 'uniform CallData call_data' or 'ParameterBlock<CallData> call_data'"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_p2_thread_count_in_calldata(device_type: spy.DeviceType):
    """_thread_count accessed via call_data. prefix. Breaks at Step 2.2."""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", "int add(int a, int b) { return a + b; }", 1, 2)
    assert_contains(code, "call_data._thread_count")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_p2_trampoline_present_for_prim(device_type: spy.DeviceType):
    """Prim-mode kernel has a _trampoline function. Breaks at Step 2.3."""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", "int add(int a, int b) { return a + b; }", 1, 2)
    assert_contains(code, "void _trampoline(")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_p2_kernel_calls_trampoline(device_type: spy.DeviceType):
    """compute_main calls _trampoline(). Breaks at Step 2.3."""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", "int add(int a, int b) { return a + b; }", 1, 2)
    # Extract compute_main body and check it calls _trampoline
    main_idx = code.index("void compute_main(")
    main_body = code[main_idx:]
    assert "_trampoline(__slangpy_context__" in main_body


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_p2_sv_group_id_present(device_type: spy.DeviceType):
    """SV_GroupID present in compute_main signature even for dim-0. Breaks at Step 2.2."""
    device = helpers.get_device(device_type)
    code = generate_code(device, "add", "int add(int a, int b) { return a + b; }", 1, 2)
    assert_contains(code, "SV_GroupID")


# -- Phase 2 negative gate — must REMAIN passing after Phase 2 --


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gate_p2_wanghasharg_keeps_load(device_type: spy.DeviceType):
    """Non-direct-bind WangHashArg still uses __slangpy_load after Phase 2."""
    device = helpers.get_device(device_type)
    src = "uint3 rng(uint3 input) { return input; }"
    code = generate_code(device, "rng", src, WangHashArg(3))
    assert_contains(code, "__slangpy_load")


# ===========================================================================
# Step 2.1 tests — fast vs fallback path determination
# ===========================================================================


def build_call_data(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> Any:
    """Build CallData and return the full CallData object."""
    func = helpers.create_function_from_module(device, func_name, module_source)
    return func.debug_build_call_data(*args, **kwargs)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_step21_scalar_uses_direct_args(device_type: spy.DeviceType):
    """Simple scalar call has small inline-uniform size → use_direct_args=True."""
    device = helpers.get_device(device_type)
    cd = build_call_data(device, "add", "int add(int a, int b) { return a + b; }", 1, 2)
    # Two ints (4+4) + RWValueRef for _result (descriptor, ~0 inline) + uint3 _thread_count (12)
    # Should be well under any backend's threshold
    assert cd.use_direct_args is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_step21_threshold_property_positive(device_type: spy.DeviceType):
    """Device has a positive max_entry_point_uniform_size threshold."""
    device = helpers.get_device(device_type)
    threshold = device.info.limits.max_entry_point_uniform_size
    assert threshold > 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_step21_vector_uses_direct_args(device_type: spy.DeviceType):
    """float3 args are small enough for direct args."""
    device = helpers.get_device(device_type)
    cd = build_call_data(
        device,
        "scale",
        "float3 scale(float3 v, float s) { return v * s; }",
        spy.math.float3(1, 2, 3),
        2.0,
    )
    assert cd.use_direct_args is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_step21_struct_uses_direct_args(device_type: spy.DeviceType):
    """All-scalar struct dict has small inline-uniform size."""
    device = helpers.get_device(device_type)
    src = """
struct S { float x; float y; };
float sum(S s) { return s.x + s.y; }
"""
    cd = build_call_data(device, "sum", src, {"_type": "S", "x": 1.0, "y": 2.0})
    assert cd.use_direct_args is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_step21_tensor_uses_direct_args(device_type: spy.DeviceType):
    """Tensor args contribute descriptor-only (0 inline bytes) → direct args."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    cd = build_call_data(
        device,
        "sum_all",
        "float sum_all(float x) { return x; }",
        tensor,
    )
    assert cd.use_direct_args is True


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_step21_many_float4x4_may_exceed_vulkan(device_type: spy.DeviceType):
    """Many float4x4 params may exceed Vulkan's 128-byte threshold.

    8 × float4x4 = 8 × 64 bytes = 512 bytes inline + 12 bytes _thread_count = 524 bytes.
    This exceeds Vulkan (128) and D3D12 (256) but not CUDA (4096).
    """
    device = helpers.get_device(device_type)
    src = """
float4x4 sum8(float4x4 a, float4x4 b, float4x4 c, float4x4 d,
              float4x4 e, float4x4 f, float4x4 g, float4x4 h) {
    return a + b + c + d + e + f + g + h;
}
"""
    identity = spy.math.float4x4.identity()
    cd = build_call_data(
        device,
        "sum8",
        src,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
    )
    threshold = device.info.limits.max_entry_point_uniform_size
    if threshold >= 524:
        assert cd.use_direct_args is True
    else:
        assert cd.use_direct_args is False


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_step21_wanghasharg_uses_direct_args(device_type: spy.DeviceType):
    """WangHashArg (non-direct-bind) still counts its inline-uniform size.
    Its wrapper type has a small inline footprint, so use_direct_args should be True.
    """
    device = helpers.get_device(device_type)
    cd = build_call_data(
        device,
        "rng",
        "uint3 rng(uint3 input) { return input; }",
        WangHashArg(3),
    )
    assert cd.use_direct_args is True


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
