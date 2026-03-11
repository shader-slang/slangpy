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
    # Scalars now use direct binding: typealias to raw type, no ValueType wrapper
    assert_not_contains(code, "ValueType<int>")
    assert_contains(code, "typealias _t_a = int;", "typealias _t_b = int;")
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
    assert_contains(code, "typealias _t_x = float;", "typealias _t_y = float;")


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
    assert_contains(code, "typealias _t_v = vector<float,3>;")


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
    assert_contains(code, "typealias _t_m = matrix<float,4,4>;")


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
    assert_contains(code, "typealias _t_a = ")


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
    # Read-only ValueRef uses raw type alias (direct-bind)
    assert_contains(code, "typealias _t_v = float;")
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
    # Direct-bind struct: uses raw type alias, no inline struct with __slangpy_load
    assert_not_contains(code, "__slangpy_load")
    assert_contains(code, "typealias _t_s = S;")
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
    # WangHashArg uses wrapper type. Check the type alias is present.
    assert_contains(code, "_t_input")


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
    # 'a' is direct-bind (scalar dim-0): raw typealias, direct trampoline load
    assert_contains(code, "typealias _t_a = float;")
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
    # Child y is direct-bind: raw type alias, direct assignment in __slangpy_load
    assert_contains(code, "typealias _t_y = float;")
    assert_contains(code, "value.y = y;")
    assert_not_contains(code, "ValueType<float>")
    # Child x should use tensor type
    assert_contains(code, "Tensor<float, 1>")
    # Scalar arg 'scale' is independent — should still be direct-bind
    assert_contains(code, "typealias _t_scale = float;")


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
    # Type alias should use Tensor<float, 1>
    assert_contains(code, "typealias _t_t = Tensor<float, 1>;")
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
    assert_contains(code, "typealias _t_y = float;")
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


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
