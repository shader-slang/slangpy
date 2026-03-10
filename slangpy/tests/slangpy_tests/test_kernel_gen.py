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
    # _result is auto-created as RWValueRef — now uses RWStructuredBuffer
    assert_not_contains(code, "RWValueRef<int>")
    assert_contains(code, "RWStructuredBuffer<int>")


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
    # Read-only ValueRef now uses raw type alias, not ValueRef<float>
    assert_not_contains(code, "ValueRef<float>")
    assert_contains(code, "typealias _t_v = float;")
    # Direct assignment in trampoline
    assert_trampoline_has(code, "v = __calldata__.v;")


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
    # Auto-created _result uses RWStructuredBuffer instead of RWValueRef
    assert_not_contains(code, "RWValueRef<int>")
    assert_contains(code, "RWStructuredBuffer<int>")
    # Trampoline uses buffer load/store
    assert_trampoline_has(code, "_result = __calldata__._result[0];")


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
        "static const int _m__result = 0",
    )


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


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
