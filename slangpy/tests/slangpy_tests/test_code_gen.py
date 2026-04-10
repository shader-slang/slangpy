# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
These tests exercise different code paths for kernel generation, verifying both
generated code patterns and binding flags in a single pass per scenario. Each
test calls ``debug_build_call_data`` once and asserts on both ``.code`` and
``.debug_only_bindings``.

Negative-gate tests (``test_*_not_direct_bind``, ``test_*_keeps_wrapper``) must
remain passing - they cover types that are NOT direct-bind eligible.

Functional dispatch tests are included only for scenarios that are not covered
by other test files (``test_simple_function_call.py``, ``test_tensor.py``, etc.).
"""

from typing import Any

import numpy as np
import os
import pytest

import slangpy as spy
from slangpy.testing import helpers
from slangpy.types import ValueRef, Tensor, diffPair
from slangpy.types.wanghasharg import WangHashArg

PRINT_CODE = os.getenv("PRINT_TEST_KERNEL_GEN", "0") == "1"


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


def assert_load_statement(code: str, *var_names: str) -> None:
    """Assert that load statements exist for the given variables.

    Handles both code paths:
    - Fast path (entry-point params): ``__tmp_x = x;``
    - Fallback path (ParameterBlock<CallData>): ``__tmp_x = call_data.x;``
    """
    for name in var_names:
        fast = f"__tmp_{name} = {name};"
        fallback = f"__tmp_{name} = call_data.{name};"
        assert fast in code or fallback in code, f"Expected load for '{name}': {fast} or {fallback}"


def generate_code_and_bindings(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> tuple[str, Any]:
    """Generate code and return ``(code_str, bindings)`` from a single ``debug_build_call_data`` call."""
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.debug_build_call_data(*args, **kwargs)
    if PRINT_CODE:
        print(cd.code)
    return cd.code, cd.debug_only_bindings


def generate_bwds_code_and_bindings(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> tuple[str, Any]:
    """Generate backwards-mode code and return ``(code_str, bindings)``."""
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.bwds.debug_build_call_data(*args, **kwargs)
    if PRINT_CODE:
        print(cd.code)
    return cd.code, cd.debug_only_bindings


def build_call_data_full(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> tuple[str, Any, Any]:
    """Build CallData and return ``(code_str, bindings, call_data)``."""
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.debug_build_call_data(*args, **kwargs)
    if PRINT_CODE:
        print(cd.code)
    return cd.code, cd.debug_only_bindings, cd


def build_bwds_call_data_full(
    device: spy.Device, func_name: str, module_source: str, *args: Any, **kwargs: Any
) -> tuple[str, Any, Any]:
    """Build bwds CallData and return ``(code_str, bindings, call_data)``."""
    func = helpers.create_function_from_module(device, func_name, module_source)
    cd = func.bwds.debug_build_call_data(*args, **kwargs)
    if PRINT_CODE:
        print(cd.code)
    return cd.code, cd.debug_only_bindings, cd


# ===========================================================================
# Codegen + binding flag tests (1-21)
# ===========================================================================


# 1 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_direct_bind(device_type: spy.DeviceType):
    """int/float scalar dim-0: direct-bind, _result writable RWValueRef.

    Merges: test_gate_scalar_uses_valuetype, test_gate_float_scalar_uses_valuetype,
    test_gate_valueref_write_uses_wrapper, test_gate_mapping_constants_present,
    test_gate_context_map_in_trampoline, test_result_binding_not_direct_bind.
    """
    device = helpers.get_device(device_type)
    code, bindings = generate_code_and_bindings(
        device, "add", "int add(int a, int b) { return a + b; }", 1, 2
    )

    # --- codegen assertions ---
    # Scalars use raw type directly, no wrapper
    assert_not_contains(code, "ValueType<int>")
    assert_not_contains(code, "typealias _t_a", "typealias _t_b")
    # Direct assignment - loaded into __tmp_ locals
    assert_load_statement(code, "a", "b")
    # _result is auto-created writable RWValueRef
    assert_contains(code, "RWValueRef<int>")
    assert_contains(code, "__slangpy_store")
    # No mapping constants for direct-bind args; _result keeps its mapping constant
    assert_not_contains(code, "static const int _m_a = 0", "static const int _m_b = 0")
    assert_contains(code, "static const int _m__result = 0")
    # No context.map for direct-bind args
    assert_not_contains(code, "__slangpy_context__.map(_m_a)")

    # --- binding flag assertions ---
    assert bindings.args[0].direct_bind is True
    assert bindings.args[0].call_dimensionality == 0
    assert bindings.args[1].direct_bind is True
    assert bindings.kwargs["_result"].direct_bind is False
    assert bindings.kwargs["_result"].call_dimensionality == 0


# 2 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vector_direct_bind(device_type: spy.DeviceType):
    """float3 dim-0: direct-bind, type used directly."""
    device = helpers.get_device(device_type)
    code, bindings = generate_code_and_bindings(
        device,
        "scale",
        "float3 scale(float3 v, float s) { return v * s; }",
        spy.math.float3(1, 2, 3),
        1.0,
    )

    assert_not_contains(code, "VectorValueType<float,3>")
    assert_not_contains(code, "typealias _t_v")
    assert_contains(code, "vector<float,3> __tmp_v;")

    assert bindings.args[0].direct_bind is True
    assert bindings.args[0].call_dimensionality == 0
    assert bindings.args[1].direct_bind is True


# 3 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_matrix_direct_bind(device_type: spy.DeviceType):
    """float4x4 dim-0: direct-bind."""
    device = helpers.get_device(device_type)
    code, bindings = generate_code_and_bindings(
        device,
        "ident",
        "float4x4 ident(float4x4 m) { return m; }",
        spy.math.float4x4.identity(),
    )

    assert_not_contains(code, "ValueType<matrix<float,4,4>>")
    assert_not_contains(code, "typealias _t_m")
    assert_contains(code, "matrix<float,4,4> __tmp_m;")

    assert bindings.args[0].direct_bind is True
    assert bindings.args[0].call_dimensionality == 0


# 4 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_array_direct_bind(device_type: spy.DeviceType):
    """float[4] dim-0: direct-bind."""
    device = helpers.get_device(device_type)
    code, bindings = generate_code_and_bindings(
        device,
        "process",
        "void process(float a[4]) { }",
        [1.0, 2.0, 3.0, 4.0],
    )

    assert_not_contains(code, "ValueType<")
    assert_not_contains(code, "typealias _t_a")

    assert bindings.args[0].direct_bind is True
    assert bindings.args[0].call_dimensionality == 0


# 5 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_valueref_read_direct_bind(device_type: spy.DeviceType):
    """Read-only ValueRef: direct-bind, raw type."""
    device = helpers.get_device(device_type)
    code, bindings = generate_code_and_bindings(
        device,
        "read_val",
        "float read_val(float v) { return v; }",
        ValueRef(1.0),
    )

    assert_not_contains(code, "typealias _t_v")
    assert_contains(code, "float __tmp_v;")
    assert_load_statement(code, "v")
    assert_contains(code, "RWValueRef<float>")

    assert bindings.args[0].direct_bind is True
    assert bindings.args[0].call_dimensionality == 0


# 6 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_writable_valueref_not_direct_bind(device_type: spy.DeviceType):
    """Writable ValueRef (inout) must not be direct-bind - needs buffer read/write."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "inc", "void inc(inout int v) { v += 1; }")
    cd = func.debug_build_call_data(ValueRef(5))
    code, bindings = cd.code, cd.debug_only_bindings

    assert_contains(code, "RWValueRef<int>")
    assert_not_contains(code, "typealias _t_v = int;")

    assert bindings.args[0].direct_bind is False
    assert bindings.args[0].call_dimensionality == 0


# 7 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_all_scalar_direct_bind(device_type: spy.DeviceType):
    """S{float x, y} via dict - all-scalar, direct-bind.

    Merges: test_gate_struct_uses_slangpy_load, test_struct_all_scalars_binding_flag.
    """
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
float sum(S s) { return s.x + s.y; }
"""
    code, bindings = generate_code_and_bindings(
        device, "sum", src, {"_type": "S", "x": 1.0, "y": 2.0}
    )

    # Direct-bind struct - raw type, no __slangpy_load
    assert_not_contains(code, "__slangpy_load")
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "S __tmp_s;")
    assert_load_statement(code, "s")

    s = bindings.args[0]
    assert s.direct_bind is True
    assert s.children["x"].direct_bind is True
    assert s.children["y"].direct_bind is True


# 8 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "variant",
    ["vector_field", "array_field", "matrix_field"],
    ids=["vector_field", "array_field", "matrix_field"],
)
def test_struct_composite_fields_direct_bind(device_type: spy.DeviceType, variant: str):
    """Struct with composite field (vector / array / matrix) all dim-0 -> direct-bind."""
    device = helpers.get_device(device_type)

    if variant == "vector_field":
        src = """
struct S {
    float3 pos;
    float scale;
};
float3 apply(S s) { return s.pos * s.scale; }
"""
        arg = {"_type": "S", "pos": spy.math.float3(1, 2, 3), "scale": 2.0}
        func_name = "apply"
        child_name = "pos"
    elif variant == "array_field":
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
        arg = {"_type": "Foo", "vals": [1, 2, 3, 4]}
        func_name = "sum_inner"
        child_name = "vals"
    else:
        src = """
struct S {
    float4x4 m;
    float scale;
};
float4x4 apply(S s) { return s.m * s.scale; }
"""
        arg = {"_type": "S", "m": spy.math.float4x4.identity(), "scale": 2.0}
        func_name = "apply"
        child_name = "m"

    code, bindings = generate_code_and_bindings(device, func_name, src, arg)

    # Struct is direct-bind - raw type, no __slangpy_load
    assert_not_contains(code, "__slangpy_load")
    param_name = "foo" if variant == "array_field" else "s"
    assert_not_contains(code, f"typealias _t_{param_name}")
    assert_load_statement(code, param_name)

    s = bindings.args[0]
    assert s.direct_bind is True
    assert s.children[child_name].direct_bind is True


# 9 -------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_deeply_nested_struct_direct_bind(device_type: spy.DeviceType):
    """3-level deep Top{Mid{Bot}} - all-scalar, direct-bind at every level.

    Subsumes 2-level nested struct tests. Merges: test_gate_deeply_nested_struct_codegen,
    test_gate_deeply_nested_struct_binding_flags.
    """
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
    arg = {
        "_type": "Top",
        "mid": {"_type": "Mid", "bot": {"_type": "Bot", "v": 2.0}, "c": 3},
        "s": 4.0,
    }
    code, bindings = generate_code_and_bindings(device, "compute", src, arg)

    assert_not_contains(code, "typealias _t_t")
    assert_contains(code, "Top __tmp_t;")
    assert_not_contains(code, "__slangpy_load")
    assert_not_contains(code, "struct _t_t")
    assert_load_statement(code, "t")

    t = bindings.args[0]
    assert t.direct_bind is True
    assert t.children["mid"].direct_bind is True
    assert t.children["mid"].children["bot"].direct_bind is True
    assert t.children["mid"].children["bot"].children["v"].direct_bind is True
    assert t.children["mid"].children["c"].direct_bind is True
    assert t.children["s"].direct_bind is True


# 10 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_mixed_fields(device_type: spy.DeviceType):
    """S{x(tensor), y(scalar)} - struct NOT direct-bind, scalar child keeps direct-bind.

    Merges: test_gate_struct_mixed_fields_codegen, test_mixed_children_direct_bind_codegen,
    test_gate_struct_mixed_fields_binding_flags.
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
    code, bindings = generate_code_and_bindings(
        device, "apply", src, {"_type": "S", "x": tensor_x, "y": 1.0}, 2.0
    )

    # Struct NOT direct-bind - inline struct with __slangpy_load
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "struct _t_s")
    assert_not_contains(code, "typealias _t_s = S;")
    # Child y direct-bind - type used directly, direct assignment
    assert_not_contains(code, "typealias _t_y")
    assert_contains(code, "float y;")
    assert_contains(code, "value.y = y;")
    assert_not_contains(code, "ValueType<float>")
    assert_not_contains(code, "_m_y")
    # Child x - tensor
    assert_contains(code, "Tensor<float, 1>")
    assert_contains(code, "x.__slangpy_load(context.map(_m_x),value.x)")
    # Independent scalar arg 'scale' - direct-bind
    assert_not_contains(code, "typealias _t_scale")
    assert_contains(code, "float __tmp_scale;")

    # Binding flags
    s = bindings.args[0]
    assert s.direct_bind is False
    assert s.children["x"].direct_bind is False
    assert s.children["y"].direct_bind is True
    assert bindings.args[1].direct_bind is True  # scale


# 11 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_nested_struct_with_tensor_child(device_type: spy.DeviceType):
    """Outer{Inner{x(tensor),y(scalar)},s} - Outer/Inner NOT direct-bind, scalar leaves are.

    Merges: test_gate_nested_struct_with_tensor_child_codegen,
    test_gate_nested_struct_with_tensor_child_binding_flags.
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
    code, bindings = generate_code_and_bindings(
        device,
        "compute",
        src,
        {"_type": "Outer", "inner": {"_type": "Inner", "x": tensor_x, "y": 10.0}, "s": 2.0},
    )

    # Outer/Inner NOT direct-bind
    assert_contains(code, "struct _t_o")
    assert_contains(code, "__slangpy_load")
    assert_not_contains(code, "typealias _t_o = Outer;")
    # Scalar children retain direct-bind
    assert_not_contains(code, "typealias _t_y")
    assert_contains(code, "float y;")
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "float s;")
    assert_contains(code, "value.y = y;")
    assert_contains(code, "_m_x")

    o = bindings.args[0]
    assert o.direct_bind is False
    assert o.children["inner"].direct_bind is False
    assert o.children["inner"].children["x"].direct_bind is False
    assert o.children["inner"].children["y"].direct_bind is True
    assert o.children["s"].direct_bind is True


# 12 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_return_not_direct_bind(device_type: spy.DeviceType):
    """Function returning struct - _result uses wrapper, NOT direct-bind.

    Merges: test_gate_struct_return_codegen, test_gate_struct_return_binding_flags.
    """
    device = helpers.get_device(device_type)
    src = """
struct S {
    int x;
    int y;
};
S make_struct(int a, int b) { return { a, b }; }
"""
    code, bindings = generate_code_and_bindings(device, "make_struct", src, 4, 5)

    # Scalar inputs direct-bind
    assert_not_contains(code, "typealias _t_a", "typealias _t_b")
    # _result writable - uses wrapper
    assert_contains(code, "__slangpy_store")
    assert_contains(code, "_m__result")

    result_binding = bindings.kwargs["_result"]
    assert result_binding.direct_bind is False
    assert bindings.args[0].direct_bind is True
    assert bindings.args[1].direct_bind is True


# 13 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_vectorized_2d_child(device_type: spy.DeviceType):
    """S{float3 v (2D tensor->float3), float s (scalar)} - struct NOT direct-bind."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float3 v;
    float s;
};
float3 apply(S st) { return st.v * st.s; }
"""
    tensor_v = Tensor.from_numpy(device, np.ones((5, 3), dtype=np.float32))
    code, bindings = generate_code_and_bindings(
        device, "apply", src, {"_type": "S", "v": tensor_v, "s": 2.0}
    )

    assert_contains(code, "struct _t_st")
    assert_contains(code, "__slangpy_load")
    assert_not_contains(code, "typealias _t_st = S;")
    # Scalar child s direct-bind
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "float s;")
    assert_contains(code, "value.s = s;")
    assert_contains(code, "_m_v")


# 14 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mixed_scalar_and_tensor(device_type: spy.DeviceType):
    """Scalar + tensor args - scalar direct-bind, tensor not.

    Merges: test_gate_mixed_args_scalar_and_tensor, test_gate_mixed_args_direct_bind_flags.
    """
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code, bindings = generate_code_and_bindings(
        device, "add", "float add(float a, float b) { return a + b; }", 1.0, tensor
    )

    # 'a' direct-bind
    assert_not_contains(code, "typealias _t_a")
    assert_not_contains(code, "ValueType<float>")
    assert_load_statement(code, "a")
    # 'b' NOT direct-bind (vectorized tensor)
    assert_contains(code, "Tensor<float, 1>")
    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_b")

    assert bindings.args[0].direct_bind is True
    assert bindings.args[0].call_dimensionality == 0
    assert bindings.args[1].direct_bind is False
    assert bindings.args[1].call_dimensionality == 1


# 15 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_dim0_direct_bind(device_type: spy.DeviceType):
    """Tensor<float,1> at dim-0: whole tensor passed, direct-bind.

    Merges: test_gate_tensor_dim0_codegen, test_gate_tensor_dim0_binding_flags.
    """
    device = helpers.get_device(device_type)
    src = """
float tensor_read(Tensor<float,1> t) {
    return t[0];
}
"""
    tensor = Tensor.from_numpy(device, np.array([42, 2, 3], dtype=np.float32))
    code, bindings = generate_code_and_bindings(device, "tensor_read", src, tensor)

    assert_not_contains(code, "typealias _t_t")
    assert_contains(code, "Tensor<float, 1> __tmp_t;")
    assert_load_statement(code, "t")
    assert_not_contains(code, "ValueType<")

    t = bindings.args[0]
    assert t.direct_bind is True
    assert t.call_dimensionality == 0


# 16 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_2d_tensor_to_vector(device_type: spy.DeviceType):
    """2D Tensor (10,3) -> float3: trailing dim consumed by vector, outer dispatched.

    Merges: test_gate_2d_tensor_to_vector_codegen, test_gate_2d_tensor_to_vector_binding_flags.
    """
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((10, 3), dtype=np.float32))
    code, bindings = generate_code_and_bindings(
        device, "scale", "float3 scale(float3 v, float s) { return v * s; }", tensor, 2.0
    )

    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_v")
    assert_not_contains(code, "typealias _t_s")
    assert_contains(code, "float __tmp_s;")

    v = bindings.args[0]
    assert v.call_dimensionality == 1
    assert v.direct_bind is False
    assert v.vector_type is not None
    assert v.vector_type.full_name == "vector<float,3>"
    s = bindings.args[1]
    assert s.call_dimensionality == 0
    assert s.direct_bind is True


# 17 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_3d_tensor_to_vector(device_type: spy.DeviceType):
    """3D Tensor (2,5,3) -> float3: two outer dims dispatched (call_dim=2).

    Merges: test_gate_3d_tensor_to_vector_codegen, test_gate_3d_tensor_to_vector_binding_flags.
    """
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((2, 5, 3), dtype=np.float32))
    code, bindings = generate_code_and_bindings(
        device, "negate", "float3 negate(float3 v) { return -v; }", tensor
    )

    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_v")

    v = bindings.args[0]
    assert v.call_dimensionality == 2
    assert v.direct_bind is False
    assert v.vector_type.full_name == "vector<float,3>"


# 18 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_2d_tensor_to_scalar(device_type: spy.DeviceType):
    """2D Tensor (4,5) -> float: both dims dispatched (call_dim=2).

    Merges: test_gate_2d_tensor_to_scalar_codegen, test_gate_2d_tensor_to_scalar_binding_flags.
    """
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((4, 5), dtype=np.float32))
    code, bindings = generate_code_and_bindings(
        device, "square", "float square(float x) { return x * x; }", tensor
    )

    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_x")

    v = bindings.args[0]
    assert v.call_dimensionality == 2
    assert v.direct_bind is False


# 19 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_2d_tensor_to_array(device_type: spy.DeviceType):
    """2D Tensor (4,8) -> half[8]: trailing dim consumed by array, outer dispatched.

    Merges: test_gate_2d_tensor_to_1d_array_codegen, test_gate_2d_tensor_to_1d_array_binding_flags.
    """
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.ones((4, 8), dtype=np.float16))
    src = r"""
half[NumChannels] tensor_test_channels<let NumChannels : int>(half[NumChannels] data)
{
    [ForceUnroll]
    for (int i = 0; i < NumChannels; ++i)
    {
        data[i] = 2.h * data[i];
    }
    return data;
}
"""
    code, bindings = generate_code_and_bindings(device, "tensor_test_channels<8>", src, tensor)

    assert_contains(code, "__slangpy_load")
    assert_contains(code, "_m_data")

    v = bindings.args[0]
    assert v.call_dimensionality == 1
    assert v.direct_bind is False


# 20 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_mixed_vectorized_dim0_tensor(device_type: spy.DeviceType):
    """One tensor vectorized (2D->float3) and another at dim-0 (Tensor<float,1> param).

    Merges: test_gate_mixed_vectorized_and_dim0_tensor_codegen,
    test_gate_mixed_vectorized_and_dim0_tensor_binding_flags.
    """
    device = helpers.get_device(device_type)
    src = """
float dot_lookup(float3 v, Tensor<float,1> weights) {
    return v.x * weights[0] + v.y * weights[1] + v.z * weights[2];
}
"""
    vec_tensor = Tensor.from_numpy(device, np.ones((5, 3), dtype=np.float32))
    weight_tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code, bindings = generate_code_and_bindings(
        device, "dot_lookup", src, vec_tensor, weight_tensor
    )

    # v: vectorized dim-1 (2D->float3)
    assert_contains(code, "_m_v")
    assert_contains(code, "__slangpy_load")
    # weights: dim-0 direct-bind
    assert_not_contains(code, "typealias _t_weights")
    assert_contains(code, "Tensor<float, 1> __tmp_weights;")
    assert_load_statement(code, "weights")

    v = bindings.args[0]
    assert v.call_dimensionality == 1
    assert v.direct_bind is False
    w = bindings.args[1]
    assert w.call_dimensionality == 0
    assert w.direct_bind is True


# 21 ------------------------------------------------------------------------
# Long type name heuristic constants
_LONG_STRUCT_NAME = "MyVeryLongStructNameThatExceedsSixtyCharactersForTesting12345"
assert len(_LONG_STRUCT_NAME) > 60
_SHORT_STRUCT_NAME = "S"
assert len(_SHORT_STRUCT_NAME) <= 60


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_long_type_name_typealias(device_type: spy.DeviceType):
    """Long struct name (>60 chars) emits typealias; short name inlines.
    Also covers long wrapper name for _result.

    Merges: test_gate_long_struct_name_gets_typealias, test_gate_short_struct_name_inlined,
    test_gate_long_scalar_type_name_gets_typealias.
    """
    device = helpers.get_device(device_type)

    # --- Long name -> typealias emitted ---
    long_src = f"""
struct {_LONG_STRUCT_NAME} {{
    float x;
    float y;
}};
float sum({_LONG_STRUCT_NAME} s) {{ return s.x + s.y; }}
"""
    code_long, _ = generate_code_and_bindings(
        device, "sum", long_src, {"_type": _LONG_STRUCT_NAME, "x": 1.0, "y": 2.0}
    )
    assert_contains(code_long, f"typealias _t_s = {_LONG_STRUCT_NAME};")
    # Typealias used in entry-point param or CallData field
    assert (
        "_t_s s;" in code_long or "uniform _t_s s" in code_long
    ), "Expected typealias usage (_t_s s; or uniform _t_s s) not found"

    # --- Short name -> no typealias ---
    short_src = f"""
struct {_SHORT_STRUCT_NAME} {{
    float x;
    float y;
}};
float sum({_SHORT_STRUCT_NAME} s) {{ return s.x + s.y; }}
"""
    code_short, _ = generate_code_and_bindings(
        device, "sum", short_src, {"_type": _SHORT_STRUCT_NAME, "x": 1.0, "y": 2.0}
    )
    assert_not_contains(code_short, "typealias _t_s")
    assert_contains(code_short, f"{_SHORT_STRUCT_NAME} __tmp_s;")

    # --- Long wrapper name for _result ---
    identity_src = f"""
struct {_LONG_STRUCT_NAME} {{
    float x;
    float y;
}};
{_LONG_STRUCT_NAME} identity({_LONG_STRUCT_NAME} s) {{ return s; }}
"""
    code_id, _ = generate_code_and_bindings(
        device, "identity", identity_src, {"_type": _LONG_STRUCT_NAME, "x": 1.0, "y": 2.0}
    )
    result_type = f"RWValueRef<{_LONG_STRUCT_NAME}>"
    assert len(result_type) > 60
    assert_contains(code_id, f"typealias _t__result = {result_type};")


# ===========================================================================
# Negative gates (22-24) - must remain passing
# ===========================================================================


# 22 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_wanghasharg_not_direct_bind(device_type: spy.DeviceType):
    """WangHashArg is NOT direct-bind, standalone and as struct child.

    Merges: test_gate_wanghasharg_uses_wrapper, test_wanghasharg_binding_flag,
    test_struct_with_wanghash_child_not_direct_bind.
    """
    device = helpers.get_device(device_type)

    # --- Standalone WangHashArg ---
    code_s, bindings_s = generate_code_and_bindings(
        device, "rng", "uint3 rng(uint3 input) { return input; }", WangHashArg(3)
    )
    assert_contains(code_s, "WangHashArg<")
    assert_contains(code_s, "input")
    assert bindings_s.args[0].direct_bind is False
    assert bindings_s.args[0].call_dimensionality == 0

    # --- As struct child ---
    struct_src = """
struct S { uint3 seed; float scale; };
float apply(S s) { return float(s.seed.x) * s.scale; }
"""
    func = helpers.create_function_from_module(device, "apply", struct_src)
    cd = func.debug_build_call_data({"_type": "S", "seed": WangHashArg(3), "scale": 1.0})
    code_c, bindings_c = cd.code, cd.debug_only_bindings

    s = bindings_c.args[0]
    assert s.direct_bind is False
    assert s.children["scale"].direct_bind is True
    assert_contains(code_c, "struct _t_s")
    assert_not_contains(code_c, "typealias _t_s = S;")


# 23 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorized_scalar_keeps_wrapper(device_type: spy.DeviceType):
    """1D tensor -> float: vectorized, keeps __slangpy_load."""
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code, _ = generate_code_and_bindings(
        device, "square", "float square(float x) { return x * x; }", tensor
    )
    assert_contains(code, "__slangpy_load")


# 24 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vectorized_dict_keeps_wrapper(device_type: spy.DeviceType):
    """Dict with tensor children: vectorized, keeps __slangpy_load."""
    device = helpers.get_device(device_type)
    src = """
struct S {
    float x;
    float y;
};
void apply(S s, float scale) {}
"""
    tensor_x = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    tensor_y = Tensor.from_numpy(device, np.array([4, 5, 6], dtype=np.float32))
    code, _ = generate_code_and_bindings(
        device,
        "apply",
        src,
        {"_type": "S", "x": tensor_x, "y": tensor_y},
        1.0,
    )
    assert_contains(code, "__slangpy_load")


# ===========================================================================
# Autodiff (25)
# ===========================================================================


# 25 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bwds_direct_bind(device_type: spy.DeviceType):
    """Backwards-mode: primals direct-bind, differentiable markers present.

    Merges: test_gate_bwds_scalar_uses_valuetype, test_gate_bwds_trampoline_is_differentiable,
    test_bwds_primal_binding_flags.
    """
    device = helpers.get_device(device_type)
    src = """
[Differentiable]
float polynomial(float a, float b) {
    return a * a + b + 1;
}
"""
    code, bindings = generate_bwds_code_and_bindings(device, "polynomial", src, 5.0, 10.0, 26.0)

    # No ValueType wrapper
    assert_not_contains(code, "ValueType<float>")
    # Differentiable markers
    assert_contains(code, "[Differentiable]", "bwd_diff(_trampoline)")
    # [Differentiable] appears before trampoline
    diff_idx = code.index("[Differentiable]")
    trampoline_idx = code.index("void _trampoline")
    assert diff_idx < trampoline_idx

    # Primal args direct-bind
    assert bindings.args[0].direct_bind is True  # a
    assert bindings.args[1].direct_bind is True  # b


# ===========================================================================
# Functional GPU dispatch - novel scenarios only (26-34)
# ===========================================================================


# 26 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_mixed_scalar_tensor(device_type: spy.DeviceType):
    """Dispatch mixed scalar + tensor and verify GPU result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "add", "float add(float a, float b) { return a + b; }"
    )
    tensor = Tensor.from_numpy(device, np.array([10, 20, 30], dtype=np.float32))
    result = func(5.0, tensor)
    expected = np.array([15, 25, 35], dtype=np.float32)
    np.testing.assert_allclose(result.to_numpy().flatten(), expected, atol=1e-5)


# 27 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_struct_mixed_fields(device_type: spy.DeviceType):
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


# 28 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_tensor_dim0(device_type: spy.DeviceType):
    """Dispatch whole tensor at dim-0 and verify GPU result."""
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


# 29 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_2d_tensor_to_vector(device_type: spy.DeviceType):
    """Dispatch 2D tensor -> float3 and verify GPU result."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "scale", "float3 scale(float3 v, float s) { return v * s; }"
    )
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = Tensor.from_numpy(device, data)
    result = func(tensor, 2.0)
    expected = data * 2.0
    np.testing.assert_allclose(result.to_numpy().reshape(expected.shape), expected, atol=1e-5)


# 30 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_2d_tensor_to_array(device_type: spy.DeviceType):
    """Dispatch 2D tensor -> half[8] and verify GPU doubles each element."""
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


# 31 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_mixed_vectorized_dim0_tensor(device_type: spy.DeviceType):
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


# 32 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_nested_struct_with_tensor(device_type: spy.DeviceType):
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
        {"_type": "Outer", "inner": {"_type": "Inner", "x": tensor_x, "y": 10.0}, "s": 2.0}
    )
    expected = np.array([22, 24, 26], dtype=np.float32)
    np.testing.assert_allclose(result.to_numpy().flatten(), expected, atol=1e-5)


# 33 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_struct_vectorized_2d_child(device_type: spy.DeviceType):
    """Dispatch struct with 2D tensor->float3 child and verify GPU result."""
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


# 34 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_struct_array_of_structs(device_type: spy.DeviceType):
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


# ===========================================================================
# Phase 2 - entry-point params (35-38, 40)
# ===========================================================================


# 35 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_entrypoint_params_scalar_dim0(device_type: spy.DeviceType):
    """Fast path: scalar dim-0 uses individual uniform entry-point params.

    Verifies: no struct CallData, no ParameterBlock, individual uniform params
    for a/b/_thread_count, _thread_count used directly in bounds check,
    SV_GroupID absent (dim-0 has no shape arrays), use_entrypoint_args=True.

    Merges: test_gate_p2_calldata_struct_absent_fast_path,
    test_gate_p2_individual_uniform_params, test_gate_p2_thread_count_direct,
    test_gate_p2_sv_group_id_absent_dim0, test_step21_scalar_uses_entrypoint_args.
    """
    if device_type == spy.DeviceType.metal:
        pytest.skip("Metal doesn't support entry point params.")
    device = helpers.get_device(device_type)
    code, bindings, cd = build_call_data_full(
        device, "add", "int add(int a, int b) { return a + b; }", 1, 2
    )

    # --- fast path flag ---
    assert cd.use_entrypoint_args is True

    # --- no CallData struct or ParameterBlock ---
    assert_not_contains(code, "struct CallData", "ParameterBlock<CallData>", "uniform CallData")

    # --- individual uniform params on compute_main ---
    assert_contains(code, "uniform uint3 _thread_count")
    assert_contains(code, "uniform int a")
    assert_contains(code, "uniform int b")

    # --- _thread_count used directly in bounds check ---
    assert_not_contains(code, "call_data._thread_count")
    main_idx = code.index("void compute_main(")
    main_body = code[main_idx:]
    assert ">= _thread_count)" in main_body

    # --- SV_GroupID absent for dim-0 (no shape arrays) ---
    assert_not_contains(code, "SV_GroupID")


# 36 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_entrypoint_params_vectorized(device_type: spy.DeviceType):
    """Fast path vectorized: shape arrays as entry-point params, SV_GroupID present.

    Verifies: use_entrypoint_args=True, shape arrays (_grid_stride, _grid_dim,
    _call_dim) as uniform params, SV_GroupID/SV_GroupIndex present when
    call_data_len > 0, no struct CallData.
    """
    if device_type == spy.DeviceType.metal:
        pytest.skip("Metal doesn't support entry point params.")
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))
    code, bindings, cd = build_call_data_full(
        device, "add", "float add(float a, float b) { return a + b; }", 1.0, tensor
    )

    # --- fast path ---
    assert cd.use_entrypoint_args is True

    # --- no CallData ---
    assert_not_contains(code, "struct CallData")

    # --- SV_GroupID/SV_GroupIndex present (call_data_len > 0) ---
    assert_contains(code, "SV_GroupID", "SV_GroupIndex")

    # --- shape arrays as entry-point params ---
    assert_contains(
        code, "uniform int[1] _grid_stride", "uniform int[1] _grid_dim", "uniform int[1] _call_dim"
    )

    # --- shape arrays NOT prefixed with call_data. in kernel body ---
    assert_not_contains(
        code, "call_data._grid_stride", "call_data._grid_dim", "call_data._call_dim"
    )


# 37 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_entrypoint_params_non_direct_bind(device_type: spy.DeviceType):
    """Fast path with non-direct-bind arg: no CallData, wrapper used, Context present.

    WangHashArg is NOT direct-bind but still goes as an entry-point param on the
    fast path. __slangpy_load and Context are present because the wrapper needs them.

    Merges: test_gate_p2_wanghasharg_keeps_load,
    test_step21_wanghasharg_uses_entrypoint_args.
    """
    if device_type == spy.DeviceType.metal:
        pytest.skip("Metal doesn't support entry point params.")
    device = helpers.get_device(device_type)
    code, bindings, cd = build_call_data_full(
        device, "rng", "uint3 rng(uint3 input) { return input; }", WangHashArg(3)
    )

    # --- fast path despite non-direct-bind ---
    assert cd.use_entrypoint_args is True

    # --- non-direct-bind binding ---
    assert bindings.args[0].direct_bind is False

    # --- wrapper type used ---
    assert_contains(code, "WangHashArg<")

    # --- __slangpy_load and Context present ---
    assert_contains(code, "__slangpy_load", "Context")

    # --- no CallData struct ---
    assert_not_contains(code, "struct CallData")


# 38 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bwds_entrypoint_no_diff_params(device_type: spy.DeviceType):
    """Bwds fast path: trampoline params have no_diff, bwd_diff call passes individuals.

    Verifies: use_entrypoint_args=True, trampoline params have 'no_diff' and
    '__in_' prefix, bwd_diff(_trampoline) call passes individual arg names,
    [Differentiable] before trampoline.
    """
    if device_type == spy.DeviceType.metal:
        pytest.skip("Metal doesn't support entry point params.")
    device = helpers.get_device(device_type)
    src = """
[Differentiable]
float polynomial(float a, float b) {
    return a * a + b + 1;
}
"""
    code, bindings, cd = build_bwds_call_data_full(device, "polynomial", src, 5.0, 10.0, 26.0)

    # --- fast path ---
    assert cd.use_entrypoint_args is True

    # --- trampoline params have no_diff ---
    assert_contains(code, "no_diff")
    assert_contains(code, "no_diff float a")
    assert_contains(code, "no_diff float b")

    # --- [Differentiable] before trampoline ---
    diff_idx = code.index("[Differentiable]")
    trampoline_idx = code.index("void _trampoline")
    assert diff_idx < trampoline_idx

    # --- bwd_diff call passes individual args (not just context) ---
    main_idx = code.index("void compute_main(")
    main_body = code[main_idx:]
    assert "bwd_diff(_trampoline)(__slangpy_context__" in main_body
    # Should have more than just the context arg
    bwd_call_start = main_body.index("bwd_diff(_trampoline)(")
    bwd_call_end = main_body.index(")", bwd_call_start + len("bwd_diff(_trampoline)("))
    bwd_call_args = main_body[bwd_call_start:bwd_call_end]
    assert ", a," in bwd_call_args or ", a)" in bwd_call_args

    # --- no struct CallData ---
    assert_not_contains(code, "struct CallData")


# 40 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fallback_calldata_large_params(device_type: spy.DeviceType):
    """Fallback path: many float4x4 params exceed threshold -> ParameterBlock<CallData>.

    8 x float4x4 = 512 bytes + 12 bytes _thread_count = 524 bytes.
    Exceeds Vulkan (128) and D3D12 (256); CUDA (4096) stays on fast path.
    Asserts codegen patterns match the expected path.

    Merges: test_step21_many_float4x4_may_exceed_vulkan (adds codegen assertions).
    """
    device = helpers.get_device(device_type)
    src = """
float4x4 sum8(float4x4 a, float4x4 b, float4x4 c, float4x4 d,
              float4x4 e, float4x4 f, float4x4 g, float4x4 h) {
    return a + b + c + d + e + f + g + h;
}
"""
    identity = spy.math.float4x4.identity()
    code, bindings, cd = build_call_data_full(
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
        # CUDA: fast path - no CallData, individual uniform params
        assert cd.use_entrypoint_args is True
        assert_not_contains(code, "struct CallData")
        assert_contains(code, "uniform uint3 _thread_count")
    else:
        # Vulkan/D3D12: fallback - struct CallData + ParameterBlock
        assert cd.use_entrypoint_args is False
        assert_contains(code, "struct CallData")
        assert_contains(code, "ParameterBlock<CallData> call_data")
        assert_contains(code, "call_data._thread_count")
        assert_not_contains(code, "uniform uint3 _thread_count")


# ===========================================================================
# Prim-mode trampoline elimination (41-42)
# ===========================================================================


# 41 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_prim_no_trampoline(device_type: spy.DeviceType):
    """Prim mode: no _trampoline function, call inlined in compute_main."""
    device = helpers.get_device(device_type)
    code, _ = generate_code_and_bindings(
        device, "add", "int add(int a, int b) { return a + b; }", 1, 2
    )
    # No trampoline function generated
    assert_not_contains(code, "void _trampoline(")
    # compute_main does NOT call _trampoline - it inlines the call
    main_idx = code.index("void compute_main(")
    main_body = code[main_idx:]
    assert "_trampoline(" not in main_body
    assert "add(__tmp_a, __tmp_b);" in main_body


# 42 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_array_of_structs_codegen(device_type: spy.DeviceType):
    """Struct with array-of-structs field: Outer{Inner items[4]} - all dim-0, direct-bind."""
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
    code, bindings = generate_code_and_bindings(
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
    assert_contains(code, "Outer __tmp_outer;")
    assert_not_contains(code, "__slangpy_load")
    assert_load_statement(code, "outer")

    assert bindings.args[0].direct_bind is True


# ===========================================================================
# Additional use_entrypoint_args coverage (43-46)
# ===========================================================================


# 43 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_threshold_property_positive(device_type: spy.DeviceType):
    """Device has a positive max_entry_point_uniform_size threshold."""
    device = helpers.get_device(device_type)
    threshold = device.info.limits.max_entry_point_uniform_size
    assert threshold > 0


# 44 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vector_uses_entrypoint_args(device_type: spy.DeviceType):
    """float3 args are small enough for entry-point params."""
    if device_type == spy.DeviceType.metal:
        pytest.skip("Metal doesn't support entry point params.")
    device = helpers.get_device(device_type)
    _, _, cd = build_call_data_full(
        device,
        "scale",
        "float3 scale(float3 v, float s) { return v * s; }",
        spy.math.float3(1, 2, 3),
        2.0,
    )
    assert cd.use_entrypoint_args is True


# 45 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_struct_uses_entrypoint_args(device_type: spy.DeviceType):
    """All-scalar struct dict has small inline-uniform size."""
    if device_type == spy.DeviceType.metal:
        pytest.skip("Metal doesn't support entry point params.")
    device = helpers.get_device(device_type)
    src = """
struct S { float x; float y; };
float sum(S s) { return s.x + s.y; }
"""
    _, _, cd = build_call_data_full(device, "sum", src, {"_type": "S", "x": 1.0, "y": 2.0})
    assert cd.use_entrypoint_args is True


# 46 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_uses_entrypoint_args(device_type: spy.DeviceType):
    """Tensor args contribute descriptor-only (0 inline bytes) -> entry-point params."""
    if device_type == spy.DeviceType.metal:
        pytest.skip("Metal doesn't support entry point params.")
    device = helpers.get_device(device_type)
    tensor = Tensor.from_numpy(device, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    _, _, cd = build_call_data_full(
        device,
        "sum_all",
        "float sum_all(float x) { return x; }",
        tensor,
    )
    assert cd.use_entrypoint_args is True


# ===========================================================================
# Additional functional dispatch tests (47-49)
# ===========================================================================


# 47 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_valueref_read(device_type: spy.DeviceType):
    """Dispatch with a read-only ValueRef input - direct-bind pipeline end-to-end."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "double_it", "float double_it(float v) { return v * 2; }"
    )
    result = func(ValueRef(7.0))
    assert abs(result - 14.0) < 1e-5


# 48 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_struct_return(device_type: spy.DeviceType):
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


# 49 ------------------------------------------------------------------------
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_long_struct_name(device_type: spy.DeviceType):
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


if __name__ == "__main__":
    pytest.main([__file__, "-vs"])
