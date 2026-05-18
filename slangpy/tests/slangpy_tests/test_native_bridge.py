# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.native_func import BaseModule, BaseStruct
from slangpy.native_refl import (
    ArrayType,
    Field,
    Function,
    IOType,
    Layout,
    MatrixType,
    Parameter,
    ScalarType,
    StructType,
    TensorType,
    TextureType as NativeTextureType,
    VectorType,
    VoidType,
    get_builtin_layout,
    name_for_scalar_type,
    resolve_element_type,
    resolve_layout,
)
from slangpy.testing import helpers


MODULE_SOURCE = """
import "slangpy";

struct Foo {
    float3 value;
    float eval(float scale) { return value.x * scale; }
};

[Differentiable]
float add(float lhs, float rhs) { return lhs + rhs; }

void update(inout float value, out float result, no_diff in float weight) {
    result = value * weight;
}

void use_textures(Texture2D<float4> texture, RWTexture2D<float4> rw_texture) {}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_and_struct_have_native_bridge(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_SOURCE)

    assert isinstance(module, BaseModule)
    assert module.layout.find_type_by_name("Foo") is not None

    struct = module.Foo.as_struct()

    assert isinstance(struct, BaseStruct)
    assert struct.struct.type_reflection.full_name == "Foo"
    assert struct.type_reflection.full_name == "Foo"
    assert not hasattr(struct.struct, "reflection")
    assert not hasattr(struct, "reflection")

    tensor = spy.Tensor.empty(device, (2,), dtype=struct)
    assert tensor.dtype is struct.struct


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_native_refl_layout_tracks_hot_reload_generation(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_SOURCE)
    layout = Layout(module.module.layout)

    generation = layout.generation
    layout.on_hot_reload(module.module.layout)
    module.on_hot_reload()

    assert layout.generation == generation + 1
    assert module.layout.find_type_by_name("Foo") is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_native_refl_layout_creates_semantic_types(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_SOURCE)
    layout = Layout(module.module.layout)

    float_type = layout.require_type_by_name("float")
    assert isinstance(float_type, ScalarType)
    assert float_type.full_name == "float"
    assert float_type.shape == ()

    void_type = layout.scalar_type(spy.TypeReflection.ScalarType.void)
    assert isinstance(void_type, VoidType)
    assert void_type.full_name == "void"

    vector_type = layout.require_type_by_name("vector<float,3>")
    assert isinstance(vector_type, VectorType)
    assert vector_type.element_type is float_type
    assert vector_type.num_elements == 3
    assert vector_type.shape == (3,)

    matrix_type = layout.require_type_by_name("matrix<float,3,2>")
    assert isinstance(matrix_type, MatrixType)
    assert matrix_type.inner_element_type is float_type
    assert matrix_type.shape == (3, 2)

    array_type = layout.require_type_by_name("float[4]")
    assert isinstance(array_type, ArrayType)
    assert array_type.element_type is float_type
    assert array_type.array_shape == (4,)

    struct_type = layout.require_type_by_name("Foo")
    assert isinstance(struct_type, StructType)
    assert struct_type.full_name == "Foo"
    assert isinstance(struct_type.fields["value"], Field)
    assert struct_type.fields["value"].type is vector_type

    tensor_type = layout.tensor_type(
        float_type,
        2,
        access=TensorType.Access.read_write,
        tensor_kind=TensorType.Kind.tensor,
    )
    assert isinstance(tensor_type, TensorType)
    assert tensor_type.dtype is float_type
    assert tensor_type.dims == 2
    assert tensor_type.readable
    assert tensor_type.writable
    assert tensor_type.shape == (-1, -1)

    texture_function = layout.require_function_by_name("use_textures")
    texture_type = texture_function.parameters[0].type
    rw_texture_type = texture_function.parameters[1].type
    assert isinstance(texture_type, NativeTextureType)
    assert texture_type.usage == spy.TextureUsage.shader_resource
    assert isinstance(rw_texture_type, NativeTextureType)
    assert rw_texture_type.usage == spy.TextureUsage.unordered_access


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_native_refl_layout_creates_function_metadata(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_SOURCE)
    layout = Layout(module.module.layout)

    float_type = layout.require_type_by_name("float")
    struct_type = layout.require_type_by_name("Foo")

    function = layout.require_function_by_name("add")
    assert isinstance(function, Function)
    assert function.name == "add"
    assert function.return_type is float_type
    assert function.have_return_value
    assert function.differentiable

    parameters = function.parameters
    assert len(parameters) == 2
    assert isinstance(parameters[0], Parameter)
    assert parameters[0].name == "lhs"
    assert parameters[0].index == 0
    assert parameters[0].type is float_type
    assert parameters[0].io_type == IOType.inn

    update = layout.require_function_by_name("update")
    assert update.parameters[0].io_type == IOType.inout
    assert update.parameters[1].io_type == IOType.out
    assert update.parameters[2].no_diff

    method = layout.require_function_by_name_in_type(struct_type, "eval")
    assert method.this_type is struct_type
    assert method.full_name == "eval"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_native_refl_lookup_resolves_builtin_and_struct_types(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)

    builtin_layout = get_builtin_layout(device)
    assert builtin_layout.find_type_by_name("float") is not None
    assert name_for_scalar_type(spy.TypeReflection.ScalarType.float32) == "float"

    float_type = resolve_element_type(builtin_layout, "float")
    assert isinstance(float_type, ScalarType)
    assert float_type.full_name == "float"

    generation = builtin_layout.generation
    device.reload_all_programs()
    assert builtin_layout.generation == generation + 1

    module = helpers.create_module(device, MODULE_SOURCE)
    struct = module.Foo.as_struct()
    struct_layout = resolve_layout(device, struct)
    assert struct_layout.find_type_by_name("Foo") is not None

    python_struct_type = module.layout.require_type_by_name("Foo")
    assert resolve_element_type(struct_layout, struct).full_name == "Foo"
    assert resolve_element_type(struct_layout, python_struct_type).full_name == "Foo"
    assert (
        resolve_element_type(struct_layout, python_struct_type.buffer_layout.reflection).full_name
        == "Foo"
    )
