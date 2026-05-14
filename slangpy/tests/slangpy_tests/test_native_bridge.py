# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.native_func import BaseModule, BaseStruct
from slangpy.native_refl import (
    ArrayType,
    Layout,
    MatrixType,
    ScalarType,
    StructType,
    TensorType,
    VectorType,
)
from slangpy.testing import helpers


MODULE_SOURCE = """
import "slangpy";

struct Foo {
    float3 value;
};
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_and_struct_have_native_bridge(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE_SOURCE)

    assert isinstance(module, BaseModule)
    assert module.layout.find_type_by_name("Foo") is not None

    struct = module.Foo.as_struct()

    assert isinstance(struct, BaseStruct)

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
