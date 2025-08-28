# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from slangpy import DeviceType, NDBuffer
from slangpy.types import Tensor
from slangpy.testing import helpers

MODULE = r"""
struct Foo {
    int x;
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buffer = NDBuffer.zeros(device, shape=(10, 20), dtype="float")

    # Test that repr() returns a meaningful string
    repr_str = repr(buffer)
    print(f"NDBuffer: {repr_str}")

    # Verify the repr contains expected information
    assert "NativeNDBuffer" in repr_str
    assert "shape" in repr_str
    assert "strides" in repr_str
    assert "dtype" in repr_str


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangtype_struct_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    foo_struct = module["Foo"].as_struct()

    # Access the underlying SlangType
    foo_type = foo_struct.struct

    # Test that repr() returns a meaningful string
    repr_str = repr(foo_type)
    print(f"SlangType: {repr_str}")

    # Verify the repr contains expected information
    assert "SlangType" in repr_str
    assert "name" in repr_str
    assert "Foo" in repr_str
    assert "shape" in repr_str


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangtype_vector_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    foo_struct = module["float3"].as_struct()

    # Access the underlying SlangType
    foo_type = foo_struct.struct

    # Test that repr() returns a meaningful string
    repr_str = repr(foo_type)
    print(f"SlangType: {repr_str}")

    # Verify the repr contains expected information
    assert "SlangType" in repr_str
    assert "name" in repr_str
    assert "vector<float,3>" in repr_str
    assert "shape" in repr_str


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)

    # Create a tensor without gradients
    tensor = Tensor.empty(device, dtype="float", shape=(3, 4))

    # Test that repr() returns a meaningful string - Tensor inherits from NativeTensor
    repr_str = repr(tensor)
    print(f"Tensor: {repr_str}")

    # Verify the repr contains expected information
    assert "NativeTensor" in repr_str
    assert "dtype" in repr_str
    assert "shape" in repr_str
    assert "has_grad_in=false" in repr_str
    assert "has_grad_out=false" in repr_str

    # Test with gradients
    grad_in = Tensor.zeros(device, dtype="float", shape=(3, 4))
    grad_out = Tensor.zeros(device, dtype="float", shape=(3, 4))
    tensor_with_grads = tensor.with_grads(grad_in, grad_out)

    repr_str_grads = repr(tensor_with_grads)
    print(f"Tensor with grads: {repr_str_grads}")
    assert "has_grad_in=true" in repr_str_grads
    assert "has_grad_out=true" in repr_str_grads


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)

    # Test that repr() returns a meaningful string for Module
    repr_str = repr(module)
    print(f"Module: {repr_str}")

    # Verify the repr contains expected information
    assert "Module" in repr_str
    assert "name=" in repr_str
    assert "linked_modules=" in repr_str


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slangtype_reflection_to_string(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    foo_struct = module["Foo"].as_struct()

    # Access the SlangType reflection
    foo_type_reflection = foo_struct.struct

    # Test that repr() returns a meaningful string for SlangType reflection
    repr_str = repr(foo_type_reflection)
    print(f"SlangType reflection: {repr_str}")

    # Verify the repr contains expected information - should get the Python SlangType repr
    assert "SlangType" in repr_str
    assert "name=" in repr_str
    assert "shape=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
