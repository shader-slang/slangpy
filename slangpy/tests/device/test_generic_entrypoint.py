# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_entrypoint_specialize(device_type: spy.DeviceType):
    """Test specializing a generic entrypoint with a concrete type."""
    device = helpers.get_device(type=device_type)

    module = device.load_module("test_generic_entrypoint.slang")

    # Get the generic entrypoint
    generic_entry = module.entry_point("compute_main")

    # Specialize it with DoubleIt
    specialized_entry = generic_entry.specialize([spy.SpecializationArg.from_type("DoubleIt")])

    # Link and create kernel
    program = device.link_program(modules=[module], entry_points=[specialized_entry])
    kernel = device.create_compute_kernel(program)

    # Create input data
    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer = device.create_buffer(
        data=input_data,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    # Dispatch
    kernel.dispatch(thread_count=[4, 1, 1], data=buffer, count=4)

    # Verify results (DoubleIt multiplies by 2)
    result = buffer.to_numpy().view(np.float32)
    expected = input_data * 2.0
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_entrypoint_multiple_specializations(device_type: spy.DeviceType):
    """Test specializing the same generic entrypoint with different types."""
    device = helpers.get_device(type=device_type)

    module = device.load_module("test_generic_entrypoint.slang")

    # Get the generic entrypoint
    generic_entry = module.entry_point("compute_main")

    # Specialize with DoubleIt
    double_entry = generic_entry.specialize([spy.SpecializationArg.from_type("DoubleIt")])
    double_program = device.link_program(modules=[module], entry_points=[double_entry])
    double_kernel = device.create_compute_kernel(double_program)

    # Specialize with SquareIt
    square_entry = generic_entry.specialize([spy.SpecializationArg.from_type("SquareIt")])
    square_program = device.link_program(modules=[module], entry_points=[square_entry])
    square_kernel = device.create_compute_kernel(square_program)

    # Specialize with AddTen
    add_entry = generic_entry.specialize([spy.SpecializationArg.from_type("AddTen")])
    add_program = device.link_program(modules=[module], entry_points=[add_entry])
    add_kernel = device.create_compute_kernel(add_program)

    # Test DoubleIt
    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer = device.create_buffer(
        data=input_data.copy(),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    double_kernel.dispatch(thread_count=[4, 1, 1], data=buffer, count=4)
    result = buffer.to_numpy().view(np.float32)
    assert np.allclose(
        result, input_data * 2.0
    ), f"DoubleIt failed: expected {input_data * 2.0}, got {result}"

    # Test SquareIt
    buffer = device.create_buffer(
        data=input_data.copy(),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    square_kernel.dispatch(thread_count=[4, 1, 1], data=buffer, count=4)
    result = buffer.to_numpy().view(np.float32)
    assert np.allclose(
        result, input_data * input_data
    ), f"SquareIt failed: expected {input_data * input_data}, got {result}"

    # Test AddTen
    buffer = device.create_buffer(
        data=input_data.copy(),
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    add_kernel.dispatch(thread_count=[4, 1, 1], data=buffer, count=4)
    result = buffer.to_numpy().view(np.float32)
    assert np.allclose(
        result, input_data + 10.0
    ), f"AddTen failed: expected {input_data + 10.0}, got {result}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_entrypoint_invalid_type(device_type: spy.DeviceType):
    """Test that specializing with an invalid type raises an error."""
    device = helpers.get_device(type=device_type)

    module = device.load_module("test_generic_entrypoint.slang")
    generic_entry = module.entry_point("compute_main")

    # Try to specialize with a non-existent type
    with pytest.raises(RuntimeError, match='Specialization type "NonExistentType" not found'):
        generic_entry.specialize([spy.SpecializationArg.from_type("NonExistentType")])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_entrypoint_int_generic_param(device_type: spy.DeviceType):
    """Test that entry_point() works for generic compute shaders with integer generic parameters."""
    device = helpers.get_device(type=device_type)

    module = device.load_module("test_generic_entrypoint.slang")

    # This should not raise - previously failed with untranslated nanobind exception
    generic_entry = module.entry_point("compute_int_generic")

    # Specialize with an integer expression
    specialized = generic_entry.specialize([spy.SpecializationArg.from_expr("4")])

    # Link and create kernel
    program = device.link_program(modules=[module], entry_points=[specialized])
    kernel = device.create_compute_kernel(program)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_entrypoint_int_generic_entry_points(device_type: spy.DeviceType):
    """Test that entry_points property works with generic compute shaders with integer generic parameters."""
    device = helpers.get_device(type=device_type)

    module = device.load_module("test_generic_entrypoint.slang")

    # This should not raise
    eps = module.entry_points
    assert len(eps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
