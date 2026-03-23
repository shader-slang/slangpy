# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_is_component_type(device_type: spy.DeviceType):
    """Test that SlangModule is an instance of SlangComponentType."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    assert isinstance(module, spy.SlangComponentType)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_entry_point_is_component_type(device_type: spy.DeviceType):
    """Test that SlangEntryPoint is an instance of SlangComponentType."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    entry_point = module.entry_point("compute_main")
    assert isinstance(entry_point, spy.SlangComponentType)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_component_type_session(device_type: spy.DeviceType):
    """Test that component_type.session returns the correct session."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    assert module.session is not None
    assert isinstance(module.session, spy.SlangSession)

    entry_point = module.entry_point("compute_main")
    assert entry_point.session is not None
    assert isinstance(entry_point.session, spy.SlangSession)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_component_type_layout(device_type: spy.DeviceType):
    """Test that component_type_layout() returns a valid ProgramLayout."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    layout = module.component_type_layout()
    assert layout is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_component_type_specialization_param_count(device_type: spy.DeviceType):
    """Test specialization_param_count property."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")

    # A non-generic module should have 0 specialization params.
    assert module.specialization_param_count == 0

    # A generic entry point should have specialization params.
    generic_entry = module.entry_point("compute_generic")
    assert generic_entry.specialization_param_count > 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_component_type_entry_point_count(device_type: spy.DeviceType):
    """Test entry_point_count property."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")

    # Module component type should have entry points.
    assert module.entry_point_count >= 0

    # An individual entry point component type should have exactly 1 entry point.
    entry_point = module.entry_point("compute_main")
    assert entry_point.entry_point_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_component_type_link(device_type: spy.DeviceType):
    """Test linking a component type."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    entry_point = module.entry_point("compute_main")

    # Create composite then link.
    session = module.session
    composite = session.create_composite_component_type([module, entry_point])
    linked = composite.link()
    assert linked is not None
    assert isinstance(linked, spy.SlangComponentType)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_component_type_specialize(device_type: spy.DeviceType):
    """Test specializing a generic entry point via SlangComponentType API."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")

    generic_entry = module.entry_point("compute_generic")

    # Specialize using the base class specialize_component_type method.
    specialized = generic_entry.specialize_component_type(
        [spy.SpecializationArg.from_type("ScaleBy2")]
    )
    assert specialized is not None
    assert isinstance(specialized, spy.SlangComponentType)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_composite_component_type(device_type: spy.DeviceType):
    """Test creating a composite component type from module + entry point."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    entry_point = module.entry_point("compute_main")

    session = module.session
    composite = session.create_composite_component_type([module, entry_point])
    assert composite is not None
    assert isinstance(composite, spy.SlangComponentType)

    # The composite should have at least 1 entry point.
    assert composite.entry_point_count >= 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_composite_and_link_dispatch(device_type: spy.DeviceType):
    """Test creating a composite, linking, and dispatching a compute kernel."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    entry_point = module.entry_point("compute_main")

    # Create composite and link via the new API.
    session = module.session
    composite = session.create_composite_component_type([module, entry_point])
    linked = composite.link()
    assert linked is not None

    # Also verify the traditional link_program path still works.
    program = device.link_program(modules=[module], entry_points=[entry_point])
    kernel = device.create_compute_kernel(program)

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer = device.create_buffer(
        data=input_data,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    kernel.dispatch(thread_count=[4, 1, 1], data=buffer, count=4)

    result = buffer.to_numpy().view(np.float32)
    expected = input_data + 1.0
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_type_conformance(device_type: spy.DeviceType):
    """Test creating a type conformance component type."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    session = module.session

    conformance = session.create_type_conformance("ScaleBy2", "ITransform")
    assert conformance is not None
    assert isinstance(conformance, spy.SlangTypeConformance)
    assert isinstance(conformance, spy.SlangComponentType)

    # Check the conformance descriptor.
    assert conformance.conformance.type_name == "ScaleBy2"
    assert conformance.conformance.interface_name == "ITransform"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_type_conformance_with_id(device_type: spy.DeviceType):
    """Test creating a type conformance with explicit id override."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    session = module.session

    conformance = session.create_type_conformance("ScaleBy2", "ITransform", id_override=42)
    assert conformance is not None
    assert conformance.conformance.id == 42


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_type_conformance_invalid_type(device_type: spy.DeviceType):
    """Test that creating a conformance with invalid type raises an error."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    session = module.session

    with pytest.raises(RuntimeError):
        session.create_type_conformance("NonExistent", "ITransform")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_type_conformance_invalid_interface(device_type: spy.DeviceType):
    """Test that creating a conformance with invalid interface raises an error."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    session = module.session

    with pytest.raises(RuntimeError):
        session.create_type_conformance("ScaleBy2", "IDoesNotExist")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_composite_with_type_conformance(device_type: spy.DeviceType):
    """Test creating a composite that includes a type conformance."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    session = module.session

    conformance = session.create_type_conformance("ScaleBy2", "ITransform")
    entry_point = module.entry_point("compute_main")

    # Compose module + entry_point + conformance.
    composite = session.create_composite_component_type(
        [module, entry_point, conformance]
    )
    assert composite is not None
    assert isinstance(composite, spy.SlangComponentType)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_linked_component_type_from_shader_program(device_type: spy.DeviceType):
    """Test that ShaderProgram.linked_component_type returns a SlangComponentType."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    entry_point = module.entry_point("compute_main")

    program = device.link_program(modules=[module], entry_points=[entry_point])
    linked_ct = program.linked_component_type
    assert linked_ct is not None
    assert isinstance(linked_ct, spy.SlangComponentType)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_link_with_options(device_type: spy.DeviceType):
    """Test link_with_options on a composite component type."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")
    entry_point = module.entry_point("compute_main")

    session = module.session
    composite = session.create_composite_component_type([module, entry_point])

    # link_with_options with empty/default options should succeed.
    linked = composite.link_with_options({})
    assert linked is not None
    assert isinstance(linked, spy.SlangComponentType)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_generic_entry_point_and_dispatch(device_type: spy.DeviceType):
    """Test specializing a generic entry point and running the kernel."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")

    # Get the generic entry point.
    generic_entry = module.entry_point("compute_generic")

    # Specialize with ScaleBy2.
    specialized_entry = generic_entry.specialize(
        [spy.SpecializationArg.from_type("ScaleBy2")]
    )

    program = device.link_program(modules=[module], entry_points=[specialized_entry])
    kernel = device.create_compute_kernel(program)

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buffer = device.create_buffer(
        data=input_data,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    kernel.dispatch(thread_count=[4, 1, 1], data=buffer, count=4)

    result = buffer.to_numpy().view(np.float32)
    expected = input_data * 2.0
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_specialize_generic_entry_point_addone(device_type: spy.DeviceType):
    """Test specializing a generic entry point with AddOne."""
    device = helpers.get_device(type=device_type)
    module = device.load_module("test_component_type.slang")

    generic_entry = module.entry_point("compute_generic")
    specialized_entry = generic_entry.specialize(
        [spy.SpecializationArg.from_type("AddOne")]
    )

    program = device.link_program(modules=[module], entry_points=[specialized_entry])
    kernel = device.create_compute_kernel(program)

    input_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    buffer = device.create_buffer(
        data=input_data,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    kernel.dispatch(thread_count=[4, 1, 1], data=buffer, count=4)

    result = buffer.to_numpy().view(np.float32)
    expected = input_data + 1.0
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
