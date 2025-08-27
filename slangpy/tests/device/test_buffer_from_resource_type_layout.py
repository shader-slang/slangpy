# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_buffer_from_resource_type_layout(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    module = device.load_module("test_buffer_from_resource_type_layout.slang")
    program = device.link_program(
        modules=[module], entry_points=[module.entry_point("compute_main")]
    )

    buffer_uint = device.create_buffer(
        element_count=1, resource_type_layout=program.reflection.buffer_uint
    )
    assert buffer_uint.struct_size == 4

    buffer_float4 = device.create_buffer(
        element_count=1, resource_type_layout=program.reflection.buffer_float4
    )
    assert buffer_float4.struct_size == 16

    buffer_my_struct = device.create_buffer(
        element_count=1, resource_type_layout=program.reflection.buffer_my_struct
    )
    assert buffer_my_struct.struct_size == 32

    # Passing a TypeReflection is not allowed
    with pytest.raises(TypeError):
        device.create_buffer(
            element_count=1, resource_type_layout=program.layout.find_type_by_name("MyStruct")
        )

    # Passing a TypeLayoutReflection that does not correspond to a resource type is not allowed
    with pytest.raises(TypeError):
        device.create_buffer(
            element_count=1,
            resource_type_layout=program.layout.get_type_layout(
                program.layout.find_type_by_name("MyStruct")
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
