# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Regular CI: host/shader bindings and basic pointer reads, writes and nulls."""

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers


SHADER = "test_buffer_ptr.slang"


@pytest.fixture(params=helpers.DEFAULT_DEVICE_TYPES)
def device(request: pytest.FixtureRequest) -> spy.Device:
    return helpers.get_device(request.param)


def _module(device: spy.Device) -> spy.SlangModule:
    return device.load_module(SHADER)


def _kernel(device: spy.Device, name: str) -> spy.ComputeKernel:
    return device.create_compute_kernel(program=device.load_program(SHADER, [name]))


# Host construction, cursor layout and functional binding


def test_host_and_functional_access(device: spy.Device) -> None:
    buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([10, 20, 30, 40], dtype=np.uint32)
    )
    ptr = spy.to_buffer_ptr(buffer, byte_offset=8)
    assert ptr.is_valid
    module = spy.Module(_module(device))
    assert module.read_value(src=ptr, index=0) == 30
    assert module.read_value(src=ptr, index=1) == 40
    assert module.read_advanced(src=spy.to_buffer_ptr(buffer)) == 70
    assert not spy.BufferPtr().is_valid
    assert not spy.RWBufferPtr().is_valid
    for byte_offset, alignment in [(2, 4), (4, 16), (20, 4), (0, 3), (2**32, 4)]:
        with pytest.raises(RuntimeError):
            spy.to_buffer_ptr(buffer, byte_offset=byte_offset, alignment=alignment)


def test_layout_cursor_and_nulls(device: spy.Device) -> None:
    module = _module(device)
    for name in (
        "BufferPtr<>",
        "RWBufferPtr<>",
        "BufferPtr<void,Access.ReadWrite>",
        "BufferPtr<float3,Access.Read,16>",
    ):
        typ = module.layout.find_type_by_name(f"StructuredBuffer<{name}>")
        layout = module.layout.get_type_layout(typ).element_type_layout
        assert (layout.size, layout.stride, layout.alignment) == (8, 8, 8)
    typ = module.layout.find_type_by_name("StructuredBuffer<PointerRecord>")
    layout = module.layout.get_type_layout(typ).element_type_layout
    # CUDA reports tail padding as part of size; all targets agree on array stride.
    expected_size = 24 if device.info.type == spy.DeviceType.cuda else 20
    assert (layout.size, layout.stride, layout.alignment) == (expected_size, 24, 8)
    assert [field.offset for field in layout.fields] == [0, 8, 16]
    data = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([10, 20, 30, 40], dtype=np.uint32)
    )
    record = device.create_buffer(usage=spy.BufferUsage.shader_resource, size=layout.stride)
    cursor = spy.BufferCursor(layout, record)
    cursor[0]["tag"].write(123)
    cursor[0]["ptr"].write(spy.to_buffer_ptr(data, 8))
    cursor[0]["tail"].write(456)
    cursor.apply()
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=24)
    _kernel(device, "read_record").dispatch((1, 1, 1), records=record, result=result)
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32)[:3], [123, 40, 456])
    table = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        data=np.array([spy.BufferPtr().data, 0], dtype=np.uint64),
    )
    _kernel(device, "nulls").dispatch(
        (1, 1, 1), src=spy.BufferPtr(), dst=spy.RWBufferPtr(), pointer_table=table, result=result
    )
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32), [0, 0, 0, 0, 0, 1])
    with pytest.raises(RuntimeError, match="cursor type"):
        cursor[0]["ptr"].write(spy.RWBufferPtr())
    cursor[0]["ptr"].write(spy.BufferPtr())
    cursor.apply()
    assert record.to_numpy().view(np.uint64)[1] == 2**64 - 1

    typ = module.layout.find_type_by_name("StructuredBuffer<BufferPtr<void,Access.ReadWrite>>")
    layout = module.layout.get_type_layout(typ).element_type_layout
    writable_table = device.create_buffer(usage=spy.BufferUsage.shader_resource, size=16)
    writable_cursor = spy.BufferCursor(layout, writable_table)
    writable_cursor[0].write(spy.to_rw_buffer_ptr(result, 8))
    writable_cursor[1].write(spy.RWBufferPtr())
    writable_cursor.apply()
    assert writable_table.to_numpy().view(np.uint64)[1] == 2**64 - 1
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(result, spy.ResourceState.unordered_access)
    _kernel(device, "write_table").dispatch(
        (1, 1, 1), command_encoder=encoder, pointers=writable_table
    )
    device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32), [0, 0, 0, 99, 0, 1])
    functional = spy.Module(module)
    assert functional.write_value(dst=spy.to_rw_buffer_ptr(result, 4), value=55) == 55
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32), [0, 55, 0, 99, 0, 1])


@pytest.mark.parametrize(
    "shader_type",
    ["BufferPtr<uint>", "BufferPtr<void,Access.Read,16>", "RWBufferPtr<>", "uint64_t"],
)
def test_rejected_cursor_specializations(device: spy.Device, shader_type: str) -> None:
    module = _module(device)
    layout = module.layout.get_type_layout(
        module.layout.find_type_by_name(f"StructuredBuffer<{shader_type}>")
    ).element_type_layout
    storage = device.create_buffer(usage=spy.BufferUsage.shader_resource, size=layout.stride)
    cursor = spy.BufferCursor(layout, storage)
    with pytest.raises(RuntimeError, match="cursor type"):
        cursor[0].write(spy.BufferPtr())
