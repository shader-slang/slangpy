# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.helpers import test_id  # type: ignore (pytest fixture)


# Before running more in depth link time tests below, this test simply
# verifies that the basic linking of 2 modules together with exported
# variables works.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_link_time_modules_compile(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    extra_module = device.load_module_from_source(
        f"test_link_time_basic_extra_{test_id}",
        r"""
        export static const uint NUM_THREADS = 16;
    """,
    )
    assert extra_module is not None

    main_module = device.load_module_from_source(
        f"test_link_time_basic_{test_id}",
        r"""
        extern static const uint NUM_THREADS;

        [shader("compute")]
        [numthreads(NUM_THREADS, 1, 1)]
        void compute_main(uint3 tid: SV_DispatchThreadID)
        {
        }
    """,
    )
    assert main_module is not None

    program = device.link_program(
        [main_module, extra_module], [main_module.entry_point("compute_main")]
    )
    assert program is not None


# Test modifying just 1 constant value
@pytest.mark.parametrize("value", [2, 5])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_link_time_constant_value(test_id: str, device_type: spy.DeviceType, value: int):
    device = helpers.get_device(type=device_type)

    constants = f"export static const float VALUE = {value};\n" if value is not None else ""
    extra_module = device.load_module_from_source(
        f"test_link_time_constants_value_extra_{test_id}", constants
    )

    main_module = device.load_module_from_source(
        f"test_link_time_constant_value_{test_id}",
        r"""
        extern static const float VALUE;

        RWStructuredBuffer<float> result;

        [shader("compute")]
        [numthreads(16, 1, 1)]
        void compute_main(uint3 tid: SV_DispatchThreadID)
        {
            result[tid.x] = tid.x * VALUE;
        }
    """,
    )
    program = device.link_program(
        [main_module, extra_module], [main_module.entry_point("compute_main")]
    )
    assert program is not None
    assert program.layout.entry_points[0].compute_thread_group_size == [16, 1, 1]

    kernel = device.create_compute_kernel(program)

    result = device.create_buffer(
        element_count=16,
        resource_type_layout=program.reflection.result,
        usage=spy.BufferUsage.unordered_access,
    )

    kernel.dispatch(thread_count=[16, 1, 1], vars={"result": result})

    assert np.all(
        result.to_numpy().view(dtype=np.float32).flatten() == np.linspace(0, 15, 16) * value
    )


# Test with threads and value, using code from a slang file
@pytest.mark.parametrize("value", [2, 5])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_link_time_constants(device_type: spy.DeviceType, value: int):
    device = helpers.get_device(type=device_type)

    constants = "\n".join(
        [
            f"export static const uint NUM_THREADS = {16};",
            f"export static const float VALUE = {value};" if value is not None else "",
        ]
    )

    program = device.load_program(
        module_name="test_link_time_constants.slang",
        entry_point_names=["compute_main"],
        additional_source=constants,
    )

    assert program.layout.entry_points[0].compute_thread_group_size == [16, 1, 1]

    kernel = device.create_compute_kernel(program)

    result = device.create_buffer(
        element_count=16,
        resource_type_layout=program.reflection.result,
        usage=spy.BufferUsage.unordered_access,
    )

    kernel.dispatch(thread_count=[16, 1, 1], vars={"result": result})

    assert np.all(
        result.to_numpy().view(dtype=np.float32).flatten() == np.linspace(0, 15, 16) * value
    )


@pytest.mark.parametrize("op", ["AddOp", "SubOp", "MulOp"])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_link_time_type(device_type: spy.DeviceType, op: str):
    device = helpers.get_device(type=device_type)

    constants = "\n".join(
        [
            'import "test_link_time_type_binary_op.slang";',
            f"export struct BINARY_OP : IBinaryOp = {op};",
        ]
    )

    program = device.load_program(
        module_name="test_link_time_type.slang",
        entry_point_names=["compute_main"],
        additional_source=constants,
    )

    kernel = device.create_compute_kernel(program)

    result = device.create_buffer(
        element_count=1,
        resource_type_layout=program.reflection.result,
        usage=spy.BufferUsage.unordered_access,
    )

    kernel.dispatch(thread_count=[1, 1, 1], vars={"result": result})

    expected = {"AddOp": 3, "SubOp": -1, "MulOp": 2}[op]

    assert result.to_numpy().view(dtype=np.float32)[0] == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
