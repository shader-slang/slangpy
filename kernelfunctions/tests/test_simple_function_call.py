from typing import Optional, Union
import numpy as np
import pytest
from kernelfunctions.backend import DeviceType, Device
import kernelfunctions as kf
import kernelfunctions.tests.helpers as helpers

from kernelfunctions.types.valueref import intRef
from kernelfunctions.types.diffpair import diffPair, floatDiffPair


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(int a, int b) {
}
""",
    )

    # just verify it can be called with no exceptions
    function(5, 10)

    # verify call fails with wrong number of arguments
    with pytest.raises(ValueError):
        function(5)

    # verify call fails with wrong type of arguments
    with pytest.raises(ValueError):
        function(5, False)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_returnvalue(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int add_numbers(int a, int b) {
    return a+b;
}
""",
    )

    # just verify it can be called with no exceptions
    res = function(5, 10)
    assert res == 15


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_returnvalue_with_diffpair_input(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int add_numbers(int a, int b) {
    return a+b;
}
""",
    )

    # just verify it can be called with no exceptions
    res = function(diffPair(5), 10)
    assert res == 15


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_outparam(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(int a, int b, out int c) {
    c = a+b;
}
""",
    )

    # Should fail, as pure python 'int' can't be used to receive output.
    with pytest.raises(
        ValueError, match="Cannot read back value for non-writable type"
    ):
        val_res: int = 0
        function(5, 10, val_res)

    # Using a scalar output the function should be able to output a value.
    out_res = intRef()
    function(5, 10, out_res)
    assert out_res.value == 15


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_inoutparam(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(inout int a) {
    a += 10;
}
""",
    )
    # Using a scalar output the function should be able to output a value.
    out_res = intRef(5)
    function(out_res)
    assert out_res.value == 15


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scalar_outparam_with_diffpair(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(float a, float b, out float c) {
    c = a+b;
}
""",
    )

    # Using a scalar output the function should be able to output a value.
    out_res = floatDiffPair()
    function(5.0, 10.0, out_res)
    assert out_res.primal == 15.0


def rand_array_of_ints(size: int):
    return np.random.randint(0, 100, size=size, dtype=np.int32)


def buffer_pair_test(
    device: Device,
    in_buffer_0_size: int,
    in_buffer_1_size: Optional[int] = None,
    out_buffer_size: Optional[int] = None,
):
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(int a, int b, out int c) {
    c = a + b;
}
""",
    )

    if in_buffer_1_size is None:
        in_buffer_1_size = in_buffer_0_size
    if out_buffer_size is None:
        out_buffer_size = max(in_buffer_0_size, in_buffer_1_size)

    # Setup input buffers
    in_buffer_0: Union[int, kf.NDBuffer]
    if in_buffer_0_size == 0:
        in_buffer_0 = int(rand_array_of_ints(1)[0])
    else:
        in_buffer_0 = kf.NDBuffer(
            element_count=in_buffer_0_size,
            device=device,
            element_type=int,
        )
        in_buffer_0.buffer.from_numpy(rand_array_of_ints(in_buffer_0.element_count))

    in_buffer_1: Union[int, kf.NDBuffer]
    if in_buffer_1_size == 0:
        in_buffer_1 = int(rand_array_of_ints(1)[0])
    else:
        in_buffer_1 = kf.NDBuffer(
            element_count=in_buffer_1_size,
            device=device,
            element_type=int,
        )
        in_buffer_1.buffer.from_numpy(rand_array_of_ints(in_buffer_1.element_count))

    # Setup output buffer
    out_buffer = kf.NDBuffer(
        element_count=out_buffer_size,
        device=device,
        element_type=int,
    )

    # Call function
    function(in_buffer_0, in_buffer_1, out_buffer)

    # Read output data and read-back input data to verify results
    if isinstance(in_buffer_0, int):
        in_data_0 = np.array([in_buffer_0] * out_buffer_size)
    else:
        in_data_0 = in_buffer_0.buffer.to_numpy().view(np.int32)
    if isinstance(in_buffer_1, int):
        in_data_1 = np.array([in_buffer_1] * out_buffer_size)
    else:
        in_data_1 = in_buffer_1.buffer.to_numpy().view(np.int32)
    out_data = out_buffer.buffer.to_numpy().view(np.int32)
    for i in range(32):
        assert out_data[i] == in_data_0[i] + in_data_1[i]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_with_buffer(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buffer_pair_test(device, 128)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_with_none_threadgroup_sized_buffer(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buffer_pair_test(device, 73)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_with_mismatched_size_buffers(device_type: DeviceType):
    device = helpers.get_device(device_type)
    with pytest.raises(ValueError):
        buffer_pair_test(device, 32, 64)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_function_with_broadcast(device_type: DeviceType):
    device = helpers.get_device(device_type)
    buffer_pair_test(device, 74, 0)
    buffer_pair_test(device, 0, 74)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_remap_output(device_type: DeviceType):
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "add_numbers_remap",
        r"""
void add_numbers_remap(int a, int b, out int c) {
    c = a + b;
}
""",
    )

    a = kf.NDBuffer(
        element_count=100,
        device=device,
        element_type=int,
    )
    a.buffer.from_numpy(rand_array_of_ints(a.element_count))

    b = kf.NDBuffer(
        element_count=50,
        device=device,
        element_type=int,
    )
    b.buffer.from_numpy(rand_array_of_ints(b.element_count))

    c = kf.NDBuffer(
        element_count=50,
        device=device,
        element_type=int,
    )
    c.buffer.from_numpy(rand_array_of_ints(c.element_count))

    function.transform_output({"a": (1,), "b": (0,)})(a, b, c)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_returnvalue_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int add_numbers(int a, int b) {
    return a+b;
}
""",
    )

    a = kf.NDBuffer(
        element_count=50,
        device=device,
        element_type=int,
    )
    a.buffer.from_numpy(rand_array_of_ints(a.element_count))

    b = kf.NDBuffer(
        element_count=50,
        device=device,
        element_type=int,
    )
    b.buffer.from_numpy(rand_array_of_ints(b.element_count))

    # just verify it can be called with no exceptions
    res: kf.NDBuffer = function(a, b)

    a_data = a.buffer.to_numpy().view(np.int32)
    b_data = b.buffer.to_numpy().view(np.int32)
    res_data = res.buffer.to_numpy().view(np.int32)

    assert np.all(res_data == a_data + b_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
