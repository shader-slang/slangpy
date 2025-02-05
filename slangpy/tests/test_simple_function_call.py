# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union

import numpy as np
import pytest
from sgl import float3

import slangpy.tests.helpers as helpers
from slangpy.backend import Device, DeviceType
from slangpy.types import NDBuffer, Tensor
from slangpy.types.diffpair import diffPair, floatDiffPair
from slangpy.types.valueref import intRef


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

    # verify call that slang is happy with because bool can cast to int
    function(5, False)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_call_function(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
void add_numbers(int a, int b) {
}
""",
    )

    # verify call fails due to invalid cast (float3->int)
    with pytest.raises(ValueError):
        function(5, float3(1.0, 2.0, 3.0))

    # verify call fails with wrong number of arguments
    with pytest.raises(ValueError):
        function(5)


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
def test_call_specialized_function(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "return_number<int>",
        r"""
T return_number<T: IInteger>() {
    return T(42);
}
""",
    )

    assert function.name == "return_number<int>"

    res = function()
    assert res == 42


@pytest.mark.skip("Awaiting diff-pair follow-up")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_returnvalue_with_diffpair_input(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
float add_numbers(float a, float b) {
    return a+b;
}
""",
    )

    # just verify it can be called with no exceptions
    res = function(diffPair(5.0), 10.0)
    assert res == 15.0


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
    # with pytest.raises(
    #    ValueError, match="Cannot read back value for non-writable type"
    # ):
    #    val_res: int = 0
    #    function(5, 10, val_res)

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


@pytest.mark.skip("Awaiting diff-pair follow-up")
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
    in_buffer_0: Union[int, NDBuffer]
    if in_buffer_0_size == 0:
        in_buffer_0 = int(rand_array_of_ints(1)[0])
    else:
        in_buffer_0 = NDBuffer(
            element_count=in_buffer_0_size,
            device=device,
            dtype=int,
        )
        in_buffer_0.storage.copy_from_numpy(rand_array_of_ints(in_buffer_0.element_count))

    in_buffer_1: Union[int, NDBuffer]
    if in_buffer_1_size == 0:
        in_buffer_1 = int(rand_array_of_ints(1)[0])
    else:
        in_buffer_1 = NDBuffer(
            element_count=in_buffer_1_size,
            device=device,
            dtype=int,
        )
        in_buffer_1.storage.copy_from_numpy(rand_array_of_ints(in_buffer_1.element_count))

    # Setup output buffer
    out_buffer = NDBuffer(
        element_count=out_buffer_size,
        device=device,
        dtype=int,
    )

    # Call function
    function(in_buffer_0, in_buffer_1, out_buffer)

    # Read output data and read-back input data to verify results
    if isinstance(in_buffer_0, int):
        in_data_0 = np.array([in_buffer_0] * out_buffer_size)
    else:
        in_data_0 = in_buffer_0.storage.to_numpy().view(np.int32)
    if isinstance(in_buffer_1, int):
        in_data_1 = np.array([in_buffer_1] * out_buffer_size)
    else:
        in_data_1 = in_buffer_1.storage.to_numpy().view(np.int32)
    out_data = out_buffer.storage.to_numpy().view(np.int32)
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

    a = NDBuffer(
        element_count=100,
        device=device,
        dtype=int,
    )
    a.storage.copy_from_numpy(rand_array_of_ints(a.element_count))

    b = NDBuffer(
        element_count=50,
        device=device,
        dtype=int,
    )
    b.storage.copy_from_numpy(rand_array_of_ints(b.element_count))

    c = NDBuffer(
        shape=(50, 100),
        device=device,
        dtype=int,
    )
    c.storage.copy_from_numpy(rand_array_of_ints(c.element_count))

    function.map((1,), (0,))(a, b, c)


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

    a = NDBuffer(
        element_count=50,
        device=device,
        dtype=int,
    )
    a.storage.copy_from_numpy(rand_array_of_ints(a.element_count))

    b = NDBuffer(
        element_count=50,
        device=device,
        dtype=int,
    )
    b.storage.copy_from_numpy(rand_array_of_ints(b.element_count))

    # just verify it can be called with no exceptions
    res: NDBuffer = function(a, b)

    a_data = a.storage.to_numpy().view(np.int32)
    b_data = b.storage.to_numpy().view(np.int32)
    res_data = res.storage.to_numpy().view(np.int32)

    assert np.all(res_data == a_data + b_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_buffer_to_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "add_numbers",
        r"""
int add_numbers(NDBuffer<int,1> a, NDBuffer<int,1> b) {
    return a[{0}]+b[{0}];
}
""",
    )

    a = NDBuffer(
        element_count=1,
        device=device,
        dtype=int,
    )
    a.storage.copy_from_numpy(rand_array_of_ints(a.element_count))

    b = NDBuffer(
        element_count=1,
        device=device,
        dtype=int,
    )
    b.storage.copy_from_numpy(rand_array_of_ints(b.element_count))

    # just verify it can be called with no exceptions
    res = function(a, b)

    a_data = a.storage.to_numpy().view(np.int32)
    b_data = b.storage.to_numpy().view(np.int32)

    assert np.all(res == a_data[0] + b_data[0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_return_struct_in_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    # Note: Don't use create_function_from_module here. It adds an
    # implict import slangpy; that masks the bug this is testing for
    module = helpers.create_module(
        device,
        r"""
struct Foo { int x; }
Foo create_foo(int x) { return { x }; }
""",
    )

    x = NDBuffer(
        element_count=1,
        device=device,
        dtype=int,
    )
    x.storage.copy_from_numpy(rand_array_of_ints(x.element_count))

    result: NDBuffer = module.create_foo(x)

    assert result.shape == (1, )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("scalar_type", ["float", "half", "double"])
def test_pass_float_array(device_type: DeviceType, scalar_type: str):
    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device, f"{scalar_type} first({scalar_type} x[3]) {{ return x[0]; }}")

    arg = [3.0, 4.0, 5.0]
    result = module.first(arg)

    assert np.abs(result - arg[0]) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("scalar_type", ["uint8_t", "uint16_t", "uint", "uint64_t", "int8_t", "int16_t", "int", "int64_t"])
def test_pass_int_array(device_type: DeviceType, scalar_type: str):
    if device_type == DeviceType.d3d12 and scalar_type in ("int8_t", "uint8_t"):
        pytest.skip("8-bit types are unsupported by DXC")
    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device, f"{scalar_type} first({scalar_type} x[3]) {{ return x[0]; }}")

    arg = [3, 4, 5]
    result = module.first(arg)

    assert result == arg[0]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("scalar_type", ["float", "half", "double"])
def test_pass_float_field(device_type: DeviceType, scalar_type: str):
    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device,
        f"""
struct Foo {{ {scalar_type} x; }}
{scalar_type} unwrap(Foo foo) {{ return foo.x; }}
""",
    )

    arg = 3.0
    result = module.unwrap({"x": arg})

    assert np.abs(result - arg) < 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("scalar_type", ["uint8_t", "uint16_t", "uint", "uint64_t", "int8_t", "int16_t", "int", "int64_t"])
def test_pass_int_field(device_type: DeviceType, scalar_type: str):
    if device_type == DeviceType.d3d12 and scalar_type in ("int8_t", "uint8_t"):
        pytest.skip("8-bit types are unsupported by DXC")
    device = helpers.get_device(device_type)
    module = helpers.create_module(
        device,
        f"""
struct Foo {{ {scalar_type} x; }}
{scalar_type} unwrap(Foo foo) {{ return foo.x; }}
""",
    )

    arg = 3
    result = module.unwrap({"x": arg})

    assert result == arg


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_pass_buffer_to_structured_buffer(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "copy_first",
        r"""
void copy_first(StructuredBuffer<int> a, RWStructuredBuffer<int> b) {
    b[0] = a[0];
}
""",
    )

    a = NDBuffer(
        element_count=1,
        device=device,
        dtype=int,
    )
    a.storage.copy_from_numpy(np.array([42], dtype=np.float32))

    b = NDBuffer(
        element_count=1,
        device=device,
        dtype=int,
    )
    b.storage.copy_from_numpy(np.zeros((b.element_count, ), dtype=np.int32))

    function(a, b)

    a_data = a.storage.to_numpy().view(np.int32)
    b_data = b.storage.to_numpy().view(np.int32)

    assert a_data[0] == b_data[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
