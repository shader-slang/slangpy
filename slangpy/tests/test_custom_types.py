# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import pytest

from slangpy.backend import DeviceType, float3, int3, uint3
from slangpy.tests import helpers
from slangpy.types.buffer import NDBuffer
from slangpy.types.callidarg import call_id
from slangpy.types.randfloatarg import RandFloatArg
from slangpy.types.threadidarg import ThreadIdArg, thread_id
from slangpy.types.wanghasharg import WangHashArg, wang_hash


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dimensions", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("signed", [False, True])
def test_thread_id(device_type: DeviceType, dimensions: int, signed: bool):

    inttype = 'int' if signed else 'uint'

    if dimensions > 0:
        # If dimensions > 0, test passing explicit dimensions into corresponding vector type
        type_name = f"{inttype}{dimensions}"
        elements = dimensions
        dims = dimensions
    elif dimensions == 0:
        # If dimensions == 0, test passing 1D value into corresponding scalar type
        type_name = inttype
        elements = 1
        dims = 1
    else:
        # If dimensions == -1, test passing undefined dimensions to 3d vector type
        type_name = f"{inttype}3"
        elements = 3
        dims = -1

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "thread_ids", f"""
{type_name} thread_ids({type_name} input) {{
    return input;
}}
"""
    )

    # Make buffer for results
    results = NDBuffer(
        element_count=128,
        device=device,
        dtype=kernel_output_values.module.layout.find_type_by_name(type_name)
    )

    # Call function with 3D thread arg. Pass results in, so it forces
    # a call shape.
    kernel_output_values(thread_id(dims), _result=results)

    # Should get out the thread ids
    data = results.storage.to_numpy().view("int32").reshape((-1, elements))
    if elements == 1:
        expected = [[i] for i in range(128)]
    elif elements == 2:
        expected = [[i, 0] for i in range(128)]
    elif elements == 3:
        expected = [[i, 0, 0] for i in range(128)]
    assert np.allclose(data, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dimensions", [-1, 1, 2, 3])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("array", [False, True])
def test_call_id(device_type: DeviceType, dimensions: int, signed: bool, array: bool):

    inttype = 'int' if signed else 'uint'

    if dimensions > 0:
        # If dimensions > 0, test passing explicit dimensions into corresponding vector/array type
        type_name = f"int[{dimensions}]" if array else f"{inttype}{dimensions}"
        elements = dimensions
        dims = dimensions
    elif dimensions == 0:
        if array:
            pytest.skip("Array not supported for 0D call_id")

        # If dimensions == 0, test passing 1D value into corresponding scalar type
        type_name = inttype
        elements = 1
        dims = 1
    else:
        # If dimensions == -1, test passing undefined dimensions to implicit array or 3d vector type
        type_name = f"int[3]" if array else f"{inttype}3"
        elements = 3
        dims = -1

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "call_ids", f"""
{type_name} call_ids({type_name} input) {{
    return input;
}}
"""
    )

    # Make buffer for results
    results = NDBuffer(
        shape=(16,)*elements,
        device=device,
        dtype=kernel_output_values.module.layout.find_type_by_name(type_name)
    )

    # Call function with 3D thread arg. Pass results in, so it forces
    # a call shape.
    kernel_output_values(call_id(dims), _result=results)

    # Should get out the thread ids
    data = results.storage.to_numpy().view("int32").reshape((-1, elements))
    expected = np.indices((16,)*elements).reshape(elements, -1).T

    # Reverse order of components in last dimension of expected
    # if testing a vector type
    if not array and elements > 1:
        expected = np.flip(expected, axis=1)

    assert np.allclose(data, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_wang_hash(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "wang_hashes", """
uint3 wang_hashes(uint3 input) {
    return input;
}
"""
    )

    # Make buffer for results
    results = NDBuffer(
        element_count=16,
        device=device,
        dtype=uint3
    )

    # Call function with 3D wang hash arg
    kernel_output_values(WangHashArg(3), _result=results)

    # Should get out the following precalculated wang hashes
    data = results.storage.to_numpy().view("uint32").reshape((-1, 3))
    expected = [[3232319850, 3075307816,  755367838],
                [663891101, 1738326990,  801461103],
                [3329832309,  685338552, 3175962347],
                [2278584254,   41021378, 1955303707],
                [3427349084,  820536086, 3381787118],
                [3322605197, 2681520273, 3073157428],
                [1902946834, 2388446925,  244231649],
                [851741419,   62190945, 3501556970],
                [3030050807, 4159240091,  137079654],
                [454550906, 2504917575,  811039371],
                [1415330532, 3026032642,  714972250],
                [1798297286, 1541577139, 2313671325],
                [161999925,  949220758,  846072919],
                [1881449543, 2723368086, 2785454616],
                [1220296370, 3358723660, 1136221045],
                [1603248408, 1883436325, 2091632478]]
    assert np.allclose(data, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_wang_hash_scalar(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "wang_hashes", """
uint wang_hashes(uint input) {
    return input;
}
"""
    )

    # Make buffer for results
    results = NDBuffer(
        element_count=16,
        device=device,
        dtype=kernel_output_values.module.uint
    )

    # Call function with 3D wang hash arg
    kernel_output_values(wang_hash(), _result=results)

    # Should get out the following precalculated wang hashes
    data = results.storage.to_numpy().view("uint32")
    print(data)
    expected = [3232319850,  663891101, 3329832309, 2278584254, 3427349084,
                3322605197, 1902946834,  851741419, 3030050807,  454550906,
                1415330532, 1798297286,  161999925, 1881449543, 1220296370,
                1603248408]
    assert np.allclose(data, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_rand_float(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "rand_float", """
float3 rand_float(float3 input) {
    return input;
}
"""
    )

    # Make buffer for results
    results = NDBuffer(
        element_count=16,
        device=device,
        dtype=float3
    )

    # Call function with 3D random arg
    kernel_output_values(RandFloatArg(1.0, 2.0, 3), _result=results)

    # Should get random numbers
    data = results.storage.to_numpy().view("float32").reshape((-1, 3))
    assert np.all(data >= 1.0) and np.all(data <= 2.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_rand_soa(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "rand_float_soa", """
struct Particle {
    float3 pos;
    float3 vel;
};
Particle dummy;
Particle rand_float_soa(Particle input) {
    return input;
}
"""
    )

    module = kernel_output_values.module

    # Make buffer for results
    results = NDBuffer(
        element_count=16,
        device=device,
        dtype=module.layout.find_type_by_name("Particle")
    )

    # Call function with 3D random arg
    kernel_output_values({
        'pos': RandFloatArg(-100.0, 100.0, 3),
        'vel': RandFloatArg(0.0, np.pi*2.0, 3),
    }, _result=results)

    # Should get random numbers
    data = results.storage.to_numpy().view("float32")[0:16*6].reshape((-1, 6))
    (pos, dir) = np.split(data, 2, axis=1)
    assert np.all(pos >= -100.0) and np.all(pos <= 100.0)
    assert np.all(dir >= 0) and np.all(dir <= np.pi*2)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_range(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "range_test", """
int range_test(int input) {
    return input;
}
"""
    )

    # Call function with 3D random arg
    res = kernel_output_values(range(10, 20, 2))

    # Should get random numbers
    data = res.storage.to_numpy().view("int32")
    assert np.all(data == [10, 12, 14, 16, 18])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
