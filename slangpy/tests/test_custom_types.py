# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from slangpy.backend import DeviceType, float3, int3, uint3
from slangpy.tests import helpers
from slangpy.types.buffer import NDBuffer
from slangpy.types.randfloatarg import RandFloatArg
from slangpy.types.threadidarg import ThreadIdArg
from slangpy.types.wanghasharg import WangHashArg


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_thread_id(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device, "thread_ids", """
int3 thread_ids(int3 input) {
    return input;
}
"""
    )

    # Make buffer for results
    results = NDBuffer(
        element_count=128,
        device=device,
        dtype=int3
    )

    # Call function with 3D thread arg. Pass results in, so it forces
    # a call shape.
    kernel_output_values(ThreadIdArg(3), _result=results)

    # Should get out the thread ids
    data = results.storage.to_numpy().view("int32").reshape((-1, 3))
    expected = [[i, 0, 0] for i in range(128)]
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
