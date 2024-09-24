

import pytest
from kernelfunctions.tests import helpers
from kernelfunctions.types.buffer import NDBuffer
from kernelfunctions.extensions.randfloatarg import RandFloatArg
from kernelfunctions.extensions.threadidarg import ThreadIdArg
import numpy as np

from kernelfunctions.extensions.wanghasharg import WangHashArg
from kernelfunctions.backend import DeviceType, int3, uint3, float3


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
        element_type=int3
    )

    # Call function with 3D thread arg. Pass results in, so it forces
    # a call shape.
    kernel_output_values(ThreadIdArg(3), _result=results)

    # Should get out the thread ids
    data = results.buffer.to_numpy().view("int32").reshape((-1, 3))
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
        element_type=uint3
    )

    # Call function with 3D wang hash arg
    kernel_output_values(WangHashArg(3), _result=results)

    # Should get out the following precalculated wang hashes
    data = results.buffer.to_numpy().view("uint32").reshape((-1, 3))
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
        element_type=float3
    )

    # Call function with 3D random arg
    kernel_output_values(RandFloatArg(1.0, 2.0, 3), _result=results)

    # Should get random numbers
    data = results.buffer.to_numpy().view("float32").reshape((-1, 3))
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

    sgl_module = kernel_output_values.module

    sb_layout = sgl_module.layout.get_type_layout(
        sgl_module.layout.find_type_by_name("StructuredBuffer<Particle>"))
    particle_layout = sb_layout.element_type_layout
    print(particle_layout.size)

    pt = kernel_output_values.module.layout.get_type_layout(
        kernel_output_values.module.layout.find_type_by_name("Particle"))

    # Make buffer for results
    results = NDBuffer(
        element_count=16,
        device=device,
        element_type=pt
    )

    # Call function with 3D random arg
    kernel_output_values({
        'pos': RandFloatArg(-100.0, 100.0, 3),
        'vel': RandFloatArg(0.0, np.pi*2.0, 3),
    }, _result=results)

    # Should get random numbers
    data = results.buffer.to_numpy().view("float32")[0:16*6].reshape((-1, 6))
    (pos, dir) = np.split(data, 2, axis=1)
    assert np.all(pos >= -100.0) and np.all(pos <= 100.0)
    assert np.all(dir >= 0) and np.all(dir <= np.pi*2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])