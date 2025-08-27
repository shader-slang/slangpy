# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import numpy.typing as npt
from slangpy import DeviceType, float3, uint3
from slangpy.experimental.gridarg import grid
from slangpy.types.buffer import NDBuffer
from slangpy.types.callidarg import call_id
from slangpy.types.randfloatarg import RandFloatArg, rand_float
from slangpy.types.threadidarg import thread_id
from slangpy.types.wanghasharg import WangHashArg, calc_wang_hash_numpy, wang_hash
from slangpy.testing import helpers

from typing import Any, Callable


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("dimensions", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("signed", [False, True])
def test_thread_id(device_type: DeviceType, dimensions: int, signed: bool):

    inttype = "int" if signed else "uint"

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
        device,
        "thread_ids",
        f"""
{type_name} thread_ids({type_name} input) {{
    return input;
}}
""",
    )

    # Make buffer for results
    results = NDBuffer(
        element_count=128,
        device=device,
        dtype=kernel_output_values.module.layout.find_type_by_name(type_name),
    )

    # Call function with 3D thread arg. Pass results in, so it forces
    # a call shape.
    kernel_output_values(thread_id(dims), _result=results)

    # Should get out the thread ids
    data = helpers.read_ndbuffer_from_numpy(results).reshape((-1, elements))

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

    inttype = "int" if signed else "uint"

    if dimensions > 0:
        # If dimensions > 0, test passing explicit dimensions into corresponding vector/array type
        type_name = f"int[{dimensions}]" if array else f"{inttype}{dimensions}"
        elements = dimensions
        dims = dimensions
    elif dimensions == 0:
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
        device,
        "call_ids",
        f"""
{type_name} call_ids({type_name} input) {{
    return input;
}}
""",
    )

    # Make buffer for results
    results = NDBuffer(
        shape=(16,) * elements,
        device=device,
        dtype=kernel_output_values.module.layout.find_type_by_name(type_name),
    )

    # Call function with 3D thread arg. Pass results in, so it forces
    # a call shape.
    kernel_output_values(call_id(dims), _result=results)

    # Should get out the thread ids
    data = helpers.read_ndbuffer_from_numpy(results).reshape((-1, elements))
    expected = np.indices((16,) * elements).reshape(elements, -1).T

    # Reverse order of components in last dimension of expected
    # if testing a vector type
    if not array and elements > 1:
        expected = np.flip(expected, axis=1)

    assert np.allclose(data, expected)


def calc_wang_hash(seed: int):
    seed = (seed ^ 61) ^ (seed >> 16)
    seed *= 9
    seed = seed ^ (seed >> 4)
    seed *= 0x27D4EB2D
    seed = seed ^ (seed >> 15)
    return seed


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("warmup", [0, 1, 2])
@pytest.mark.parametrize("hash_seed", [False, True])
@pytest.mark.parametrize("seed", [0, 2640457667])
def test_wang_hash(device_type: DeviceType, warmup: int, hash_seed: bool, seed: int):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "wang_hashes",
        """
uint3 wang_hashes(uint3 input) {
    return input;
}
""",
    )

    # Make buffer for results
    results = NDBuffer(element_count=16, device=device, dtype=uint3)

    # Call function with 3D wang hash arg
    kernel_output_values(
        WangHashArg(3, seed=seed, warmup=warmup, hash_seed=hash_seed), _result=results
    )

    # Calculate expected results
    thread_ids = np.indices((16,), dtype=np.uint32) * 3
    np_seeds = np.full((16,), seed, dtype=np.uint32)
    if hash_seed:
        np_seeds = calc_wang_hash_numpy(np_seeds)
    thread_hash = thread_ids
    for i in range(warmup):
        thread_hash = calc_wang_hash_numpy(thread_hash)
    expected_d0 = calc_wang_hash_numpy(thread_hash ^ np_seeds)
    expected_d1 = calc_wang_hash_numpy(expected_d0)
    expected_d2 = calc_wang_hash_numpy(expected_d1)

    # combine the 3 expected arrays into a single array
    expected = np.stack((expected_d0, expected_d1, expected_d2), axis=-1)

    # Should get out the following precalculated wang hashes
    data = helpers.read_ndbuffer_from_numpy(results).reshape((-1, 3))
    assert np.allclose(data, expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("warmup", [0, 1, 2])
@pytest.mark.parametrize("hash_seed", [False, True])
@pytest.mark.parametrize("seed", [0, 2640457667])
def test_wang_hash_scalar(device_type: DeviceType, warmup: int, hash_seed: bool, seed: int):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "wang_hashes",
        """
uint wang_hashes(uint input) {
    return input;
}
""",
    )

    # Make buffer for results
    results = NDBuffer(element_count=16, device=device, dtype=kernel_output_values.module.uint)

    # Call function with 3D wang hash arg
    kernel_output_values(wang_hash(warmup=warmup, hash_seed=hash_seed, seed=seed), _result=results)

    # Calculate expected results
    thread_ids = np.indices((16,), dtype=np.uint32)
    np_seeds = np.full((16,), seed, dtype=np.uint32)
    if hash_seed:
        np_seeds = calc_wang_hash_numpy(np_seeds)
    thread_hash = thread_ids
    for i in range(warmup):
        thread_hash = calc_wang_hash_numpy(thread_hash)
    expected = calc_wang_hash_numpy(thread_hash ^ np_seeds)

    # Should get out matching hashes
    data = helpers.read_ndbuffer_from_numpy(results)
    assert np.allclose(data, expected)


# Dumb test just to make sure hashes aren't completely broken!


def measure_sequential_hash_quality(hash_func: Callable[[int], npt.NDArray[Any]]):
    hashes = hash_func(0)
    hashes2 = hash_func(1)
    combined_array = np.concatenate((hashes, hashes2))
    unique, counts = np.unique(combined_array, return_counts=True)
    duplicates = np.sum(counts > 1)
    return 1 - duplicates / len(hashes)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_wang_seq_seeds(device_type: DeviceType):
    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "wang_hashes",
        """
uint wang_hashes(uint input) {
    return input;
}
""",
    )

    def read_values(seed: int):
        results = NDBuffer(
            element_count=16,  # 3840 * 2160,
            device=device,
            dtype=kernel_output_values.module.uint,
        )
        kernel_output_values(wang_hash(warmup=0, hash_seed=False, seed=seed), _result=results)
        return results.storage.to_numpy().view("uint32")

    normal_quality = measure_sequential_hash_quality(read_values)

    def read_values_hash_seed(seed: int):
        results = NDBuffer(
            element_count=16,  # 3840 * 2160,
            device=device,
            dtype=kernel_output_values.module.uint,
        )
        kernel_output_values(wang_hash(warmup=0, hash_seed=True, seed=seed), _result=results)
        return results.storage.to_numpy().view("uint32")

    hash_seed_quality = measure_sequential_hash_quality(read_values_hash_seed)

    def read_values_warmup(seed: int):
        results = NDBuffer(
            element_count=3840 * 2160,
            device=device,
            dtype=kernel_output_values.module.uint,
        )
        kernel_output_values(wang_hash(warmup=1, hash_seed=False, seed=seed), _result=results)
        return results.storage.to_numpy().view("uint32")

    warmup_quality = measure_sequential_hash_quality(read_values_warmup)

    # We know with no seed hash and no warmup, completely sequential seeds are temporarilly coherent
    assert normal_quality < 0.01

    # Both hashing the seed and/or warmup makes sequential seeds fine
    assert hash_seed_quality > 0.99999
    assert warmup_quality > 0.99999


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("warmup", [0, 1, 2])
@pytest.mark.parametrize("hash_seed", [False, True])
@pytest.mark.parametrize("seed", [0, 2640457667])
def test_rand_float(device_type: DeviceType, warmup: int, hash_seed: bool, seed: int):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "rand_float",
        """
float3 rand_float(float3 input) {
    return input;
}
""",
    )

    count = 1000

    # Make buffer for results
    results = NDBuffer(element_count=count, device=device, dtype=float3)

    # Call function with 3D random arg
    kernel_output_values(
        RandFloatArg(1.0, 3.0, 3, warmup=warmup, hash_seed=hash_seed, seed=seed),
        _result=results,
    )

    # Calculate expected results
    thread_ids = np.indices((count,), dtype=np.uint32) * 3
    np_seeds = np.full((count,), seed, dtype=np.uint32)
    if hash_seed:
        np_seeds = calc_wang_hash_numpy(np_seeds)
    thread_hash = thread_ids
    for i in range(warmup):
        thread_hash = calc_wang_hash_numpy(thread_hash)
    hash_d0 = calc_wang_hash_numpy(thread_hash ^ np_seeds)
    hash_d1 = calc_wang_hash_numpy(hash_d0)
    hash_d2 = calc_wang_hash_numpy(hash_d1)
    hash = np.stack((hash_d0, hash_d1, hash_d2), axis=-1)
    k = hash & 0x7FFFFF
    u = k / float(0x800000)
    values = 1.0 + 2.0 * u

    # Should get random numbers
    data = helpers.read_ndbuffer_from_numpy(results).reshape((-1, 3))
    assert np.allclose(data, values)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_rand_float_uniformity(device_type: DeviceType):
    bucket_size = 17

    # Create function that atomically increments counts
    device = helpers.get_device(device_type)
    bucket_values = helpers.create_function_from_module(
        device,
        "add_to_bucket",
        f"""
void add_to_bucket(int id, RWByteAddressBuffer bucket, float value) {{
    int idx = int(value * ( {bucket_size - 1}));
    bucket.InterlockedAdd(idx * 4, 1);
}}
""",
    )

    # Make buffer for bucket of counts
    buckets = NDBuffer(element_count=bucket_size, device=device, dtype=int)
    buckets.clear()

    # Run bucketer with 1M random floats
    # TODO: Find out why this took insanely long to run on CUDA with 1B floats
    count = 1 * 1000 * 1000
    bucket_values(grid((count,)), buckets.storage, rand_float())

    # Verify the distribution of values in range [0,1) is roughly even
    res = buckets.to_numpy().view("int32")
    expected_count_per_bucket = count / (bucket_size - 1)
    for i in range(bucket_size - 1):
        bucket = float(res[i])
        rel_diff = abs(bucket - expected_count_per_bucket) / expected_count_per_bucket
        assert rel_diff < 0.005

    # Verify 1 never turns up
    assert res[bucket_size - 1] == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_rand_soa(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "rand_float_soa",
        """
struct Particle {
    float3 pos;
    float3 vel;
};
Particle dummy;
Particle rand_float_soa(Particle input) {
    return input;
}
""",
    )

    module = kernel_output_values.module

    # Make buffer for results
    results = NDBuffer(
        element_count=16,
        device=device,
        dtype=module.layout.find_type_by_name("Particle"),
    )

    # Call function with 3D random arg
    kernel_output_values(
        {
            "pos": RandFloatArg(-100.0, 100.0, 3),
            "vel": RandFloatArg(0.0, np.pi * 2.0, 3),
        },
        _result=results,
    )

    # Should get random numbers
    data = helpers.read_ndbuffer_from_numpy(results)
    print(data)

    pos = np.array([item["pos"] for item in data])
    dir = np.array([item["vel"] for item in data])
    assert np.all(pos >= -100.0) and np.all(pos <= 100.0)
    assert np.all(dir >= 0) and np.all(dir <= np.pi * 2)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_range(device_type: DeviceType):

    # Create function that just dumps input to output
    device = helpers.get_device(device_type)
    kernel_output_values = helpers.create_function_from_module(
        device,
        "range_test",
        """
int range_test(int input) {
    return input;
}
""",
    )

    # Call function with 3D random arg
    res = kernel_output_values(range(10, 20, 2))

    # Should get random numbers
    data = res.storage.to_numpy().view("int32")
    assert np.all(data == [10, 12, 14, 16, 18])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
