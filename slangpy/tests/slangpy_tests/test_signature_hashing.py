# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from time import time

from slangpy import DeviceType, float3, Module
import slangpy.core.function as kff
from slangpy.types import Tensor
from slangpy.testing import helpers

# We mess with cache in this suite, so make sure it gets turned on correctly before each test


@pytest.fixture(autouse=True)
def run_after_each_test():
    kff.ENABLE_CALLDATA_CACHE = False
    yield


def load_module(device_type: DeviceType, name: str = "test_modules.slang") -> Module:
    device = helpers.get_device(device_type)
    return Module(device.load_module(name))


# @pytest.mark.skip(reason="Perf test only")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_kernel_reuse(device_type: DeviceType):
    add_vectors = load_module(device_type).add_vectors.as_func()

    encoder = Tensor.empty(helpers.get_device(device_type), dtype=int, shape=(4,))

    count = 1000

    # kff.ENABLE_CALLDATA_CACHE = False
    #
    # start = time()
    # for i in range(0,count):
    #    res = add_vectors(float3(1,2,3),float3(4,5,6))
    # end = time()
    # print(f"Time taken uncached: {1000.0*(end-start)/count}ms")

    kff.ENABLE_CALLDATA_CACHE = True

    a = Tensor.empty(helpers.get_device(device_type), dtype=float3, shape=(1,))
    b = Tensor.empty(helpers.get_device(device_type), dtype=float3, shape=(1,))
    res = Tensor.empty(helpers.get_device(device_type), dtype=float3, shape=(1,))

    add_vectors(a, b, _result=res)

    encoder = helpers.get_device(device_type).create_command_encoder()

    start = time()
    for i in range(0, int(count / 10)):
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
        add_vectors.append_to(encoder, a, b, _result=res)
    end = time()

    encoder.finish()

    print(f"Time taken cached: {1000.0*(end-start)/count}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
