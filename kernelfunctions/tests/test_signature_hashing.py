from time import time
import pytest
from kernelfunctions.core import hash_signature
from kernelfunctions.backend import DeviceType, float3
from kernelfunctions.function import Function
from kernelfunctions.module import Module
from kernelfunctions.tests import helpers
from kernelfunctions.types.buffer import NDBuffer
import kernelfunctions.function as kff

# We mess with cache in this suite, so make sure it gets turned on correctly before each test


@pytest.fixture(autouse=True)
def run_after_each_test():
    kff.ENABLE_CALLDATA_CACHE = False
    yield


def load_module(device_type: DeviceType, name: str = "test_modules.slang") -> Module:
    device = helpers.get_device(device_type)
    return Module(device.load_module(name))


@pytest.mark.skip(reason="Perf test only")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_signature_gen(device_type: DeviceType):
    func: Function = load_module(device_type).get_particle_quad  # type: ignore

    buffer = NDBuffer(helpers.get_device(device_type), int, 4)

    start = time()
    count = 10000
    for i in range(0, count):
        hash_text = hash_signature(func, buffer, 1, 2, 3, 4)
    end = time()
    print(f"Time taken per signature: {1000.0*(end-start)/count}ms")
    print(hash_text)
    pass


@pytest.mark.skip(reason="Perf test only")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_kernel_reuse(device_type: DeviceType):
    add_vectors: Function = load_module(device_type).add_vectors  # type: ignore

    buffer = NDBuffer(helpers.get_device(device_type), int, 4)

    count = 10000

    # kff.ENABLE_CALLDATA_CACHE = False
#
    # start = time()
    # for i in range(0,count):
    #    res = add_vectors(float3(1,2,3),float3(4,5,6))
    # end = time()
    # print(f"Time taken uncached: {1000.0*(end-start)/count}ms")

    kff.ENABLE_CALLDATA_CACHE = True

    a = NDBuffer(helpers.get_device(device_type), float3, 1)
    b = NDBuffer(helpers.get_device(device_type), float3, 1)
    res = NDBuffer(helpers.get_device(device_type), float3, 1)

    add_vectors(a, b, _result=res)

    start = time()
    for i in range(0, int(count/10)):
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
        add_vectors(a, b, _result=res)
    end = time()
    print(f"Time taken cached: {1000.0*(end-start)/count}ms")

    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
