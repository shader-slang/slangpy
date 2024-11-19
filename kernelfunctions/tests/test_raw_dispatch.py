from typing import Any
import pytest
from kernelfunctions.backend import DeviceType, uint3
import kernelfunctions.tests.helpers as helpers
from kernelfunctions.types.buffer import NDBuffer
import numpy as np

MODULE = r"""
import "slangpy";

void func_noparams() {

}

void func_threadparam( uint3 dispatchThreadID, RWStructuredBuffer<uint3> buffer ) {
    buffer[dispatchThreadID.x] = dispatchThreadID;
}

void ndbuffer_threadparam( uint3 dispatchThreadID, RWNDBuffer<uint3,1> buffer ) {
    buffer[{dispatchThreadID.x}] = dispatchThreadID;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void func_entrypoint(uint3 dispatchThreadID: SV_DispatchThreadID, RWStructuredBuffer<uint3> buffer) {
    buffer[dispatchThreadID.x] = dispatchThreadID;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ndbuffer_entrypoint(uint3 dispatchThreadID: SV_DispatchThreadID, RWNDBuffer<uint3,1> buffer) {
    buffer[{dispatchThreadID.x}] = dispatchThreadID;
}


void ndbuffer_multiply( uint3 dispatchThreadID, RWNDBuffer<uint3,1> buffer, uint amount ) {
    buffer[{dispatchThreadID.x}] = dispatchThreadID * amount;
}

"""


def load_test_module(device_type: DeviceType, link: list[Any] = [], options: dict[str, Any] = {}):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, MODULE, link=link, options=options)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_entrypoint(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = NDBuffer(mod.device, mod.uint3, 32)
    mod.func_entrypoint.dispatch(uint3(32, 1, 1), buffer=buffer.buffer)
    data = buffer.to_numpy().view(np.uint32).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_ndbuffer_entrypoint(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = NDBuffer(mod.device, mod.uint3, 32)
    mod.ndbuffer_entrypoint.dispatch(uint3(32, 1, 1), buffer=buffer)
    data = buffer.to_numpy().view(np.uint32).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_func(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = NDBuffer(mod.device, mod.uint3, 32)
    mod.func_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer.buffer)
    data = buffer.to_numpy().view(np.uint32).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_ndbuffer_func(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = NDBuffer(mod.device, mod.uint3, 32)
    mod.ndbuffer_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer)
    data = buffer.to_numpy().view(np.uint32).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_override_threadgroup(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = NDBuffer(mod.device, mod.uint3, 32)
    mod.func_threadparam.thread_group_size(uint3(1, 1, 1)).dispatch(
        uint3(32, 1, 1), buffer=buffer.buffer)
    data = buffer.to_numpy().view(np.uint32).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_multiply_scalar(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = NDBuffer(mod.device, mod.uint3, 32)
    mod.ndbuffer_multiply.dispatch(uint3(32, 1, 1), buffer=buffer, amount=10)
    data = buffer.to_numpy().view(np.uint32).reshape(-1, 3)
    expected = np.array([[i*10, 0, 0] for i in range(32)])
    assert np.all(data == expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
