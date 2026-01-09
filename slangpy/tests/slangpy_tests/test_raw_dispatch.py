# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

from slangpy import DeviceType, uint3
from slangpy.types import Tensor
from slangpy.testing import helpers

from typing import Any

MODULE = r"""
import "slangpy";

void func_noparams() {

}

void func_threadparam( uint3 dispatchThreadID, RWStructuredBuffer<uint3> buffer ) {
    buffer[dispatchThreadID.x] = dispatchThreadID;
}

void tensor_threadparam( uint3 dispatchThreadID, RWTensor<uint3,1> buffer ) {
    buffer[dispatchThreadID.x] = dispatchThreadID;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void func_entrypoint(uint3 dispatchThreadID: SV_DispatchThreadID, RWStructuredBuffer<uint3> buffer) {
    buffer[dispatchThreadID.x] = dispatchThreadID;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void tensor_entrypoint(uint3 dispatchThreadID: SV_DispatchThreadID, RWTensor<uint3,1> buffer) {
    buffer[dispatchThreadID.x] = dispatchThreadID;
}


void tensor_multiply( uint3 dispatchThreadID, RWTensor<uint3,1> buffer, uint amount ) {
    buffer[dispatchThreadID.x] = dispatchThreadID * amount;
}

extern static const int VAL;
void tensor_multiply_const( uint3 dispatchThreadID, RWTensor<uint3,1> buffer ) {
    buffer[dispatchThreadID.x] = dispatchThreadID * VAL;
}

struct Params {
    int k;
}
ParameterBlock<Params> params;

void tensor_multiply_uniform(uint3 dispatchThreadID, RWTensor<uint3,1> buffer) {
    buffer[dispatchThreadID.x] = dispatchThreadID * params.k;
}

"""


def load_test_module(device_type: DeviceType, link: list[Any] = [], options: dict[str, Any] = {}):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, MODULE, link=link, options=options)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_entrypoint(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)
    mod.func_entrypoint.dispatch(uint3(32, 1, 1), buffer=buffer.storage)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)

    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_tensor_entrypoint(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)
    mod.tensor_entrypoint.dispatch(uint3(32, 1, 1), buffer=buffer)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_func(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)
    mod.func_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer.storage)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_tensor_func(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)
    mod.tensor_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_override_threadgroup(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)
    mod.func_threadparam.thread_group_size(uint3(1, 1, 1)).dispatch(
        uint3(32, 1, 1), buffer=buffer.storage
    )
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_multiply_scalar(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)
    mod.tensor_multiply.dispatch(uint3(32, 1, 1), buffer=buffer, amount=10)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i * 10, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_multiply_const(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)
    mod.tensor_multiply_const.constants({"VAL": 5}).dispatch(
        uint3(32, 1, 1), buffer=buffer, amount=10
    )
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i * 5, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_set(device_type: DeviceType):
    mod = load_test_module(device_type)
    assert mod is not None

    func = mod.tensor_multiply_uniform.as_func()
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)

    func = func.set({"params": {"k": 20}})
    func.dispatch(uint3(32, 1, 1), buffer=buffer)

    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i * 20, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_set_with_callback(device_type: DeviceType):
    mod = load_test_module(device_type)
    assert mod is not None

    func = mod.tensor_multiply_uniform.as_func()
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, element_count=32)

    func = func.set(lambda x: {"params": {"k": 30}})
    func.dispatch(uint3(32, 1, 1), buffer=buffer)

    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i * 30, 0, 0] for i in range(32)])
    assert np.all(data == expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
