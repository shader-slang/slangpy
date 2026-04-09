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

float func_with_return(uint3 dispatchThreadID) { return 0.0; }

void func_with_out(uint3 dispatchThreadID, out float result) { result = 0.0; }

void func_wrong_param(uint3 threadID) {}

"""


def load_test_module(device_type: DeviceType, link: list[Any] = [], options: dict[str, Any] = {}):
    device = helpers.get_device(device_type)
    return helpers.create_module(device, MODULE, link=link, options=options)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_entrypoint(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
    mod.func_entrypoint.dispatch(uint3(32, 1, 1), buffer=buffer.storage)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)

    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_tensor_entrypoint(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
    mod.tensor_entrypoint.dispatch(uint3(32, 1, 1), buffer=buffer)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_func(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
    mod.func_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer.storage)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_tensor_func(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
    mod.tensor_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_override_threadgroup(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
    mod.func_threadparam.thread_group_size(uint3(1, 1, 1)).dispatch(
        uint3(32, 1, 1), buffer=buffer.storage
    )
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_multiply_scalar(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
    mod.tensor_multiply.dispatch(uint3(32, 1, 1), buffer=buffer, amount=10)
    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i * 10, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_multiply_const(device_type: DeviceType):
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
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
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))

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
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))

    func = func.set(lambda x: {"params": {"k": 30}})
    func.dispatch(uint3(32, 1, 1), buffer=buffer)

    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i * 30, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_return_value_error(device_type: DeviceType):
    """Dispatching a function with a return value should raise ValueError."""
    mod = load_test_module(device_type)
    with pytest.raises(ValueError, match="return value"):
        mod.func_with_return.dispatch(uint3(1, 1, 1))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_out_param_error(device_type: DeviceType):
    """Dispatching a function with out parameters should raise ValueError."""
    mod = load_test_module(device_type)
    with pytest.raises(ValueError, match="out or inout"):
        mod.func_with_out.dispatch(uint3(1, 1, 1), result=0.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_no_params_error(device_type: DeviceType):
    """Dispatching a zero-parameter function should raise ValueError."""
    mod = load_test_module(device_type)
    with pytest.raises(ValueError, match="first parameter must be a thread id"):
        mod.func_noparams.dispatch(uint3(1, 1, 1))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_wrong_param_name_error(device_type: DeviceType):
    """Dispatching a function where first param isn't dispatchThreadID should raise ValueError."""
    mod = load_test_module(device_type)
    with pytest.raises(ValueError, match="dispatchThreadID"):
        mod.func_wrong_param.dispatch(uint3(1, 1, 1))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_with_writer(device_type: DeviceType):
    """Dispatch with .write() exercises the writer tuple path in DispatchData.dispatch."""
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))
    func = mod.tensor_multiply_uniform.as_func()

    def writer(cursor, k_value):
        cursor.write({"params": {"k": k_value}})

    func = func.write(writer, 42)
    func.dispatch(uint3(32, 1, 1), buffer=buffer)

    data = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)
    expected = np.array([[i * 42, 0, 0] for i in range(32)])
    assert np.all(data == expected)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dispatch_cache_hit(device_type: DeviceType):
    """Dispatching same function twice should hit the dispatch data cache."""
    mod = load_test_module(device_type)
    buffer = Tensor.empty(mod.device, dtype=mod.uint3, shape=(32,))

    mod.tensor_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer)
    data1 = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)

    mod.tensor_threadparam.dispatch(uint3(32, 1, 1), buffer=buffer)
    data2 = helpers.read_tensor_from_numpy(buffer).reshape(-1, 3)

    expected = np.array([[i, 0, 0] for i in range(32)])
    assert np.all(data1 == expected)
    assert np.all(data2 == expected)


TORCH_DISPATCH_SRC = r"""
import "slangpy";

void copy_vals(uint3 dispatchThreadID, RWTensor<float,1> output, Tensor<float,1> input) {
    output[dispatchThreadID.x] = input[dispatchThreadID.x] * 2.0;
}
"""


@pytest.mark.parametrize(
    "device_type", [t for t in helpers.DEFAULT_DEVICE_TYPES if t == DeviceType.cuda]
)
def test_dispatch_torch_tensor(device_type: DeviceType):
    """Raw dispatch with torch tensor triggers create_dispatchdata."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import slangpy.torchintegration.torchtensormarshall  # noqa: F401

    device = helpers.get_device(device_type)
    from slangpy import Module

    torch_mod = Module.load_from_source(device, "torch_raw_dispatch", TORCH_DISPATCH_SRC)

    inp = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
    out = torch.zeros(4, device="cuda", dtype=torch.float32)
    torch_mod.copy_vals.dispatch(uint3(4, 1, 1), output=out, input=inp)
    expected = torch.tensor([2.0, 4.0, 6.0, 8.0], device="cuda", dtype=torch.float32)
    assert torch.allclose(out, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
