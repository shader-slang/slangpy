# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.benchmark import BenchmarkSlangFunction


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_add_simple(
    device_type: spy.DeviceType, benchmark_slang_function: BenchmarkSlangFunction
):
    device = helpers.get_device(device_type)
    a = np.random.rand(1024, 1024).astype(np.float32)
    b = np.random.rand(1024, 1024).astype(np.float32)
    tensor_a = spy.Tensor.from_numpy(device, a)
    tensor_b = spy.Tensor.from_numpy(device, b)
    tensor_c = spy.Tensor.empty(device, shape=(1024, 1024), dtype=float)

    module = spy.Module(device.load_module("test_benchmark_tensor.slang"))
    func = module.require_function(f"add")

    benchmark_slang_function(device, func, a=tensor_a, b=tensor_b, _result=tensor_c)
    assert np.allclose(tensor_c.to_numpy(), a + b)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("count", [2, 4, 8, 16, 32])
def test_tensor_sum(
    device_type: spy.DeviceType, count: int, benchmark_slang_function: BenchmarkSlangFunction
):
    device = helpers.get_device(device_type)
    inputs = [np.random.rand(1024, 1024).astype(np.float32) for _ in range(count)]
    input_tensors = [spy.Tensor.from_numpy(device, input) for input in inputs]
    result_tensor = spy.Tensor.empty(device, shape=(1024, 1024), dtype=float)

    module = spy.Module(device.load_module("test_benchmark_tensor.slang"))
    func = module.require_function(f"sum<{count}>")

    benchmark_slang_function(
        device, func, tid=spy.call_id(), tensors=input_tensors, _result=result_tensor
    )
    assert np.allclose(result_tensor.to_numpy(), sum(inputs))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("count", [2, 4, 8, 16, 32])
def test_tensor_sum_indirect(
    device_type: spy.DeviceType, count: int, benchmark_slang_function: BenchmarkSlangFunction
):
    device = helpers.get_device(device_type)
    inputs = [np.random.rand(1024, 1024).astype(np.float32) for _ in range(count)]
    input_tensors = [spy.Tensor.from_numpy(device, input) for input in inputs]
    result_tensor = spy.Tensor.empty(device, shape=(1024, 1024), dtype=float)

    module = spy.Module(device.load_module("test_benchmark_tensor.slang"))
    func = module.require_function(f"sum_indirect<{count}>")

    tensor_list = {"tensors": input_tensors}
    tensor_indices = list(range(count))

    benchmark_slang_function(
        device,
        func,
        tid=spy.call_id(),
        tensor_list=tensor_list,
        tensor_indices=tensor_indices,
        _result=result_tensor,
    )
    assert np.allclose(result_tensor.to_numpy(), sum(inputs))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
