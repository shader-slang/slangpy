# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Benchmarks the cost model of Slang's CUDA dynamic-index legalization
# (shader-slang/slang#11939), which rewrites a runtime index into a by-value
# kernel parameter array to go through an eager, whole-aggregate per-thread
# local copy made in the kernel prologue.
#
# The reviewer question this answers: the copy is paid up front for the whole
# array, per thread, per dispatch - it is NOT made on demand - so where is the
# crossover between the copy cost and the serial `.param` load chain it
# replaces, and does the copy itself become a regression for large parameters
# with few accesses?
#
# Three kernels x a parameter-size sweep:
#   - pick_one:    1 dynamic access per thread  (worst case: pay N, use 1)
#   - sum_dynamic: N dynamic accesses per thread (best case: copy amortized)
#   - sum_static:  statically indexed control    (legalization must not fire)
#
# Run the same file against a pre-#11939 Slang build to get the serial-chain
# baseline, and against a #11939 build for the local-copy numbers; the
# sum_static control should be identical on both, and any pick_one regression
# on the new build quantifies the eager-copy tax. Sizes stay <= 512 floats
# (2 KB) so the argument blob stays under CUDA's 4 KB entry-point-argument
# threshold and the functional API keeps the by-value fast path being measured.

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.benchmark import BenchmarkSlangFunction

PARAM_SIZES = [16, 64, 256, 512]
CALL_SHAPE = (1024, 1024)


def _make_inputs(device: spy.Device, size: int):
    weights = np.random.rand(size).astype(np.float32)
    indices = np.random.randint(0, size, size=CALL_SHAPE).astype(np.uint32)
    indices_tensor = spy.Tensor.from_numpy(device, indices)
    result_tensor = spy.Tensor.empty(device, shape=CALL_SHAPE, dtype=float)
    return weights, indices, indices_tensor, result_tensor


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("size", PARAM_SIZES)
def test_param_array_pick_one(
    device_type: spy.DeviceType, size: int, benchmark_slang_function: BenchmarkSlangFunction
):
    device = helpers.get_device(device_type)
    weights, indices, indices_tensor, result_tensor = _make_inputs(device, size)

    module = spy.Module(device.load_module("test_benchmark_param_array.slang"))
    func = module.require_function(f"pick_one<{size}>")

    benchmark_slang_function(
        device,
        func,
        tid=spy.call_id(),
        weights=weights.tolist(),
        indices=indices_tensor,
        _result=result_tensor,
    )
    assert np.allclose(result_tensor.to_numpy(), weights[indices])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("size", PARAM_SIZES)
def test_param_array_sum_dynamic(
    device_type: spy.DeviceType, size: int, benchmark_slang_function: BenchmarkSlangFunction
):
    device = helpers.get_device(device_type)
    weights, _, indices_tensor, result_tensor = _make_inputs(device, size)

    module = spy.Module(device.load_module("test_benchmark_param_array.slang"))
    func = module.require_function(f"sum_dynamic<{size}>")

    benchmark_slang_function(
        device,
        func,
        tid=spy.call_id(),
        weights=weights.tolist(),
        indices=indices_tensor,
        _result=result_tensor,
    )
    # Every thread sums all N elements (rotated by its start index).
    assert np.allclose(result_tensor.to_numpy(), np.sum(weights), atol=1e-3)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("size", PARAM_SIZES)
def test_param_array_sum_static(
    device_type: spy.DeviceType, size: int, benchmark_slang_function: BenchmarkSlangFunction
):
    device = helpers.get_device(device_type)
    weights, _, _, result_tensor = _make_inputs(device, size)

    module = spy.Module(device.load_module("test_benchmark_param_array.slang"))
    func = module.require_function(f"sum_static<{size}>")

    benchmark_slang_function(
        device,
        func,
        tid=spy.call_id(),
        weights=weights.tolist(),
        _result=result_tensor,
    )
    assert np.allclose(result_tensor.to_numpy(), np.sum(weights), atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
