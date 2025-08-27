# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
from slangpy import Module
from slangpy import DeviceType, float3
from slangpy.experimental.diffbuffer import NDDifferentiableBuffer
from slangpy.testing import helpers


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_modules.slang"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("use_arg", [True, False])
def test_command_buffer(device_type: DeviceType, use_arg: bool):
    m = load_test_module(device_type)
    assert m is not None

    polynomial = m.polynomial.as_func()

    command_encoder = m.device.create_command_encoder()

    a = NDDifferentiableBuffer(m.device, float3, 10, requires_grad=True)
    b = NDDifferentiableBuffer(m.device, float3, 10, requires_grad=True)
    res = NDDifferentiableBuffer(m.device, float3, 10, requires_grad=True)
    assert a.grad is not None
    assert b.grad is not None
    assert res.grad is not None

    a_data = np.random.rand(10, 3).astype(np.float32)
    b_data = np.random.rand(10, 3).astype(np.float32)
    res_data = np.zeros((10, 3), dtype=np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten(), 3)
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten(), 3)
    helpers.write_ndbuffer_from_numpy(res, res_data.flatten(), 3)
    helpers.write_ndbuffer_from_numpy(res.grad, np.ones_like(res_data).flatten(), 3)

    if use_arg:
        polynomial(a, b, _result=res, _append_to=command_encoder)
        polynomial.bwds(a, b, _result=res, _append_to=command_encoder)
    else:
        polynomial.append_to(command_encoder, a, b, _result=res)
        polynomial.bwds.append_to(command_encoder, a, b, _result=res)

    # Nothing should have happened yet if command buffer is not submitted!
    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(-1, 3)
    assert not np.allclose(res_data, a_data * a_data + b_data + 1)

    # Submit the command buffer to execute the operations
    m.device.submit_command_buffer(command_encoder.finish())

    # Now the result should be computed
    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(-1, 3)
    assert np.allclose(res_data, a_data * a_data + b_data + 1)

    a_grad = helpers.read_ndbuffer_from_numpy(a.grad).reshape(-1, 3)
    b_grad = helpers.read_ndbuffer_from_numpy(b.grad).reshape(-1, 3)
    assert np.allclose(a_grad, 2 * a_data)
    assert np.allclose(b_grad, np.ones_like(b_data))

    # Also check nothing dies when calling function directly with a None encoder
    polynomial(a, b, _result=res, _append_to=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
