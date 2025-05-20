# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import pytest

from . import helpers
from slangpy import Module
from slangpy import DeviceType, float3
from slangpy.experimental.diffbuffer import NDDifferentiableBuffer


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_modules.slang"))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_command_buffer(device_type: DeviceType):
    m = load_test_module(device_type)
    assert m is not None

    polynomial = m.polynomial.as_func()

    command_encoder = m.device.create_command_encoder()

    a = NDDifferentiableBuffer(m.device, float3, 10, requires_grad=True)
    b = NDDifferentiableBuffer(m.device, float3, 10, requires_grad=True)
    res = NDDifferentiableBuffer(m.device, float3, 10, requires_grad=True)

    a_data = np.random.rand(10, 3).astype(np.float32)
    b_data = np.random.rand(10, 3).astype(np.float32)
    res_data = np.zeros((10, 3), dtype=np.float32)

    helpers.write_ndbuffer_from_numpy(a, a_data.flatten(), 3)
    helpers.write_ndbuffer_from_numpy(b, b_data.flatten(), 3)
    helpers.write_ndbuffer_from_numpy(res, res_data.flatten(), 3)
    helpers.write_ndbuffer_from_numpy(res.grad, np.ones_like(res_data).flatten(), 3)

    polynomial.append_to(command_encoder, a, b, _result=res)
    polynomial.bwds.append_to(command_encoder, a, b, _result=res)

    m.device.submit_command_buffer(command_encoder.finish())

    res_data = helpers.read_ndbuffer_from_numpy(res).reshape(-1, 3)
    assert np.allclose(res_data, a_data * a_data + b_data + 1)

    a_grad = helpers.read_ndbuffer_from_numpy(a.grad).reshape(-1, 3)
    b_grad = helpers.read_ndbuffer_from_numpy(b.grad).reshape(-1, 3)

    assert np.allclose(a_grad, 2 * a_data)
    assert np.allclose(b_grad, np.ones_like(b_data))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
