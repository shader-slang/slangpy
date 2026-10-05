# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Regression tests for slangpy#1204.

Backward gradients must flow through the array/vector load/store requirements
of IDiffTensor, IWDiffTensor and IRWDiffTensor when called via an interface-typed
or generic parameter, not only via the concrete tensor type.
"""

import pytest
import numpy as np
from pathlib import Path

from slangpy import DeviceType, Tensor
from slangpy.core.module import Module
from slangpy.testing import helpers

N = 4

FUNCTIONS = [
    "iface_load_array",
    "iface_load_vector",
    "generic_load",
    "rw_iface_load_array",
    "rw_iface_load_vector",
    "iface_store_array",
    "iface_store_vector",
    "generic_store",
    "rw_iface_store_array",
    "rw_iface_store_vector",
]


def load_module(device_type: DeviceType) -> Module:
    device = helpers.get_device(device_type)
    return Module.load_from_file(
        device,
        str(Path(__file__).parent / "test_difftensor_interface_bwd.slang"),
    )


@pytest.mark.parametrize("func_name", FUNCTIONS)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_interface_load_store_bwd(device_type: DeviceType, func_name: str) -> None:
    module = load_module(device_type)
    device = module.device
    func = module.find_function(func_name)
    assert func is not None

    x_np = np.arange(1, N + 1, dtype=np.float32)
    x = Tensor.from_numpy(device, x_np).with_grads(zero=True)
    out = Tensor.zeros(device, shape=(N,), dtype=float).with_grads(zero=True)
    idx = Tensor.from_numpy(device, np.arange(N, dtype=np.int32))

    func(x, out, idx)
    assert np.allclose(out.to_numpy(), x_np * 3.0)

    assert out.grad_in is not None
    out.grad_in.storage.copy_from_numpy(np.ones(N, dtype=np.float32))
    func.bwds(x, out, idx)

    assert x.grad_out is not None
    x_grad = x.grad_out.to_numpy()
    assert np.allclose(x_grad, 3.0), f"{func_name}: expected grad 3.0, got {x_grad}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
