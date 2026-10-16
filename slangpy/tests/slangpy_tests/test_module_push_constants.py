# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from slangpy import DeviceType
from slangpy.types import Tensor
from slangpy.testing import helpers

# https://github.com/shader-slang/slangpy/issues/1232
PUSH_CONSTANT_MODULE = r"""
import "slangpy";
struct PC { uint count; float scale; };
[[vk::push_constant]] uniform PC push;
float unused(float x) { return x * 2.0; }
float read(float x) { return x * push.scale + float(push.count); }
"""

PLAIN_UNIFORM_MODULE = PUSH_CONSTANT_MODULE.replace("[[vk::push_constant]] ", "")


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_push_constant(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, PUSH_CONSTANT_MODULE)
    x = Tensor.from_numpy(device, np.arange(4, dtype=np.float32))

    if device_type == DeviceType.vulkan:
        # Even a function that never reads 'push' returned garbage on Vulkan.
        with pytest.raises(ValueError, match=r"push constant 'push'"):
            module.unused(x, _result="numpy")
        with pytest.raises(ValueError, match=r"push constant 'push'"):
            module.read.set(push={"count": 10, "scale": 3.0})(x, _result="numpy")
    else:
        assert np.allclose(module.unused(x, _result="numpy"), [0, 2, 4, 6])
        result = module.read.set(push={"count": 10, "scale": 3.0})(x, _result="numpy")
        assert np.allclose(result, [10, 13, 16, 19])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_plain_uniform(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, PLAIN_UNIFORM_MODULE)
    x = Tensor.from_numpy(device, np.arange(4, dtype=np.float32))

    assert np.allclose(module.unused(x, _result="numpy"), [0, 2, 4, 6])
    result = module.read.set(push={"count": 10, "scale": 3.0})(x, _result="numpy")
    assert np.allclose(result, [10, 13, 16, 19])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
