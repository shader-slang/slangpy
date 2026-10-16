# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from slangpy import Buffer, BufferUsage, Device, DeviceType, SlangPyError, uint3
from slangpy.types import Tensor
from slangpy.testing import helpers

# https://github.com/shader-slang/slangpy/issues/1232
PUSH_CONSTANT_MODULE = r"""
import "slangpy";
struct PC { uint count; float scale; };
[[vk::push_constant]] uniform PC push;
float unused(float x) { return x * 2.0; }
float read(float x) { return x * push.scale + float(push.count); }
void raw_scale(uint3 dispatchThreadID, RWStructuredBuffer<float> buf, float k) {
    buf[dispatchThreadID.x] = float(dispatchThreadID.x) * k;
}
void raw_read(uint3 dispatchThreadID, RWStructuredBuffer<float> buf) {
    buf[dispatchThreadID.x] = float(dispatchThreadID.x) * push.scale + float(push.count);
}
struct Params { float k; };
ParameterBlock<Params> params;
void raw_block(uint3 dispatchThreadID, RWStructuredBuffer<float> buf) {
    buf[dispatchThreadID.x] = float(dispatchThreadID.x) * params.k;
}
"""

PLAIN_UNIFORM_MODULE = PUSH_CONSTANT_MODULE.replace("[[vk::push_constant]] ", "")

# A push-constant block holding a resource reflects as a "mixed" category.
MIXED_PUSH_CONSTANT_MODULE = r"""
import "slangpy";
struct PC { uint count; Texture2D tex; };
[[vk::push_constant]] uniform PC push;
float unused(float x) { return x * 2.0; }
"""

ERROR = r"push constant 'push' is not supported .* plain 'uniform' .* function\.set\(push=\.\.\.\)"


def make_buffer(device: Device) -> Buffer:
    return device.create_buffer(
        element_count=4,
        struct_size=4,
        usage=BufferUsage.unordered_access | BufferUsage.shader_resource,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_push_constant(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, PUSH_CONSTANT_MODULE)
    x = Tensor.from_numpy(device, np.arange(4, dtype=np.float32))

    if device_type == DeviceType.vulkan:
        # Even a function that never reads 'push' returned garbage on Vulkan.
        with pytest.raises(SlangPyError, match=ERROR):
            module.unused(x, _result="numpy")
        with pytest.raises(SlangPyError, match=ERROR):
            module.read.set(push={"count": 10, "scale": 3.0})(x, _result="numpy")
    else:
        assert np.allclose(module.unused(x, _result="numpy"), [0, 2, 4, 6])
        result = module.read.set(push={"count": 10, "scale": 3.0})(x, _result="numpy")
        assert np.allclose(result, [10, 13, 16, 19])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_mixed_push_constant(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MIXED_PUSH_CONSTANT_MODULE)
    x = Tensor.from_numpy(device, np.arange(4, dtype=np.float32))

    if device_type == DeviceType.vulkan:
        with pytest.raises(SlangPyError, match=ERROR):
            module.unused(x, _result="numpy")
    else:
        assert np.allclose(module.unused(x, _result="numpy"), [0, 2, 4, 6])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_push_constant_raw_dispatch(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, PUSH_CONSTANT_MODULE)

    # A kernel with uniform arguments has its own push constants on Vulkan.
    buf = make_buffer(device)
    if device_type == DeviceType.vulkan:
        with pytest.raises(SlangPyError, match=ERROR):
            module.raw_scale.dispatch(uint3(4, 1, 1), buf=buf, k=2.0)
    else:
        module.raw_scale.dispatch(uint3(4, 1, 1), buf=buf, k=2.0)
        assert np.allclose(buf.to_numpy().view(np.float32), [0, 2, 4, 6])

    # Resource-only kernels are not rejected and give correct results here.
    buf = make_buffer(device)
    module.raw_read.set(push={"count": 10, "scale": 3.0}).dispatch(uint3(4, 1, 1), buf=buf)
    assert np.allclose(buf.to_numpy().view(np.float32), [10, 13, 16, 19])

    buf = make_buffer(device)
    module.raw_block.set(params={"k": 2.0}).dispatch(uint3(4, 1, 1), buf=buf)
    assert np.allclose(buf.to_numpy().view(np.float32), [0, 2, 4, 6])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_plain_uniform(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, PLAIN_UNIFORM_MODULE)
    x = Tensor.from_numpy(device, np.arange(4, dtype=np.float32))

    assert np.allclose(module.unused(x, _result="numpy"), [0, 2, 4, 6])
    result = module.read.set(push={"count": 10, "scale": 3.0})(x, _result="numpy")
    assert np.allclose(result, [10, 13, 16, 19])

    buf = make_buffer(device)
    module.raw_scale.dispatch(uint3(4, 1, 1), buf=buf, k=2.0)
    assert np.allclose(buf.to_numpy().view(np.float32), [0, 2, 4, 6])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
