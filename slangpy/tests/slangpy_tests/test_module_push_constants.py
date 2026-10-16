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
RWStructuredBuffer<float> gbuf;
void raw_global(uint3 dispatchThreadID) {
    gbuf[dispatchThreadID.x] = float(dispatchThreadID.x) * push.scale + float(push.count);
}
"""

# A global ParameterBlock next to a module push constant gets the wrong descriptor set on Vulkan.
PARAMETER_BLOCK_MODULE = r"""
import "slangpy";
struct PC { uint count; float scale; };
[[vk::push_constant]] uniform PC push;
struct Params { float k; RWStructuredBuffer<float> out; };
ParameterBlock<Params> params;
void raw_block(uint3 dispatchThreadID) {
    params.out[dispatchThreadID.x] = float(dispatchThreadID.x) * params.k;
}
"""

# A ParameterBlock entry-point parameter is shifted the same way.
ENTRY_POINT_BLOCK_MODULE = r"""
import "slangpy";
struct PC { uint count; float scale; };
[[vk::push_constant]] uniform PC push;
struct Params { float k; RWStructuredBuffer<float> out; };
void raw_block(uint3 dispatchThreadID, ParameterBlock<Params> params) {
    params.out[dispatchThreadID.x] = float(dispatchThreadID.x) * params.k;
}
"""

# The push constant can come from a module that is only imported, not composed or passed via link=.
IMPORTED_PUSH_CONSTANT_MODULE = r"""
struct PC { uint count; float scale; };
[[vk::push_constant]] uniform PC push;
public float push_value() { return push.scale + float(push.count); }
"""

IMPORTING_MODULE = r"""
import "slangpy";
import IMPORTED;
float read(float x) { return x + push_value(); }
void raw_read(uint3 dispatchThreadID, RWStructuredBuffer<float> buf, float k) {
    buf[dispatchThreadID.x] = k + push_value();
}
"""

PLAIN_UNIFORM_MODULE = PUSH_CONSTANT_MODULE.replace("[[vk::push_constant]] ", "")

CONSTANT_BUFFER_MODULE = r"""
import "slangpy";
struct PC { uint count; float scale; };
ConstantBuffer<PC> cb;
float unused(float x) { return x * 2.0; }
float read(float x) { return x * cb.scale + float(cb.count); }
"""

# A push-constant block holding a resource reflects as a "mixed" category.
MIXED_PUSH_CONSTANT_MODULE = r"""
import "slangpy";
struct PC { uint count; Texture2D tex; };
[[vk::push_constant]] uniform PC push;
float unused(float x) { return x * 2.0; }
"""

ERROR = r"push constant 'push' is not supported .* Remove \[\[vk::push_constant\]\] .* function\.set\(push=\.\.\.\)"


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

    # Uniform kernel arguments are a second push-constant block on Vulkan.
    buf = make_buffer(device)
    if device_type == DeviceType.vulkan:
        with pytest.raises(SlangPyError, match=ERROR):
            module.raw_scale.dispatch(uint3(4, 1, 1), buf=buf, k=2.0)
    else:
        module.raw_scale.dispatch(uint3(4, 1, 1), buf=buf, k=2.0)
        assert np.allclose(buf.to_numpy().view(np.float32), [0, 2, 4, 6])

    # Kernels whose arguments are all resources read the module push constant correctly.
    buf = make_buffer(device)
    module.raw_read.set(push={"count": 10, "scale": 3.0}).dispatch(uint3(4, 1, 1), buf=buf)
    assert np.allclose(buf.to_numpy().view(np.float32), [10, 13, 16, 19])

    buf = make_buffer(device)
    module.raw_global.set(push={"count": 10, "scale": 3.0}, gbuf=buf).dispatch(uint3(4, 1, 1))
    assert np.allclose(buf.to_numpy().view(np.float32), [10, 13, 16, 19])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_push_constant_raw_dispatch_parameter_block(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, PARAMETER_BLOCK_MODULE)

    buf = make_buffer(device)
    dispatch = module.raw_block.set(params={"k": 2.0, "out": buf})
    if device_type == DeviceType.vulkan:
        with pytest.raises(SlangPyError, match=ERROR):
            dispatch.dispatch(uint3(4, 1, 1))
    else:
        dispatch.dispatch(uint3(4, 1, 1))
        assert np.allclose(buf.to_numpy().view(np.float32), [0, 2, 4, 6])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_push_constant_raw_dispatch_entry_point_block(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, ENTRY_POINT_BLOCK_MODULE)

    buf = make_buffer(device)
    params = {"k": 2.0, "out": buf}
    if device_type == DeviceType.vulkan:
        with pytest.raises(SlangPyError, match=ERROR):
            module.raw_block.dispatch(uint3(4, 1, 1), params=params)
    else:
        module.raw_block.dispatch(uint3(4, 1, 1), params=params)
        assert np.allclose(buf.to_numpy().view(np.float32), [0, 2, 4, 6])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_imported_module_push_constant(device_type: DeviceType):
    device = helpers.get_device(device_type)
    imported = f"imported_push_constant_{device_type.name}"
    device.load_module_from_source(imported, IMPORTED_PUSH_CONSTANT_MODULE)
    module = helpers.create_module(device, IMPORTING_MODULE.replace("IMPORTED", imported))
    x = Tensor.from_numpy(device, np.arange(4, dtype=np.float32))
    push = {"count": 10, "scale": 3.0}

    buf = make_buffer(device)
    if device_type == DeviceType.vulkan:
        with pytest.raises(SlangPyError, match=ERROR):
            module.read.set(push=push)(x, _result="numpy")
        with pytest.raises(SlangPyError, match=ERROR):
            module.raw_read.set(push=push).dispatch(uint3(4, 1, 1), buf=buf, k=2.0)
    else:
        assert np.allclose(module.read.set(push=push)(x, _result="numpy"), [13, 14, 15, 16])
        module.raw_read.set(push=push).dispatch(uint3(4, 1, 1), buf=buf, k=2.0)
        assert np.allclose(buf.to_numpy().view(np.float32), [15, 15, 15, 15])


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

    buf = make_buffer(device)
    module.raw_read.set(push={"count": 10, "scale": 3.0}).dispatch(uint3(4, 1, 1), buf=buf)
    assert np.allclose(buf.to_numpy().view(np.float32), [10, 13, 16, 19])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_module_constant_buffer(device_type: DeviceType):
    device = helpers.get_device(device_type)
    module = helpers.create_module(device, CONSTANT_BUFFER_MODULE)
    x = Tensor.from_numpy(device, np.arange(4, dtype=np.float32))

    assert np.allclose(module.unused(x, _result="numpy"), [0, 2, 4, 6])
    result = module.read.set(cb={"count": 10, "scale": 3.0})(x, _result="numpy")
    assert np.allclose(result, [10, 13, 16, 19])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
