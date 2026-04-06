# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for DescriptorMarshall (builtin/descriptor.py).

Exercises the functional API path for DescriptorHandle values,
covering __init__, resolve_type, resolve_dimensionality, gen_calldata,
reduce_type, and build_shader_object.
"""

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing.helpers import get_device, create_function_from_module


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
class TestDescriptorMarshallTexture:
    """Test DescriptorMarshall with texture descriptor handles on CUDA."""

    def _skip_if_no_bindless(self, device: spy.Device):
        if not device.has_feature(spy.Feature.bindless):
            pytest.skip("Bindless not supported on this device")

    def test_read_texture_via_descriptor(self, device_type: spy.DeviceType):
        """Pass a texture descriptor handle through the functional API and read a value."""
        device = get_device(device_type)
        self._skip_if_no_bindless(device)

        texture = device.create_texture(
            width=4,
            height=1,
            format=spy.Format.r32_float,
            usage=spy.TextureUsage.shader_resource,
            data=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        )
        view = texture.create_view()
        handle = view.descriptor_handle_ro

        func = create_function_from_module(
            device,
            "read_descriptor",
            """
            float read_descriptor(DescriptorHandle<Texture2D<float>> tex) {
                Texture2D<float> t = tex;
                return t.Load(int3(0, 0, 0));
            }
            """,
        )

        result = func(handle)
        assert result is not None

    def test_sampler_descriptor(self, device_type: spy.DeviceType):
        """Pass a sampler descriptor handle through the functional API."""
        device = get_device(device_type)
        self._skip_if_no_bindless(device)

        sampler = device.create_sampler()
        handle = sampler.descriptor_handle

        func = create_function_from_module(
            device,
            "use_sampler",
            """
            int use_sampler(DescriptorHandle<SamplerState> s) {
                return 42;
            }
            """,
        )

        result = func(handle)
        assert result == 42


class TestDescriptorMarshallErrors:
    """Test DescriptorMarshall error paths."""

    def test_reduce_type_nonzero_dimensions(self):
        """reduce_type raises ValueError when dimensions > 0."""
        device = get_device(spy.DeviceType.cuda)

        if not device.has_feature(spy.Feature.bindless):
            pytest.skip("Bindless not supported on this device")

        texture = device.create_texture(
            width=2,
            height=1,
            format=spy.Format.r32_float,
            usage=spy.TextureUsage.shader_resource,
            data=np.array([1.0, 2.0], dtype=np.float32),
        )
        view = texture.create_view()
        handle = view.descriptor_handle_ro

        from slangpy.builtin.descriptor import DescriptorMarshall

        slang_module = device.load_module_from_source(
            "descriptor_test_reduce",
            'import "slangpy";',
        )
        marshall = DescriptorMarshall(slang_module.layout, handle.type)

        with pytest.raises(ValueError, match="Cannot reduce dimensions"):
            marshall.reduce_type(None, 1)

    def test_init_type_not_found(self):
        """__init__ raises ValueError when DescriptorHandle type not found in layout."""
        device = get_device(spy.DeviceType.cuda)

        slang_module = device.load_module_from_source(
            "descriptor_test_notype",
            "float dummy() { return 0; }",
        )

        with pytest.raises(ValueError, match="Could not find DescriptorHandle"):
            from slangpy.builtin.descriptor import DescriptorMarshall

            DescriptorMarshall(slang_module.layout, spy.DescriptorHandleType.texture)
