# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy


def test_format_info_rgb32_float():
    info = spy.get_format_info(spy.Format.rgb32_float)
    assert info.format == spy.Format.rgb32_float
    assert info.bytes_per_block == 12
    assert info.channel_count == 3
    assert info.type == spy.FormatType.float
    assert info.is_depth == False
    assert info.is_stencil == False
    assert info.is_compressed == False
    assert info.block_width == 1
    assert info.block_height == 1
    assert info.channel_bit_count == [32, 32, 32, 0]
    assert info.dxgi_format == 6
    assert info.vk_format == 106
    assert info.is_depth_stencil() == False
    assert info.is_float_format() == True
    assert info.is_integer_format() == False
    assert info.is_normalized_format() == False
    assert info.is_srgb_format() == False
    assert info.get_channels() == spy.FormatChannels.rgb
    assert info.get_channel_bits(spy.FormatChannels.r) == 32
    assert info.get_channel_bits(spy.FormatChannels.g) == 32
    assert info.get_channel_bits(spy.FormatChannels.b) == 32
    assert info.get_channel_bits(spy.FormatChannels.a) == 0
    assert info.get_channel_bits(spy.FormatChannels.rgba) == 96
    assert info.has_equal_channel_bits() == True


def test_format_info_d32_float_s8_uint():
    info = spy.get_format_info(spy.Format.d32_float_s8_uint)
    assert info.format == spy.Format.d32_float_s8_uint
    assert info.bytes_per_block == 8
    assert info.channel_count == 2
    assert info.type == spy.FormatType.float
    assert info.is_depth == True
    assert info.is_stencil == True
    assert info.is_compressed == False
    assert info.block_width == 1
    assert info.block_height == 1
    assert info.channel_bit_count == [32, 8, 0, 0]
    assert info.dxgi_format == 20
    assert info.vk_format == 130
    assert info.is_depth_stencil() == True
    assert info.is_float_format() == True
    assert info.is_integer_format() == False
    assert info.is_normalized_format() == False
    assert info.is_srgb_format() == False


def test_format_info_bc7_unorm_srgb():
    info = spy.get_format_info(spy.Format.bc7_unorm_srgb)
    assert info.format == spy.Format.bc7_unorm_srgb
    assert info.bytes_per_block == 16
    assert info.channel_count == 4
    assert info.type == spy.FormatType.unorm_srgb
    assert info.is_depth == False
    assert info.is_stencil == False
    assert info.is_compressed == True
    assert info.block_width == 4
    assert info.block_height == 4
    assert info.channel_bit_count == [128, 0, 0, 0]
    assert info.dxgi_format == 99
    assert info.vk_format == 146
    assert info.is_depth_stencil() == False
    assert info.is_float_format() == False
    assert info.is_integer_format() == False
    assert info.is_normalized_format() == True
    assert info.is_srgb_format() == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
