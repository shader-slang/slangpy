// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/formats.h"
#include "sgl/device/native_formats.h"
#include "sgl/device/helpers.h"

using namespace sgl;

TEST_SUITE_BEGIN("formats");

TEST_CASE("host_type_to_format")
{
    CHECK_EQ(host_type_to_format<int8_t>(), Format::r8_sint);
    CHECK_EQ(host_type_to_format<uint8_t>(), Format::r8_uint);
    CHECK_EQ(host_type_to_format<int16_t>(), Format::r16_sint);
    CHECK_EQ(host_type_to_format<uint16_t>(), Format::r16_uint);
    CHECK_EQ(host_type_to_format<int>(), Format::r32_sint);
    CHECK_EQ(host_type_to_format<int2>(), Format::rg32_sint);
    CHECK_EQ(host_type_to_format<int4>(), Format::rgba32_sint);
    CHECK_EQ(host_type_to_format<uint>(), Format::r32_uint);
    CHECK_EQ(host_type_to_format<uint2>(), Format::rg32_uint);
    CHECK_EQ(host_type_to_format<uint4>(), Format::rgba32_uint);
    CHECK_EQ(host_type_to_format<float16_t>(), Format::r16_float);
    CHECK_EQ(host_type_to_format<float16_t2>(), Format::rg16_float);
    CHECK_EQ(host_type_to_format<float16_t4>(), Format::rgba16_float);
    CHECK_EQ(host_type_to_format<float>(), Format::r32_float);
    CHECK_EQ(host_type_to_format<float2>(), Format::rg32_float);
    CHECK_EQ(host_type_to_format<float3>(), Format::rgb32_float);
    CHECK_EQ(host_type_to_format<float4>(), Format::rgba32_float);
}

TEST_CASE("validate_format_infos")
{
    auto format_type_to_slang_type = [](FormatType type) -> SlangScalarType
    {
        switch (type) {
        case FormatType::unknown:
            return SLANG_SCALAR_TYPE_NONE;
        case FormatType::float_:
            return SLANG_SCALAR_TYPE_FLOAT32;
        case FormatType::unorm:
            return SLANG_SCALAR_TYPE_FLOAT32;
        case FormatType::unorm_srgb:
            return SLANG_SCALAR_TYPE_FLOAT32;
        case FormatType::snorm:
            return SLANG_SCALAR_TYPE_FLOAT32;
        case FormatType::uint:
            return SLANG_SCALAR_TYPE_UINT32;
        case FormatType::sint:
            return SLANG_SCALAR_TYPE_INT32;
        default:
            return SLANG_SCALAR_TYPE_NONE;
        }
    };

    for (uint32_t i = 0; i < uint32_t(Format::count); ++i) {
        const FormatInfo& info = get_format_info(Format(i));
        const rhi::FormatInfo& rhi_info = rhi::getFormatInfo(rhi::Format(i));

        CAPTURE(info.name);
        // CHECK_EQ(format_type_to_slang_type(info.type), rhi_info.channelType);
        CHECK_EQ(info.bytes_per_block, rhi_info.blockSizeInBytes);
        CHECK_EQ(info.channel_count, rhi_info.channelCount);
        CHECK_EQ(info.block_width * info.block_height, rhi_info.pixelsPerBlock);
        CHECK_EQ(info.block_width, rhi_info.blockWidth);
        CHECK_EQ(info.block_height, rhi_info.blockHeight);
    }
}

TEST_CASE("rgba32_float")
{
    const FormatInfo& info = get_format_info(Format::rgba32_float);
    CHECK_EQ(info.format, Format::rgba32_float);
    CHECK_EQ(info.name, "rgba32_float");
    CHECK_EQ(info.bytes_per_block, 16);
    CHECK_EQ(info.channel_count, 4);
    CHECK_EQ(info.type, FormatType::float_);
    CHECK_EQ(info.is_depth, false);
    CHECK_EQ(info.is_stencil, false);
    CHECK_EQ(info.is_compressed, false);
    CHECK_EQ(info.block_width, 1);
    CHECK_EQ(info.block_height, 1);
    CHECK_EQ(info.channel_bit_count[0], 32);
    CHECK_EQ(info.channel_bit_count[1], 32);
    CHECK_EQ(info.channel_bit_count[2], 32);
    CHECK_EQ(info.channel_bit_count[3], 32);
    CHECK_EQ(info.dxgi_format, DXGI_FORMAT_R32G32B32A32_FLOAT);
    CHECK_EQ(info.vk_format, VK_FORMAT_R32G32B32A32_SFLOAT);

    CHECK_EQ(info.is_depth_stencil(), false);
    CHECK_EQ(info.is_float_format(), true);
    CHECK_EQ(info.is_integer_format(), false);
    CHECK_EQ(info.is_normalized_format(), false);
    CHECK_EQ(info.is_srgb_format(), false);
    CHECK_EQ(info.get_channels(), FormatChannels::rgba);
    CHECK_EQ(info.get_channel_bits(FormatChannels::r), 32);
    CHECK_EQ(info.get_channel_bits(FormatChannels::g), 32);
    CHECK_EQ(info.get_channel_bits(FormatChannels::b), 32);
    CHECK_EQ(info.get_channel_bits(FormatChannels::a), 32);
    CHECK_EQ(info.get_channel_bits(FormatChannels::rgba), 128);
}

TEST_CASE("rg16_uint")
{
    const FormatInfo& info = get_format_info(Format::rg16_uint);
    CHECK_EQ(info.format, Format::rg16_uint);
    CHECK_EQ(info.name, "rg16_uint");
    CHECK_EQ(info.bytes_per_block, 4);
    CHECK_EQ(info.channel_count, 2);
    CHECK_EQ(info.type, FormatType::uint);
    CHECK_EQ(info.is_depth, false);
    CHECK_EQ(info.is_stencil, false);
    CHECK_EQ(info.is_compressed, false);
    CHECK_EQ(info.block_width, 1);
    CHECK_EQ(info.block_height, 1);
    CHECK_EQ(info.channel_bit_count[0], 16);
    CHECK_EQ(info.channel_bit_count[1], 16);
    CHECK_EQ(info.channel_bit_count[2], 0);
    CHECK_EQ(info.channel_bit_count[3], 0);
    CHECK_EQ(info.dxgi_format, DXGI_FORMAT_R16G16_UINT);
    CHECK_EQ(info.vk_format, VK_FORMAT_R16G16_UINT);

    CHECK_EQ(info.is_depth_stencil(), false);
    CHECK_EQ(info.is_float_format(), false);
    CHECK_EQ(info.is_integer_format(), true);
    CHECK_EQ(info.is_normalized_format(), false);
    CHECK_EQ(info.is_srgb_format(), false);
    CHECK_EQ(info.get_channels(), FormatChannels::rg);
    CHECK_EQ(info.get_channel_bits(FormatChannels::r), 16);
    CHECK_EQ(info.get_channel_bits(FormatChannels::g), 16);
    CHECK_EQ(info.get_channel_bits(FormatChannels::b), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::a), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::rgba), 32);
}

TEST_CASE("r8_snorm")
{
    const FormatInfo& info = get_format_info(Format::r8_snorm);
    CHECK_EQ(info.format, Format::r8_snorm);
    CHECK_EQ(info.name, "r8_snorm");
    CHECK_EQ(info.bytes_per_block, 1);
    CHECK_EQ(info.channel_count, 1);
    CHECK_EQ(info.type, FormatType::snorm);
    CHECK_EQ(info.is_depth, false);
    CHECK_EQ(info.is_stencil, false);
    CHECK_EQ(info.is_compressed, false);
    CHECK_EQ(info.block_width, 1);
    CHECK_EQ(info.block_height, 1);
    CHECK_EQ(info.channel_bit_count[0], 8);
    CHECK_EQ(info.channel_bit_count[1], 0);
    CHECK_EQ(info.channel_bit_count[2], 0);
    CHECK_EQ(info.channel_bit_count[3], 0);
    CHECK_EQ(info.dxgi_format, DXGI_FORMAT_R8_SNORM);
    CHECK_EQ(info.vk_format, VK_FORMAT_R8_SNORM);

    CHECK_EQ(info.is_depth_stencil(), false);
    CHECK_EQ(info.is_float_format(), false);
    CHECK_EQ(info.is_integer_format(), false);
    CHECK_EQ(info.is_normalized_format(), true);
    CHECK_EQ(info.is_srgb_format(), false);
    CHECK_EQ(info.get_channels(), FormatChannels::r);
    CHECK_EQ(info.get_channel_bits(FormatChannels::r), 8);
    CHECK_EQ(info.get_channel_bits(FormatChannels::g), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::b), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::a), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::rgba), 8);
}

TEST_CASE("d32_float_s8_uint")
{
    const FormatInfo& info = get_format_info(Format::d32_float_s8_uint);
    CHECK_EQ(info.format, Format::d32_float_s8_uint);
    CHECK_EQ(info.name, "d32_float_s8_uint");
    CHECK_EQ(info.bytes_per_block, 8);
    CHECK_EQ(info.channel_count, 2);
    CHECK_EQ(info.type, FormatType::float_);
    CHECK_EQ(info.is_depth, true);
    CHECK_EQ(info.is_stencil, true);
    CHECK_EQ(info.is_compressed, false);
    CHECK_EQ(info.block_width, 1);
    CHECK_EQ(info.block_height, 1);
    CHECK_EQ(info.channel_bit_count[0], 32);
    CHECK_EQ(info.channel_bit_count[1], 8);
    CHECK_EQ(info.channel_bit_count[2], 0);
    CHECK_EQ(info.channel_bit_count[3], 0);
    CHECK_EQ(info.dxgi_format, DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
    CHECK_EQ(info.vk_format, VK_FORMAT_D32_SFLOAT_S8_UINT);

    CHECK_EQ(info.is_depth_stencil(), true);
    CHECK_EQ(info.is_float_format(), true);
    CHECK_EQ(info.is_integer_format(), false);
    CHECK_EQ(info.is_normalized_format(), false);
    CHECK_EQ(info.is_srgb_format(), false);
    CHECK_EQ(info.get_channels(), FormatChannels::rg);
    CHECK_EQ(info.get_channel_bits(FormatChannels::r), 32);
    CHECK_EQ(info.get_channel_bits(FormatChannels::g), 8);
    CHECK_EQ(info.get_channel_bits(FormatChannels::b), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::a), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::rgba), 40);
}

TEST_CASE("rgb10a2_uint")
{
    const FormatInfo& info = get_format_info(Format::rgb10a2_uint);
    CHECK_EQ(info.format, Format::rgb10a2_uint);
    CHECK_EQ(info.name, "rgb10a2_uint");
    CHECK_EQ(info.bytes_per_block, 4);
    CHECK_EQ(info.channel_count, 4);
    CHECK_EQ(info.type, FormatType::uint);
    CHECK_EQ(info.is_depth, false);
    CHECK_EQ(info.is_stencil, false);
    CHECK_EQ(info.is_compressed, false);
    CHECK_EQ(info.block_width, 1);
    CHECK_EQ(info.block_height, 1);
    CHECK_EQ(info.channel_bit_count[0], 10);
    CHECK_EQ(info.channel_bit_count[1], 10);
    CHECK_EQ(info.channel_bit_count[2], 10);
    CHECK_EQ(info.channel_bit_count[3], 2);
    CHECK_EQ(info.dxgi_format, DXGI_FORMAT_R10G10B10A2_UINT);
    CHECK_EQ(info.vk_format, VK_FORMAT_A2R10G10B10_UINT_PACK32);

    CHECK_EQ(info.is_depth_stencil(), false);
    CHECK_EQ(info.is_float_format(), false);
    CHECK_EQ(info.is_integer_format(), true);
    CHECK_EQ(info.is_normalized_format(), false);
    CHECK_EQ(info.is_srgb_format(), false);
    CHECK_EQ(info.get_channels(), FormatChannels::rgba);
    CHECK_EQ(info.get_channel_bits(FormatChannels::r), 10);
    CHECK_EQ(info.get_channel_bits(FormatChannels::g), 10);
    CHECK_EQ(info.get_channel_bits(FormatChannels::b), 10);
    CHECK_EQ(info.get_channel_bits(FormatChannels::a), 2);
    CHECK_EQ(info.get_channel_bits(FormatChannels::rgba), 32);
}

TEST_CASE("bc7_unorm_srgb")
{
    const FormatInfo& info = get_format_info(Format::bc7_unorm_srgb);
    CHECK_EQ(info.format, Format::bc7_unorm_srgb);
    CHECK_EQ(info.name, "bc7_unorm_srgb");
    CHECK_EQ(info.bytes_per_block, 16);
    CHECK_EQ(info.channel_count, 4);
    CHECK_EQ(info.type, FormatType::unorm_srgb);
    CHECK_EQ(info.is_depth, false);
    CHECK_EQ(info.is_stencil, false);
    CHECK_EQ(info.is_compressed, true);
    CHECK_EQ(info.block_width, 4);
    CHECK_EQ(info.block_height, 4);
    CHECK_EQ(info.channel_bit_count[0], 128);
    CHECK_EQ(info.channel_bit_count[1], 0);
    CHECK_EQ(info.channel_bit_count[2], 0);
    CHECK_EQ(info.channel_bit_count[3], 0);
    CHECK_EQ(info.dxgi_format, DXGI_FORMAT_BC7_UNORM_SRGB);
    CHECK_EQ(info.vk_format, VK_FORMAT_BC7_SRGB_BLOCK);

    CHECK_EQ(info.is_depth_stencil(), false);
    CHECK_EQ(info.is_float_format(), false);
    CHECK_EQ(info.is_integer_format(), false);
    CHECK_EQ(info.is_normalized_format(), true);
    CHECK_EQ(info.is_srgb_format(), true);
    CHECK_EQ(info.get_channels(), FormatChannels::rgba);
    CHECK_EQ(info.get_channel_bits(FormatChannels::r), 128);
    CHECK_EQ(info.get_channel_bits(FormatChannels::g), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::b), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::a), 0);
    CHECK_EQ(info.get_channel_bits(FormatChannels::rgba), 128);
}

TEST_SUITE_END();
