// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/enum.h"
#include "sgl/math/vector_types.h"

#include <slang-rhi.h>

#include <array>

namespace sgl {

/// Resource formats.
enum class Format : uint32_t {
    undefined = static_cast<uint32_t>(rhi::Format::Undefined),

    r8_uint = static_cast<uint32_t>(rhi::Format::R8Uint),
    r8_sint = static_cast<uint32_t>(rhi::Format::R8Sint),
    r8_unorm = static_cast<uint32_t>(rhi::Format::R8Unorm),
    r8_snorm = static_cast<uint32_t>(rhi::Format::R8Snorm),

    rg8_uint = static_cast<uint32_t>(rhi::Format::RG8Uint),
    rg8_sint = static_cast<uint32_t>(rhi::Format::RG8Sint),
    rg8_unorm = static_cast<uint32_t>(rhi::Format::RG8Unorm),
    rg8_snorm = static_cast<uint32_t>(rhi::Format::RG8Snorm),

    rgba8_uint = static_cast<uint32_t>(rhi::Format::RGBA8Uint),
    rgba8_sint = static_cast<uint32_t>(rhi::Format::RGBA8Sint),
    rgba8_unorm = static_cast<uint32_t>(rhi::Format::RGBA8Unorm),
    rgba8_unorm_srgb = static_cast<uint32_t>(rhi::Format::RGBA8UnormSrgb),
    rgba8_snorm = static_cast<uint32_t>(rhi::Format::RGBA8Snorm),

    bgra8_unorm = static_cast<uint32_t>(rhi::Format::BGRA8Unorm),
    bgra8_unorm_srgb = static_cast<uint32_t>(rhi::Format::BGRA8UnormSrgb),
    bgrx8_unorm = static_cast<uint32_t>(rhi::Format::BGRX8Unorm),
    bgrx8_unorm_srgb = static_cast<uint32_t>(rhi::Format::BGRX8UnormSrgb),

    r16_uint = static_cast<uint32_t>(rhi::Format::R16Uint),
    r16_sint = static_cast<uint32_t>(rhi::Format::R16Sint),
    r16_unorm = static_cast<uint32_t>(rhi::Format::R16Unorm),
    r16_snorm = static_cast<uint32_t>(rhi::Format::R16Snorm),
    r16_float = static_cast<uint32_t>(rhi::Format::R16Float),

    rg16_uint = static_cast<uint32_t>(rhi::Format::RG16Uint),
    rg16_sint = static_cast<uint32_t>(rhi::Format::RG16Sint),
    rg16_unorm = static_cast<uint32_t>(rhi::Format::RG16Unorm),
    rg16_snorm = static_cast<uint32_t>(rhi::Format::RG16Snorm),
    rg16_float = static_cast<uint32_t>(rhi::Format::RG16Float),

    rgba16_uint = static_cast<uint32_t>(rhi::Format::RGBA16Uint),
    rgba16_sint = static_cast<uint32_t>(rhi::Format::RGBA16Sint),
    rgba16_unorm = static_cast<uint32_t>(rhi::Format::RGBA16Unorm),
    rgba16_snorm = static_cast<uint32_t>(rhi::Format::RGBA16Snorm),
    rgba16_float = static_cast<uint32_t>(rhi::Format::RGBA16Float),

    r32_uint = static_cast<uint32_t>(rhi::Format::R32Uint),
    r32_sint = static_cast<uint32_t>(rhi::Format::R32Sint),
    r32_float = static_cast<uint32_t>(rhi::Format::R32Float),

    rg32_uint = static_cast<uint32_t>(rhi::Format::RG32Uint),
    rg32_sint = static_cast<uint32_t>(rhi::Format::RG32Sint),
    rg32_float = static_cast<uint32_t>(rhi::Format::RG32Float),

    rgb32_uint = static_cast<uint32_t>(rhi::Format::RGB32Uint),
    rgb32_sint = static_cast<uint32_t>(rhi::Format::RGB32Sint),
    rgb32_float = static_cast<uint32_t>(rhi::Format::RGB32Float),

    rgba32_uint = static_cast<uint32_t>(rhi::Format::RGBA32Uint),
    rgba32_sint = static_cast<uint32_t>(rhi::Format::RGBA32Sint),
    rgba32_float = static_cast<uint32_t>(rhi::Format::RGBA32Float),

    r64_uint = static_cast<uint32_t>(rhi::Format::R64Uint),
    r64_sint = static_cast<uint32_t>(rhi::Format::R64Sint),

    bgra4_unorm = static_cast<uint32_t>(rhi::Format::BGRA4Unorm),
    b5g6r5_unorm = static_cast<uint32_t>(rhi::Format::B5G6R5Unorm),
    bgr5a1_unorm = static_cast<uint32_t>(rhi::Format::BGR5A1Unorm),

    rgb9e5_ufloat = static_cast<uint32_t>(rhi::Format::RGB9E5Ufloat),
    rgb10a2_uint = static_cast<uint32_t>(rhi::Format::RGB10A2Uint),
    rgb10a2_unorm = static_cast<uint32_t>(rhi::Format::RGB10A2Unorm),
    r11g11b10_float = static_cast<uint32_t>(rhi::Format::R11G11B10Float),

    d32_float = static_cast<uint32_t>(rhi::Format::D32Float),
    d16_unorm = static_cast<uint32_t>(rhi::Format::D16Unorm),
    d32_float_s8_uint = static_cast<uint32_t>(rhi::Format::D32FloatS8Uint),

    bc1_unorm = static_cast<uint32_t>(rhi::Format::BC1Unorm),
    bc1_unorm_srgb = static_cast<uint32_t>(rhi::Format::BC1UnormSrgb),
    bc2_unorm = static_cast<uint32_t>(rhi::Format::BC2Unorm),
    bc2_unorm_srgb = static_cast<uint32_t>(rhi::Format::BC2UnormSrgb),
    bc3_unorm = static_cast<uint32_t>(rhi::Format::BC3Unorm),
    bc3_unorm_srgb = static_cast<uint32_t>(rhi::Format::BC3UnormSrgb),
    bc4_unorm = static_cast<uint32_t>(rhi::Format::BC4Unorm),
    bc4_snorm = static_cast<uint32_t>(rhi::Format::BC4Snorm),
    bc5_unorm = static_cast<uint32_t>(rhi::Format::BC5Unorm),
    bc5_snorm = static_cast<uint32_t>(rhi::Format::BC5Snorm),
    bc6h_ufloat = static_cast<uint32_t>(rhi::Format::BC6HUfloat),
    bc6h_sfloat = static_cast<uint32_t>(rhi::Format::BC6HSfloat),
    bc7_unorm = static_cast<uint32_t>(rhi::Format::BC7Unorm),
    bc7_unorm_srgb = static_cast<uint32_t>(rhi::Format::BC7UnormSrgb),

    count,
};

SGL_ENUM_INFO(
    Format,
    {
        {Format::undefined, "undefined"},

        {Format::r8_uint, "r8_uint"},
        {Format::r8_sint, "r8_sint"},
        {Format::r8_unorm, "r8_unorm"},
        {Format::r8_snorm, "r8_snorm"},

        {Format::rg8_uint, "rg8_uint"},
        {Format::rg8_sint, "rg8_sint"},
        {Format::rg8_unorm, "rg8_unorm"},
        {Format::rg8_snorm, "rg8_snorm"},

        {Format::rgba8_uint, "rgba8_uint"},
        {Format::rgba8_sint, "rgba8_sint"},
        {Format::rgba8_unorm, "rgba8_unorm"},
        {Format::rgba8_unorm_srgb, "rgba8_unorm_srgb"},
        {Format::rgba8_snorm, "rgba8_snorm"},

        {Format::bgra8_unorm, "bgra8_unorm"},
        {Format::bgra8_unorm_srgb, "bgra8_unorm_srgb"},
        {Format::bgrx8_unorm, "bgrx8_unorm"},
        {Format::bgrx8_unorm_srgb, "bgrx8_unorm_srgb"},

        {Format::r16_uint, "r16_uint"},
        {Format::r16_sint, "r16_sint"},
        {Format::r16_unorm, "r16_unorm"},
        {Format::r16_snorm, "r16_snorm"},
        {Format::r16_float, "r16_float"},

        {Format::rg16_uint, "rg16_uint"},
        {Format::rg16_sint, "rg16_sint"},
        {Format::rg16_unorm, "rg16_unorm"},
        {Format::rg16_snorm, "rg16_snorm"},
        {Format::rg16_float, "rg16_float"},

        {Format::rgba16_uint, "rgba16_uint"},
        {Format::rgba16_sint, "rgba16_sint"},
        {Format::rgba16_unorm, "rgba16_unorm"},
        {Format::rgba16_snorm, "rgba16_snorm"},
        {Format::rgba16_float, "rgba16_float"},

        {Format::r32_uint, "r32_uint"},
        {Format::r32_sint, "r32_sint"},
        {Format::r32_float, "r32_float"},

        {Format::rg32_uint, "rg32_uint"},
        {Format::rg32_sint, "rg32_sint"},
        {Format::rg32_float, "rg32_float"},

        {Format::rgb32_uint, "rgb32_uint"},
        {Format::rgb32_sint, "rgb32_sint"},
        {Format::rgb32_float, "rgb32_float"},

        {Format::rgba32_uint, "rgba32_uint"},
        {Format::rgba32_sint, "rgba32_sint"},
        {Format::rgba32_float, "rgba32_float"},

        {Format::r64_uint, "r64_uint"},
        {Format::r64_sint, "r64_sint"},

        {Format::bgra4_unorm, "bgra4_unorm"},
        {Format::b5g6r5_unorm, "b5g6r5_unorm"},
        {Format::bgr5a1_unorm, "bgr5a1_unorm"},

        {Format::rgb9e5_ufloat, "rgb9e5_ufloat"},
        {Format::rgb10a2_uint, "rgb10a2_uint"},
        {Format::rgb10a2_unorm, "rgb10a2_unorm"},
        {Format::r11g11b10_float, "r11g11b10_float"},

        {Format::d32_float, "d32_float"},
        {Format::d16_unorm, "d16_unorm"},
        {Format::d32_float_s8_uint, "d32_float_s8_uint"},

        {Format::bc1_unorm, "bc1_unorm"},
        {Format::bc1_unorm_srgb, "bc1_unorm_srgb"},
        {Format::bc2_unorm, "bc2_unorm"},
        {Format::bc2_unorm_srgb, "bc2_unorm_srgb"},
        {Format::bc3_unorm, "bc3_unorm"},
        {Format::bc3_unorm_srgb, "bc3_unorm_srgb"},
        {Format::bc4_unorm, "bc4_unorm"},
        {Format::bc4_snorm, "bc4_snorm"},
        {Format::bc5_unorm, "bc5_unorm"},
        {Format::bc5_snorm, "bc5_snorm"},
        {Format::bc6h_ufloat, "bc6h_ufloat"},
        {Format::bc6h_sfloat, "bc6h_sfloat"},
        {Format::bc7_unorm, "bc7_unorm"},
        {Format::bc7_unorm_srgb, "bc7_unorm_srgb"},

    }
);
SGL_ENUM_REGISTER(Format);
static_assert(EnumInfo<Format>::items.size() == static_cast<size_t>(Format::count));

/// Resource format types.
enum class FormatType {
    /// Unknown format.
    unknown,
    /// Floating-point formats.
    float_,
    /// Unsigned normalized formats.
    unorm,
    /// Unsigned normalized SRGB formats.
    unorm_srgb,
    /// Signed normalized formats.
    snorm,
    /// Unsigned integer formats.
    uint,
    /// Signed integer formats.
    sint
};

SGL_ENUM_INFO(
    FormatType,
    {
        {FormatType::unknown, "unknown"},
        {FormatType::float_, "float"},
        {FormatType::unorm, "unorm"},
        {FormatType::unorm_srgb, "unorm_srgb"},
        {FormatType::snorm, "snorm"},
        {FormatType::uint, "uint"},
        {FormatType::sint, "sint"},
    }
);
SGL_ENUM_REGISTER(FormatType);

enum class FormatChannels : uint32_t {
    none = 0x0,
    r = 0x1,
    g = 0x2,
    b = 0x4,
    a = 0x8,
    rg = r | g,
    rgb = r | g | b,
    rgba = r | g | b | a,
};

SGL_ENUM_CLASS_OPERATORS(FormatChannels);
SGL_ENUM_INFO(
    FormatChannels,
    {
        {FormatChannels::none, "none"},
        {FormatChannels::r, "r"},
        {FormatChannels::g, "g"},
        {FormatChannels::b, "b"},
        {FormatChannels::a, "a"},
        {FormatChannels::rg, "rg"},
        {FormatChannels::rgb, "rgb"},
        {FormatChannels::rgba, "rgba"},
    }
);
SGL_ENUM_REGISTER(FormatChannels);

/// Resource format information.
struct SGL_API FormatInfo {
    /// Resource format.
    Format format;
    /// Format name.
    std::string name;
    /// Number of bytes per block (compressed) or pixel (uncompressed).
    uint32_t bytes_per_block;
    /// Number of channels.
    uint32_t channel_count;
    /// Format type (typeless, float, unorm, unorm_srgb, snorm, uint, sint).
    FormatType type;
    /// True if format has a depth component.
    bool is_depth;
    /// True if format has a stencil component.
    bool is_stencil;
    /// True if format is compressed.
    bool is_compressed;
    /// Block width for compressed formats (1 for uncompressed formats).
    uint32_t block_width;
    /// Block height for compressed formats (1 for uncompressed formats).
    uint32_t block_height;
    /// Number of bits per channel.
    std::array<uint32_t, 4> channel_bit_count;
    /// DXGI format.
    uint32_t dxgi_format;
    /// Vulkan format.
    uint32_t vk_format;
    /// Slang format.
    const char* slang_format;

    /// True if format has a depth or stencil component.
    bool is_depth_stencil() const { return is_depth || is_stencil; }

    /// True if format is floating point.
    bool is_float_format() const { return type == FormatType::float_; }
    /// True if format is integer.
    bool is_integer_format() const { return type == FormatType::uint || type == FormatType::sint; }
    /// True if format is normalized.
    bool is_normalized_format() const
    {
        return type == FormatType::unorm || type == FormatType::unorm_srgb || type == FormatType::snorm;
    }
    /// True if format is sRGB.
    bool is_srgb_format() const { return type == FormatType::unorm_srgb; }

    /// Get the channels for the format (only for color formats).
    FormatChannels get_channels() const
    {
        FormatChannels channels = FormatChannels::none;
        channels |= channel_count >= 1 ? FormatChannels::r : FormatChannels::none;
        channels |= channel_count >= 2 ? FormatChannels::g : FormatChannels::none;
        channels |= channel_count >= 3 ? FormatChannels::b : FormatChannels::none;
        channels |= channel_count >= 4 ? FormatChannels::a : FormatChannels::none;
        return channels;
    }

    /// Get the number of bits for the specified channels.
    uint32_t get_channel_bits(FormatChannels channels) const
    {
        uint32_t bits = 0;
        bits += (is_set(channels, FormatChannels::r)) ? channel_bit_count[0] : 0;
        bits += (is_set(channels, FormatChannels::g)) ? channel_bit_count[1] : 0;
        bits += (is_set(channels, FormatChannels::b)) ? channel_bit_count[2] : 0;
        bits += (is_set(channels, FormatChannels::a)) ? channel_bit_count[3] : 0;
        return bits;
    }

    /// Check if all channels have the same number of bits.
    bool has_equal_channel_bits() const
    {
        uint32_t bits = channel_bit_count[0];
        for (uint32_t i = 1; i < channel_count; ++i)
            if (channel_bit_count[i] != bits)
                return false;
        return true;
    }

    std::string to_string() const;
};

SGL_API const FormatInfo& get_format_info(Format format);

enum class FormatSupport : uint32_t {
    none = static_cast<uint32_t>(rhi::FormatSupport::None),
    copy_source = static_cast<uint32_t>(rhi::FormatSupport::CopySource),
    copy_destination = static_cast<uint32_t>(rhi::FormatSupport::CopyDestination),
    texture = static_cast<uint32_t>(rhi::FormatSupport::Texture),
    depth_stencil = static_cast<uint32_t>(rhi::FormatSupport::DepthStencil),
    render_target = static_cast<uint32_t>(rhi::FormatSupport::RenderTarget),
    blendable = static_cast<uint32_t>(rhi::FormatSupport::Blendable),
    multisampling = static_cast<uint32_t>(rhi::FormatSupport::Multisampling),
    resolvable = static_cast<uint32_t>(rhi::FormatSupport::Resolvable),
    shader_load = static_cast<uint32_t>(rhi::FormatSupport::ShaderLoad),
    shader_sample = static_cast<uint32_t>(rhi::FormatSupport::ShaderSample),
    shader_uav_load = static_cast<uint32_t>(rhi::FormatSupport::ShaderUavLoad),
    shader_uav_store = static_cast<uint32_t>(rhi::FormatSupport::ShaderUavStore),
    shader_atomic = static_cast<uint32_t>(rhi::FormatSupport::ShaderAtomic),
    buffer = static_cast<uint32_t>(rhi::FormatSupport::Buffer),
    index_buffer = static_cast<uint32_t>(rhi::FormatSupport::IndexBuffer),
    vertex_buffer = static_cast<uint32_t>(rhi::FormatSupport::VertexBuffer),
};
SGL_ENUM_CLASS_OPERATORS(FormatSupport);

SGL_ENUM_INFO(
    FormatSupport,
    {
        {FormatSupport::none, "none"},
        {FormatSupport::copy_source, "copy_source"},
        {FormatSupport::copy_destination, "copy_destination"},
        {FormatSupport::texture, "texture"},
        {FormatSupport::depth_stencil, "depth_stencil"},
        {FormatSupport::render_target, "render_target"},
        {FormatSupport::blendable, "blendable"},
        {FormatSupport::multisampling, "multisampling"},
        {FormatSupport::resolvable, "resolvable"},
        {FormatSupport::shader_load, "shader_load"},
        {FormatSupport::shader_sample, "shader_sample"},
        {FormatSupport::shader_uav_load, "shader_uav_load"},
        {FormatSupport::shader_uav_store, "shader_uav_store"},
        {FormatSupport::shader_atomic, "shader_atomic"},
        {FormatSupport::buffer, "buffer"},
        {FormatSupport::index_buffer, "index_buffer"},
        {FormatSupport::vertex_buffer, "vertex_buffer"},
    }
);
SGL_ENUM_REGISTER(FormatSupport);

namespace detail {
    template<typename T>
    struct HostTypeToFormat {
        static constexpr Format value = Format::undefined;
    };

#define SGL_HOST_TYPE_TO_FORMAT(type, format)                                                                          \
    template<>                                                                                                         \
    struct HostTypeToFormat<type> {                                                                                    \
        static constexpr Format value = format;                                                                        \
    }

    SGL_HOST_TYPE_TO_FORMAT(int8_t, Format::r8_sint);
    SGL_HOST_TYPE_TO_FORMAT(uint8_t, Format::r8_uint);
    SGL_HOST_TYPE_TO_FORMAT(int16_t, Format::r16_sint);
    SGL_HOST_TYPE_TO_FORMAT(uint16_t, Format::r16_uint);
    SGL_HOST_TYPE_TO_FORMAT(int, Format::r32_sint);
    SGL_HOST_TYPE_TO_FORMAT(int2, Format::rg32_sint);
    SGL_HOST_TYPE_TO_FORMAT(int4, Format::rgba32_sint);
    SGL_HOST_TYPE_TO_FORMAT(uint, Format::r32_uint);
    SGL_HOST_TYPE_TO_FORMAT(uint2, Format::rg32_uint);
    SGL_HOST_TYPE_TO_FORMAT(uint4, Format::rgba32_uint);
    SGL_HOST_TYPE_TO_FORMAT(float16_t, Format::r16_float);
    SGL_HOST_TYPE_TO_FORMAT(float16_t2, Format::rg16_float);
    SGL_HOST_TYPE_TO_FORMAT(float16_t4, Format::rgba16_float);
    SGL_HOST_TYPE_TO_FORMAT(float, Format::r32_float);
    SGL_HOST_TYPE_TO_FORMAT(float2, Format::rg32_float);
    SGL_HOST_TYPE_TO_FORMAT(float3, Format::rgb32_float);
    SGL_HOST_TYPE_TO_FORMAT(float4, Format::rgba32_float);

#undef SGL_HOST_TYPE_TO_FORMAT
} // namespace detail

template<typename T>
inline constexpr Format host_type_to_format()
{
    return detail::HostTypeToFormat<T>::value;
}

} // namespace sgl
