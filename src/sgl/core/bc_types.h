// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/data_struct.h"
#include "sgl/core/enum.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/rfilter.h"
#include "sgl/device/formats.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace sgl {

//
// Utility functions
//

/// Bytes per 4x4 compressed block.
inline uint32_t bc_format_bytes_per_block(Format f)
{
    switch (f) {
    case Format::bc1_unorm:
    case Format::bc1_unorm_srgb:
    case Format::bc4_unorm:
    case Format::bc4_snorm:
        return 8;
    case Format::bc2_unorm:
    case Format::bc2_unorm_srgb:
    case Format::bc3_unorm:
    case Format::bc3_unorm_srgb:
    case Format::bc5_unorm:
    case Format::bc5_snorm:
    case Format::bc6h_ufloat:
    case Format::bc6h_sfloat:
    case Format::bc7_unorm:
    case Format::bc7_unorm_srgb:
        return 16;
    default:
        return 0;
    }
}

/// Number of 4-pixel blocks needed for one image dimension.
inline uint32_t bc_block_count(uint32_t dimension)
{
    return dimension / 4 + (dimension % 4 != 0);
}

/// Total compressed size in bytes for one mip level (ceiling division for non-multiple-of-4).
inline size_t bc_compressed_size(uint32_t width, uint32_t height, Format format)
{
    size_t blocks_x = bc_block_count(width);
    size_t blocks_y = bc_block_count(height);
    return static_cast<size_t>(blocks_x) * blocks_y * bc_format_bytes_per_block(format);
}

/// Full mip chain level count: floor(log2(max(w,h))) + 1.
inline uint32_t bc_mip_count(uint32_t width, uint32_t height)
{
    uint32_t m = width > height ? width : height;
    if (m == 0)
        return 0;
    // Use integer bit counting instead of float log2 to avoid precision loss for dimensions > 2^24.
    uint32_t count = 0;
    uint32_t v = m;
    while (v >>= 1)
        ++count;
    return count + 1;
}

//
// Image views
//

using BCComponentType = DataStruct::Type;

/// Non-owning immutable view into CPU pixel data in R, RG, RGB, or RGBA order.
/// Missing color channels are zero and missing alpha is one. Components are
/// normalized integers or floating-point values. Color values must already use
/// the transfer function of the destination format (sRGB for *_srgb, linear otherwise).
struct BCImage {
    const void* data;
    uint32_t width;
    uint32_t height;
    uint32_t row_pitch;     ///< Bytes per row (allows stride).
    uint32_t channel_count; ///< 1-4.
    BCComponentType component_type;
};

/// Non-owning mutable view into CPU pixel data.
struct BCMutableImage {
    void* data;
    uint32_t width;
    uint32_t height;
    uint32_t row_pitch;
    uint32_t channel_count;
    BCComponentType component_type;
};

/// Create a raw BCImage view from a Bitmap, without interpreting pixel format or gamma.
/// Use BCEncoder::encode(const Bitmap*, ...) to honor bitmap metadata.
inline BCImage bc_image_from_bitmap(const Bitmap& bmp)
{
    return BCImage{
        .data = bmp.data(),
        .width = bmp.width(),
        .height = bmp.height(),
        .row_pitch = static_cast<uint32_t>(bmp.width() * bmp.bytes_per_pixel()),
        .channel_count = bmp.channel_count(),
        .component_type = bmp.component_type(),
    };
}

//
// Encode options & result types
//

/// BC compression quality and encoding effort.
enum class BCEncodeQuality {
    /// Favor encoding speed over quality.
    fastest,
    /// Balance encoding speed and quality.
    normal,
    /// Spend more encoding effort for production-quality output.
    production,
    /// Use the highest available encoding effort.
    highest,
};
SGL_ENUM_INFO(
    BCEncodeQuality,
    {
        {BCEncodeQuality::fastest, "fastest"},
        {BCEncodeQuality::normal, "normal"},
        {BCEncodeQuality::production, "production"},
        {BCEncodeQuality::highest, "highest"},
    }
);
SGL_ENUM_REGISTER(BCEncodeQuality);

/// Reconstruction filters supported by all BC encoder backends.
using BCMipFilter = std::variant<BoxFilter, TentFilter, MitchellFilter>;

/// Options shared by the software and NVTT encoders.
struct BCEncodeOptions {
    /// Encoding quality and effort. Defaults to normal.
    BCEncodeQuality quality = BCEncodeQuality::normal;
    /// Generate the full mip chain from the source image. Defaults to false.
    bool generate_mipmaps = false;
    /// Mipmap reconstruction filter: BoxFilter, TentFilter, or MitchellFilter.
    /// Defaults to BoxFilter.
    BCMipFilter mip_filter = BoxFilter{};
    /// Positive RGBA channel weights. Defaults to equal weights.
    std::array<uint32_t, 4> channel_weights = {1, 1, 1, 1};
    /// Hint that the image has alpha. Defaults to true; handling depends on the format and backend.
    bool has_alpha = true;
};

/// An owning compressed mip level with read-only pixel dimensions and block data.
class SGL_API BCCompressedMip {
public:
    BCCompressedMip(uint32_t width, uint32_t height, std::vector<uint8_t> data);

    /// Width of this mip level in pixels.
    uint32_t width() const { return m_width; }
    /// Height of this mip level in pixels.
    uint32_t height() const { return m_height; }
    /// Read-only compressed block data.
    /// In Python, returns a uint8 NumPy view that keeps the mip alive; bytes(data) makes a copy.
    const std::vector<uint8_t>& data() const { return m_data; }

private:
    uint32_t m_width;
    uint32_t m_height;
    std::vector<uint8_t> m_data;
};

/// An owning BC-compressed 2D image, with one or more consecutive mip levels.
class SGL_API BCCompressedImage {
public:
    /// Validate and take ownership of compressed mip levels, starting at level zero.
    /// Python mip inputs are copied. All levels must use the supplied format.
    BCCompressedImage(Format format, std::vector<BCCompressedMip> mip_levels);

    /// Load all mip levels from a BC-compressed 2D DDS file.
    /// Array, cube, and volume textures are not supported.
    explicit BCCompressedImage(const std::filesystem::path& path);
    /// Read all mip levels from a BC-compressed 2D DDS stream.
    /// Array, cube, and volume textures are not supported. The stream is not retained.
    explicit BCCompressedImage(Stream* stream);

    /// BC format shared by all mip levels.
    Format format() const { return m_format; }
    /// Width of the largest mip level in pixels.
    uint32_t width() const { return m_mip_levels.front().width(); }
    /// Height of the largest mip level in pixels.
    uint32_t height() const { return m_mip_levels.front().height(); }
    /// Number of mip levels.
    uint32_t mip_count() const { return static_cast<uint32_t>(m_mip_levels.size()); }
    /// Mip levels from largest to smallest.
    /// In Python, each mip keeps its image alive.
    std::span<const BCCompressedMip> mip_levels() const { return m_mip_levels; }

    /// Write all mip levels to a DDS file.
    void write_dds(const std::filesystem::path& path) const;
    /// Write all mip levels to a DDS stream.
    void write_dds(Stream* stream) const;

    /// Decode a mip level to an independent Bitmap without changing its transfer function.
    /// BC1/2/3/7 produce RGBA uint8; BC4/5 produce R/RG uint8 or int8 for SNORM;
    /// BC6H produces RGB float16. The sRGB flag is preserved. Throws for an invalid mip index.
    ref<Bitmap> decode(uint32_t mip_level = 0) const;

private:
    Format m_format;
    std::vector<BCCompressedMip> m_mip_levels;
};

} // namespace sgl
