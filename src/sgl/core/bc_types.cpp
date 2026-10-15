// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/bc_types.h"
#include "sgl/core/bc_codec.h"
#include "sgl/core/error.h"

#include <algorithm>
#include <limits>

namespace sgl {

BCCompressedMip::BCCompressedMip(uint32_t width, uint32_t height, std::vector<uint8_t> data)
    : m_width(width)
    , m_height(height)
    , m_data(std::move(data))
{
    SGL_CHECK(width > 0 && height > 0, "BCCompressedMip: dimensions must be non-zero");
}

BCCompressedImage::BCCompressedImage(Format format, std::vector<BCCompressedMip> mip_levels)
    : m_format(format)
    , m_mip_levels(std::move(mip_levels))
{
    SGL_CHECK(bc_format_bytes_per_block(format) != 0, "BCCompressedImage: expected a BC format");
    SGL_CHECK(!m_mip_levels.empty(), "BCCompressedImage: image has no mip levels");
    SGL_CHECK(m_mip_levels.size() <= bc_mip_count(width(), height()), "BCCompressedImage: too many mip levels");
    uint32_t expected_width = width();
    uint32_t expected_height = height();
    for (size_t i = 0; i < m_mip_levels.size(); ++i) {
        const auto& mip = m_mip_levels[i];
        SGL_CHECK(
            mip.width() == expected_width && mip.height() == expected_height,
            "BCCompressedImage: invalid dimensions for mip level {}",
            i
        );
        SGL_CHECK(
            mip.data().size() == bc_compressed_size(expected_width, expected_height, format),
            "BCCompressedImage: invalid data size for mip level {}",
            i
        );
        expected_width = std::max(1u, expected_width / 2);
        expected_height = std::max(1u, expected_height / 2);
    }
}

ref<Bitmap> BCCompressedImage::decode(uint32_t mip_level) const
{
    SGL_CHECK(mip_level < mip_count(), "BCCompressedImage::decode: mip level out of range");
    const auto& mip = m_mip_levels[mip_level];
    Bitmap::PixelFormat pixel_format = Bitmap::PixelFormat::rgba;
    Bitmap::ComponentType component_type = Bitmap::ComponentType::uint8;
    switch (m_format) {
    case Format::bc4_unorm:
    case Format::bc4_snorm:
        pixel_format = Bitmap::PixelFormat::r;
        if (m_format == Format::bc4_snorm)
            component_type = Bitmap::ComponentType::int8;
        break;
    case Format::bc5_unorm:
    case Format::bc5_snorm:
        pixel_format = Bitmap::PixelFormat::rg;
        if (m_format == Format::bc5_snorm)
            component_type = Bitmap::ComponentType::int8;
        break;
    case Format::bc6h_ufloat:
    case Format::bc6h_sfloat:
        pixel_format = Bitmap::PixelFormat::rgb;
        component_type = Bitmap::ComponentType::float16;
        break;
    default:
        break;
    }

    auto bitmap = make_ref<Bitmap>(pixel_format, component_type, mip.width(), mip.height());
    bitmap->set_srgb_gamma(get_format_info(m_format).type == FormatType::unorm_srgb);
    size_t row_pitch = static_cast<size_t>(mip.width()) * bitmap->bytes_per_pixel();
    SGL_CHECK(row_pitch <= std::numeric_limits<uint32_t>::max(), "BCCompressedImage::decode: row pitch overflow");
    BCMutableImage dst{
        bitmap->data(),
        mip.width(),
        mip.height(),
        static_cast<uint32_t>(row_pitch),
        bitmap->channel_count(),
        component_type,
    };
    decode_bc(mip.data().data(), mip.data().size(), m_format, mip.width(), mip.height(), dst);
    return bitmap;
}

} // namespace sgl
