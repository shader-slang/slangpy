// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/bc_types.h"
#include "sgl/core/error.h"

#include <algorithm>

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

} // namespace sgl
