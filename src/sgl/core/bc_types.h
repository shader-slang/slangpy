// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/data_struct.h"
#include "sgl/core/bitmap.h"
#include "sgl/device/formats.h"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>
#include <vector>

namespace sgl {

// ────────────────────────────────────────────────────────────────────────────
// BCFormat
// ────────────────────────────────────────────────────────────────────────────

enum class BCFormat {
    bc1_unorm,
    bc1_unorm_srgb,
    bc2_unorm,
    bc2_unorm_srgb,
    bc3_unorm,
    bc3_unorm_srgb,
    bc4_unorm,
    bc4_snorm,
    bc5_unorm,
    bc5_snorm,
    bc6h_ufloat,
    bc6h_sfloat,
    bc7_unorm,
    bc7_unorm_srgb,
};

/// Convert BCFormat to RHI Format.
inline Format bc_format_to_format(BCFormat f)
{
    switch (f) {
    case BCFormat::bc1_unorm:
        return Format::bc1_unorm;
    case BCFormat::bc1_unorm_srgb:
        return Format::bc1_unorm_srgb;
    case BCFormat::bc2_unorm:
        return Format::bc2_unorm;
    case BCFormat::bc2_unorm_srgb:
        return Format::bc2_unorm_srgb;
    case BCFormat::bc3_unorm:
        return Format::bc3_unorm;
    case BCFormat::bc3_unorm_srgb:
        return Format::bc3_unorm_srgb;
    case BCFormat::bc4_unorm:
        return Format::bc4_unorm;
    case BCFormat::bc4_snorm:
        return Format::bc4_snorm;
    case BCFormat::bc5_unorm:
        return Format::bc5_unorm;
    case BCFormat::bc5_snorm:
        return Format::bc5_snorm;
    case BCFormat::bc6h_ufloat:
        return Format::bc6h_ufloat;
    case BCFormat::bc6h_sfloat:
        return Format::bc6h_sfloat;
    case BCFormat::bc7_unorm:
        return Format::bc7_unorm;
    case BCFormat::bc7_unorm_srgb:
        return Format::bc7_unorm_srgb;
    default:
        return Format::undefined;
    }
}

/// Convert RHI Format to BCFormat (returns nullopt if not a BC format).
inline std::optional<BCFormat> format_to_bc_format(Format f)
{
    switch (f) {
    case Format::bc1_unorm:
        return BCFormat::bc1_unorm;
    case Format::bc1_unorm_srgb:
        return BCFormat::bc1_unorm_srgb;
    case Format::bc2_unorm:
        return BCFormat::bc2_unorm;
    case Format::bc2_unorm_srgb:
        return BCFormat::bc2_unorm_srgb;
    case Format::bc3_unorm:
        return BCFormat::bc3_unorm;
    case Format::bc3_unorm_srgb:
        return BCFormat::bc3_unorm_srgb;
    case Format::bc4_unorm:
        return BCFormat::bc4_unorm;
    case Format::bc4_snorm:
        return BCFormat::bc4_snorm;
    case Format::bc5_unorm:
        return BCFormat::bc5_unorm;
    case Format::bc5_snorm:
        return BCFormat::bc5_snorm;
    case Format::bc6h_ufloat:
        return BCFormat::bc6h_ufloat;
    case Format::bc6h_sfloat:
        return BCFormat::bc6h_sfloat;
    case Format::bc7_unorm:
        return BCFormat::bc7_unorm;
    case Format::bc7_unorm_srgb:
        return BCFormat::bc7_unorm_srgb;
    default:
        return std::nullopt;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Utility functions
// ────────────────────────────────────────────────────────────────────────────

/// Bytes per 4×4 compressed block.
inline uint32_t bc_format_bytes_per_block(BCFormat f)
{
    switch (f) {
    case BCFormat::bc1_unorm:
    case BCFormat::bc1_unorm_srgb:
    case BCFormat::bc4_unorm:
    case BCFormat::bc4_snorm:
        return 8;
    default:
        return 16;
    }
}

/// Total compressed size in bytes for one mip level (ceiling division for non-multiple-of-4).
inline size_t bc_compressed_size(uint32_t width, uint32_t height, BCFormat format)
{
    uint32_t blocks_x = (width + 3) / 4;
    uint32_t blocks_y = (height + 3) / 4;
    return static_cast<size_t>(blocks_x) * blocks_y * bc_format_bytes_per_block(format);
}

/// Full mip chain level count: floor(log2(max(w,h))) + 1.
inline uint32_t bc_mip_count(uint32_t width, uint32_t height)
{
    uint32_t m = width > height ? width : height;
    if (m == 0)
        return 0;
    return static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(m)))) + 1;
}

// ────────────────────────────────────────────────────────────────────────────
// Image views
// ────────────────────────────────────────────────────────────────────────────

using BCComponentType = DataStruct::Type;

/// Non-owning immutable view into CPU pixel data.
struct BCImage {
    const void* data;
    uint32_t width;
    uint32_t height;
    uint32_t row_pitch;     ///< Bytes per row (allows stride).
    uint32_t channel_count; ///< 1–4.
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

/// Create a BCImage from a Bitmap.
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

// ────────────────────────────────────────────────────────────────────────────
// Encode options & result types
// ────────────────────────────────────────────────────────────────────────────

enum class BCEncodeQuality {
    fastest,
    normal,
    production,
    highest,
};

struct BCEncodeOptions {
    BCEncodeQuality quality = BCEncodeQuality::normal;
    bool generate_mipmaps = false;
    ReconstructionFilter mip_filter = BoxFilter{};
    uint32_t channel_weights[4] = {1, 1, 1, 1};
    bool has_alpha = true;
};

struct BCCompressedMip {
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> data;
};

struct BCCompressedImage {
    BCFormat format;
    std::vector<BCCompressedMip> mip_levels;
};

} // namespace sgl
