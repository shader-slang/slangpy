// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <cstddef>
#include <cstdint>

namespace sgl {

/**
 * \brief Downsample an interleaved float image by exactly 2x in each dimension using a 2x2 box filter.
 *
 * The source image is `2 * dst_width` by `2 * dst_height` pixels. Results match `Bitmap::resample` with a
 * `BoxFilter` for the same dimensions. Rows may be padded and any row range can be processed independently,
 * which allows callers to split the work across threads.
 *
 * \param src Source pixels.
 * \param src_row_pitch Distance between source rows in bytes.
 * \param dst Destination pixels.
 * \param dst_row_pitch Distance between destination rows in bytes.
 * \param dst_width Destination width in pixels.
 * \param dst_height Destination height in pixels.
 * \param channel_count Number of interleaved channels per pixel.
 */
SGL_API void downsample_2x_box(
    const float* src,
    size_t src_row_pitch,
    float* dst,
    size_t dst_row_pitch,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channel_count
);

/**
 * \brief Downsample an interleaved 8-bit unorm image by exactly 2x in each dimension using a 2x2 box filter.
 *
 * Filtering happens in linear light: with `srgb` set, every channel except the alpha channel is decoded from sRGB
 * before averaging and encoded back afterwards. Results are within one unit of decoding to float, resampling with
 * a `BoxFilter` and re-encoding with `Bitmap::convert`, without allocating float intermediates.
 *
 * \param src Source pixels, `2 * dst_width` by `2 * dst_height`.
 * \param src_row_pitch Distance between source rows in bytes.
 * \param dst Destination pixels.
 * \param dst_row_pitch Distance between destination rows in bytes.
 * \param dst_width Destination width in pixels.
 * \param dst_height Destination height in pixels.
 * \param channel_count Number of interleaved channels per pixel.
 * \param srgb Whether color channels are sRGB encoded.
 * \param has_alpha Whether the last channel is a linear alpha channel.
 */
SGL_API void downsample_2x_box_unorm8(
    const uint8_t* src,
    size_t src_row_pitch,
    uint8_t* dst,
    size_t dst_row_pitch,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channel_count,
    bool srgb,
    bool has_alpha = false
);

/**
 * \brief Downsample an interleaved 16-bit unorm image by exactly 2x in each dimension using a 2x2 box filter.
 *
 * All channels are averaged as linear values, computing `(a + b + c + d + 2) >> 2`. This is within one unit of
 * decoding to float, resampling with a `BoxFilter` and re-encoding with `Bitmap::convert`. sRGB-encoded 16-bit data is
 * not supported.
 *
 * \param src Source pixels, `2 * dst_width` by `2 * dst_height`.
 * \param src_row_pitch Distance between source rows in bytes.
 * \param dst Destination pixels.
 * \param dst_row_pitch Distance between destination rows in bytes.
 * \param dst_width Destination width in pixels.
 * \param dst_height Destination height in pixels.
 * \param channel_count Number of interleaved channels per pixel.
 */
SGL_API void downsample_2x_box_unorm16(
    const uint16_t* src,
    size_t src_row_pitch,
    uint16_t* dst,
    size_t dst_row_pitch,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channel_count
);

namespace detail {
    /// Scalar-only variant of `downsample_2x_box_unorm8`, kept as a reference for testing the SIMD paths.
    SGL_API void downsample_2x_box_unorm8_scalar(
        const uint8_t* src,
        size_t src_row_pitch,
        uint8_t* dst,
        size_t dst_row_pitch,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t channel_count,
        bool srgb,
        bool has_alpha = false
    );
} // namespace detail

} // namespace sgl
