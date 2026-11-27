// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/downsample.h"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("downsample");

TEST_CASE("downsample_2x_box_unorm8_matches_float_reference")
{
    using PF = Bitmap::PixelFormat;
    using CT = Bitmap::ComponentType;
    const uint32_t dst_width = 32, dst_height = 16;

    for (bool srgb : {false, true}) {
        for (PF pixel_format : {PF::y, PF::ya, PF::rgb, PF::rgba}) {
            CAPTURE(srgb);
            const int pixel_format_index = int(pixel_format);
            CAPTURE(pixel_format_index);
            Bitmap source(pixel_format, CT::uint8, 2 * dst_width, 2 * dst_height, 0, {}, nullptr, srgb);
            std::mt19937 rng(7);
            for (size_t i = 0; i < source.buffer_size(); ++i)
                source.uint8_data()[i] = static_cast<uint8_t>(rng());

            const uint32_t channels = source.channel_count();
            const bool has_alpha = pixel_format == PF::ya || pixel_format == PF::rgba;
            Bitmap result(pixel_format, CT::uint8, dst_width, dst_height);
            downsample_2x_box_unorm8(
                source.uint8_data(),
                size_t(source.width()) * channels,
                result.uint8_data(),
                size_t(dst_width) * channels,
                dst_width,
                dst_height,
                channels,
                srgb,
                has_alpha
            );

            // Reference: decode with the converter, box filter in float, encode with the converter.
            ref<Bitmap> linear = source.convert(pixel_format, CT::float32, false);
            ref<Bitmap> expected = linear->resample(dst_width, dst_height)->convert(pixel_format, CT::uint8, srgb);

            int max_diff = 0;
            for (size_t i = 0; i < result.buffer_size(); ++i)
                max_diff = std::max(max_diff, std::abs(int(result.uint8_data()[i]) - int(expected->uint8_data()[i])));
            CHECK(max_diff <= 1);
        }
    }
}

TEST_CASE("downsample_2x_box_unorm16_matches_float_reference")
{
    using PF = Bitmap::PixelFormat;
    using CT = Bitmap::ComponentType;
    const uint32_t dst_width = 33, dst_height = 7;

    for (PF pixel_format : {PF::y, PF::ya, PF::rgb, PF::rgba}) {
        const int pixel_format_index = int(pixel_format);
        CAPTURE(pixel_format_index);
        Bitmap source(pixel_format, CT::uint16, 2 * dst_width, 2 * dst_height, 0, {}, nullptr, false);
        std::mt19937 rng(11);
        for (size_t i = 0; i < source.pixel_count() * source.channel_count(); ++i)
            source.data_as<uint16_t>()[i] = static_cast<uint16_t>(rng());

        // Pad source and destination rows to check that the row pitches are honored.
        const uint32_t channels = source.channel_count();
        const size_t src_row = size_t(source.width()) * channels;
        const size_t src_stride = src_row + 3;
        std::vector<uint16_t> src(src_stride * source.height());
        for (uint32_t y = 0; y < source.height(); ++y)
            std::copy_n(source.data_as<uint16_t>() + y * src_row, src_row, src.begin() + y * src_stride);
        const size_t dst_stride = size_t(dst_width) * channels + 1;
        std::vector<uint16_t> result(dst_stride * dst_height, 0xcdcd);
        downsample_2x_box_unorm16(
            src.data(),
            src_stride * sizeof(uint16_t),
            result.data(),
            dst_stride * sizeof(uint16_t),
            dst_width,
            dst_height,
            channels
        );

        ref<Bitmap> linear = source.convert(pixel_format, CT::float32, false);
        ref<Bitmap> expected = linear->resample(dst_width, dst_height)->convert(pixel_format, CT::uint16, false);
        int max_diff = 0;
        for (uint32_t y = 0; y < dst_height; ++y) {
            for (uint32_t i = 0; i < dst_width * channels; ++i) {
                const int e = expected->data_as<uint16_t>()[y * dst_width * channels + i];
                max_diff = std::max(max_diff, std::abs(int(result[y * dst_stride + i]) - e));
            }
            CHECK(result[y * dst_stride + dst_width * channels] == 0xcdcd);
        }
        CHECK(max_diff <= 1);
    }
}

TEST_CASE("downsample_2x_box_unorm8_simd_matches_scalar")
{
    // SIMD is only used for linear rows with 1-4 channels. Widths cover empty SIMD loops, exact multiples of the vector
    // step and scalar tails.
    for (uint32_t channels = 1; channels <= 4; ++channels) {
        for (uint32_t dst_width : {1u, 3u, 4u, 7u, 8u, 15u, 16u, 17u, 33u, 61u}) {
            CAPTURE(channels);
            CAPTURE(dst_width);
            const uint32_t dst_height = 3;
            const size_t src_pitch = size_t(2) * dst_width * channels + 5;
            const size_t dst_pitch = size_t(dst_width) * channels + 3;
            std::vector<uint8_t> src(src_pitch * 2 * dst_height);
            std::mt19937 rng(dst_width * 16 + channels);
            for (uint8_t& value : src)
                value = static_cast<uint8_t>(rng());
            std::vector<uint8_t> expected(dst_pitch * dst_height, 0xcd);
            std::vector<uint8_t> actual(dst_pitch * dst_height, 0xcd);
            detail::downsample_2x_box_unorm8_scalar(
                src.data(),
                src_pitch,
                expected.data(),
                dst_pitch,
                dst_width,
                dst_height,
                channels,
                false
            );
            downsample_2x_box_unorm8(
                src.data(),
                src_pitch,
                actual.data(),
                dst_pitch,
                dst_width,
                dst_height,
                channels,
                false
            );
            CHECK(actual == expected);
        }
    }
}

TEST_CASE("downsample_2x_box_unorm8_srgb_alpha")
{
    // Independent of the kernel dispatch: the last (alpha) channel must be a plain linear average and all other
    // channels must match the sRGB result without alpha.
    const uint32_t dst_width = 19, dst_height = 3;
    for (uint32_t channels = 1; channels <= 5; ++channels) {
        const size_t src_pitch = size_t(2) * dst_width * channels;
        const size_t dst_pitch = size_t(dst_width) * channels;
        std::vector<uint8_t> src(src_pitch * 2 * dst_height);
        std::mt19937 rng(channels);
        for (uint8_t& value : src)
            value = static_cast<uint8_t>(rng());
        std::vector<uint8_t> no_alpha(dst_pitch * dst_height);
        downsample_2x_box_unorm8(
            src.data(),
            src_pitch,
            no_alpha.data(),
            dst_pitch,
            dst_width,
            dst_height,
            channels,
            true
        );

        CAPTURE(channels);
        const uint32_t alpha_channel = channels - 1;
        std::vector<uint8_t> result(dst_pitch * dst_height);
        downsample_2x_box_unorm8(
            src.data(),
            src_pitch,
            result.data(),
            dst_pitch,
            dst_width,
            dst_height,
            channels,
            true,
            true
        );
        bool ok = true;
        for (uint32_t y = 0; y < dst_height; ++y) {
            for (uint32_t x = 0; x < dst_width; ++x) {
                for (uint32_t c = 0; c < channels; ++c) {
                    const size_t i = y * dst_pitch + x * channels + c;
                    uint32_t expected = no_alpha[i];
                    if (c == alpha_channel) {
                        auto at = [&](uint32_t sx, uint32_t sy)
                        {
                            return uint32_t(src[sy * src_pitch + sx * channels + c]);
                        };
                        expected = (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1)
                                    + at(2 * x + 1, 2 * y + 1) + 2)
                            >> 2;
                    }
                    ok &= result[i] == expected;
                }
            }
        }
        CHECK(ok);
    }
}

TEST_SUITE_END();
