// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/downsample.h"

#include <chrono>
#include <cstdlib>
#include <random>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("downsample");

TEST_CASE("downsample_2x_box_padded_rows")
{
    for (uint32_t channels = 1; channels <= 5; ++channels) {
        CAPTURE(channels);
        const uint32_t dst_width = 5, dst_height = 3;
        // Pad both source and destination rows to check that the row pitches are honored.
        const size_t src_stride = 2 * dst_width * channels + 3;
        const size_t dst_stride = dst_width * channels + 2;
        std::vector<float> src(src_stride * 2 * dst_height);
        std::mt19937 rng(channels);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (float& value : src)
            value = dist(rng);
        std::vector<float> dst(dst_stride * dst_height, 42.f);

        downsample_2x_box(
            src.data(),
            src_stride * sizeof(float),
            dst.data(),
            dst_stride * sizeof(float),
            dst_width,
            dst_height,
            channels
        );

        for (uint32_t y = 0; y < dst_height; ++y) {
            for (uint32_t x = 0; x < dst_width; ++x) {
                for (uint32_t c = 0; c < channels; ++c) {
                    auto at = [&](uint32_t sx, uint32_t sy)
                    {
                        return src[sy * src_stride + sx * channels + c];
                    };
                    const float expected = 0.25f
                        * (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1));
                    CHECK(dst[y * dst_stride + x * channels + c] == doctest::Approx(expected).epsilon(1e-6));
                }
            }
            // Padding must stay untouched.
            for (size_t i = dst_width * channels; i < dst_stride; ++i)
                CHECK(dst[y * dst_stride + i] == 42.f);
        }
    }
}

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

        // Pad destination rows to check that the row pitch is honored.
        const uint32_t channels = source.channel_count();
        const size_t dst_stride = size_t(dst_width) * channels + 1;
        std::vector<uint16_t> result(dst_stride * dst_height, 0xcdcd);
        downsample_2x_box_unorm16(
            source.data_as<uint16_t>(),
            size_t(source.width()) * channels * sizeof(uint16_t),
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
    // Widths cover empty SIMD loops, exact multiples of the vector step and scalar tails.
    for (uint32_t channels = 1; channels <= 5; ++channels) {
        for (uint32_t dst_width : {1u, 3u, 4u, 7u, 8u, 15u, 16u, 17u, 33u, 61u}) {
            for (bool srgb : {false, true}) {
                for (bool has_alpha : {false, true}) {
                    CAPTURE(channels);
                    CAPTURE(dst_width);
                    CAPTURE(srgb);
                    CAPTURE(has_alpha);
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
                        srgb,
                        has_alpha
                    );
                    downsample_2x_box_unorm8(
                        src.data(),
                        src_pitch,
                        actual.data(),
                        dst_pitch,
                        dst_width,
                        dst_height,
                        channels,
                        srgb,
                        has_alpha
                    );
                    CHECK(actual == expected);
                }
            }
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

// Run with `sgl_tests -tc=downsample_benchmark --no-skip` in a release build.
TEST_CASE("downsample_benchmark" * doctest::skip())
{
    const uint32_t dst_width = 2048, dst_height = 2048;
    const int iterations = 10;
    auto report = [&](const char* name, uint32_t channels, bool srgb, size_t bytes, auto&& fn)
    {
        fn();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
            fn();
        double ms
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iterations;
        MESSAGE(
            fmt::format(
                "{} channels={} srgb={}: {:.2f} ms ({:.1f} GB/s source)",
                name,
                channels,
                srgb,
                ms,
                bytes / (ms * 1e6)
            )
        );
    };
    for (uint32_t channels = 1; channels <= 4; ++channels) {
        const bool has_alpha = channels == 2 || channels == 4;
        std::vector<uint8_t> src(size_t(4) * dst_width * dst_height * channels);
        std::mt19937 rng(1);
        for (uint8_t& value : src)
            value = static_cast<uint8_t>(rng());
        std::vector<uint8_t> dst(size_t(dst_width) * dst_height * channels);
        for (bool srgb : {false, true}) {
            auto args = [&](auto kernel)
            {
                return [&, kernel]
                {
                    kernel(
                        src.data(),
                        size_t(2) * dst_width * channels,
                        dst.data(),
                        size_t(dst_width) * channels,
                        dst_width,
                        dst_height,
                        channels,
                        srgb,
                        has_alpha
                    );
                };
            };
            report("unorm8 scalar", channels, srgb, src.size(), args(detail::downsample_2x_box_unorm8_scalar));
            report("unorm8       ", channels, srgb, src.size(), args(downsample_2x_box_unorm8));
        }
        std::vector<uint16_t> src_u16(src.begin(), src.end());
        std::vector<uint16_t> dst_u16(dst.size());
        report(
            "unorm16      ",
            channels,
            false,
            src_u16.size() * sizeof(uint16_t),
            [&]
            {
                downsample_2x_box_unorm16(
                    src_u16.data(),
                    size_t(2) * dst_width * channels * sizeof(uint16_t),
                    dst_u16.data(),
                    size_t(dst_width) * channels * sizeof(uint16_t),
                    dst_width,
                    dst_height,
                    channels
                );
            }
        );
        std::vector<float> src_float(src.size());
        std::vector<float> dst_float(dst.size());
        report(
            "float32      ",
            channels,
            false,
            src_float.size() * sizeof(float),
            [&]
            {
                downsample_2x_box(
                    src_float.data(),
                    size_t(2) * dst_width * channels * sizeof(float),
                    dst_float.data(),
                    size_t(dst_width) * channels * sizeof(float),
                    dst_width,
                    dst_height,
                    channels
                );
            }
        );
    }
}

TEST_SUITE_END();
