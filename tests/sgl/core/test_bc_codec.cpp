// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/bc_types.h"
#include "sgl/core/bc_codec.h"
#include "sgl/math/float16.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("bc_codec");

using BCEncoderTestFunc = void (*)(BCEncoderBackend backend);

static void run_bc_encoder_test(BCEncoderBackend backend, BCEncoderTestFunc func)
{
    if (backend == BCEncoderBackend::nvtt_gpu && !testing::device_tests_enabled())
        SKIP("device tests disabled by -skip-device-tests");

    if (!BCEncoder::is_backend_available(backend)) {
        if (backend == BCEncoderBackend::nvtt_cpu)
            SKIP("NVTT CPU backend is not available");
        if (backend == BCEncoderBackend::nvtt_gpu)
            SKIP("NVTT GPU backend is not available");
        FAIL("BC encoder backend is not available");
    }

    func(backend);
}

#define BC_ENCODER_TEST_CASE_IMPL(name, func)                                                                          \
    static void func(BCEncoderBackend backend);                                                                        \
    TEST_CASE(name ".software")                                                                                        \
    {                                                                                                                  \
        run_bc_encoder_test(BCEncoderBackend::software, func);                                                         \
    }                                                                                                                  \
    TEST_CASE(name ".nvtt_cpu")                                                                                        \
    {                                                                                                                  \
        run_bc_encoder_test(BCEncoderBackend::nvtt_cpu, func);                                                         \
    }                                                                                                                  \
    TEST_CASE(name ".nvtt_gpu")                                                                                        \
    {                                                                                                                  \
        run_bc_encoder_test(BCEncoderBackend::nvtt_gpu, func);                                                         \
    }                                                                                                                  \
    static void func(BCEncoderBackend backend)

#define BC_ENCODER_TEST_CASE(name) BC_ENCODER_TEST_CASE_IMPL(name, DOCTEST_ANONYMOUS(bc_encoder_test_))

//
// 1. Utility functions
//

TEST_CASE("bc_format_bytes_per_block")
{
    CHECK(bc_format_bytes_per_block(BCFormat::bc1_unorm) == 8);
    CHECK(bc_format_bytes_per_block(BCFormat::bc1_unorm_srgb) == 8);
    CHECK(bc_format_bytes_per_block(BCFormat::bc4_unorm) == 8);
    CHECK(bc_format_bytes_per_block(BCFormat::bc4_snorm) == 8);

    CHECK(bc_format_bytes_per_block(BCFormat::bc2_unorm) == 16);
    CHECK(bc_format_bytes_per_block(BCFormat::bc3_unorm) == 16);
    CHECK(bc_format_bytes_per_block(BCFormat::bc5_unorm) == 16);
    CHECK(bc_format_bytes_per_block(BCFormat::bc6h_ufloat) == 16);
    CHECK(bc_format_bytes_per_block(BCFormat::bc7_unorm) == 16);
    CHECK(bc_format_bytes_per_block(static_cast<BCFormat>(0xffffffffu)) == 0);
}

TEST_CASE("bc_compressed_size")
{
    // 4x4 -> 1 block
    CHECK(bc_compressed_size(4, 4, BCFormat::bc1_unorm) == 8);
    CHECK(bc_compressed_size(4, 4, BCFormat::bc7_unorm) == 16);

    // 8x8 -> 4 blocks
    CHECK(bc_compressed_size(8, 8, BCFormat::bc1_unorm) == 4 * 8);

    // Non-multiple-of-4: 5x5 -> ceil(5/4)^2 = 2*2 = 4 blocks
    CHECK(bc_compressed_size(5, 5, BCFormat::bc1_unorm) == 4 * 8);

    // 13x7 -> ceil(13/4)*ceil(7/4) = 4*2 = 8 blocks
    CHECK(bc_compressed_size(13, 7, BCFormat::bc3_unorm) == 8 * 16);

    // 1x1 -> 1 block
    CHECK(bc_compressed_size(1, 1, BCFormat::bc7_unorm) == 16);

    // 256x256 -> 64*64 = 4096 blocks
    CHECK(bc_compressed_size(256, 256, BCFormat::bc1_unorm) == 4096 * 8);
}

TEST_CASE("bc_mip_count")
{
    CHECK(bc_mip_count(1, 1) == 1);
    CHECK(bc_mip_count(2, 2) == 2);
    CHECK(bc_mip_count(4, 4) == 3);
    CHECK(bc_mip_count(64, 64) == 7);
    CHECK(bc_mip_count(256, 128) == 9);
    CHECK(bc_mip_count(1, 64) == 7);
    CHECK(bc_mip_count(0, 0) == 0);
}

//
// 2. BCFormat  Format conversion
//

TEST_CASE("BCFormat_Format_conversion")
{
    BCFormat all_bc[] = {
        BCFormat::bc1_unorm,
        BCFormat::bc1_unorm_srgb,
        BCFormat::bc2_unorm,
        BCFormat::bc2_unorm_srgb,
        BCFormat::bc3_unorm,
        BCFormat::bc3_unorm_srgb,
        BCFormat::bc4_unorm,
        BCFormat::bc4_snorm,
        BCFormat::bc5_unorm,
        BCFormat::bc5_snorm,
        BCFormat::bc6h_ufloat,
        BCFormat::bc6h_sfloat,
        BCFormat::bc7_unorm,
        BCFormat::bc7_unorm_srgb,
    };

    for (BCFormat bcf : all_bc) {
        Format f = bc_format_to_format(bcf);
        CHECK(f != Format::undefined);
        auto rt = format_to_bc_format(f);
        REQUIRE(rt.has_value());
        CHECK(rt.value() == bcf);
    }

    // Non-BC format -> nullopt
    CHECK(!format_to_bc_format(Format::rgba8_unorm).has_value());
}

//
// Helper: create a synthetic RGBA uint8 gradient image
//

static std::vector<uint8_t> make_gradient_rgba(uint32_t w, uint32_t h)
{
    std::vector<uint8_t> pixels(w * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            size_t idx = (y * w + x) * 4;
            pixels[idx + 0] = static_cast<uint8_t>(x * 255 / std::max(w - 1, 1u));
            pixels[idx + 1] = static_cast<uint8_t>(y * 255 / std::max(h - 1, 1u));
            pixels[idx + 2] = 128;
            pixels[idx + 3] = 255;
        }
    }
    return pixels;
}

static BCImage make_rgba_image(const std::vector<uint8_t>& pixels, uint32_t w, uint32_t h)
{
    return BCImage{
        .data = pixels.data(),
        .width = w,
        .height = h,
        .row_pitch = w * 4,
        .channel_count = 4,
        .component_type = BCComponentType::uint8,
    };
}

//
// Helper: compute PSNR between two images
//

static double compute_psnr(const uint8_t* a, const uint8_t* b, uint32_t w, uint32_t h, uint32_t channels)
{
    double mse = 0.0;
    size_t count = static_cast<size_t>(w) * h * channels;
    for (size_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(count);
    if (mse < 1e-10)
        return 100.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

static double compute_interleaved_psnr(
    const uint8_t* reference,
    uint32_t reference_channels,
    const uint8_t* decoded,
    uint32_t decoded_channels,
    uint32_t w,
    uint32_t h
)
{
    SGL_ASSERT(decoded_channels <= reference_channels);
    std::vector<uint8_t> reference_subset(static_cast<size_t>(w) * h * decoded_channels);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        for (uint32_t c = 0; c < decoded_channels; ++c)
            reference_subset[i * decoded_channels + c] = reference[i * reference_channels + c];
    }
    return compute_psnr(reference_subset.data(), decoded, w, h, decoded_channels);
}

static std::vector<uint8_t> decode_unorm(const BCCompressedMip& mip, BCFormat format, uint32_t channels)
{
    std::vector<uint8_t> decoded(static_cast<size_t>(mip.width) * mip.height * channels);
    BCMutableImage dst{
        decoded.data(),
        mip.width,
        mip.height,
        mip.width * channels,
        channels,
        BCComponentType::uint8,
    };
    decode_bc(mip.data.data(), mip.data.size(), format, mip.width, mip.height, dst);
    return decoded;
}

static bool has_near_opaque_alpha(const std::vector<uint8_t>& pixels, uint32_t channels)
{
    if (channels < 4)
        return false;
    for (size_t i = 3; i < pixels.size(); i += channels) {
        if (pixels[i] < 250)
            return false;
    }
    return true;
}

//
// 3. Roundtrip per format (4x4 block)
//

BC_ENCODER_TEST_CASE("roundtrip_4x4")
{
    BCEncoder encoder(backend);
    auto pixels = make_gradient_rgba(4, 4);
    BCImage src = make_rgba_image(pixels, 4, 4);

    struct FormatInfo {
        BCFormat format;
        uint32_t decoded_channels;
        size_t compressed_size;
        double min_psnr;
    };
    const FormatInfo formats[] = {
        {BCFormat::bc1_unorm, 4, 8, 13.0},
        {BCFormat::bc2_unorm, 4, 16, 13.0},
        {BCFormat::bc3_unorm, 4, 16, 13.0},
        {BCFormat::bc4_unorm, 1, 8, 20.0},
        {BCFormat::bc5_unorm, 2, 16, 20.0},
        {BCFormat::bc7_unorm, 4, 16, 20.0},
    };

    for (const FormatInfo& info : formats) {
        CAPTURE(static_cast<int>(info.format));
        BCCompressedImage compressed = encoder.encode(src, info.format);
        REQUIRE_EQ(compressed.mip_levels.size(), 1);
        const BCCompressedMip& mip = compressed.mip_levels[0];
        CHECK_EQ(mip.width, 4);
        CHECK_EQ(mip.height, 4);
        CHECK_EQ(mip.data.size(), info.compressed_size);

        std::vector<uint8_t> decoded = decode_unorm(mip, info.format, info.decoded_channels);
        CHECK(compute_interleaved_psnr(pixels.data(), 4, decoded.data(), info.decoded_channels, 4, 4) >= info.min_psnr);
        if (info.decoded_channels == 4)
            CHECK(has_near_opaque_alpha(decoded, info.decoded_channels));
    }
}

//
// 4. Roundtrip larger image (64x64) with PSNR check
//

BC_ENCODER_TEST_CASE("roundtrip_64x64")
{
    BCEncoder encoder(backend);
    const uint32_t W = 64, H = 64;
    auto pixels = make_gradient_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    struct FormatInfo {
        BCFormat format;
        uint32_t decoded_channels;
        double min_psnr;
    };

    FormatInfo formats[] = {
        {BCFormat::bc1_unorm, 4, 20.0},
        {BCFormat::bc2_unorm, 4, 20.0},
        {BCFormat::bc3_unorm, 4, 20.0},
        {BCFormat::bc4_unorm, 1, 20.0},
        {BCFormat::bc5_unorm, 2, 20.0},
        {BCFormat::bc7_unorm, 4, 25.0},
    };

    for (auto& fi : formats) {
        CAPTURE(static_cast<int>(fi.format));

        auto compressed = encoder.encode(src, fi.format);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        uint32_t ch = fi.decoded_channels;
        std::vector<uint8_t> decoded(W * H * ch, 0);
        BCMutableImage dst{decoded.data(), W, H, W * ch, ch, BCComponentType::uint8};
        decode_bc(compressed.mip_levels[0].data.data(), compressed.mip_levels[0].data.size(), fi.format, W, H, dst);

        // Build per-channel reference from the source for PSNR.
        std::vector<uint8_t> ref_data(W * H * ch);
        for (uint32_t i = 0; i < W * H; ++i)
            for (uint32_t c = 0; c < ch; ++c)
                ref_data[i * ch + c] = pixels[i * 4 + c];

        double psnr = compute_psnr(ref_data.data(), decoded.data(), W, H, ch);
        CHECK(psnr >= fi.min_psnr);
    }
}

//
// 5. Non-multiple-of-4 sizes
//

BC_ENCODER_TEST_CASE("non_multiple_of_4")
{
    BCEncoder encoder(backend);
    auto pixels = make_gradient_rgba(13, 7);
    BCImage src = make_rgba_image(pixels, 13, 7);

    auto compressed = encoder.encode(src, BCFormat::bc1_unorm);
    REQUIRE(compressed.mip_levels.size() == 1);
    CHECK(compressed.mip_levels[0].width == 13);
    CHECK(compressed.mip_levels[0].height == 7);
    // 4*2 = 8 blocks * 8 bytes = 64 bytes
    CHECK(compressed.mip_levels[0].data.size() == 64);

    std::vector<uint8_t> decoded(13 * 7 * 4, 0);
    BCMutableImage dst{decoded.data(), 13, 7, 13 * 4, 4, BCComponentType::uint8};
    decode_bc(
        compressed.mip_levels[0].data.data(),
        compressed.mip_levels[0].data.size(),
        BCFormat::bc1_unorm,
        13,
        7,
        dst
    );

    CHECK(compute_interleaved_psnr(pixels.data(), 4, decoded.data(), 4, 13, 7) >= 20.0);
    CHECK(has_near_opaque_alpha(decoded, 4));
}

//
// 6. Small images (1x1, 2x2, 3x3, 4x4)
//

BC_ENCODER_TEST_CASE("small_images")
{
    BCEncoder encoder(backend);

    uint32_t sizes[] = {1, 2, 3, 4};
    for (uint32_t s : sizes) {
        CAPTURE(s);
        auto pixels = make_gradient_rgba(s, s);
        BCImage src = make_rgba_image(pixels, s, s);

        auto compressed = encoder.encode(src, BCFormat::bc7_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].data.size() == 16); // one block

        std::vector<uint8_t> decoded(s * s * 4, 0);
        BCMutableImage dst{decoded.data(), s, s, s * 4, 4, BCComponentType::uint8};
        decode_bc(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc7_unorm,
            s,
            s,
            dst
        );

        CHECK(compute_interleaved_psnr(pixels.data(), 4, decoded.data(), 4, s, s) >= 18.0);
        CHECK(has_near_opaque_alpha(decoded, 4));
    }
}

//
// 7. Encode with mipmaps
//

BC_ENCODER_TEST_CASE("encode_with_mipmaps")
{
    BCEncoder encoder(backend);
    const uint32_t W = 64, H = 64;
    auto pixels = make_gradient_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncodeOptions opts;
    opts.generate_mipmaps = true;

    auto compressed = encoder.encode(src, BCFormat::bc1_unorm, opts);

    // bc_mip_count(64,64) = 7 levels (64, 32, 16, 8, 4, 2, 1)
    REQUIRE(compressed.mip_levels.size() == 7);

    // Verify per-level dimensions.
    uint32_t expected_w = W, expected_h = H;
    for (size_t i = 0; i < compressed.mip_levels.size(); ++i) {
        CHECK(compressed.mip_levels[i].width == expected_w);
        CHECK(compressed.mip_levels[i].height == expected_h);
        CHECK(compressed.mip_levels[i].data.size() == bc_compressed_size(expected_w, expected_h, BCFormat::bc1_unorm));
        expected_w = std::max(1u, expected_w / 2);
        expected_h = std::max(1u, expected_h / 2);
    }
}

//
// 8. Quality levels
//

TEST_CASE("quality_levels")
{
    BCEncoder encoder(BCEncoderBackend::software);
    constexpr uint32_t W = 64;
    constexpr uint32_t H = 64;
    auto pixels = make_gradient_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncodeOptions opts_fast;
    opts_fast.quality = BCEncodeQuality::fastest;

    BCEncodeOptions opts_high;
    opts_high.quality = BCEncodeQuality::highest;

    auto fast = encoder.encode(src, BCFormat::bc1_unorm, opts_fast);
    auto high = encoder.encode(src, BCFormat::bc1_unorm, opts_high);

    REQUIRE(fast.mip_levels.size() == 1);
    REQUIRE(high.mip_levels.size() == 1);
    CHECK(fast.mip_levels[0].data.size() == high.mip_levels[0].data.size());
    CHECK(fast.mip_levels[0].data != high.mip_levels[0].data);

    std::vector<uint8_t> fast_decoded = decode_unorm(fast.mip_levels[0], BCFormat::bc1_unorm, 4);
    std::vector<uint8_t> high_decoded = decode_unorm(high.mip_levels[0], BCFormat::bc1_unorm, 4);
    double fast_psnr = compute_interleaved_psnr(pixels.data(), 4, fast_decoded.data(), 4, W, H);
    double high_psnr = compute_interleaved_psnr(pixels.data(), 4, high_decoded.data(), 4, W, H);
    CHECK(high_psnr >= fast_psnr);
}

//
// 9. Channel weights (BC7)
//

TEST_CASE("channel_weights_bc7")
{
    BCEncoder encoder(BCEncoderBackend::software);
    constexpr uint32_t W = 16;
    constexpr uint32_t H = 16;
    std::vector<uint8_t> pixels(W * H * 4);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            size_t offset = (static_cast<size_t>(y) * W + x) * 4;
            pixels[offset + 0] = static_cast<uint8_t>((x * 37 + y * 17) & 0xff);
            pixels[offset + 1] = static_cast<uint8_t>((x * 11 + y * 53) & 0xff);
            pixels[offset + 2] = static_cast<uint8_t>((x * 71 + y * 7) & 0xff);
            pixels[offset + 3] = 255;
        }
    }
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncodeOptions uniform_options;
    BCEncodeOptions red_options;
    red_options.channel_weights[0] = 64;

    BCCompressedImage uniform = encoder.encode(src, BCFormat::bc7_unorm, uniform_options);
    BCCompressedImage red_weighted = encoder.encode(src, BCFormat::bc7_unorm, red_options);
    REQUIRE_EQ(uniform.mip_levels.size(), 1);
    REQUIRE_EQ(red_weighted.mip_levels.size(), 1);
    CHECK(uniform.mip_levels[0].data != red_weighted.mip_levels[0].data);
}

//
// 10. has_alpha hint (BC7)
//

TEST_CASE("has_alpha_hint_bc7")
{
    BCEncoder encoder(BCEncoderBackend::software);
    constexpr uint32_t W = 8;
    constexpr uint32_t H = 8;
    auto pixels = make_gradient_rgba(W, H);
    for (size_t i = 3; i < pixels.size(); i += 4)
        pixels[i] = static_cast<uint8_t>((i / 4) * 255 / (W * H - 1));
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncodeOptions alpha_options;
    alpha_options.has_alpha = true;
    BCEncodeOptions opaque_options;
    opaque_options.has_alpha = false;

    BCCompressedImage alpha = encoder.encode(src, BCFormat::bc7_unorm, alpha_options);
    BCCompressedImage opaque = encoder.encode(src, BCFormat::bc7_unorm, opaque_options);
    REQUIRE_EQ(alpha.mip_levels.size(), 1);
    REQUIRE_EQ(opaque.mip_levels.size(), 1);
    CHECK(alpha.mip_levels[0].data != opaque.mip_levels[0].data);

    std::vector<uint8_t> alpha_decoded = decode_unorm(alpha.mip_levels[0], BCFormat::bc7_unorm, 4);
    std::vector<uint8_t> opaque_decoded = decode_unorm(opaque.mip_levels[0], BCFormat::bc7_unorm, 4);
    CHECK(compute_interleaved_psnr(pixels.data(), 4, alpha_decoded.data(), 4, W, H) >= 20.0);
    CHECK(has_near_opaque_alpha(opaque_decoded, 4));
}

//
// 11. Backend selection and capabilities
//

TEST_CASE("backend_selection")
{
    BCEncoder software_encoder(BCEncoderBackend::software);
    CHECK_EQ(software_encoder.backend(), BCEncoderBackend::software);
    CHECK(BCEncoder::is_backend_available(BCEncoderBackend::automatic));
    CHECK(BCEncoder::is_backend_available(BCEncoderBackend::software));

    if (testing::device_tests_enabled()) {
        BCEncoder automatic_encoder;
        BCEncoderBackend expected_backend = BCEncoder::is_backend_available(BCEncoderBackend::nvtt_gpu)
            ? BCEncoderBackend::nvtt_gpu
            : (BCEncoder::is_backend_available(BCEncoderBackend::nvtt_cpu) ? BCEncoderBackend::nvtt_cpu
                                                                           : BCEncoderBackend::software);
        CHECK_EQ(automatic_encoder.backend(), expected_backend);
    }

    if (BCEncoder::is_backend_available(BCEncoderBackend::nvtt_cpu)) {
        BCEncoder nvtt_cpu_encoder(BCEncoderBackend::nvtt_cpu);
        CHECK_EQ(nvtt_cpu_encoder.backend(), BCEncoderBackend::nvtt_cpu);
    } else {
        CHECK_THROWS(BCEncoder{BCEncoderBackend::nvtt_cpu});
    }

    if (testing::device_tests_enabled()) {
        if (BCEncoder::is_backend_available(BCEncoderBackend::nvtt_gpu)) {
            BCEncoder nvtt_gpu_encoder(BCEncoderBackend::nvtt_gpu);
            CHECK_EQ(nvtt_gpu_encoder.backend(), BCEncoderBackend::nvtt_gpu);
        } else {
            CHECK_THROWS(BCEncoder{BCEncoderBackend::nvtt_gpu});
        }
    }
}

TEST_CASE("can_encode")
{
    BCEncoder software_encoder(BCEncoderBackend::software);
    const BCFormat invalid_format = static_cast<BCFormat>(0xffffffffu);

    // SW-encodable formats.
    CHECK(software_encoder.can_encode(BCFormat::bc1_unorm));
    CHECK(software_encoder.can_encode(BCFormat::bc3_unorm));
    CHECK(software_encoder.can_encode(BCFormat::bc4_unorm));
    CHECK(software_encoder.can_encode(BCFormat::bc5_unorm));
    CHECK(software_encoder.can_encode(BCFormat::bc7_unorm));
    CHECK_FALSE(software_encoder.can_encode(BCFormat::bc4_snorm));
    CHECK_FALSE(software_encoder.can_encode(BCFormat::bc5_snorm));
    CHECK_FALSE(software_encoder.can_encode(BCFormat::bc6h_ufloat));
    CHECK_FALSE(software_encoder.can_encode(BCFormat::bc6h_sfloat));
    CHECK_FALSE(software_encoder.can_encode(invalid_format));

    if (BCEncoder::is_backend_available(BCEncoderBackend::nvtt_cpu)) {
        BCEncoder nvtt_encoder(BCEncoderBackend::nvtt_cpu);
        CHECK_FALSE(nvtt_encoder.can_encode(BCFormat::bc4_snorm));
        CHECK_FALSE(nvtt_encoder.can_encode(BCFormat::bc5_snorm));
        CHECK(nvtt_encoder.can_encode(BCFormat::bc6h_ufloat));
        CHECK(nvtt_encoder.can_encode(BCFormat::bc6h_sfloat));
        CHECK_FALSE(nvtt_encoder.can_encode(invalid_format));

        auto pixels = make_gradient_rgba(4, 4);
        BCImage src = make_rgba_image(pixels, 4, 4);
        CHECK_THROWS(nvtt_encoder.encode(src, invalid_format));
    }
}

static std::vector<uint8_t> make_filter_pattern_rgba(uint32_t w, uint32_t h)
{
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 4;
            uint32_t hash = (x * 0x1f123bb5u) ^ (y * 0x9e3779b9u) ^ ((x + y) * 0x85ebca6bu);
            pixels[idx + 0] = static_cast<uint8_t>(hash);
            pixels[idx + 1] = static_cast<uint8_t>(hash >> 8);
            pixels[idx + 2] = static_cast<uint8_t>(hash >> 16);
            pixels[idx + 3] = 255;
        }
    }
    return pixels;
}

TEST_CASE("decode_snorm")
{
    // Every selector is zero, so this block decodes entirely to endpoint 0 (-127).
    uint8_t block[8] = {0x81, 0x7f, 0, 0, 0, 0, 0, 0};

    std::vector<int8_t> decoded_int(4 * 4);
    BCMutableImage int_dst{decoded_int.data(), 4, 4, 4, 1, BCComponentType::int8};
    decode_bc(block, sizeof(block), BCFormat::bc4_snorm, 4, 4, int_dst);
    for (int8_t value : decoded_int)
        CHECK_EQ(value, -127);

    std::vector<float> decoded_float(4 * 4);
    BCMutableImage float_dst{
        decoded_float.data(),
        4,
        4,
        4 * sizeof(float),
        1,
        BCComponentType::float32,
    };
    decode_bc(block, sizeof(block), BCFormat::bc4_snorm, 4, 4, float_dst);
    for (float value : decoded_float)
        CHECK(value == doctest::Approx(-1.0f));
}

TEST_CASE("decode_validates_destination_layout")
{
    uint8_t block[8] = {};
    std::vector<uint8_t> decoded(4 * 4 * 4);

    BCMutableImage bad_channels{decoded.data(), 4, 4, 16, 0, BCComponentType::uint8};
    CHECK_THROWS(decode_bc(block, sizeof(block), BCFormat::bc1_unorm, 4, 4, bad_channels));

    BCMutableImage bad_pitch{decoded.data(), 4, 4, 15, 4, BCComponentType::uint8};
    CHECK_THROWS(decode_bc(block, sizeof(block), BCFormat::bc1_unorm, 4, 4, bad_pitch));

    BCMutableImage bad_component_type{
        decoded.data(),
        4,
        4,
        16,
        4,
        static_cast<BCComponentType>(0xffffffffu),
    };
    CHECK_THROWS(decode_bc(block, sizeof(block), BCFormat::bc1_unorm, 4, 4, bad_component_type));

    BCMutableImage valid_dst{decoded.data(), 4, 4, 16, 4, BCComponentType::uint8};
    CHECK_THROWS(decode_bc(block, sizeof(block), static_cast<BCFormat>(0xffffffffu), 4, 4, valid_dst));
}

TEST_CASE("unaligned_component_access")
{
    BCEncoder encoder(BCEncoderBackend::software);
    constexpr uint32_t W = 4;
    constexpr uint32_t H = 4;
    constexpr uint32_t CHANNELS = 4;
    constexpr uint32_t TIGHT_ROW_PITCH = W * CHANNELS * sizeof(float);
    constexpr uint32_t UNALIGNED_ROW_PITCH = TIGHT_ROW_PITCH + 1;

    std::vector<float> tight_pixels(W * H * CHANNELS);
    for (size_t i = 0; i < tight_pixels.size(); ++i)
        tight_pixels[i] = static_cast<float>(i) / static_cast<float>(tight_pixels.size() - 1);

    std::vector<uint8_t> unaligned_pixels(1 + UNALIGNED_ROW_PITCH * H, 0xcd);
    for (uint32_t y = 0; y < H; ++y) {
        std::memcpy(
            unaligned_pixels.data() + 1 + y * UNALIGNED_ROW_PITCH,
            tight_pixels.data() + y * W * CHANNELS,
            TIGHT_ROW_PITCH
        );
    }

    BCImage tight_src{
        .data = tight_pixels.data(),
        .width = W,
        .height = H,
        .row_pitch = TIGHT_ROW_PITCH,
        .channel_count = CHANNELS,
        .component_type = BCComponentType::float32,
    };
    BCImage unaligned_src{
        .data = unaligned_pixels.data() + 1,
        .width = W,
        .height = H,
        .row_pitch = UNALIGNED_ROW_PITCH,
        .channel_count = CHANNELS,
        .component_type = BCComponentType::float32,
    };

    BCCompressedImage tight_encoded = encoder.encode(tight_src, BCFormat::bc4_unorm);
    BCCompressedImage unaligned_encoded = encoder.encode(unaligned_src, BCFormat::bc4_unorm);
    REQUIRE_EQ(tight_encoded.mip_levels.size(), 1);
    REQUIRE_EQ(unaligned_encoded.mip_levels.size(), 1);
    CHECK(tight_encoded.mip_levels[0].data == unaligned_encoded.mip_levels[0].data);

    constexpr uint32_t TIGHT_DST_ROW_PITCH = W * sizeof(float);
    constexpr uint32_t UNALIGNED_DST_ROW_PITCH = TIGHT_DST_ROW_PITCH + 1;
    std::vector<float> tight_decoded(W * H);
    std::vector<uint8_t> unaligned_decoded(1 + UNALIGNED_DST_ROW_PITCH * H, 0xcd);
    BCMutableImage tight_dst{
        .data = tight_decoded.data(),
        .width = W,
        .height = H,
        .row_pitch = TIGHT_DST_ROW_PITCH,
        .channel_count = 1,
        .component_type = BCComponentType::float32,
    };
    BCMutableImage unaligned_dst{
        .data = unaligned_decoded.data() + 1,
        .width = W,
        .height = H,
        .row_pitch = UNALIGNED_DST_ROW_PITCH,
        .channel_count = 1,
        .component_type = BCComponentType::float32,
    };

    const BCCompressedMip& mip = tight_encoded.mip_levels[0];
    decode_bc(mip.data.data(), mip.data.size(), BCFormat::bc4_unorm, W, H, tight_dst);
    decode_bc(mip.data.data(), mip.data.size(), BCFormat::bc4_unorm, W, H, unaligned_dst);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float value;
            std::memcpy(
                &value,
                unaligned_decoded.data() + 1 + y * UNALIGNED_DST_ROW_PITCH + x * sizeof(float),
                sizeof(value)
            );
            CHECK(value == tight_decoded[y * W + x]);
        }
    }
}

TEST_CASE("software_mipmaps_honor_row_pitch")
{
    BCEncoder encoder(BCEncoderBackend::software);
    constexpr uint32_t W = 8;
    constexpr uint32_t H = 8;
    constexpr uint32_t ROW_SIZE = W * 4;
    constexpr uint32_t PADDED_ROW_SIZE = ROW_SIZE + 13;

    auto tight_pixels = make_gradient_rgba(W, H);
    std::vector<uint8_t> padded_pixels(PADDED_ROW_SIZE * H, 0xcd);
    for (uint32_t y = 0; y < H; ++y)
        std::memcpy(padded_pixels.data() + y * PADDED_ROW_SIZE, tight_pixels.data() + y * ROW_SIZE, ROW_SIZE);

    BCImage tight = make_rgba_image(tight_pixels, W, H);
    BCImage padded{
        .data = padded_pixels.data(),
        .width = W,
        .height = H,
        .row_pitch = PADDED_ROW_SIZE,
        .channel_count = 4,
        .component_type = BCComponentType::uint8,
    };
    BCEncodeOptions options;
    options.generate_mipmaps = true;

    BCCompressedImage tight_result = encoder.encode(tight, BCFormat::bc7_unorm, options);
    BCCompressedImage padded_result = encoder.encode(padded, BCFormat::bc7_unorm, options);
    REQUIRE_EQ(tight_result.mip_levels.size(), padded_result.mip_levels.size());
    for (size_t i = 0; i < tight_result.mip_levels.size(); ++i)
        CHECK(tight_result.mip_levels[i].data == padded_result.mip_levels[i].data);
}

TEST_CASE("software_mipmaps_preserve_constant_color")
{
    BCEncoder encoder(BCEncoderBackend::software);
    constexpr uint32_t W = 8;
    constexpr uint32_t H = 8;
    std::vector<uint8_t> pixels(W * H * 4, 128);
    for (size_t i = 3; i < pixels.size(); i += 4)
        pixels[i] = 255;
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncodeOptions options;
    options.generate_mipmaps = true;
    for (BCFormat format : {BCFormat::bc7_unorm, BCFormat::bc7_unorm_srgb}) {
        BCCompressedImage compressed = encoder.encode(src, format, options);
        for (const BCCompressedMip& mip : compressed.mip_levels) {
            std::vector<uint8_t> decoded(mip.width * mip.height * 4);
            BCMutableImage dst{decoded.data(), mip.width, mip.height, mip.width * 4, 4, BCComponentType::uint8};
            decode_bc(mip.data.data(), mip.data.size(), format, mip.width, mip.height, dst);
            for (size_t i = 0; i < decoded.size(); i += 4) {
                CHECK(decoded[i] >= 120);
                CHECK(decoded[i] <= 136);
            }
        }
    }
}

//
// 12. BC6H encode error
//

TEST_CASE("bc6h_encode_error")
{
    BCEncoder encoder(BCEncoderBackend::software);
    auto pixels = make_gradient_rgba(4, 4);
    BCImage src = make_rgba_image(pixels, 4, 4);

    CHECK_THROWS(encoder.encode(src, BCFormat::bc6h_ufloat));
    CHECK_THROWS(encoder.encode(src, BCFormat::bc6h_sfloat));
}

//
// 13. Decode output format
//

BC_ENCODER_TEST_CASE("decode_output_format")
{
    BCEncoder encoder(backend);

    SUBCASE("BC4 -> 1ch uint8")
    {
        auto pixels = make_gradient_rgba(4, 4);
        BCImage src = make_rgba_image(pixels, 4, 4);
        auto compressed = encoder.encode(src, BCFormat::bc4_unorm);

        std::vector<uint8_t> decoded(4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4, 1, BCComponentType::uint8};
        decode_bc(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc4_unorm,
            4,
            4,
            dst
        );

        CHECK(compute_interleaved_psnr(pixels.data(), 4, decoded.data(), 1, 4, 4) >= 20.0);
    }

    SUBCASE("BC5 -> 2ch uint8")
    {
        auto pixels = make_gradient_rgba(4, 4);
        BCImage src = make_rgba_image(pixels, 4, 4);
        auto compressed = encoder.encode(src, BCFormat::bc5_unorm);

        std::vector<uint8_t> decoded(4 * 4 * 2, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 2, 2, BCComponentType::uint8};
        decode_bc(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc5_unorm,
            4,
            4,
            dst
        );

        CHECK(compute_interleaved_psnr(pixels.data(), 4, decoded.data(), 2, 4, 4) >= 20.0);
    }

    SUBCASE("BC7 -> 4ch RGBA uint8")
    {
        auto pixels = make_gradient_rgba(4, 4);
        BCImage src = make_rgba_image(pixels, 4, 4);
        auto compressed = encoder.encode(src, BCFormat::bc7_unorm);

        std::vector<uint8_t> decoded(4 * 4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 4, 4, BCComponentType::uint8};
        decode_bc(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc7_unorm,
            4,
            4,
            dst
        );

        CHECK(compute_interleaved_psnr(pixels.data(), 4, decoded.data(), 4, 4, 4) >= 20.0);
        CHECK(has_near_opaque_alpha(decoded, 4));
    }
}

//
// 14. NVTT3 encode (all formats)
//

TEST_CASE("bc_codec_nvtt3_encode" * doctest::skip(!BCEncoder::is_backend_available(BCEncoderBackend::nvtt_cpu)))
{
    BCEncoder encoder(BCEncoderBackend::nvtt_cpu);
    const uint32_t W = 64, H = 64;
    auto pixels = make_gradient_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncodeOptions opts;

    struct FormatInfo {
        BCFormat format;
        uint32_t decoded_channels;
        double min_psnr;
    };

    FormatInfo formats[] = {
        {BCFormat::bc1_unorm, 4, 20.0},
        {BCFormat::bc2_unorm, 4, 20.0},
        {BCFormat::bc3_unorm, 4, 20.0},
        {BCFormat::bc4_unorm, 1, 20.0},
        {BCFormat::bc5_unorm, 2, 20.0},
        {BCFormat::bc7_unorm, 4, 25.0},
    };

    for (auto& fi : formats) {
        CAPTURE(static_cast<int>(fi.format));

        auto compressed = encoder.encode(src, fi.format, opts);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        uint32_t ch = fi.decoded_channels;
        std::vector<uint8_t> decoded(W * H * ch, 0);
        BCMutableImage dst{decoded.data(), W, H, W * ch, ch, BCComponentType::uint8};
        decode_bc(compressed.mip_levels[0].data.data(), compressed.mip_levels[0].data.size(), fi.format, W, H, dst);

        std::vector<uint8_t> ref_data(W * H * ch);
        for (uint32_t i = 0; i < W * H; ++i)
            for (uint32_t c = 0; c < ch; ++c)
                ref_data[i * ch + c] = pixels[i * 4 + c];

        double psnr = compute_psnr(ref_data.data(), decoded.data(), W, H, ch);
        CHECK(psnr >= fi.min_psnr);
    }
}

//
// 15. NVTT3 BC6H encode+decode roundtrip
//

static std::vector<float> make_hdr_float32_rgb(uint32_t w, uint32_t h)
{
    // Synthetic HDR image: 3-channel float32 with values > 1.0.
    std::vector<float> pixels(w * h * 3);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            size_t idx = (y * w + x) * 3;
            pixels[idx + 0] = static_cast<float>(x) / static_cast<float>(w) * 4.0f;
            pixels[idx + 1] = static_cast<float>(y) / static_cast<float>(h) * 4.0f;
            pixels[idx + 2] = 1.5f;
        }
    }
    return pixels;
}

static std::vector<float> make_signed_hdr_float32_rgb(uint32_t w, uint32_t h)
{
    std::vector<float> pixels(static_cast<size_t>(w) * h * 3);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            pixels[idx + 0] = -2.0f + static_cast<float>(x) / static_cast<float>(std::max(w - 1, 1u)) * 4.0f;
            pixels[idx + 1] = -2.0f + static_cast<float>(y) / static_cast<float>(std::max(h - 1, 1u)) * 4.0f;
            pixels[idx + 2] = 1.5f;
        }
    }
    return pixels;
}

static std::vector<float> decode_bc6h(const BCCompressedMip& mip, BCFormat format)
{
    std::vector<uint16_t> half_data(static_cast<size_t>(mip.width) * mip.height * 3);
    BCMutableImage dst{
        .data = half_data.data(),
        .width = mip.width,
        .height = mip.height,
        .row_pitch = static_cast<uint32_t>(mip.width * 3 * sizeof(uint16_t)),
        .channel_count = 3,
        .component_type = BCComponentType::float16,
    };
    decode_bc(mip.data.data(), mip.data.size(), format, mip.width, mip.height, dst);

    std::vector<float> result(half_data.size());
    std::transform(
        half_data.begin(),
        half_data.end(),
        result.begin(),
        [](uint16_t value)
        {
            return math::float16_to_float32(value);
        }
    );
    return result;
}

static void check_hdr_decode_matches(const std::vector<float>& decoded, const std::vector<float>& reference)
{
    REQUIRE_EQ(decoded.size(), reference.size());
    float decoded_min = std::numeric_limits<float>::max();
    float decoded_max = std::numeric_limits<float>::lowest();
    float reference_min = std::numeric_limits<float>::max();
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t i = 0; i < decoded.size(); ++i) {
        decoded_min = std::min(decoded_min, decoded[i]);
        decoded_max = std::max(decoded_max, decoded[i]);
        reference_min = std::min(reference_min, reference[i]);
        double error = static_cast<double>(decoded[i]) - reference[i];
        squared_error += error * error;
        squared_reference += static_cast<double>(reference[i]) * reference[i];
    }

    CHECK(decoded_max > 1.0f);
    if (reference_min < -1.0f)
        CHECK(decoded_min < -1.0f);
    double relative_rmse = std::sqrt(squared_error / std::max(squared_reference, 1e-20));
    CHECK(relative_rmse < 0.2);
}

static std::vector<float> downsample_box_rgb(const std::vector<float>& source, uint32_t width, uint32_t height)
{
    uint32_t target_width = std::max(1u, width / 2);
    uint32_t target_height = std::max(1u, height / 2);
    std::vector<float> result(static_cast<size_t>(target_width) * target_height * 3);
    for (uint32_t y = 0; y < target_height; ++y) {
        uint32_t y_begin = y * 2;
        uint32_t y_end = std::min(y_begin + 2, height);
        for (uint32_t x = 0; x < target_width; ++x) {
            uint32_t x_begin = x * 2;
            uint32_t x_end = std::min(x_begin + 2, width);
            float sum[3] = {};
            uint32_t count = 0;
            for (uint32_t source_y = y_begin; source_y < y_end; ++source_y) {
                for (uint32_t source_x = x_begin; source_x < x_end; ++source_x) {
                    size_t source_index = (static_cast<size_t>(source_y) * width + source_x) * 3;
                    for (uint32_t channel = 0; channel < 3; ++channel)
                        sum[channel] += source[source_index + channel];
                    ++count;
                }
            }
            size_t target_index = (static_cast<size_t>(y) * target_width + x) * 3;
            for (uint32_t channel = 0; channel < 3; ++channel)
                result[target_index + channel] = sum[channel] / static_cast<float>(count);
        }
    }
    return result;
}

TEST_CASE("bc_codec_nvtt3_bc6h" * doctest::skip(!BCEncoder::is_backend_available(BCEncoderBackend::nvtt_cpu)))
{
    BCEncoder encoder(BCEncoderBackend::nvtt_cpu);
    const uint32_t W = 64, H = 64;
    auto pixels = make_hdr_float32_rgb(W, H);
    BCImage src{
        .data = pixels.data(),
        .width = W,
        .height = H,
        .row_pitch = W * 3 * sizeof(float),
        .channel_count = 3,
        .component_type = BCComponentType::float32,
    };

    BCEncodeOptions opts;

    SUBCASE("bc6h_ufloat")
    {
        auto compressed = encoder.encode(src, BCFormat::bc6h_ufloat, opts);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        std::vector<float> decoded = decode_bc6h(compressed.mip_levels[0], BCFormat::bc6h_ufloat);
        check_hdr_decode_matches(decoded, pixels);
    }

    SUBCASE("bc6h_sfloat")
    {
        std::vector<float> signed_pixels = make_signed_hdr_float32_rgb(W, H);
        BCImage signed_src{
            .data = signed_pixels.data(),
            .width = W,
            .height = H,
            .row_pitch = W * 3 * sizeof(float),
            .channel_count = 3,
            .component_type = BCComponentType::float32,
        };
        auto compressed = encoder.encode(signed_src, BCFormat::bc6h_sfloat, opts);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        std::vector<float> decoded = decode_bc6h(compressed.mip_levels[0], BCFormat::bc6h_sfloat);
        check_hdr_decode_matches(decoded, signed_pixels);
    }
}

//
// 16. NVTT3 vs SW comparison
//

TEST_CASE("bc_codec_nvtt3_vs_sw" * doctest::skip(!BCEncoder::is_backend_available(BCEncoderBackend::nvtt_cpu)))
{
    const uint32_t W = 64, H = 64;
    auto pixels = make_gradient_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncoder software_encoder(BCEncoderBackend::software);
    BCEncoder nvtt_encoder(BCEncoderBackend::nvtt_cpu);

    BCFormat formats[] = {
        BCFormat::bc1_unorm,
        BCFormat::bc3_unorm,
        BCFormat::bc7_unorm,
    };

    for (BCFormat fmt : formats) {
        CAPTURE(static_cast<int>(fmt));

        auto sw_compressed = software_encoder.encode(src, fmt);
        auto nvtt_compressed = nvtt_encoder.encode(src, fmt);

        REQUIRE(sw_compressed.mip_levels.size() == 1);
        REQUIRE(nvtt_compressed.mip_levels.size() == 1);

        // Both should produce same-sized output.
        CHECK(sw_compressed.mip_levels[0].data.size() == nvtt_compressed.mip_levels[0].data.size());

        // Decode both and verify similar quality.
        std::vector<uint8_t> sw_decoded(W * H * 4, 0);
        std::vector<uint8_t> nvtt_decoded(W * H * 4, 0);
        BCMutableImage sw_dst{sw_decoded.data(), W, H, W * 4, 4, BCComponentType::uint8};
        BCMutableImage nvtt_dst{nvtt_decoded.data(), W, H, W * 4, 4, BCComponentType::uint8};

        decode_bc(sw_compressed.mip_levels[0].data.data(), sw_compressed.mip_levels[0].data.size(), fmt, W, H, sw_dst);
        decode_bc(
            nvtt_compressed.mip_levels[0].data.data(),
            nvtt_compressed.mip_levels[0].data.size(),
            fmt,
            W,
            H,
            nvtt_dst
        );

        // Both decodings should be reasonable quality against the original.
        std::vector<uint8_t> ref_data(W * H * 4);
        for (uint32_t i = 0; i < W * H; ++i)
            for (uint32_t c = 0; c < 4; ++c)
                ref_data[i * 4 + c] = pixels[i * 4 + c];

        double sw_psnr = compute_psnr(ref_data.data(), sw_decoded.data(), W, H, 4);
        double nvtt_psnr = compute_psnr(ref_data.data(), nvtt_decoded.data(), W, H, 4);
        CHECK(sw_psnr >= 20.0);
        CHECK(nvtt_psnr >= 20.0);
    }
}

//
// 17. NVTT3 mipmap generation + BC6H encoding
//

TEST_CASE("bc_codec_nvtt3_mipmaps" * doctest::skip(!BCEncoder::is_backend_available(BCEncoderBackend::nvtt_cpu)))
{
    BCEncoder encoder(BCEncoderBackend::nvtt_cpu);
    const uint32_t W = 64, H = 64;
    auto pixels = make_hdr_float32_rgb(W, H);
    BCImage src{
        .data = pixels.data(),
        .width = W,
        .height = H,
        .row_pitch = W * 3 * sizeof(float),
        .channel_count = 3,
        .component_type = BCComponentType::float32,
    };

    BCEncodeOptions opts;
    opts.generate_mipmaps = true;

    auto compressed = encoder.encode(src, BCFormat::bc6h_ufloat, opts);

    // bc_mip_count(64,64) = 7 levels (64, 32, 16, 8, 4, 2, 1)
    REQUIRE(compressed.mip_levels.size() == 7);

    uint32_t expected_w = W, expected_h = H;
    std::vector<float> expected_pixels = pixels;
    for (size_t i = 0; i < compressed.mip_levels.size(); ++i) {
        CHECK(compressed.mip_levels[i].width == expected_w);
        CHECK(compressed.mip_levels[i].height == expected_h);
        CHECK(
            compressed.mip_levels[i].data.size() == bc_compressed_size(expected_w, expected_h, BCFormat::bc6h_ufloat)
        );

        std::vector<float> decoded = decode_bc6h(compressed.mip_levels[i], BCFormat::bc6h_ufloat);
        check_hdr_decode_matches(decoded, expected_pixels);

        expected_pixels = downsample_box_rgb(expected_pixels, expected_w, expected_h);
        expected_w = std::max(1u, expected_w / 2);
        expected_h = std::max(1u, expected_h / 2);
    }
}

TEST_CASE("bc_codec_nvtt3_gpu_mipmaps")
{
    if (!testing::device_tests_enabled())
        SKIP("device tests disabled by -skip-device-tests");
    if (!BCEncoder::is_backend_available(BCEncoderBackend::nvtt_gpu))
        SKIP("NVTT GPU backend is not available");

    BCEncoder encoder(BCEncoderBackend::nvtt_gpu);
    const uint32_t W = 64, H = 64;
    auto pixels = make_hdr_float32_rgb(W, H);
    BCImage src{
        .data = pixels.data(),
        .width = W,
        .height = H,
        .row_pitch = W * 3 * sizeof(float),
        .channel_count = 3,
        .component_type = BCComponentType::float32,
    };

    BCEncodeOptions options;
    options.generate_mipmaps = true;
    BCCompressedImage compressed = encoder.encode(src, BCFormat::bc6h_ufloat, options);
    REQUIRE_EQ(compressed.mip_levels.size(), bc_mip_count(W, H));

    uint32_t expected_w = W;
    uint32_t expected_h = H;
    std::vector<float> expected_pixels = pixels;
    for (const BCCompressedMip& mip : compressed.mip_levels) {
        CHECK_EQ(mip.width, expected_w);
        CHECK_EQ(mip.height, expected_h);
        CHECK_EQ(mip.data.size(), bc_compressed_size(expected_w, expected_h, BCFormat::bc6h_ufloat));

        std::vector<float> decoded = decode_bc6h(mip, BCFormat::bc6h_ufloat);
        check_hdr_decode_matches(decoded, expected_pixels);

        expected_pixels = downsample_box_rgb(expected_pixels, expected_w, expected_h);
        expected_w = std::max(1u, expected_w / 2);
        expected_h = std::max(1u, expected_h / 2);
    }
}

BC_ENCODER_TEST_CASE("bc_codec_mipmap_filters")
{
    constexpr uint32_t W = 32;
    constexpr uint32_t H = 16;
    auto pixels = make_filter_pattern_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    const BCMipFilter filters[] = {
        BoxFilter{},
        TentFilter{1.25f},
        MitchellFilter{0.25f, 0.5f},
    };

    std::vector<std::vector<uint8_t>> filtered_mips;
    for (const BCMipFilter& filter : filters) {
        CAPTURE(filter.index());
        BCEncoder encoder(backend);
        BCEncodeOptions options;
        options.generate_mipmaps = true;
        options.mip_filter = filter;

        BCCompressedImage compressed = encoder.encode(src, BCFormat::bc7_unorm_srgb, options);
        REQUIRE_EQ(compressed.mip_levels.size(), bc_mip_count(W, H));
        CHECK_EQ(compressed.mip_levels.back().width, 1);
        CHECK_EQ(compressed.mip_levels.back().height, 1);
        filtered_mips.push_back(compressed.mip_levels[1].data);
    }
    REQUIRE_EQ(filtered_mips.size(), 3);
    CHECK(filtered_mips[0] != filtered_mips[1]);
    CHECK(filtered_mips[0] != filtered_mips[2]);
    CHECK(filtered_mips[1] != filtered_mips[2]);
}

TEST_SUITE_END();
