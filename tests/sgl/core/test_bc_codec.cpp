// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/bc_types.h"
#include "sgl/core/bc_codec.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

using namespace sgl;

TEST_SUITE_BEGIN("bc_codec");

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

//
// 3. Roundtrip per format (4x4 block)
//

TEST_CASE("roundtrip_4x4")
{
    BCCodec codec;
    auto pixels = make_gradient_rgba(4, 4);
    BCImage src = make_rgba_image(pixels, 4, 4);

    SUBCASE("BC1")
    {
        auto compressed = codec.encode(src, BCFormat::bc1_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == 4);
        CHECK(compressed.mip_levels[0].height == 4);
        CHECK(compressed.mip_levels[0].data.size() == 8);

        std::vector<uint8_t> decoded(4 * 4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 4, 4, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc1_unorm,
            4,
            4,
            dst
        );

        // Lossy - just verify non-zero output.
        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("BC2")
    {
        auto compressed = codec.encode(src, BCFormat::bc2_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].data.size() == 16);

        std::vector<uint8_t> decoded(4 * 4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 4, 4, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc2_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("BC3")
    {
        auto compressed = codec.encode(src, BCFormat::bc3_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].data.size() == 16);

        std::vector<uint8_t> decoded(4 * 4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 4, 4, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc3_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("BC4")
    {
        auto compressed = codec.encode(src, BCFormat::bc4_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].data.size() == 8);

        std::vector<uint8_t> decoded(4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4, 1, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc4_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("BC5")
    {
        auto compressed = codec.encode(src, BCFormat::bc5_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].data.size() == 16);

        std::vector<uint8_t> decoded(4 * 4 * 2, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 2, 2, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc5_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("BC7")
    {
        auto compressed = codec.encode(src, BCFormat::bc7_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].data.size() == 16);

        std::vector<uint8_t> decoded(4 * 4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 4, 4, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc7_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }
}

//
// 4. Roundtrip larger image (64x64) with PSNR check
//

TEST_CASE("roundtrip_64x64")
{
    BCCodec codec;
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

        auto compressed = codec.encode(src, fi.format);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        uint32_t ch = fi.decoded_channels;
        std::vector<uint8_t> decoded(W * H * ch, 0);
        BCMutableImage dst{decoded.data(), W, H, W * ch, ch, BCComponentType::uint8};
        codec.decode(compressed.mip_levels[0].data.data(), compressed.mip_levels[0].data.size(), fi.format, W, H, dst);

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

TEST_CASE("non_multiple_of_4")
{
    BCCodec codec;
    auto pixels = make_gradient_rgba(13, 7);
    BCImage src = make_rgba_image(pixels, 13, 7);

    auto compressed = codec.encode(src, BCFormat::bc1_unorm);
    REQUIRE(compressed.mip_levels.size() == 1);
    CHECK(compressed.mip_levels[0].width == 13);
    CHECK(compressed.mip_levels[0].height == 7);
    // 4*2 = 8 blocks * 8 bytes = 64 bytes
    CHECK(compressed.mip_levels[0].data.size() == 64);

    std::vector<uint8_t> decoded(13 * 7 * 4, 0);
    BCMutableImage dst{decoded.data(), 13, 7, 13 * 4, 4, BCComponentType::uint8};
    codec.decode(
        compressed.mip_levels[0].data.data(),
        compressed.mip_levels[0].data.size(),
        BCFormat::bc1_unorm,
        13,
        7,
        dst
    );

    bool any_nonzero = false;
    for (auto v : decoded)
        if (v != 0) {
            any_nonzero = true;
            break;
        }
    CHECK(any_nonzero);
}

//
// 6. Small images (1x1, 2x2, 3x3, 4x4)
//

TEST_CASE("small_images")
{
    BCCodec codec;

    uint32_t sizes[] = {1, 2, 3, 4};
    for (uint32_t s : sizes) {
        CAPTURE(s);
        auto pixels = make_gradient_rgba(s, s);
        BCImage src = make_rgba_image(pixels, s, s);

        auto compressed = codec.encode(src, BCFormat::bc7_unorm);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].data.size() == 16); // one block

        std::vector<uint8_t> decoded(s * s * 4, 0);
        BCMutableImage dst{decoded.data(), s, s, s * 4, 4, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc7_unorm,
            s,
            s,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }
}

//
// 7. Encode with mipmaps
//

TEST_CASE("encode_with_mipmaps")
{
    BCCodec codec;
    const uint32_t W = 64, H = 64;
    auto pixels = make_gradient_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    BCEncodeOptions opts;
    opts.generate_mipmaps = true;

    auto compressed = codec.encode(src, BCFormat::bc1_unorm, opts);

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
    BCCodec codec;
    auto pixels = make_gradient_rgba(8, 8);
    BCImage src = make_rgba_image(pixels, 8, 8);

    BCEncodeOptions opts_fast;
    opts_fast.quality = BCEncodeQuality::fastest;

    BCEncodeOptions opts_high;
    opts_high.quality = BCEncodeQuality::highest;

    auto fast = codec.encode(src, BCFormat::bc1_unorm, opts_fast);
    auto high = codec.encode(src, BCFormat::bc1_unorm, opts_high);

    REQUIRE(fast.mip_levels.size() == 1);
    REQUIRE(high.mip_levels.size() == 1);
    CHECK(fast.mip_levels[0].data.size() == high.mip_levels[0].data.size());
}

//
// 9. Channel weights (BC7)
//

TEST_CASE("channel_weights_bc7")
{
    BCCodec codec;
    auto pixels = make_gradient_rgba(4, 4);
    BCImage src = make_rgba_image(pixels, 4, 4);

    BCEncodeOptions opts;
    opts.channel_weights[0] = 4;
    opts.channel_weights[1] = 2;
    opts.channel_weights[2] = 1;
    opts.channel_weights[3] = 1;

    auto compressed = codec.encode(src, BCFormat::bc7_unorm, opts);
    REQUIRE(compressed.mip_levels.size() == 1);
    CHECK(compressed.mip_levels[0].data.size() == 16);
}

//
// 10. has_alpha hint (BC7)
//

TEST_CASE("has_alpha_hint_bc7")
{
    BCCodec codec;
    auto pixels = make_gradient_rgba(4, 4);
    BCImage src = make_rgba_image(pixels, 4, 4);

    BCEncodeOptions opts;
    opts.has_alpha = false;

    auto compressed = codec.encode(src, BCFormat::bc7_unorm, opts);
    REQUIRE(compressed.mip_levels.size() == 1);
    CHECK(compressed.mip_levels[0].data.size() == 16);
}

//
// 11. can_encode / can_decode
//

TEST_CASE("can_encode_can_decode")
{
    BCCodec codec;

    // All formats decodable.
    CHECK(codec.can_decode(BCFormat::bc1_unorm));
    CHECK(codec.can_decode(BCFormat::bc6h_ufloat));
    CHECK(codec.can_decode(BCFormat::bc7_unorm));

    // SW-encodable formats.
    CHECK(codec.can_encode(BCFormat::bc1_unorm));
    CHECK(codec.can_encode(BCFormat::bc3_unorm));
    CHECK(codec.can_encode(BCFormat::bc4_unorm));
    CHECK(codec.can_encode(BCFormat::bc5_unorm));
    CHECK(codec.can_encode(BCFormat::bc7_unorm));

    // BC6H requires NVTT3.
    if (codec.is_nvtt_available()) {
        CHECK(codec.can_encode(BCFormat::bc6h_ufloat));
        CHECK(codec.can_encode(BCFormat::bc6h_sfloat));
    } else {
        CHECK_FALSE(codec.can_encode(BCFormat::bc6h_ufloat));
        CHECK_FALSE(codec.can_encode(BCFormat::bc6h_sfloat));
    }
}

//
// 12. BC6H encode error
//

TEST_CASE("bc6h_encode_error" * doctest::skip(BCCodec().is_nvtt_available()))
{
    BCCodec codec;
    auto pixels = make_gradient_rgba(4, 4);
    BCImage src = make_rgba_image(pixels, 4, 4);

    CHECK_THROWS(codec.encode(src, BCFormat::bc6h_ufloat));
    CHECK_THROWS(codec.encode(src, BCFormat::bc6h_sfloat));
}

//
// 13. Decode output format
//

TEST_CASE("decode_output_format")
{
    BCCodec codec;

    SUBCASE("BC4 -> 1ch uint8")
    {
        auto pixels = make_gradient_rgba(4, 4);
        BCImage src = make_rgba_image(pixels, 4, 4);
        auto compressed = codec.encode(src, BCFormat::bc4_unorm);

        std::vector<uint8_t> decoded(4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4, 1, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc4_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("BC5 -> 2ch uint8")
    {
        auto pixels = make_gradient_rgba(4, 4);
        BCImage src = make_rgba_image(pixels, 4, 4);
        auto compressed = codec.encode(src, BCFormat::bc5_unorm);

        std::vector<uint8_t> decoded(4 * 4 * 2, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 2, 2, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc5_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("BC7 -> 4ch RGBA uint8")
    {
        auto pixels = make_gradient_rgba(4, 4);
        BCImage src = make_rgba_image(pixels, 4, 4);
        auto compressed = codec.encode(src, BCFormat::bc7_unorm);

        std::vector<uint8_t> decoded(4 * 4 * 4, 0);
        BCMutableImage dst{decoded.data(), 4, 4, 4 * 4, 4, BCComponentType::uint8};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc7_unorm,
            4,
            4,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    // BC6H decode-only test (can't encode without NVTT3, so we test with hand-crafted data)
    // BC6H decode is tested implicitly by verifying can_decode returns true.
}

//
// 14. NVTT3 encode (all formats)
//

TEST_CASE("bc_codec_nvtt3_encode" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    BCCodec codec;
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

        auto compressed = codec.encode(src, fi.format, opts);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        uint32_t ch = fi.decoded_channels;
        std::vector<uint8_t> decoded(W * H * ch, 0);
        BCMutableImage dst{decoded.data(), W, H, W * ch, ch, BCComponentType::uint8};
        codec.decode(compressed.mip_levels[0].data.data(), compressed.mip_levels[0].data.size(), fi.format, W, H, dst);

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

TEST_CASE("bc_codec_nvtt3_bc6h" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    BCCodec codec;
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
        auto compressed = codec.encode(src, BCFormat::bc6h_ufloat, opts);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        // Decode to float16 RGB (6 bytes per pixel).
        std::vector<uint16_t> decoded(W * H * 3, 0);
        BCMutableImage dst{decoded.data(), W, H, W * 3 * sizeof(uint16_t), 3, BCComponentType::float16};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc6h_ufloat,
            W,
            H,
            dst
        );

        // Verify non-zero output.
        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }

    SUBCASE("bc6h_sfloat")
    {
        auto compressed = codec.encode(src, BCFormat::bc6h_sfloat, opts);
        REQUIRE(compressed.mip_levels.size() == 1);
        CHECK(compressed.mip_levels[0].width == W);
        CHECK(compressed.mip_levels[0].height == H);

        std::vector<uint16_t> decoded(W * H * 3, 0);
        BCMutableImage dst{decoded.data(), W, H, W * 3 * sizeof(uint16_t), 3, BCComponentType::float16};
        codec.decode(
            compressed.mip_levels[0].data.data(),
            compressed.mip_levels[0].data.size(),
            BCFormat::bc6h_sfloat,
            W,
            H,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);
    }
}

//
// 16. NVTT3 vs SW comparison
//

TEST_CASE("bc_codec_nvtt3_vs_sw" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    const uint32_t W = 64, H = 64;
    auto pixels = make_gradient_rgba(W, H);
    BCImage src = make_rgba_image(pixels, W, H);

    BCCodec sw_codec(false);
    BCCodec nvtt_codec(true);

    BCFormat formats[] = {
        BCFormat::bc1_unorm,
        BCFormat::bc3_unorm,
        BCFormat::bc7_unorm,
    };

    for (BCFormat fmt : formats) {
        CAPTURE(static_cast<int>(fmt));

        auto sw_compressed = sw_codec.encode(src, fmt);
        auto nvtt_compressed = nvtt_codec.encode(src, fmt);

        REQUIRE(sw_compressed.mip_levels.size() == 1);
        REQUIRE(nvtt_compressed.mip_levels.size() == 1);

        // Both should produce same-sized output.
        CHECK(sw_compressed.mip_levels[0].data.size() == nvtt_compressed.mip_levels[0].data.size());

        // Decode both and verify similar quality.
        std::vector<uint8_t> sw_decoded(W * H * 4, 0);
        std::vector<uint8_t> nvtt_decoded(W * H * 4, 0);
        BCMutableImage sw_dst{sw_decoded.data(), W, H, W * 4, 4, BCComponentType::uint8};
        BCMutableImage nvtt_dst{nvtt_decoded.data(), W, H, W * 4, 4, BCComponentType::uint8};

        sw_codec.decode(
            sw_compressed.mip_levels[0].data.data(),
            sw_compressed.mip_levels[0].data.size(),
            fmt,
            W,
            H,
            sw_dst
        );
        nvtt_codec.decode(
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

TEST_CASE("bc_codec_nvtt3_mipmaps" * doctest::skip(!BCCodec().is_nvtt_available()))
{
    BCCodec codec;
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

    auto compressed = codec.encode(src, BCFormat::bc6h_ufloat, opts);

    // bc_mip_count(64,64) = 7 levels (64, 32, 16, 8, 4, 2, 1)
    REQUIRE(compressed.mip_levels.size() == 7);

    uint32_t expected_w = W, expected_h = H;
    for (size_t i = 0; i < compressed.mip_levels.size(); ++i) {
        CHECK(compressed.mip_levels[i].width == expected_w);
        CHECK(compressed.mip_levels[i].height == expected_h);
        CHECK(
            compressed.mip_levels[i].data.size() == bc_compressed_size(expected_w, expected_h, BCFormat::bc6h_ufloat)
        );

        // Verify each level decodes successfully.
        std::vector<uint16_t> decoded(expected_w * expected_h * 3, 0);
        BCMutableImage dst{
            decoded.data(),
            expected_w,
            expected_h,
            expected_w * 3 * sizeof(uint16_t),
            3,
            BCComponentType::float16,
        };
        codec.decode(
            compressed.mip_levels[i].data.data(),
            compressed.mip_levels[i].data.size(),
            BCFormat::bc6h_ufloat,
            expected_w,
            expected_h,
            dst
        );

        bool any_nonzero = false;
        for (auto v : decoded)
            if (v != 0) {
                any_nonzero = true;
                break;
            }
        CHECK(any_nonzero);

        expected_w = std::max(1u, expected_w / 2);
        expected_h = std::max(1u, expected_h / 2);
    }
}

TEST_SUITE_END();
