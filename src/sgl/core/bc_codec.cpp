// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/bc_codec.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/config.h"
#include "sgl/core/error.h"

#include "sgl/math/float16.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <type_traits>

// nvtt
#if SGL_HAS_NVTT
#define NVTT_SHARED 1
#include <nvtt/nvtt.h>
#include <nvtt/nvtt_wrapper.h>
#endif

//  bcdec (header-only decoder)
#define BCDEC_BC4BC5_PRECISE
#define BCDEC_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif
#include <bcdec.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

//  rgbcx + bc7enc (SW encoders)
#include <rgbcx.h>
#include <bc7enc.h>

namespace sgl {

//
// One-time initialization
//

static std::once_flag g_sw_init_flag;

static void ensure_sw_init()
{
    std::call_once(
        g_sw_init_flag,
        []
        {
            rgbcx::init();
            bc7enc_compress_block_init();
        }
    );
}

//
// Helpers
//

static constexpr uint32_t BLOCK_DIM = 4;

/// Map BCEncodeQuality -> rgbcx level (0-18).
static uint32_t quality_to_rgbcx_level(BCEncodeQuality q)
{
    switch (q) {
    case BCEncodeQuality::fastest:
        return 0;
    case BCEncodeQuality::normal:
        return 10;
    case BCEncodeQuality::production:
        return 14;
    case BCEncodeQuality::highest:
        return 18;
    default:
        return 10;
    }
}

/// Map BCEncodeQuality -> bc7enc uber level (0-BC7ENC_MAX_UBER_LEVEL).
static uint32_t quality_to_bc7_uber(BCEncodeQuality q)
{
    switch (q) {
    case BCEncodeQuality::fastest:
        return 0;
    case BCEncodeQuality::normal:
        return 0;
    case BCEncodeQuality::production:
        return 2;
    case BCEncodeQuality::highest:
        return BC7ENC_MAX_UBER_LEVEL;
    default:
        return 0;
    }
}

/// Map BCEncodeQuality -> bc7enc max_partitions.
static uint32_t quality_to_bc7_partitions(BCEncodeQuality q)
{
    switch (q) {
    case BCEncodeQuality::fastest:
        return 0;
    case BCEncodeQuality::normal:
        return 32;
    case BCEncodeQuality::production:
        return BC7ENC_MAX_PARTITIONS;
    case BCEncodeQuality::highest:
        return BC7ENC_MAX_PARTITIONS;
    default:
        return 32;
    }
}

/// True if the BCFormat is a BC6H variant.
static bool is_bc6h(BCFormat f)
{
    return f == BCFormat::bc6h_ufloat || f == BCFormat::bc6h_sfloat;
}

/// Return the number of bytes per component for BCComponentType.
static uint32_t component_byte_size(BCComponentType t)
{
    switch (t) {
    case BCComponentType::uint8:
    case BCComponentType::int8:
        return 1;
    case BCComponentType::uint16:
    case BCComponentType::int16:
    case BCComponentType::float16:
        return 2;
    case BCComponentType::uint32:
    case BCComponentType::int32:
    case BCComponentType::float32:
        return 4;
    case BCComponentType::uint64:
    case BCComponentType::int64:
    case BCComponentType::float64:
        return 8;
    default:
        return 1;
    }
}

/// Read a single component from a pixel as a normalized float in [0,1] (for integer types) or raw float.
static float read_component_as_float(const uint8_t* pixel, uint32_t channel, BCComponentType type)
{
    switch (type) {
    case BCComponentType::uint8:
        return pixel[channel] / 255.0f;
    case BCComponentType::int8:
        return (reinterpret_cast<const int8_t*>(pixel)[channel] + 128.0f) / 255.0f;
    case BCComponentType::uint16:
        return reinterpret_cast<const uint16_t*>(pixel)[channel] / 65535.0f;
    case BCComponentType::int16:
        return (reinterpret_cast<const int16_t*>(pixel)[channel] + 32768.0f) / 65535.0f;
    case BCComponentType::float16:
        return math::float16_to_float32(reinterpret_cast<const uint16_t*>(pixel)[channel]);
    case BCComponentType::uint32:
        return static_cast<float>(reinterpret_cast<const uint32_t*>(pixel)[channel] / 4294967295.0);
    case BCComponentType::int32:
        return static_cast<float>((reinterpret_cast<const int32_t*>(pixel)[channel] + 2147483648.0) / 4294967295.0);
    case BCComponentType::float32:
        return reinterpret_cast<const float*>(pixel)[channel];
    case BCComponentType::float64:
        return static_cast<float>(reinterpret_cast<const double*>(pixel)[channel]);
    case BCComponentType::uint64:
        return static_cast<float>(reinterpret_cast<const uint64_t*>(pixel)[channel] / 18446744073709551615.0);
    case BCComponentType::int64:
        return static_cast<float>(
            (static_cast<double>(reinterpret_cast<const int64_t*>(pixel)[channel]) + 9223372036854775808.0)
            / 18446744073709551615.0
        );
    default:
        return 0.0f;
    }
}

/// Write a float value to a pixel component in the given type.
/// For integer types, the float is treated as normalized [0,1].
/// For float types, the value is written directly.
static void write_component_from_float(uint8_t* pixel, uint32_t channel, BCComponentType type, float value)
{
    switch (type) {
    case BCComponentType::uint8:
        pixel[channel] = static_cast<uint8_t>(std::clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
        break;
    case BCComponentType::int8:
        reinterpret_cast<int8_t*>(pixel)[channel]
            = static_cast<int8_t>(std::clamp(value * 255.0f - 128.0f + 0.5f, -128.0f, 127.0f));
        break;
    case BCComponentType::uint16:
        reinterpret_cast<uint16_t*>(pixel)[channel]
            = static_cast<uint16_t>(std::clamp(value * 65535.0f + 0.5f, 0.0f, 65535.0f));
        break;
    case BCComponentType::int16:
        reinterpret_cast<int16_t*>(pixel)[channel]
            = static_cast<int16_t>(std::clamp(value * 65535.0f - 32768.0f + 0.5f, -32768.0f, 32767.0f));
        break;
    case BCComponentType::float16:
        reinterpret_cast<uint16_t*>(pixel)[channel] = math::float32_to_float16(value);
        break;
    case BCComponentType::uint32:
        reinterpret_cast<uint32_t*>(pixel)[channel]
            = static_cast<uint32_t>(std::clamp(static_cast<double>(value) * 4294967295.0 + 0.5, 0.0, 4294967295.0));
        break;
    case BCComponentType::int32:
        reinterpret_cast<int32_t*>(pixel)[channel] = static_cast<int32_t>(
            std::clamp(static_cast<double>(value) * 4294967295.0 - 2147483648.0 + 0.5, -2147483648.0, 2147483647.0)
        );
        break;
    case BCComponentType::float32:
        reinterpret_cast<float*>(pixel)[channel] = value;
        break;
    case BCComponentType::float64:
        reinterpret_cast<double*>(pixel)[channel] = static_cast<double>(value);
        break;
    default:
        break;
    }
}

/// Extract a 4x4 RGBA8 block from the source image, padding if the block
/// extends beyond the image boundary (edge clamp).
static void extract_rgba8_block(
    const BCImage& src,
    uint32_t block_x,
    uint32_t block_y,
    uint8_t out_block[BLOCK_DIM * BLOCK_DIM * 4]
)
{
    uint32_t src_pixel_bytes = src.channel_count * component_byte_size(src.component_type);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src.data);

    for (uint32_t by = 0; by < BLOCK_DIM; ++by) {
        uint32_t py = std::min(block_y * BLOCK_DIM + by, src.height - 1);
        for (uint32_t bx = 0; bx < BLOCK_DIM; ++bx) {
            uint32_t px = std::min(block_x * BLOCK_DIM + bx, src.width - 1);
            const uint8_t* pixel = src_bytes + py * src.row_pitch + px * src_pixel_bytes;
            uint8_t* dst = out_block + (by * BLOCK_DIM + bx) * 4;

            if (src.component_type == BCComponentType::uint8) {
                uint32_t nc = std::min(src.channel_count, 4u);
                for (uint32_t c = 0; c < nc; ++c)
                    dst[c] = pixel[c];
                // Fill missing channels.
                for (uint32_t c = nc; c < 4; ++c)
                    dst[c] = (c == 3) ? 255 : 0;
            } else {
                // For non-uint8, read via type-aware helper and quantize to [0,255].
                uint32_t nc = std::min(src.channel_count, 4u);
                for (uint32_t c = 0; c < nc; ++c) {
                    float v = read_component_as_float(pixel, c, src.component_type);
                    dst[c] = static_cast<uint8_t>(std::clamp(v * 255.0f + 0.5f, 0.0f, 255.0f));
                }
                for (uint32_t c = nc; c < 4; ++c)
                    dst[c] = (c == 3) ? 255 : 0;
            }
        }
    }
}

//
// Encoding helpers (per-format)
//

static void encode_image_bc1(const BCImage& src, uint8_t* dst, BCEncodeQuality quality)
{
    uint32_t level = quality_to_rgbcx_level(quality);
    uint32_t blocks_x = (src.width + 3) / 4;
    uint32_t blocks_y = (src.height + 3) / 4;
    uint8_t block_pixels[BLOCK_DIM * BLOCK_DIM * 4];

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            extract_rgba8_block(src, bx, by, block_pixels);
            rgbcx::encode_bc1(level, dst, block_pixels, false, false);
            dst += 8;
        }
    }
}

static void encode_image_bc2(const BCImage& src, uint8_t* dst, BCEncodeQuality quality)
{
    uint32_t level = quality_to_rgbcx_level(quality);
    uint32_t blocks_x = (src.width + 3) / 4;
    uint32_t blocks_y = (src.height + 3) / 4;
    uint8_t block_pixels[BLOCK_DIM * BLOCK_DIM * 4];

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            extract_rgba8_block(src, bx, by, block_pixels);

            // BC2: first 8 bytes = explicit 4-bit alpha per pixel.
            uint8_t alpha_block[8] = {};
            for (uint32_t i = 0; i < 16; ++i) {
                uint8_t a4 = static_cast<uint8_t>((block_pixels[i * 4 + 3] * 15 + 127) / 255);
                if (i % 2 == 0)
                    alpha_block[i / 2] = a4;
                else
                    alpha_block[i / 2] |= static_cast<uint8_t>(a4 << 4);
            }
            std::memcpy(dst, alpha_block, 8);

            // BC2: next 8 bytes = BC1 color block (no alpha).
            rgbcx::encode_bc1(level, dst + 8, block_pixels, false, false);
            dst += 16;
        }
    }
}

static void encode_image_bc3(const BCImage& src, uint8_t* dst, BCEncodeQuality quality)
{
    uint32_t level = quality_to_rgbcx_level(quality);
    uint32_t blocks_x = (src.width + 3) / 4;
    uint32_t blocks_y = (src.height + 3) / 4;
    uint8_t block_pixels[BLOCK_DIM * BLOCK_DIM * 4];

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            extract_rgba8_block(src, bx, by, block_pixels);
            rgbcx::encode_bc3(level, dst, block_pixels);
            dst += 16;
        }
    }
}

static void encode_image_bc4(const BCImage& src, uint8_t* dst)
{
    uint32_t blocks_x = (src.width + 3) / 4;
    uint32_t blocks_y = (src.height + 3) / 4;
    uint8_t block_pixels[BLOCK_DIM * BLOCK_DIM * 4];

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            extract_rgba8_block(src, bx, by, block_pixels);
            // rgbcx::encode_bc4 takes stride=4 (RGBA pixels), encodes channel 0.
            rgbcx::encode_bc4(dst, block_pixels, 4);
            dst += 8;
        }
    }
}

static void encode_image_bc5(const BCImage& src, uint8_t* dst)
{
    uint32_t blocks_x = (src.width + 3) / 4;
    uint32_t blocks_y = (src.height + 3) / 4;
    uint8_t block_pixels[BLOCK_DIM * BLOCK_DIM * 4];

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            extract_rgba8_block(src, bx, by, block_pixels);
            // rgbcx::encode_bc5 takes chan0=0, chan1=1, stride=4.
            rgbcx::encode_bc5(dst, block_pixels, 0, 1, 4);
            dst += 16;
        }
    }
}

static void encode_image_bc7(const BCImage& src, uint8_t* dst, const BCEncodeOptions& options)
{
    uint32_t blocks_x = (src.width + 3) / 4;
    uint32_t blocks_y = (src.height + 3) / 4;
    uint8_t block_pixels[BLOCK_DIM * BLOCK_DIM * 4];

    bc7enc_compress_block_params params;
    bc7enc_compress_block_params_init(&params);
    params.m_uber_level = quality_to_bc7_uber(options.quality);
    params.m_max_partitions = quality_to_bc7_partitions(options.quality);
    params.m_force_alpha = options.has_alpha;
    for (int i = 0; i < 4; ++i)
        params.m_weights[i] = options.channel_weights[i];

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            extract_rgba8_block(src, bx, by, block_pixels);
            bc7enc_compress_block(dst, block_pixels, &params);
            dst += 16;
        }
    }
}

//
// Decode helpers (per-format)
//

/// Write decoded pixels (one 4x4 block) into the destination image, clipping
/// to image boundaries. Converts between decoded and destination component types.
static void copy_block_to_dst(
    const BCMutableImage& dst,
    uint32_t block_x,
    uint32_t block_y,
    const void* decoded_block,
    uint32_t decoded_pitch,      // bytes per row of the decoded 4x4 block
    uint32_t decoded_channels,   // number of channels in decoded data
    BCComponentType decoded_type // component type of decoded data
)
{
    const uint8_t* src = static_cast<const uint8_t*>(decoded_block);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst.data);
    uint32_t dst_pixel_bytes = dst.channel_count * component_byte_size(dst.component_type);
    uint32_t decoded_pixel_bytes = decoded_channels * component_byte_size(decoded_type);

    // Fast path: if types and channel counts match, use memcpy.
    bool can_memcpy = (decoded_type == dst.component_type) && (decoded_channels == dst.channel_count);

    for (uint32_t by = 0; by < BLOCK_DIM; ++by) {
        uint32_t py = block_y * BLOCK_DIM + by;
        if (py >= dst.height)
            break;
        for (uint32_t bx = 0; bx < BLOCK_DIM; ++bx) {
            uint32_t px = block_x * BLOCK_DIM + bx;
            if (px >= dst.width)
                break;
            const uint8_t* sp = src + by * decoded_pitch + bx * decoded_pixel_bytes;
            uint8_t* dp = dst_bytes + py * dst.row_pitch + px * dst_pixel_bytes;

            if (can_memcpy) {
                std::memcpy(dp, sp, dst_pixel_bytes);
            } else {
                // Convert per-channel through float.
                uint32_t nc = std::min(decoded_channels, dst.channel_count);
                for (uint32_t c = 0; c < nc; ++c) {
                    float v = read_component_as_float(sp, c, decoded_type);
                    write_component_from_float(dp, c, dst.component_type, v);
                }
                // Fill missing destination channels with defaults (0 for RGB, 1 for alpha).
                for (uint32_t c = nc; c < dst.channel_count; ++c) {
                    float v = (c == 3) ? 1.0f : 0.0f;
                    write_component_from_float(dp, c, dst.component_type, v);
                }
            }
        }
    }
}

#if SGL_HAS_NVTT

//
// NVTT3 format / quality mappings
//

static NvttFormat bc_format_to_nvtt(BCFormat f)
{
    switch (f) {
    case BCFormat::bc1_unorm:
    case BCFormat::bc1_unorm_srgb:
        return NVTT_Format_BC1;
    case BCFormat::bc2_unorm:
    case BCFormat::bc2_unorm_srgb:
        return NVTT_Format_BC2;
    case BCFormat::bc3_unorm:
    case BCFormat::bc3_unorm_srgb:
        return NVTT_Format_BC3;
    case BCFormat::bc4_unorm:
        return NVTT_Format_BC4;
    case BCFormat::bc4_snorm:
        return NVTT_Format_BC4S;
    case BCFormat::bc5_unorm:
        return NVTT_Format_BC5;
    case BCFormat::bc5_snorm:
        return NVTT_Format_BC5S;
    case BCFormat::bc6h_ufloat:
        return NVTT_Format_BC6U;
    case BCFormat::bc6h_sfloat:
        return NVTT_Format_BC6S;
    case BCFormat::bc7_unorm:
    case BCFormat::bc7_unorm_srgb:
        return NVTT_Format_BC7;
    default:
        return NVTT_Format_BC1;
    }
}

static NvttQuality quality_to_nvtt(BCEncodeQuality q)
{
    switch (q) {
    case BCEncodeQuality::fastest:
        return NVTT_Quality_Fastest;
    case BCEncodeQuality::normal:
        return NVTT_Quality_Normal;
    case BCEncodeQuality::production:
        return NVTT_Quality_Production;
    case BCEncodeQuality::highest:
        return NVTT_Quality_Highest;
    default:
        return NVTT_Quality_Normal;
    }
}

/// Map ReconstructionFilter to NvttMipmapFilter.
static NvttMipmapFilter mip_filter_to_nvtt(const ReconstructionFilter& filter)
{
    return std::visit(
        [](auto&& f) -> NvttMipmapFilter
        {
            using T = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<T, BoxFilter>)
                return NVTT_MipmapFilter_Box;
            else if constexpr (std::is_same_v<T, TentFilter>)
                return NVTT_MipmapFilter_Triangle;
            else if constexpr (std::is_same_v<T, MitchellFilter>)
                return NVTT_MipmapFilter_Mitchell;
            else
                // GaussianFilter and LanczosFilter have no exact NVTT equivalent; use Kaiser.
                return NVTT_MipmapFilter_Kaiser;
        },
        filter
    );
}

//
// NVTT3 encode helpers
//

/// Encode a single mip level via NVTT3's low-level CPU API.
static void nvtt_encode_single(
    const float* rgba_data,
    uint32_t width,
    uint32_t height,
    uint32_t channel_count,
    BCFormat format,
    const BCEncodeOptions& options,
    std::vector<uint8_t>& out_data
)
{
    NvttRefImage ref_image{};
    ref_image.data = rgba_data;
    ref_image.width = static_cast<int>(width);
    ref_image.height = static_cast<int>(height);
    ref_image.depth = 1;
    ref_image.num_channels = static_cast<int>(std::min(channel_count, 4u));
    ref_image.channel_swizzle[0] = NVTT_ChannelOrder_Red;
    ref_image.channel_swizzle[1] = NVTT_ChannelOrder_Green;
    ref_image.channel_swizzle[2] = NVTT_ChannelOrder_Blue;
    ref_image.channel_swizzle[3] = NVTT_ChannelOrder_Alpha;
    ref_image.channel_interleave = NVTT_True;

    unsigned num_tiles = 0;
    NvttCPUInputBuffer* input_buffer = nvttCreateCPUInputBuffer(
        &ref_image,
        NVTT_ValueType_FLOAT32,
        1,    // numImages
        4,    // tile_w (BC block width)
        4,    // tile_h (BC block height)
        static_cast<float>(options.channel_weights[0]),
        static_cast<float>(options.channel_weights[1]),
        static_cast<float>(options.channel_weights[2]),
        static_cast<float>(options.channel_weights[3]),
        nullptr, // timing_context
        &num_tiles
    );
    SGL_CHECK(input_buffer != nullptr, "NVTT3: failed to create CPU input buffer");

    NvttEncodeSettings settings{};
    settings.sType = 1; // NVTT_EncodeSettings_Version_1
    settings.format = bc_format_to_nvtt(format);
    settings.quality = quality_to_nvtt(options.quality);
    settings.rgb_pixel_type = is_bc6h(format) ? NVTT_PixelType_Float : NVTT_PixelType_UnsignedNorm;
    settings.timing_context = nullptr;
    settings.encode_flags = options.has_alpha ? static_cast<uint32_t>(NVTT_EncodeFlags_None)
                                              : static_cast<uint32_t>(NVTT_EncodeFlags_Opaque);

    size_t compressed_bytes = bc_compressed_size(width, height, format);
    out_data.resize(compressed_bytes);

    NvttBoolean ok = nvttEncodeCPU(input_buffer, out_data.data(), &settings);
    nvttDestroyCPUInputBuffer(input_buffer);

    SGL_CHECK(ok == NVTT_True, "NVTT3: encoding failed");
}

/// Convert a BCImage to interleaved float32 RGBA.
static std::vector<float> bc_image_to_float32_rgba(const BCImage& src)
{
    uint32_t pixel_count = src.width * src.height;
    std::vector<float> result(pixel_count * 4, 0.0f);
    uint32_t src_pixel_bytes = src.channel_count * component_byte_size(src.component_type);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src.data);

    for (uint32_t y = 0; y < src.height; ++y) {
        for (uint32_t x = 0; x < src.width; ++x) {
            const uint8_t* pixel = src_bytes + y * src.row_pitch + x * src_pixel_bytes;
            float* dst = result.data() + (y * src.width + x) * 4;

            uint32_t nc = std::min(src.channel_count, 4u);
            for (uint32_t c = 0; c < nc; ++c) {
                float v = read_component_as_float(pixel, c, src.component_type);
                // For integer types, read_component_as_float returns normalized [0,1] which is correct for NVTT.
                // For float types (float16/32/64), it returns the raw value which is also correct.
                dst[c] = v;
            }
            // Fill missing channels: RGB default 0, Alpha default 1.
            for (uint32_t c = nc; c < 4; ++c)
                dst[c] = (c == 3) ? 1.0f : 0.0f;
        }
    }
    return result;
}

/// Convert NVTT3 planar surface data to interleaved float32 RGBA.
static std::vector<float> nvtt_planar_to_interleaved(const float* planar_data, uint32_t width, uint32_t height)
{
    uint32_t pixel_count = width * height;
    std::vector<float> interleaved(pixel_count * 4);
    // NVTT Surface stores data as: R plane (w*h floats), G plane, B plane, A plane.
    const float* r_plane = planar_data;
    const float* g_plane = planar_data + pixel_count;
    const float* b_plane = planar_data + pixel_count * 2;
    const float* a_plane = planar_data + pixel_count * 3;
    for (uint32_t i = 0; i < pixel_count; ++i) {
        interleaved[i * 4 + 0] = r_plane[i];
        interleaved[i * 4 + 1] = g_plane[i];
        interleaved[i * 4 + 2] = b_plane[i];
        interleaved[i * 4 + 3] = a_plane[i];
    }
    return interleaved;
}

/// Full NVTT3 encode path (handles mipmaps internally).
static BCCompressedImage nvtt_encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options)
{
    BCCompressedImage result;
    result.format = format;

    // Convert source to interleaved float32 RGBA for NVTT3.
    std::vector<float> float_data = bc_image_to_float32_rgba(src);

    // Encode mip level 0.
    {
        BCCompressedMip mip;
        mip.width = src.width;
        mip.height = src.height;
        nvtt_encode_single(float_data.data(), src.width, src.height, 4, format, options, mip.data);
        result.mip_levels.push_back(std::move(mip));
    }

    if (!options.generate_mipmaps || (src.width <= 1 && src.height <= 1))
        return result;

    // Use NVTT3 Surface for mipmap generation.
    NvttSurface* surface = nvttCreateSurface();
    SGL_CHECK(surface != nullptr, "NVTT3: failed to create surface");

    NvttBoolean ok = nvttSurfaceSetImageData(
        surface,
        NVTT_InputFormat_RGBA_32F,
        static_cast<int>(src.width),
        static_cast<int>(src.height),
        1,
        float_data.data(),
        NVTT_False, // not a reference - copy the data
        nullptr     // timing_context
    );

    if (ok != NVTT_True) {
        nvttDestroySurface(surface);
        SGL_THROW("NVTT3: failed to set surface image data");
    }

    uint32_t mip_w = src.width;
    uint32_t mip_h = src.height;
    NvttMipmapFilter nvtt_mip_filter = mip_filter_to_nvtt(options.mip_filter);
    while (mip_w > 1 || mip_h > 1) {
        ok = nvttSurfaceBuildNextMipmapDefaults(surface, nvtt_mip_filter, 1, nullptr);
        if (ok != NVTT_True)
            break;

        mip_w = static_cast<uint32_t>(nvttSurfaceWidth(surface));
        mip_h = static_cast<uint32_t>(nvttSurfaceHeight(surface));

        // Get planar data from surface and convert to interleaved.
        float* planar = nvttSurfaceData(surface);
        std::vector<float> interleaved = nvtt_planar_to_interleaved(planar, mip_w, mip_h);

        BCCompressedMip mip;
        mip.width = mip_w;
        mip.height = mip_h;
        nvtt_encode_single(interleaved.data(), mip_w, mip_h, 4, format, options, mip.data);
        result.mip_levels.push_back(std::move(mip));
    }

    nvttDestroySurface(surface);
    return result;
}

//
// BCCodecImpl - abstract interface with SW and NVTT implementations
//

struct BCCodecImpl {
    virtual ~BCCodecImpl() = default;
    virtual BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options) = 0;
    virtual bool can_encode(BCFormat format) const = 0;
    virtual bool can_decode(BCFormat format) const = 0;

    /// Default decode using bcdec (header-only). Subclasses may override.
    virtual void
    decode(const void* data, size_t size, BCFormat format, uint32_t width, uint32_t height, const BCMutableImage& dst);
};

/// Software encoder using rgbcx (BC1-5) and bc7enc (BC7).
struct BCCodecSWImpl : BCCodecImpl {
    BCCodecSWImpl() { ensure_sw_init(); }

    bool can_encode(BCFormat format) const override { return !is_bc6h(format); }

    bool can_decode(BCFormat) const override { return true; }

    BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options) override
    {
        if (is_bc6h(format))
            SGL_THROW("BCCodecSW::encode: BC6H encoding requires NVTT3");

        BCCompressedImage result;
        result.format = format;

        // Build the list of mip levels to encode.
        struct MipLevel {
            BCImage image;
            ref<Bitmap> owned_bitmap; // keeps data alive for generated mips
        };

        std::vector<MipLevel> levels;

        // Level 0 is always the source.
        levels.push_back({src, nullptr});

        if (options.generate_mipmaps && (src.width > 1 || src.height > 1)) {
            // Convert source to a Bitmap for mip generation.
            Bitmap::PixelFormat pf;
            switch (src.channel_count) {
            case 1:
                pf = Bitmap::PixelFormat::r;
                break;
            case 2:
                pf = Bitmap::PixelFormat::rg;
                break;
            case 3:
                pf = Bitmap::PixelFormat::rgb;
                break;
            default:
                pf = Bitmap::PixelFormat::rgba;
                break;
            }

            // Bitmap::resample requires float16 or float32.
            // Convert to float32 for mip generation.
            // Non-owning wrap of source data - safe because src outlives this scope.
            auto src_bmp = make_ref<Bitmap>(
                pf,
                src.component_type,
                src.width,
                src.height,
                src.channel_count,
                std::vector<std::string>{},
                const_cast<void*>(src.data)
            );

            ref<Bitmap> float_bmp;
            if (src.component_type != Bitmap::ComponentType::float32
                && src.component_type != Bitmap::ComponentType::float16) {
                float_bmp = src_bmp->convert(pf, Bitmap::ComponentType::float32, false);
            } else {
                float_bmp = src_bmp;
            }

            // Generate mip chain by iteratively downsampling with Bitmap::resample.
            ref<Bitmap> current = float_bmp;
            uint32_t mip_w = src.width;
            uint32_t mip_h = src.height;
            while (mip_w > 1 || mip_h > 1) {
                mip_w = std::max(mip_w / 2, 1u);
                mip_h = std::max(mip_h / 2, 1u);
                current = current->resample(mip_w, mip_h, options.mip_filter);

                // Convert back to source component type if needed for encoding.
                ref<Bitmap> enc_bmp;
                if (src.component_type != Bitmap::ComponentType::float32
                    && src.component_type != Bitmap::ComponentType::float16) {
                    enc_bmp = current->convert(pf, src.component_type, false);
                } else {
                    enc_bmp = current;
                }
                BCImage mip_img = bc_image_from_bitmap(*enc_bmp);
                levels.push_back({mip_img, enc_bmp});
            }
        }

        // Encode each level.
        for (auto& lvl : levels) {
            size_t compressed_bytes = bc_compressed_size(lvl.image.width, lvl.image.height, format);
            BCCompressedMip mip;
            mip.width = lvl.image.width;
            mip.height = lvl.image.height;
            mip.data.resize(compressed_bytes);

            switch (format) {
            case BCFormat::bc1_unorm:
            case BCFormat::bc1_unorm_srgb:
                encode_image_bc1(lvl.image, mip.data.data(), options.quality);
                break;
            case BCFormat::bc2_unorm:
            case BCFormat::bc2_unorm_srgb:
                encode_image_bc2(lvl.image, mip.data.data(), options.quality);
                break;
            case BCFormat::bc3_unorm:
            case BCFormat::bc3_unorm_srgb:
                encode_image_bc3(lvl.image, mip.data.data(), options.quality);
                break;
            case BCFormat::bc4_unorm:
            case BCFormat::bc4_snorm:
                encode_image_bc4(lvl.image, mip.data.data());
                break;
            case BCFormat::bc5_unorm:
            case BCFormat::bc5_snorm:
                encode_image_bc5(lvl.image, mip.data.data());
                break;
            case BCFormat::bc7_unorm:
            case BCFormat::bc7_unorm_srgb:
                encode_image_bc7(lvl.image, mip.data.data(), options);
                break;
            default:
                SGL_THROW("BCCodecSW::encode: unsupported format");
            }

            result.mip_levels.push_back(std::move(mip));
        }

        return result;
    }
};

/// NVTT3 encoder - delegates to the linked NVTT3 library.
struct BCCodecNVTTImpl : BCCodecImpl {
    bool can_encode(BCFormat) const override { return true; }

    bool can_decode(BCFormat) const override { return true; }

    BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options) override
    {
        return nvtt_encode(src, format, options);
    }
};

#endif // SGL_HAS_NVTT

//
// BCCodec public interface
//

BCCodec::BCCodec(bool prefer_nvtt)
{
#if SGL_HAS_NVTT
    if (prefer_nvtt)
        m_impl = std::make_unique<BCCodecNVTTImpl>();
    else
        m_impl = std::make_unique<BCCodecSWImpl>();
#else
    (void)prefer_nvtt;
    m_impl = std::make_unique<BCCodecSWImpl>();
#endif
}

BCCodec::~BCCodec() = default;

bool BCCodec::is_nvtt_available()
{
    return SGL_HAS_NVTT;
}

bool BCCodec::can_encode(BCFormat format) const
{
    return m_impl->can_encode(format);
}

bool BCCodec::can_decode(BCFormat format) const
{
    return m_impl->can_decode(format);
}

//  Encode

BCCompressedImage BCCodec::encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options)
{
    SGL_CHECK(src.data != nullptr, "BCCodec::encode: source data is null");
    SGL_CHECK(src.width > 0 && src.height > 0, "BCCodec::encode: invalid source dimensions");
    SGL_CHECK(src.channel_count >= 1 && src.channel_count <= 4, "BCCodec::encode: channel_count must be 1-4");
    SGL_CHECK(
        src.row_pitch >= src.width * src.channel_count * component_byte_size(src.component_type),
        "BCCodec::encode: row_pitch is too small for the given width, channel count, and component type"
    );
    for (int i = 0; i < 4; ++i)
        SGL_CHECK(options.channel_weights[i] > 0, "BCCodec::encode: channel_weights must be non-zero");

    return m_impl->encode(src, format, options);
}

//  Decode

void BCCodec::decode(
    const void* data,
    size_t size,
    BCFormat format,
    uint32_t width,
    uint32_t height,
    const BCMutableImage& dst
)
{
    m_impl->decode(data, size, format, width, height, dst);
}

//  BCCodecImpl default decode (bcdec)

void BCCodecImpl::decode(
    const void* data,
    size_t size,
    BCFormat format,
    uint32_t width,
    uint32_t height,
    const BCMutableImage& dst
)
{
    SGL_CHECK(data != nullptr, "BCCodec::decode: source data is null");
    SGL_CHECK(dst.data != nullptr, "BCCodec::decode: destination data is null");
    SGL_CHECK(width > 0 && height > 0, "BCCodec::decode: invalid dimensions");
    SGL_CHECK(size >= bc_compressed_size(width, height, format), "BCCodec::decode: insufficient source data");
    SGL_CHECK(dst.width >= width && dst.height >= height, "BCCodec::decode: destination too small");

    uint32_t blocks_x = (width + 3) / 4;
    uint32_t blocks_y = (height + 3) / 4;
    uint32_t bpb = bc_format_bytes_per_block(format);
    const uint8_t* src = static_cast<const uint8_t*>(data);

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            const uint8_t* block = src + (static_cast<size_t>(by) * blocks_x + bx) * bpb;

            switch (format) {
            case BCFormat::bc1_unorm:
            case BCFormat::bc1_unorm_srgb: {
                // Output: RGBA uint8 (4 bytes per pixel), 4x4 block.
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc1(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc2_unorm:
            case BCFormat::bc2_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc2(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc3_unorm:
            case BCFormat::bc3_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc3(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc4_unorm: {
                // Output: R uint8 (1 byte per pixel).
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM];
                bcdec_bc4(block, decoded, BLOCK_DIM, 0);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM, 1, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc4_snorm: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM];
                bcdec_bc4(block, decoded, BLOCK_DIM, 1);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM, 1, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc5_unorm: {
                // Output: RG uint8 (2 bytes per pixel).
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 2];
                bcdec_bc5(block, decoded, BLOCK_DIM * 2, 0);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 2, 2, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc5_snorm: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 2];
                bcdec_bc5(block, decoded, BLOCK_DIM * 2, 1);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 2, 2, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc6h_ufloat: {
                // Output: RGB float16 (6 bytes per pixel) - lossless via bcdec_bc6h_half().
                // NOTE: bcdec_bc6h_half uses unsigned short* internally, so destinationPitch is in elements, not bytes.
                uint16_t decoded[BLOCK_DIM * BLOCK_DIM * 3];
                bcdec_bc6h_half(block, decoded, BLOCK_DIM * 3, 0);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 3 * sizeof(uint16_t), 3, BCComponentType::float16);
                break;
            }
            case BCFormat::bc6h_sfloat: {
                uint16_t decoded[BLOCK_DIM * BLOCK_DIM * 3];
                bcdec_bc6h_half(block, decoded, BLOCK_DIM * 3, 1);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 3 * sizeof(uint16_t), 3, BCComponentType::float16);
                break;
            }
            case BCFormat::bc7_unorm:
            case BCFormat::bc7_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc7(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            }
        }
    }
}

} // namespace sgl
