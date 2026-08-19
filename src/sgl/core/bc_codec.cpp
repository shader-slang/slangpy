// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/bc_codec.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/config.h"
#include "sgl/core/error.h"

#include "sgl/math/float16.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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

/// True if the BC format stores signed-normalized values.
static bool is_snorm(BCFormat f)
{
    return f == BCFormat::bc4_snorm || f == BCFormat::bc5_snorm;
}

/// True if the BC format stores sRGB-encoded color values.
static bool is_srgb(BCFormat f)
{
    return f == BCFormat::bc1_unorm_srgb || f == BCFormat::bc2_unorm_srgb || f == BCFormat::bc3_unorm_srgb
        || f == BCFormat::bc7_unorm_srgb;
}

/// True if the value names a supported BC format.
static bool is_valid_bc_format(BCFormat f)
{
    return bc_format_bytes_per_block(f) != 0;
}

/// Return the number of bytes per component for BCComponentType.
static uint32_t component_byte_size(BCComponentType t)
{
    return static_cast<uint32_t>(DataStruct::type_size(t));
}

template<typename T>
static T read_unaligned_component(const uint8_t* pixel, uint32_t channel)
{
    T value;
    std::memcpy(&value, pixel + static_cast<size_t>(channel) * sizeof(T), sizeof(T));
    return value;
}

template<typename T>
static void write_unaligned_component(uint8_t* pixel, uint32_t channel, T value)
{
    std::memcpy(pixel + static_cast<size_t>(channel) * sizeof(T), &value, sizeof(T));
}

/// Read a component as unsigned-normalized [0,1], signed-normalized [-1,1], or a raw floating-point value.
static float read_component_as_float(const uint8_t* pixel, uint32_t channel, BCComponentType type)
{
    switch (type) {
    case BCComponentType::uint8:
        return pixel[channel] / 255.0f;
    case BCComponentType::int8:
        return std::max(-1.0f, reinterpret_cast<const int8_t*>(pixel)[channel] / 127.0f);
    case BCComponentType::uint16:
        return read_unaligned_component<uint16_t>(pixel, channel) / 65535.0f;
    case BCComponentType::int16:
        return std::max(-1.0f, read_unaligned_component<int16_t>(pixel, channel) / 32767.0f);
    case BCComponentType::float16:
        return math::float16_to_float32(read_unaligned_component<uint16_t>(pixel, channel));
    case BCComponentType::uint32:
        return static_cast<float>(read_unaligned_component<uint32_t>(pixel, channel) / 4294967295.0);
    case BCComponentType::int32:
        return std::max(-1.0f, static_cast<float>(read_unaligned_component<int32_t>(pixel, channel) / 2147483647.0));
    case BCComponentType::float32:
        return read_unaligned_component<float>(pixel, channel);
    case BCComponentType::float64:
        return static_cast<float>(read_unaligned_component<double>(pixel, channel));
    case BCComponentType::uint64:
        return static_cast<float>(read_unaligned_component<uint64_t>(pixel, channel) / 18446744073709551615.0);
    case BCComponentType::int64:
        return std::max(
            -1.0f,
            static_cast<float>(
                static_cast<long double>(read_unaligned_component<int64_t>(pixel, channel))
                / static_cast<long double>(std::numeric_limits<int64_t>::max())
            )
        );
    default:
        SGL_THROW("read_component_as_float: unsupported component type");
    }
}

/// Write a float value to a pixel component in the given type.
/// For unsigned integer types, the float is treated as normalized [0,1].
/// For signed integer types, the float is treated as normalized [-1,1].
/// For float types, the value is written directly.
static void write_component_from_float(uint8_t* pixel, uint32_t channel, BCComponentType type, float value)
{
    switch (type) {
    case BCComponentType::uint8:
        pixel[channel] = static_cast<uint8_t>(std::clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
        break;
    case BCComponentType::int8:
        reinterpret_cast<int8_t*>(pixel)[channel]
            = static_cast<int8_t>(std::clamp(std::round(value * 127.0f), -128.0f, 127.0f));
        break;
    case BCComponentType::uint16:
        write_unaligned_component<uint16_t>(
            pixel,
            channel,
            static_cast<uint16_t>(std::clamp(value * 65535.0f + 0.5f, 0.0f, 65535.0f))
        );
        break;
    case BCComponentType::int16:
        write_unaligned_component<int16_t>(
            pixel,
            channel,
            static_cast<int16_t>(std::clamp(std::round(value * 32767.0f), -32768.0f, 32767.0f))
        );
        break;
    case BCComponentType::float16:
        write_unaligned_component<uint16_t>(pixel, channel, math::float32_to_float16(value));
        break;
    case BCComponentType::uint32:
        write_unaligned_component<uint32_t>(
            pixel,
            channel,
            static_cast<uint32_t>(std::clamp(static_cast<double>(value) * 4294967295.0 + 0.5, 0.0, 4294967295.0))
        );
        break;
    case BCComponentType::int32:
        write_unaligned_component<int32_t>(
            pixel,
            channel,
            static_cast<int32_t>(
                std::clamp(std::round(static_cast<double>(value) * 2147483647.0), -2147483648.0, 2147483647.0)
            )
        );
        break;
    case BCComponentType::float32:
        write_unaligned_component<float>(pixel, channel, value);
        break;
    case BCComponentType::float64:
        write_unaligned_component<double>(pixel, channel, static_cast<double>(value));
        break;
    case BCComponentType::uint64:
        if (value <= 0.0f)
            write_unaligned_component<uint64_t>(pixel, channel, 0);
        else if (value >= 1.0f)
            write_unaligned_component<uint64_t>(pixel, channel, std::numeric_limits<uint64_t>::max());
        else
            write_unaligned_component<uint64_t>(
                pixel,
                channel,
                static_cast<uint64_t>(
                    static_cast<long double>(value) * static_cast<long double>(std::numeric_limits<uint64_t>::max())
                )
            );
        break;
    case BCComponentType::int64:
        if (value <= -1.0f)
            write_unaligned_component<int64_t>(pixel, channel, std::numeric_limits<int64_t>::min());
        else if (value >= 1.0f)
            write_unaligned_component<int64_t>(pixel, channel, std::numeric_limits<int64_t>::max());
        else
            write_unaligned_component<int64_t>(
                pixel,
                channel,
                static_cast<int64_t>(std::round(
                    static_cast<long double>(value) * static_cast<long double>(std::numeric_limits<int64_t>::max())
                ))
            );
        break;
    default:
        SGL_THROW("write_component_from_float: unsupported component type");
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
    size_t src_pixel_bytes = static_cast<size_t>(src.channel_count) * component_byte_size(src.component_type);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src.data);

    for (uint32_t by = 0; by < BLOCK_DIM; ++by) {
        uint32_t py = std::min(block_y * BLOCK_DIM + by, src.height - 1);
        for (uint32_t bx = 0; bx < BLOCK_DIM; ++bx) {
            uint32_t px = std::min(block_x * BLOCK_DIM + bx, src.width - 1);
            const uint8_t* pixel
                = src_bytes + static_cast<size_t>(py) * src.row_pitch + static_cast<size_t>(px) * src_pixel_bytes;
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
    uint32_t blocks_x = bc_block_count(src.width);
    uint32_t blocks_y = bc_block_count(src.height);
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
    uint32_t blocks_x = bc_block_count(src.width);
    uint32_t blocks_y = bc_block_count(src.height);
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
    uint32_t blocks_x = bc_block_count(src.width);
    uint32_t blocks_y = bc_block_count(src.height);
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
    uint32_t blocks_x = bc_block_count(src.width);
    uint32_t blocks_y = bc_block_count(src.height);
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
    uint32_t blocks_x = bc_block_count(src.width);
    uint32_t blocks_y = bc_block_count(src.height);
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
    uint32_t blocks_x = bc_block_count(src.width);
    uint32_t blocks_y = bc_block_count(src.height);
    uint8_t block_pixels[BLOCK_DIM * BLOCK_DIM * 4];

    bc7enc_compress_block_params params;
    bc7enc_compress_block_params_init(&params);
    params.m_uber_level = quality_to_bc7_uber(options.quality);
    params.m_max_partitions = quality_to_bc7_partitions(options.quality);
    for (int i = 0; i < 4; ++i)
        params.m_weights[i] = options.channel_weights[i];

    for (uint32_t by = 0; by < blocks_y; ++by) {
        for (uint32_t bx = 0; bx < blocks_x; ++bx) {
            extract_rgba8_block(src, bx, by, block_pixels);
            if (!options.has_alpha) {
                for (uint32_t i = 0; i < BLOCK_DIM * BLOCK_DIM; ++i)
                    block_pixels[i * 4 + 3] = 255;
            }
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
    size_t dst_pixel_bytes = static_cast<size_t>(dst.channel_count) * component_byte_size(dst.component_type);
    size_t decoded_pixel_bytes = static_cast<size_t>(decoded_channels) * component_byte_size(decoded_type);

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
            const uint8_t* sp
                = src + static_cast<size_t>(by) * decoded_pitch + static_cast<size_t>(bx) * decoded_pixel_bytes;
            uint8_t* dp
                = dst_bytes + static_cast<size_t>(py) * dst.row_pitch + static_cast<size_t>(px) * dst_pixel_bytes;

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
    case BCFormat::bc5_unorm:
        return NVTT_Format_BC5;
    case BCFormat::bc6h_ufloat:
        return NVTT_Format_BC6U;
    case BCFormat::bc6h_sfloat:
        return NVTT_Format_BC6S;
    case BCFormat::bc7_unorm:
    case BCFormat::bc7_unorm_srgb:
        return NVTT_Format_BC7;
    default:
        SGL_THROW("NVTT3: unsupported BC format");
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

/// Convert a BCImage to interleaved float32 RGBA.
static std::vector<float> bc_image_to_float32_rgba(const BCImage& src)
{
    size_t pixel_count = static_cast<size_t>(src.width) * src.height;
    std::vector<float> result;
    SGL_CHECK(pixel_count <= result.max_size() / 4, "BCEncoderNVTT::encode: source dimensions are too large");
    result.resize(pixel_count * 4, 0.0f);
    uint32_t src_pixel_bytes = src.channel_count * component_byte_size(src.component_type);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src.data);

    for (uint32_t y = 0; y < src.height; ++y) {
        for (uint32_t x = 0; x < src.width; ++x) {
            const uint8_t* pixel
                = src_bytes + static_cast<size_t>(y) * src.row_pitch + static_cast<size_t>(x) * src_pixel_bytes;
            float* dst = result.data() + (static_cast<size_t>(y) * src.width + x) * 4;

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

static bool is_nvtt_gpu_available()
{
    static const bool available = nvttIsCudaSupported() == NVTT_True;
    return available;
}

static bool nvtt_build_next_mipmap(nvtt::Surface& surface, const BCMipFilter& filter)
{
    return std::visit(
        [&](const auto& f)
        {
            using Filter = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<Filter, BoxFilter>) {
                return surface.buildNextMipmap(nvtt::MipmapFilter_Box);
            } else if constexpr (std::is_same_v<Filter, TentFilter>) {
                return surface.buildNextMipmap(nvtt::MipmapFilter_Triangle, f.radius());
            } else {
                float params[2] = {f.b(), f.c()};
                return surface.buildNextMipmap(nvtt::MipmapFilter_Mitchell, f.radius(), params);
            }
        },
        filter
    );
}

static void nvtt_encode_surface(
    const nvtt::Surface& surface,
    bool use_gpu,
    BCFormat format,
    const BCEncodeOptions& options,
    std::vector<uint8_t>& out_data
)
{
    NvttRefImage ref_image{};
    ref_image.data = use_gpu ? surface.gpuData() : surface.data();
    ref_image.width = surface.width();
    ref_image.height = surface.height();
    ref_image.depth = 1;
    ref_image.num_channels = 4;
    ref_image.channel_swizzle[0] = NVTT_ChannelOrder_Red;
    ref_image.channel_swizzle[1] = NVTT_ChannelOrder_Green;
    ref_image.channel_swizzle[2] = NVTT_ChannelOrder_Blue;
    ref_image.channel_swizzle[3] = NVTT_ChannelOrder_Alpha;
    ref_image.channel_interleave = NVTT_False;
    SGL_CHECK(ref_image.data != nullptr, "NVTT3: surface data is unavailable");

    NvttEncodeSettings settings{};
    settings.sType = NVTT_EncodeSettings_Version_1;
    settings.format = bc_format_to_nvtt(format);
    settings.quality = quality_to_nvtt(options.quality);
    settings.rgb_pixel_type = is_bc6h(format) ? NVTT_PixelType_Float : NVTT_PixelType_UnsignedNorm;
    settings.timing_context = nullptr;
    settings.encode_flags
        = use_gpu ? static_cast<uint32_t>(NVTT_EncodeFlags_UseGPU) : static_cast<uint32_t>(NVTT_EncodeFlags_None);
    if (!options.has_alpha)
        settings.encode_flags |= static_cast<uint32_t>(NVTT_EncodeFlags_Opaque);

    out_data.resize(
        bc_compressed_size(static_cast<uint32_t>(surface.width()), static_cast<uint32_t>(surface.height()), format)
    );

    unsigned num_tiles = 0;
    NvttBoolean ok = NVTT_False;
    if (use_gpu) {
        std::unique_ptr<NvttGPUInputBuffer, decltype(&nvttDestroyGPUInputBuffer)> input_buffer(
            nvttCreateGPUInputBuffer(
                &ref_image,
                NVTT_ValueType_FLOAT32,
                1,
                BLOCK_DIM,
                BLOCK_DIM,
                static_cast<float>(options.channel_weights[0]),
                static_cast<float>(options.channel_weights[1]),
                static_cast<float>(options.channel_weights[2]),
                static_cast<float>(options.channel_weights[3]),
                nullptr,
                &num_tiles
            ),
            nvttDestroyGPUInputBuffer
        );
        SGL_CHECK(input_buffer != nullptr, "NVTT3: failed to create GPU input buffer");
        ok = nvttEncodeGPU(input_buffer.get(), out_data.data(), &settings);
    } else {
        std::unique_ptr<NvttCPUInputBuffer, decltype(&nvttDestroyCPUInputBuffer)> input_buffer(
            nvttCreateCPUInputBuffer(
                &ref_image,
                NVTT_ValueType_FLOAT32,
                1,
                BLOCK_DIM,
                BLOCK_DIM,
                static_cast<float>(options.channel_weights[0]),
                static_cast<float>(options.channel_weights[1]),
                static_cast<float>(options.channel_weights[2]),
                static_cast<float>(options.channel_weights[3]),
                nullptr,
                &num_tiles
            ),
            nvttDestroyCPUInputBuffer
        );
        SGL_CHECK(input_buffer != nullptr, "NVTT3: failed to create CPU input buffer");
        ok = nvttEncodeCPU(input_buffer.get(), out_data.data(), &settings);
    }

    SGL_CHECK(ok == NVTT_True, "NVTT3: encoding failed");
}

#endif // SGL_HAS_NVTT

//
// Backend-independent mip preparation
//

struct PreparedMip {
    BCImage image;
    ref<Bitmap> owned_bitmap;
};

static std::vector<PreparedMip> prepare_mips(const BCImage& src, BCFormat format, const BCEncodeOptions& options)
{
    std::vector<PreparedMip> levels;
    levels.push_back({src, nullptr});

    if (!options.generate_mipmaps || (src.width <= 1 && src.height <= 1))
        return levels;

    Bitmap::PixelFormat pixel_format;
    switch (src.channel_count) {
    case 1:
        pixel_format = Bitmap::PixelFormat::r;
        break;
    case 2:
        pixel_format = Bitmap::PixelFormat::rg;
        break;
    case 3:
        pixel_format = Bitmap::PixelFormat::rgb;
        break;
    default:
        pixel_format = Bitmap::PixelFormat::rgba;
        break;
    }

    // Bitmap has no row-pitch parameter, so copy the source into tightly packed storage.
    // The source is expected to use the transfer function implied by the destination BC format.
    bool srgb = is_srgb(format);
    auto src_bitmap = make_ref<Bitmap>(
        pixel_format,
        src.component_type,
        src.width,
        src.height,
        src.channel_count,
        std::vector<std::string>{},
        nullptr,
        srgb
    );
    size_t src_row_size = static_cast<size_t>(src.width) * src.channel_count * component_byte_size(src.component_type);
    for (uint32_t y = 0; y < src.height; ++y) {
        std::memcpy(
            src_bitmap->uint8_data() + y * src_row_size,
            static_cast<const uint8_t*>(src.data) + static_cast<size_t>(y) * src.row_pitch,
            src_row_size
        );
    }

    // Always filter in linear float32. Convert generated mips back to the transfer function
    // expected by the destination BC format before handing them to an encoder backend.
    ref<Bitmap> current = src_bitmap->convert(pixel_format, Bitmap::ComponentType::float32, false);
    ReconstructionFilter mip_filter = std::visit(
        [](const auto& filter) -> ReconstructionFilter
        {
            return filter;
        },
        options.mip_filter
    );
    uint32_t mip_width = src.width;
    uint32_t mip_height = src.height;
    while (mip_width > 1 || mip_height > 1) {
        mip_width = std::max(mip_width / 2, 1u);
        mip_height = std::max(mip_height / 2, 1u);
        current = current->resample(mip_width, mip_height, mip_filter);

        ref<Bitmap> encoded_bitmap
            = srgb ? current->convert(pixel_format, Bitmap::ComponentType::float32, true) : current;
        levels.push_back({bc_image_from_bitmap(*encoded_bitmap), encoded_bitmap});
    }

    return levels;
}

//
// BCEncoderImpl - abstract interface with software and NVTT implementations
//

struct BCEncoderImpl {
    virtual ~BCEncoderImpl() = default;
    virtual bool can_encode(BCFormat format) const = 0;
    virtual BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options) = 0;
};

/// Software encoder using rgbcx (BC1-5) and bc7enc (BC7).
struct BCEncoderSWImpl : BCEncoderImpl {
    BCEncoderSWImpl() { ensure_sw_init(); }

    bool can_encode(BCFormat format) const override
    {
        return is_valid_bc_format(format) && !is_bc6h(format) && !is_snorm(format);
    }

    BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options) override
    {
        std::vector<PreparedMip> levels = prepare_mips(src, format, options);
        BCCompressedImage result;
        result.format = format;
        result.mip_levels.reserve(levels.size());
        for (const PreparedMip& level : levels) {
            BCCompressedMip mip;
            mip.width = level.image.width;
            mip.height = level.image.height;
            mip.data.resize(bc_compressed_size(mip.width, mip.height, format));
            encode_mip(level.image, format, options, mip.data);
            result.mip_levels.push_back(std::move(mip));
        }
        return result;
    }

    void encode_mip(const BCImage& src, BCFormat format, const BCEncodeOptions& options, std::vector<uint8_t>& output)
    {
        if (is_bc6h(format))
            SGL_THROW("BCEncoderSW::encode: BC6H encoding requires NVTT3");
        if (is_snorm(format))
            SGL_THROW("BCEncoderSW::encode: BC4/BC5 SNORM encoding is not implemented");

        switch (format) {
        case BCFormat::bc1_unorm:
        case BCFormat::bc1_unorm_srgb:
            encode_image_bc1(src, output.data(), options.quality);
            break;
        case BCFormat::bc2_unorm:
        case BCFormat::bc2_unorm_srgb:
            encode_image_bc2(src, output.data(), options.quality);
            break;
        case BCFormat::bc3_unorm:
        case BCFormat::bc3_unorm_srgb:
            encode_image_bc3(src, output.data(), options.quality);
            break;
        case BCFormat::bc4_unorm:
        case BCFormat::bc4_snorm:
            encode_image_bc4(src, output.data());
            break;
        case BCFormat::bc5_unorm:
        case BCFormat::bc5_snorm:
            encode_image_bc5(src, output.data());
            break;
        case BCFormat::bc7_unorm:
        case BCFormat::bc7_unorm_srgb:
            encode_image_bc7(src, output.data(), options);
            break;
        default:
            SGL_THROW("BCEncoderSW::encode: unsupported format");
        }
    }
};

#if SGL_HAS_NVTT

/// NVTT3 encoder with a CPU- or GPU-resident Surface pipeline.
struct BCEncoderNVTTImpl : BCEncoderImpl {
    explicit BCEncoderNVTTImpl(bool use_gpu)
        : m_use_gpu(use_gpu)
    {
    }

    bool can_encode(BCFormat format) const override { return is_valid_bc_format(format) && !is_snorm(format); }

    BCCompressedImage encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options) override
    {
        // The NVTT Surface pipeline supplies planar float input, which does not
        // produce valid BC4S/BC5S output. Do not expose those formats until signed
        // input preparation is implemented for both CPU and GPU paths.
        if (is_snorm(format))
            SGL_THROW("BCEncoderNVTT::encode: BC4/BC5 SNORM encoding is not implemented");

        SGL_CHECK(
            src.width <= static_cast<uint32_t>(std::numeric_limits<int>::max())
                && src.height <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
            "BCEncoderNVTT::encode: source dimensions are too large"
        );

        std::vector<float> float_data = bc_image_to_float32_rgba(src);
        nvtt::Surface current;
        SGL_CHECK(
            current.setImage(
                nvtt::InputFormat_RGBA_32F,
                static_cast<int>(src.width),
                static_cast<int>(src.height),
                1,
                float_data.data()
            ),
            "NVTT3: failed to create source surface"
        );
        current.setWrapMode(nvtt::WrapMode_Clamp);
        current.setAlphaMode(options.has_alpha ? nvtt::AlphaMode_Transparency : nvtt::AlphaMode_None);

        if (m_use_gpu) {
            current.ToGPU();
            SGL_CHECK(current.gpuData() != nullptr, "NVTT3: failed to move source surface to the GPU");
        }

        BCCompressedImage result;
        result.format = format;
        result.mip_levels.reserve(options.generate_mipmaps ? bc_mip_count(src.width, src.height) : 1);

        auto append_mip = [&](const nvtt::Surface& surface)
        {
            BCCompressedMip mip;
            mip.width = static_cast<uint32_t>(surface.width());
            mip.height = static_cast<uint32_t>(surface.height());
            nvtt_encode_surface(surface, m_use_gpu, format, options, mip.data);
            result.mip_levels.push_back(std::move(mip));
        };

        // Preserve the source transfer function for level zero.
        append_mip(current);

        if (options.generate_mipmaps && (src.width > 1 || src.height > 1)) {
            const bool srgb = is_srgb(format);
            if (srgb)
                current.toLinearFromSrgb();

            while (nvtt_build_next_mipmap(current, options.mip_filter)) {
                if (srgb) {
                    nvtt::Surface encoded = current;
                    encoded.toSrgb();
                    append_mip(encoded);
                } else {
                    append_mip(current);
                }
            }
        }

        return result;
    }

private:
    bool m_use_gpu;
};

#endif // SGL_HAS_NVTT

//
// BCEncoder public interface
//

BCEncoder::BCEncoder(BCEncoderBackend backend)
{
    if (backend == BCEncoderBackend::automatic) {
#if SGL_HAS_NVTT
        backend = is_nvtt_gpu_available() ? BCEncoderBackend::nvtt_gpu : BCEncoderBackend::nvtt_cpu;
#else
        backend = BCEncoderBackend::software;
#endif
    }

    switch (backend) {
    case BCEncoderBackend::software:
        m_impl = std::make_unique<BCEncoderSWImpl>();
        break;
    case BCEncoderBackend::nvtt_cpu:
#if SGL_HAS_NVTT
        m_impl = std::make_unique<BCEncoderNVTTImpl>(false);
#else
        SGL_THROW("BCEncoder: NVTT CPU backend is not available in this build");
#endif
        break;
    case BCEncoderBackend::nvtt_gpu:
#if SGL_HAS_NVTT
        SGL_CHECK(is_nvtt_gpu_available(), "BCEncoder: NVTT GPU backend requires CUDA support");
        m_impl = std::make_unique<BCEncoderNVTTImpl>(true);
#else
        SGL_THROW("BCEncoder: NVTT GPU backend is not available in this build");
#endif
        break;
    default:
        SGL_THROW("BCEncoder: invalid backend");
    }
    m_backend = backend;
}

BCEncoder::~BCEncoder() = default;

bool BCEncoder::is_backend_available(BCEncoderBackend backend)
{
    switch (backend) {
    case BCEncoderBackend::automatic:
    case BCEncoderBackend::software:
        return true;
    case BCEncoderBackend::nvtt_cpu:
        return SGL_HAS_NVTT;
    case BCEncoderBackend::nvtt_gpu:
#if SGL_HAS_NVTT
        return is_nvtt_gpu_available();
#else
        return false;
#endif
    default:
        return false;
    }
}

bool BCEncoder::can_encode(BCFormat format) const
{
    return m_impl->can_encode(format);
}

BCCompressedImage BCEncoder::encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options)
{
    SGL_CHECK(src.data != nullptr, "BCEncoder::encode: source data is null");
    SGL_CHECK(src.width > 0 && src.height > 0, "BCEncoder::encode: invalid source dimensions");
    SGL_CHECK(src.channel_count >= 1 && src.channel_count <= 4, "BCEncoder::encode: channel_count must be 1-4");
    size_t min_row_pitch = static_cast<size_t>(src.width) * src.channel_count * component_byte_size(src.component_type);
    SGL_CHECK(
        min_row_pitch <= std::numeric_limits<uint32_t>::max() && src.row_pitch >= min_row_pitch,
        "BCEncoder::encode: row_pitch is too small for the given width, channel count, and component type"
    );
    for (int i = 0; i < 4; ++i)
        SGL_CHECK(options.channel_weights[i] > 0, "BCEncoder::encode: channel_weights must be positive");
    SGL_CHECK(can_encode(format), "BCEncoder::encode: selected backend cannot encode the requested format");

    return m_impl->encode(src, format, options);
}

//
// Stateless bcdec decoder
//

void decode_bc(
    const void* data,
    size_t size,
    BCFormat format,
    uint32_t width,
    uint32_t height,
    const BCMutableImage& dst
)
{
    SGL_CHECK(data != nullptr, "decode_bc: source data is null");
    SGL_CHECK(dst.data != nullptr, "decode_bc: destination data is null");
    SGL_CHECK(width > 0 && height > 0, "decode_bc: invalid dimensions");
    uint32_t bpb = bc_format_bytes_per_block(format);
    SGL_CHECK(bpb > 0, "decode_bc: unsupported BC format");
    SGL_CHECK(size >= bc_compressed_size(width, height, format), "decode_bc: insufficient source data");
    SGL_CHECK(dst.width >= width && dst.height >= height, "decode_bc: destination too small");
    SGL_CHECK(dst.channel_count >= 1 && dst.channel_count <= 4, "decode_bc: destination channel_count must be 1-4");
    uint32_t dst_component_size = component_byte_size(dst.component_type);
    size_t min_row_pitch = static_cast<size_t>(width) * dst.channel_count * dst_component_size;
    SGL_CHECK(
        min_row_pitch <= std::numeric_limits<uint32_t>::max() && dst.row_pitch >= min_row_pitch,
        "decode_bc: destination row_pitch is too small"
    );

    // Only the requested image extent is writable, even when the supplied destination view is larger.
    BCMutableImage clipped_dst = dst;
    clipped_dst.width = width;
    clipped_dst.height = height;

    uint32_t blocks_x = bc_block_count(width);
    uint32_t blocks_y = bc_block_count(height);
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
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc2_unorm:
            case BCFormat::bc2_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc2(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc3_unorm:
            case BCFormat::bc3_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc3(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc4_unorm: {
                // Output: R uint8 (1 byte per pixel).
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM];
                bcdec_bc4(block, decoded, BLOCK_DIM, 0);
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM, 1, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc4_snorm: {
                int8_t decoded[BLOCK_DIM * BLOCK_DIM];
                bcdec_bc4(block, decoded, BLOCK_DIM, 1);
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM, 1, BCComponentType::int8);
                break;
            }
            case BCFormat::bc5_unorm: {
                // Output: RG uint8 (2 bytes per pixel).
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 2];
                bcdec_bc5(block, decoded, BLOCK_DIM * 2, 0);
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM * 2, 2, BCComponentType::uint8);
                break;
            }
            case BCFormat::bc5_snorm: {
                int8_t decoded[BLOCK_DIM * BLOCK_DIM * 2];
                bcdec_bc5(block, decoded, BLOCK_DIM * 2, 1);
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM * 2, 2, BCComponentType::int8);
                break;
            }
            case BCFormat::bc6h_ufloat: {
                // Output: RGB float16 (6 bytes per pixel) - lossless via bcdec_bc6h_half().
                // NOTE: bcdec_bc6h_half uses unsigned short* internally, so destinationPitch is in elements, not bytes.
                uint16_t decoded[BLOCK_DIM * BLOCK_DIM * 3];
                bcdec_bc6h_half(block, decoded, BLOCK_DIM * 3, 0);
                copy_block_to_dst(
                    clipped_dst,
                    bx,
                    by,
                    decoded,
                    BLOCK_DIM * 3 * sizeof(uint16_t),
                    3,
                    BCComponentType::float16
                );
                break;
            }
            case BCFormat::bc6h_sfloat: {
                uint16_t decoded[BLOCK_DIM * BLOCK_DIM * 3];
                bcdec_bc6h_half(block, decoded, BLOCK_DIM * 3, 1);
                copy_block_to_dst(
                    clipped_dst,
                    bx,
                    by,
                    decoded,
                    BLOCK_DIM * 3 * sizeof(uint16_t),
                    3,
                    BCComponentType::float16
                );
                break;
            }
            case BCFormat::bc7_unorm:
            case BCFormat::bc7_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc7(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(clipped_dst, bx, by, decoded, BLOCK_DIM * 4, 4, BCComponentType::uint8);
                break;
            }
            default:
                SGL_THROW("decode_bc: unsupported BC format");
            }
        }
    }
}

} // namespace sgl
