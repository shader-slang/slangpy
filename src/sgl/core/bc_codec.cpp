// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/core/bc_codec.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/error.h"
#include "sgl/core/logger.h"
#include "sgl/core/nvtt_api.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

// ── bcdec (header-only decoder) ─────────────────────────────────────────────
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

// ── rgbcx + bc7enc (SW encoders) ────────────────────────────────────────────
#include <rgbcx.h>
#include <bc7enc.h>

namespace sgl {

// ────────────────────────────────────────────────────────────────────────────
// One-time initialization
// ────────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

static constexpr uint32_t BLOCK_DIM = 4;

/// Map BCEncodeQuality → rgbcx level (0–18).
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

/// Map BCEncodeQuality → bc7enc uber level (0–BC7ENC_MAX_UBER_LEVEL).
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

/// Map BCEncodeQuality → bc7enc max_partitions.
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

/// Extract a 4×4 RGBA8 block from the source image, padding if the block
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
                // For non-uint8, assume float32 and quantize to [0,255].
                uint32_t nc = std::min(src.channel_count, 4u);
                for (uint32_t c = 0; c < nc; ++c) {
                    float v = 0.0f;
                    if (src.component_type == BCComponentType::float32)
                        v = reinterpret_cast<const float*>(pixel)[c];
                    else if (src.component_type == BCComponentType::float64)
                        v = static_cast<float>(reinterpret_cast<const double*>(pixel)[c]);
                    dst[c] = static_cast<uint8_t>(std::clamp(v * 255.0f + 0.5f, 0.0f, 255.0f));
                }
                for (uint32_t c = nc; c < 4; ++c)
                    dst[c] = (c == 3) ? 255 : 0;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Encoding helpers (per-format)
// ────────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────────
// Decode helpers (per-format)
// ────────────────────────────────────────────────────────────────────────────

/// Write decoded pixels (one 4×4 block) into the destination image, clipping
/// to image boundaries.
static void copy_block_to_dst(
    const BCMutableImage& dst,
    uint32_t block_x,
    uint32_t block_y,
    const void* decoded_block,
    uint32_t decoded_pitch, // bytes per row of the decoded 4×4 block
    uint32_t pixel_bytes    // bytes per decoded pixel
)
{
    const uint8_t* src = static_cast<const uint8_t*>(decoded_block);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst.data);
    uint32_t dst_pixel_bytes = dst.channel_count * component_byte_size(dst.component_type);

    for (uint32_t by = 0; by < BLOCK_DIM; ++by) {
        uint32_t py = block_y * BLOCK_DIM + by;
        if (py >= dst.height)
            break;
        for (uint32_t bx = 0; bx < BLOCK_DIM; ++bx) {
            uint32_t px = block_x * BLOCK_DIM + bx;
            if (px >= dst.width)
                break;
            const uint8_t* sp = src + by * decoded_pitch + bx * pixel_bytes;
            uint8_t* dp = dst_bytes + py * dst.row_pitch + px * dst_pixel_bytes;
            std::memcpy(dp, sp, std::min(pixel_bytes, dst_pixel_bytes));
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// NVTT3 dynamic loading
// ────────────────────────────────────────────────────────────────────────────

void NvttAPI::try_load()
{
    // 1. Try runtime_directory() / library name.
#if SGL_WINDOWS
    const char* lib_name = "nvtt.dll";
#elif SGL_LINUX
    const char* lib_name = "libnvtt.so";
#elif SGL_MACOS
    const char* lib_name = "libnvtt.dylib";
#else
    const char* lib_name = nullptr;
#endif

    if (!lib_name) {
        log_info("NVTT3 not supported on this platform.");
        return;
    }

    // Try next to sgl library.
    if (load_from_path(platform::runtime_directory() / lib_name))
        goto resolve;

    // 2. Try bare name (system PATH / LD_LIBRARY_PATH).
    if (load_from_path(lib_name))
        goto resolve;

    // 3. Try SGL_NVTT_PATH environment variable.
    {
#ifdef _MSC_VER
        char* env_path = nullptr;
        size_t env_len = 0;
        _dupenv_s(&env_path, &env_len, "SGL_NVTT_PATH");
        if (env_path && env_path[0] != '\0') {
            bool ok = load_from_path(std::filesystem::path(env_path) / lib_name);
            free(env_path);
            if (ok)
                goto resolve;
        } else {
            free(env_path);
        }
#else
        const char* env_path = std::getenv("SGL_NVTT_PATH");
        if (env_path && env_path[0] != '\0') {
            if (load_from_path(std::filesystem::path(env_path) / lib_name))
                goto resolve;
        }
#endif
    }

    log_info("NVTT3 not found, BC6H encoding unavailable.");
    return;

resolve:
    if (!resolve_symbols()) {
        log_warn("NVTT3 library found but symbols could not be resolved.");
        platform::release_shared_library(library);
        library = nullptr;
        available = false;
        return;
    }

    unsigned int version = nvttVersion();
    log_info("NVTT3 loaded (version {}.{}.{}).", (version >> 16) & 0xff, (version >> 8) & 0xff, version & 0xff);
    available = true;
}

bool NvttAPI::load_from_path(const std::filesystem::path& path)
{
    library = platform::load_shared_library(path);
    return library != nullptr;
}

bool NvttAPI::resolve_symbols()
{
#define NVTT_RESOLVE(name)                                                                                             \
    name = reinterpret_cast<decltype(name)>(platform::get_proc_address(library, #name));                               \
    if (!name)                                                                                                         \
        return false;

    NVTT_RESOLVE(nvttCreateCPUInputBuffer);
    NVTT_RESOLVE(nvttDestroyCPUInputBuffer);
    NVTT_RESOLVE(nvttEncodeCPU);
    NVTT_RESOLVE(nvttCreateSurface);
    NVTT_RESOLVE(nvttDestroySurface);
    NVTT_RESOLVE(nvttSurfaceSetImageData);
    NVTT_RESOLVE(nvttSurfaceBuildNextMipmapDefaults);
    NVTT_RESOLVE(nvttSurfaceData);
    NVTT_RESOLVE(nvttSurfaceWidth);
    NVTT_RESOLVE(nvttSurfaceHeight);
    NVTT_RESOLVE(nvttVersion);

#undef NVTT_RESOLVE
    return true;
}

static NvttAPI& get_nvtt_api()
{
    static NvttAPI api;
    static std::once_flag flag;
    std::call_once(flag, [&] { api.try_load(); });
    return api;
}

// ────────────────────────────────────────────────────────────────────────────
// NVTT3 format / quality mappings
// ────────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────────
// NVTT3 encode helpers
// ────────────────────────────────────────────────────────────────────────────

/// Encode a single mip level via NVTT3's low-level CPU API.
static void nvtt_encode_single(
    NvttAPI& nvtt,
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
    NvttCPUInputBuffer* input_buffer = nvtt.nvttCreateCPUInputBuffer(
        &ref_image,
        NVTT_ValueType_FLOAT32,
        1, // numImages
        static_cast<int>(width),
        static_cast<int>(height),
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
    settings.rgb_pixel_type = is_bc6h(format) ? 4 : 0; // 4 = Float, 0 = UnsignedNorm
    settings.timing_context = nullptr;
    settings.encode_flags
        = options.has_alpha ? static_cast<uint32_t>(NVTT_EncodeFlags_None) : static_cast<uint32_t>(NVTT_EncodeFlags_Opaque);

    size_t compressed_bytes = bc_compressed_size(width, height, format);
    out_data.resize(compressed_bytes);

    NvttBoolean ok = nvtt.nvttEncodeCPU(input_buffer, out_data.data(), &settings);
    nvtt.nvttDestroyCPUInputBuffer(input_buffer);

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
                if (src.component_type == BCComponentType::uint8)
                    dst[c] = pixel[c] / 255.0f;
                else if (src.component_type == BCComponentType::float32)
                    dst[c] = reinterpret_cast<const float*>(pixel)[c];
                else if (src.component_type == BCComponentType::float64)
                    dst[c] = static_cast<float>(reinterpret_cast<const double*>(pixel)[c]);
                else
                    dst[c] = pixel[c] / 255.0f; // fallback
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
static BCCompressedImage
nvtt_encode(NvttAPI& nvtt, const BCImage& src, BCFormat format, const BCEncodeOptions& options)
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
        nvtt_encode_single(nvtt, float_data.data(), src.width, src.height, 4, format, options, mip.data);
        result.mip_levels.push_back(std::move(mip));
    }

    if (!options.generate_mipmaps || (src.width <= 1 && src.height <= 1))
        return result;

    // Use NVTT3 Surface for mipmap generation.
    NvttSurface* surface = nvtt.nvttCreateSurface();
    SGL_CHECK(surface != nullptr, "NVTT3: failed to create surface");

    NvttBoolean ok = nvtt.nvttSurfaceSetImageData(
        surface,
        NVTT_InputFormat_RGBA_32F,
        static_cast<int>(src.width),
        static_cast<int>(src.height),
        1,
        float_data.data(),
        NVTT_False, // not a reference — copy the data
        nullptr     // timing_context
    );

    if (ok != NVTT_True) {
        nvtt.nvttDestroySurface(surface);
        SGL_THROW("NVTT3: failed to set surface image data");
    }

    uint32_t mip_w = src.width;
    uint32_t mip_h = src.height;
    while (mip_w > 1 || mip_h > 1) {
        ok = nvtt.nvttSurfaceBuildNextMipmapDefaults(surface, NVTT_MipmapFilter_Box, 1, nullptr);
        if (ok != NVTT_True)
            break;

        mip_w = static_cast<uint32_t>(nvtt.nvttSurfaceWidth(surface));
        mip_h = static_cast<uint32_t>(nvtt.nvttSurfaceHeight(surface));

        // Get planar data from surface and convert to interleaved.
        float* planar = nvtt.nvttSurfaceData(surface);
        std::vector<float> interleaved = nvtt_planar_to_interleaved(planar, mip_w, mip_h);

        BCCompressedMip mip;
        mip.width = mip_w;
        mip.height = mip_h;
        nvtt_encode_single(nvtt, interleaved.data(), mip_w, mip_h, 4, format, options, mip.data);
        result.mip_levels.push_back(std::move(mip));
    }

    nvtt.nvttDestroySurface(surface);
    return result;
}

// ────────────────────────────────────────────────────────────────────────────
// BCCodec::Impl
// ────────────────────────────────────────────────────────────────────────────

struct BCCodec::Impl {
    NvttAPI& nvtt = get_nvtt_api();
};

// ────────────────────────────────────────────────────────────────────────────
// BCCodec public interface
// ────────────────────────────────────────────────────────────────────────────

BCCodec::BCCodec()
    : m_impl(std::make_unique<Impl>())
{
    ensure_sw_init();
}

BCCodec::~BCCodec() = default;

bool BCCodec::is_nvtt_available() const
{
    return m_impl->nvtt.available;
}

bool BCCodec::can_encode(BCFormat format) const
{
    if (is_bc6h(format))
        return m_impl->nvtt.available;
    return true;
}

bool BCCodec::can_decode(BCFormat) const
{
    return true;
}

// ── Encode ──────────────────────────────────────────────────────────────────

BCCompressedImage BCCodec::encode(const BCImage& src, BCFormat format, const BCEncodeOptions& options)
{
    SGL_CHECK(src.data != nullptr, "BCCodec::encode: source data is null");
    SGL_CHECK(src.width > 0 && src.height > 0, "BCCodec::encode: invalid source dimensions");

    auto& nvtt = m_impl->nvtt;
    if (nvtt.available && options.prefer_nvtt)
        return nvtt_encode(nvtt, src, format, options);

    if (is_bc6h(format))
        SGL_THROW("BCCodec::encode: BC6H encoding requires NVTT3 (not available)");

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
        // Non-owning wrap of source data — safe because src outlives this scope.
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
            SGL_THROW("BCCodec::encode: unsupported format");
        }

        result.mip_levels.push_back(std::move(mip));
    }

    return result;
}

// ── Decode ──────────────────────────────────────────────────────────────────

void BCCodec::decode(
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
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4);
                break;
            }
            case BCFormat::bc2_unorm:
            case BCFormat::bc2_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc2(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4);
                break;
            }
            case BCFormat::bc3_unorm:
            case BCFormat::bc3_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc3(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4);
                break;
            }
            case BCFormat::bc4_unorm: {
                // Output: R uint8 (1 byte per pixel).
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM];
                bcdec_bc4(block, decoded, BLOCK_DIM, 0);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM, 1);
                break;
            }
            case BCFormat::bc4_snorm: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM];
                bcdec_bc4(block, decoded, BLOCK_DIM, 1);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM, 1);
                break;
            }
            case BCFormat::bc5_unorm: {
                // Output: RG uint8 (2 bytes per pixel).
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 2];
                bcdec_bc5(block, decoded, BLOCK_DIM * 2, 0);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 2, 2);
                break;
            }
            case BCFormat::bc5_snorm: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 2];
                bcdec_bc5(block, decoded, BLOCK_DIM * 2, 1);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 2, 2);
                break;
            }
            case BCFormat::bc6h_ufloat: {
                // Output: RGB float16 (6 bytes per pixel) — lossless via bcdec_bc6h_half().
                uint16_t decoded[BLOCK_DIM * BLOCK_DIM * 3];
                bcdec_bc6h_half(block, decoded, BLOCK_DIM * 3 * sizeof(uint16_t), 0);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 3 * sizeof(uint16_t), 6);
                break;
            }
            case BCFormat::bc6h_sfloat: {
                uint16_t decoded[BLOCK_DIM * BLOCK_DIM * 3];
                bcdec_bc6h_half(block, decoded, BLOCK_DIM * 3 * sizeof(uint16_t), 1);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 3 * sizeof(uint16_t), 6);
                break;
            }
            case BCFormat::bc7_unorm:
            case BCFormat::bc7_unorm_srgb: {
                uint8_t decoded[BLOCK_DIM * BLOCK_DIM * 4];
                bcdec_bc7(block, decoded, BLOCK_DIM * 4);
                copy_block_to_dst(dst, bx, by, decoded, BLOCK_DIM * 4, 4);
                break;
            }
            }
        }
    }
}

} // namespace sgl
