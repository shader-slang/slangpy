// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "texture_loader.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"
#include "sgl/device/blit.h"
#include "sgl/device/native_formats.h"

#include "sgl/core/error.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/config.h"
#include "sgl/core/dds_file.h"
#include "sgl/core/file_stream.h"
#include "sgl/core/logger.h"
#include "sgl/core/maths.h"
#include "sgl/core/timer.h"
#include "sgl/core/thread.h"

#include "sgl/stl/bit.h"

#include <map>
#include <utility>

namespace sgl {

static constexpr size_t BATCH_SIZE = 32;

/// Holds a source image to be uploaded to a texture.
/// Data can either be a bitmap or a DDS file.
struct SourceImage {
    ref<Bitmap> bitmap;
    ref<DDSFile> dds_file;
    Format format{Format::undefined};
};

inline ref<Bitmap> convert_ya_to_rg(const Bitmap* ya_bitmap)
{
    SGL_ASSERT(ya_bitmap->pixel_format() == Bitmap::PixelFormat::ya);

    ref<Bitmap> rg_bitmap = make_ref<Bitmap>(
        Bitmap::PixelFormat::rg,
        ya_bitmap->component_type(),
        ya_bitmap->width(),
        ya_bitmap->height()
    );
    rg_bitmap->set_srgb_gamma(ya_bitmap->srgb_gamma());
    std::memcpy(rg_bitmap->data(), ya_bitmap->data(), ya_bitmap->buffer_size());

    return rg_bitmap;
}

inline TextureUsage get_effective_texture_usage(Device* device, const TextureLoader::Options& options)
{
    TextureUsage usage = options.usage;
    if (options.generate_mips) {
        usage |= device->has_feature(Feature::rasterization) ? TextureUsage::render_target
                                                             : TextureUsage::unordered_access;
    }
    return usage;
}

inline FormatSupport get_required_format_support(TextureUsage usage)
{
    FormatSupport support = FormatSupport::texture;
    if (is_set(usage, TextureUsage::render_target))
        support |= FormatSupport::render_target;
    if (is_set(usage, TextureUsage::unordered_access))
        support |= FormatSupport::shader_uav_store;
    return support;
}

inline bool has_format_support(Device* device, Format format, FormatSupport required_support)
{
    return (device->get_format_support(format) & required_support) == required_support;
}

/**
 * \brief Determine the texture format given a bitmap.
 *
 * Uses the following option flags to affect the format
 * determination:
 * - \c Options::extend_alpha
 *   RGB bitmap that has no supported format, or needs renderable usage,
 * will be determined as RGBA
 *   (if a RGBA format exists).
 * - \c Options::srgb_mode selects whether 8-bit RGBA uses \c Format::rgba8_unorm_srgb.
 * Automatic interpretation follows bitmap metadata; explicit linear/sRGB overrides it.
 *
 * - \c Options::load_as_normalized
 *   8/16-bit integer bitmap will be determined as normalized resource format.
 *
 * \param bitmap Bitmap to determine format for.
 * \param options Texture loading options.
 * \return A pair containing the determined format and flag if the bitmap needs to be converted to RGBA to match the format.
 */
inline std::pair<Format, bool>
determine_texture_format(Device* device, const Bitmap* bitmap, const TextureLoader::Options& options)
{
    SGL_ASSERT(bitmap != nullptr);

    using PixelFormat = Bitmap::PixelFormat;
    using ComponentType = Bitmap::ComponentType;

    enum class FormatFlags {
        none = 0,
        normalized = 1,
        srgb = 2,
    };

    auto make_key = [](PixelFormat pixel_format,
                       ComponentType component_type,
                       FormatFlags flags = FormatFlags::none) constexpr -> uint32_t
    {
        static_assert(Bitmap::PIXEL_FORMAT_COUNT <= 8);
        static_assert(DataStruct::TYPE_COUNT <= 16);
        uint32_t key = 0;
        key |= (1 << uint32_t(pixel_format));
        key |= (1 << uint32_t(component_type)) << 8;
        key |= uint32_t(flags) << 24;
        return key;
    };

    static const std::map<uint32_t, Format> FORMAT_TABLE{
        // PixelFormat::r
        {make_key(PixelFormat::r, ComponentType::int8), Format::r8_sint},
        {make_key(PixelFormat::r, ComponentType::int8, FormatFlags::normalized), Format::r8_snorm},
        {make_key(PixelFormat::r, ComponentType::int16), Format::r16_sint},
        {make_key(PixelFormat::r, ComponentType::int16, FormatFlags::normalized), Format::r16_snorm},
        {make_key(PixelFormat::r, ComponentType::int32), Format::r32_sint},
        {make_key(PixelFormat::r, ComponentType::uint8), Format::r8_uint},
        {make_key(PixelFormat::r, ComponentType::uint8, FormatFlags::normalized), Format::r8_unorm},
        {make_key(PixelFormat::r, ComponentType::uint16), Format::r16_uint},
        {make_key(PixelFormat::r, ComponentType::uint16, FormatFlags::normalized), Format::r16_unorm},
        {make_key(PixelFormat::r, ComponentType::uint32), Format::r32_uint},
        {make_key(PixelFormat::r, ComponentType::float16), Format::r16_float},
        {make_key(PixelFormat::r, ComponentType::float32), Format::r32_float},
        // PixelFormat::rg,
        {make_key(PixelFormat::rg, ComponentType::int8), Format::rg8_sint},
        {make_key(PixelFormat::rg, ComponentType::int8, FormatFlags::normalized), Format::rg8_snorm},
        {make_key(PixelFormat::rg, ComponentType::int16), Format::rg16_sint},
        {make_key(PixelFormat::rg, ComponentType::int16, FormatFlags::normalized), Format::rg16_snorm},
        {make_key(PixelFormat::rg, ComponentType::int32), Format::rg32_sint},
        {make_key(PixelFormat::rg, ComponentType::uint8), Format::rg8_uint},
        {make_key(PixelFormat::rg, ComponentType::uint8, FormatFlags::normalized), Format::rg8_unorm},
        {make_key(PixelFormat::rg, ComponentType::uint16), Format::rg16_uint},
        {make_key(PixelFormat::rg, ComponentType::uint16, FormatFlags::normalized), Format::rg16_unorm},
        {make_key(PixelFormat::rg, ComponentType::uint32), Format::rg32_uint},
        {make_key(PixelFormat::rg, ComponentType::float16), Format::rg16_float},
        {make_key(PixelFormat::rg, ComponentType::float32), Format::rg32_float},
        // PixelFormat::rgb,
        {make_key(PixelFormat::rgb, ComponentType::int32), Format::rgb32_sint},
        {make_key(PixelFormat::rgb, ComponentType::uint32), Format::rgb32_uint},
        {make_key(PixelFormat::rgb, ComponentType::float32), Format::rgb32_float},
        // PixelFormat::rgba,
        {make_key(PixelFormat::rgba, ComponentType::int8), Format::rgba8_sint},
        {make_key(PixelFormat::rgba, ComponentType::int8, FormatFlags::normalized), Format::rgba8_snorm},
        {make_key(PixelFormat::rgba, ComponentType::int16), Format::rgba16_sint},
        {make_key(PixelFormat::rgba, ComponentType::int16, FormatFlags::normalized), Format::rgba16_snorm},
        {make_key(PixelFormat::rgba, ComponentType::int32), Format::rgba32_sint},
        {make_key(PixelFormat::rgba, ComponentType::uint8), Format::rgba8_uint},
        {make_key(PixelFormat::rgba, ComponentType::uint8, FormatFlags::normalized), Format::rgba8_unorm},
        {make_key(PixelFormat::rgba, ComponentType::uint8, FormatFlags::srgb), Format::rgba8_unorm_srgb},
        {make_key(PixelFormat::rgba, ComponentType::uint16), Format::rgba16_uint},
        {make_key(PixelFormat::rgba, ComponentType::uint16, FormatFlags::normalized), Format::rgba16_unorm},
        {make_key(PixelFormat::rgba, ComponentType::uint32), Format::rgba32_uint},
        {make_key(PixelFormat::rgba, ComponentType::float16), Format::rgba16_float},
        {make_key(PixelFormat::rgba, ComponentType::float32), Format::rgba32_float},
    };

    ComponentType component_type = bitmap->component_type();
    PixelFormat pixel_format = bitmap->pixel_format();
    bool convert_to_rgba = false;
    if (pixel_format == PixelFormat::y) {
        if (options.y_handling == YHandling::expand_to_rgba) {
            pixel_format = PixelFormat::rgba;
            convert_to_rgba = true;
        } else {
            pixel_format = PixelFormat::r;
        }
    }
    FormatFlags format_flags = FormatFlags::none;
    if (options.load_as_normalized && DataStruct::is_integer(component_type))
        format_flags = FormatFlags::normalized;

    // Check if bitmap is RGB and we can convert to RGBA.
    if (options.extend_alpha && pixel_format == PixelFormat::rgb) {
        TextureUsage effective_usage = get_effective_texture_usage(device, options);
        FormatSupport required_support = get_required_format_support(effective_usage);
        bool prefer_rgba = is_set(effective_usage, TextureUsage::render_target)
            || is_set(effective_usage, TextureUsage::unordered_access);

        // Some backends can sample RGB formats but cannot use them for renderable mip generation.
        // When alpha extension is enabled, prefer RGBA for renderable usage to keep loading portable.
        bool rgb_format_supported = false;
        if (auto it = FORMAT_TABLE.find(make_key(PixelFormat::rgb, component_type, format_flags));
            it != FORMAT_TABLE.end() && has_format_support(device, it->second, required_support))
            rgb_format_supported = true;

        bool rgba_format_supported = false;
        if (auto it = FORMAT_TABLE.find(make_key(PixelFormat::rgba, component_type, format_flags));
            it != FORMAT_TABLE.end() && has_format_support(device, it->second, required_support))
            rgba_format_supported = true;

        if ((!rgb_format_supported || prefer_rgba) && rgba_format_supported) {
            convert_to_rgba = true;
            pixel_format = PixelFormat::rgba;
        }
    }

    // Handle YA bitmap based on ya_handling option.
    if (pixel_format == PixelFormat::ya) {
        if (options.ya_handling == YAHandling::expand_to_rgba) {
            pixel_format = PixelFormat::rgba;
            convert_to_rgba = true;
        } else {
            pixel_format = PixelFormat::rg;
        }
    }

    // Use sRGB format if requested and supported.
    const bool srgb
        = options.srgb_mode == SRGBMode::srgb || (options.srgb_mode == SRGBMode::automatic && bitmap->srgb_gamma());
    if (srgb && pixel_format == PixelFormat::rgba && component_type == ComponentType::uint8)
        format_flags = FormatFlags::srgb;

    // Find texture format.
    auto it = FORMAT_TABLE.find(make_key(pixel_format, component_type, format_flags));
    if (it == FORMAT_TABLE.end())
        SGL_THROW("Unsupported bitmap format: {} {}", pixel_format, component_type);

    return {it->second, convert_to_rgba};
}

inline std::pair<TextureType, uint32_t> get_texture_type_and_layer_count(DDSFile::TextureType type, uint32_t array_size)
{
    switch (type) {
    case DDSFile::TextureType::texture_1d:
        if (array_size > 1)
            return {TextureType::texture_1d_array, array_size};
        return {TextureType::texture_1d, 1};
    case DDSFile::TextureType::texture_2d:
        if (array_size > 1)
            return {TextureType::texture_2d_array, array_size};
        return {TextureType::texture_2d, 1};
    case DDSFile::TextureType::texture_3d:
        return {TextureType::texture_3d, 1};
    case DDSFile::TextureType::texture_cube:
        if (array_size > 1)
            return {TextureType::texture_cube_array, array_size * 6};
        return {TextureType::texture_cube, 6};
    default:
        SGL_THROW("Invalid DDS texture type {}", type);
    }
}

// Reduce in linear light, quantizing to the texture format after each halving.
inline ref<Bitmap> reduce_bitmap(ref<Bitmap> bitmap, Format format, uint32_t max_mip_count)
{
    if (max_mip_count == 0 || uint32_t(stdx::bit_width(std::max(bitmap->width(), bitmap->height()))) <= max_mip_count)
        return bitmap;

    const bool srgb = get_format_info(format).is_srgb_format();
    const auto pixel_format = bitmap->pixel_format();
    const auto component_type = bitmap->component_type();
    const bool source_gamma = bitmap->srgb_gamma();
    while (uint32_t(stdx::bit_width(std::max(bitmap->width(), bitmap->height()))) > max_mip_count) {
        // A borrowed view changes interpretation without mutating the caller's bitmap.
        Bitmap source(pixel_format, component_type, bitmap->width(), bitmap->height(), 0, {}, bitmap->data(), srgb);
        bitmap = source.convert(pixel_format, Bitmap::ComponentType::float32, false);
        const uint32_t width = std::max(1u, bitmap->width() / 2);
        const uint32_t height = std::max(1u, bitmap->height() / 2);
        bitmap = bitmap->resample(width, height);
        bitmap = bitmap->convert(pixel_format, component_type, srgb);
    }
    bitmap->set_srgb_gamma(source_gamma);
    return bitmap;
}

inline SourceImage convert_bitmap(Device* device, ref<Bitmap> bitmap, const TextureLoader::Options& options)
{
    using PixelFormat = Bitmap::PixelFormat;

    auto [format, convert_to_rgba] = determine_texture_format(device, bitmap, options);

    if (bitmap->pixel_format() == PixelFormat::ya && options.ya_handling == YAHandling::preserve_as_rg) {
        bitmap = convert_ya_to_rg(bitmap);
    } else if (convert_to_rgba) {
        bitmap = bitmap->convert(PixelFormat::rgba, bitmap->component_type(), bitmap->srgb_gamma());
    }

    return SourceImage{
        .bitmap = reduce_bitmap(std::move(bitmap), format, options.max_mip_count),
        .format = format,
    };
}

inline SourceImage load_source_image(Stream* stream)
{
    SourceImage source_image;
    if (DDSFile::detect_dds_file(stream)) {
        source_image.dds_file = ref(new DDSFile(stream));
        source_image.format = get_format(DXGI_FORMAT(source_image.dds_file->dxgi_format()));
    } else if (Bitmap::detect_file_format(stream) != Bitmap::FileFormat::unknown) {
        source_image.bitmap = ref(new Bitmap(stream));
    }
    return source_image;
}

inline SourceImage load_source_image(const std::filesystem::path& path)
{
    FileStream stream(path, FileStream::Mode::read);
    return load_source_image(&stream);
}

inline SourceImage load_and_convert_source_image(Device* device, Stream* stream, const TextureLoader::Options& options)
{
    SourceImage source_image = load_source_image(stream);
    if (source_image.bitmap) {
        source_image = convert_bitmap(device, std::move(source_image.bitmap), options);
    }
    return source_image;
}

inline SourceImage
load_and_convert_source_image(Device* device, const std::filesystem::path& path, const TextureLoader::Options& options)
{
    SourceImage source_image = load_source_image(path);
    if (source_image.bitmap) {
        source_image = convert_bitmap(device, std::move(source_image.bitmap), options);
    }
    return source_image;
}

inline ref<Texture> create_texture(
    Device* device,
    Blitter* blitter,
    CommandEncoder* command_encoder,
    SourceImage source_image,
    const TextureLoader::Options& options
)
{
    if (source_image.bitmap) {
        const Bitmap* bitmap = source_image.bitmap;
        bool allocate_mips = options.allocate_mips || options.generate_mips;

        TextureUsage usage = get_effective_texture_usage(device, options);

        ref<Texture> texture = device->create_texture({
            .type = TextureType::texture_2d,
            .format = source_image.format,
            .width = bitmap->width(),
            .height = bitmap->height(),
            .mip_count = allocate_mips ? ALL_MIPS : 1u,
            .usage = usage,
        });

        SubresourceData subresource_data{
            .data = bitmap->data(),
            .row_pitch = bitmap->width() * bitmap->bytes_per_pixel(),
        };

        command_encoder->upload_texture_data(texture, 0, 0, subresource_data);
        if (options.generate_mips) {
            blitter->generate_mips(command_encoder, texture);
        }

        return texture;
    } else if (source_image.dds_file) {
        const DDSFile* dds_file = source_image.dds_file;
        const auto& [texture_type, layer_count]
            = get_texture_type_and_layer_count(dds_file->type(), dds_file->array_size());
        uint32_t first_mip = 0;
        uint32_t width = dds_file->width();
        uint32_t height = dds_file->height();
        uint32_t depth = dds_file->depth();
        while (options.max_mip_count != 0
               && uint32_t(stdx::bit_width(std::max({width, height, depth}))) > options.max_mip_count
               && first_mip + 1 < dds_file->mip_count()) {
            ++first_mip;
            width = std::max(1u, width / 2);
            height = std::max(1u, height / 2);
            depth = std::max(1u, depth / 2);
        }
        if (options.max_mip_count != 0
            && uint32_t(stdx::bit_width(std::max({width, height, depth}))) > options.max_mip_count) {
            log_warn(
                "DDS has no mip fitting max_mip_count {}; using smallest available mip {}x{}x{}",
                options.max_mip_count,
                width,
                height,
                depth
            );
        }
        short_vector<SubresourceData, 16> subresource_data;
        for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            for (uint32_t mip_index = first_mip; mip_index < dds_file->mip_count(); ++mip_index) {
                uint32_t row_pitch;
                uint32_t slice_pitch;
                dds_file->get_subresource_pitch(mip_index, &row_pitch, &slice_pitch);
                uint32_t mip_depth = dds_file->type() == DDSFile::TextureType::texture_3d
                    ? std::max(1u, dds_file->depth() >> mip_index)
                    : 1u;
                subresource_data.push_back({
                    .data = dds_file->get_subresource_data(mip_index, layer_index),
                    .size = size_t(slice_pitch) * mip_depth,
                    .row_pitch = row_pitch,
                    .slice_pitch = slice_pitch,
                });
            }
        }

        return device->create_texture({
            .type = texture_type,
            .format = source_image.format,
            .width = width,
            .height = height,
            .depth = depth,
            .array_length = dds_file->array_size(),
            .mip_count = dds_file->mip_count() - first_mip,
            .usage = options.usage,
            .data = subresource_data,
        });
    } else {
        SGL_THROW("Unsupported source image type");
    }
}

// Keep the estimate conservative without duplicating backend format selection.
inline size_t
estimate_bitmap_memory(const Bitmap::Info& info, const TextureLoader::Options& options, size_t decode_bytes, bool owned)
{
    uint32_t channels = info.channel_count;
    if (info.pixel_format == Bitmap::PixelFormat::rgb
        || (info.pixel_format == Bitmap::PixelFormat::y && options.y_handling == YHandling::expand_to_rgba)
        || (info.pixel_format == Bitmap::PixelFormat::ya && options.ya_handling == YAHandling::expand_to_rgba))
        channels = 4;
    const size_t component_bytes = DataStruct::type_size(info.component_type);
    const size_t pixel_bytes = channels * component_bytes;
    uint32_t width = info.width, height = info.height;
    size_t current_bytes = owned ? size_t(width) * height * info.channel_count * component_bytes : 0;
    size_t peak = std::max(decode_bytes, current_bytes);
    if (channels != info.channel_count || info.pixel_format == Bitmap::PixelFormat::ya) {
        const size_t converted_bytes = size_t(width) * height * pixel_bytes;
        peak = std::max(peak, current_bytes + converted_bytes);
        current_bytes = converted_bytes;
    }
    const uint32_t mip_count = uint32_t(stdx::bit_width(std::max(width, height)));
    if (options.max_mip_count && mip_count > options.max_mip_count) {
        // The first halving bounds all later conversion and resampling peaks.
        const uint32_t next_width = std::max(1u, width / 2), next_height = std::max(1u, height / 2);
        const size_t float_pixel_bytes = size_t(channels) * sizeof(float);
        const size_t linear = size_t(width) * height * float_pixel_bytes;
        const size_t horizontal = size_t(next_width) * height * float_pixel_bytes;
        const size_t reduced = size_t(next_width) * next_height * float_pixel_bytes;
        peak = std::max({peak, current_bytes + linear, linear + horizontal + 2 * reduced});
        const uint32_t reductions = mip_count - options.max_mip_count;
        width = std::max(1u, width >> reductions);
        height = std::max(1u, height >> reductions);
    }
    const size_t output = size_t(width) * height * pixel_bytes;
    // Allow a retained result and a staging copy, including conservative row alignment.
    const size_t staging = align_to(size_t(256), size_t(width) * pixel_bytes) * height;
    const size_t estimate = peak + output + staging + 32768;
    return estimate + div_round_up(estimate, size_t(8));
}

inline size_t estimate_source_memory(const Bitmap* bitmap, const TextureLoader::Options& options)
{
    return estimate_bitmap_memory(
        {bitmap->width(), bitmap->height(), bitmap->pixel_format(), bitmap->component_type(), bitmap->channel_count()},
        options,
        0,
        false
    );
}

inline size_t estimate_source_memory(const std::filesystem::path& path, const TextureLoader::Options& options)
{
    FileStream stream(path, FileStream::Mode::read);
    const size_t encoded = stream.size();
    if (DDSFile::detect_dds_file(&stream)) {
        const size_t estimate = 2 * encoded + 65536;
        return estimate + div_round_up(estimate, size_t(8));
    }
    const auto format = Bitmap::detect_file_format(&stream);
    const auto info = Bitmap::read_info(&stream, format);
    const size_t pixels = size_t(info.width) * info.height * info.channel_count;
    const size_t decoded = pixels * DataStruct::type_size(info.component_type);
    // Cover the result plus image-sized codec scratch (including encoded input).
    size_t decode_bytes = encoded + 2 * decoded;
#if !SGL_HAS_OPENEXR
    if (format == Bitmap::FileFormat::exr)
        // TinyEXR retains encoded input, channel planes and the interleaved result.
        decode_bytes = encoded + pixels * sizeof(float) + decoded;
#endif
    return estimate_bitmap_memory(info, options, decode_bytes, true);
}

inline SourceImage
load_and_convert_source_image(Device* device, const Bitmap* bitmap, const TextureLoader::Options& options)
{
    return convert_bitmap(device, ref(const_cast<Bitmap*>(bitmap)), options);
}

/// Load ordered batches within an estimated memory budget.
/// Source is std::filesystem::path or const Bitmap*; options has one entry per source.
/// Caller-owned bitmaps must remain alive until this function returns.
/// consume(begin, images, handles) runs on the calling thread after launching each batch's tasks.
/// begin is the batch's offset in sources; images and handles use batch-local indices.
/// The callback clears each handle slot before waiting and releasing that task, then accesses its image.
/// It consumes/releases the images and finishes their GPU uploads before returning normally.
/// Both spans are borrowed for the callback only; this helper drains remaining handles on failure.
template<typename Source, typename Consume>
void load_batches(
    Device* device,
    std::span<const Source> sources,
    std::span<const TextureLoader::Options> options,
    uint64_t memory_budget,
    Consume&& consume
)
{
    for (size_t begin = 0; begin < sources.size();) {
        size_t end = begin;
        uint64_t remaining = memory_budget;
        while (end < sources.size() && remaining) {
            const uint64_t estimate = estimate_source_memory(sources[end], options[end]);
            if (end > begin && estimate > remaining)
                break;
            remaining -= std::min(estimate, remaining); // An oversized image is admitted alone.
            ++end;
        }
        std::vector<SourceImage> images(end - begin);
        std::vector<thread::TaskHandle> handles(images.size());
        try {
            for (size_t i = 0; i < images.size(); ++i)
                handles[i] = thread::do_async(
                    [&, i]
                    {
                        images[i] = load_and_convert_source_image(device, sources[begin + i], options[begin + i]);
                    }
                );
            consume(begin, std::span(images), std::span(handles));
        } catch (...) {
            // Workers borrow batch storage; drain them before propagating the original failure.
            for (thread::TaskHandle handle : handles) {
                if (handle) {
                    try {
                        thread::task_wait_and_release(handle);
                    } catch (...) {
                    }
                }
            }
            throw;
        }
        begin = end;
    }
}

inline std::vector<ref<Texture>> create_textures(
    Device* device,
    Blitter* blitter,
    std::span<SourceImage> source_images,
    std::span<thread::TaskHandle> source_image_tasks,
    std::span<const TextureLoader::Options> options
)
{
    SGL_ASSERT(source_images.size() == source_image_tasks.size());
    SGL_ASSERT(source_images.size() == options.size());
    std::vector<ref<Texture>> textures(source_images.size());
    ref<CommandEncoder> command_encoder = device->create_command_encoder();
    for (size_t i = 0; i < source_images.size(); ++i) {
        thread::task_wait_and_release(std::exchange(source_image_tasks[i], nullptr));
        textures[i] = create_texture(device, blitter, command_encoder, std::move(source_images[i]), options[i]);
        if ((i + 1) % BATCH_SIZE == 0 && (i + 1) < source_images.size()) {
            device->submit_command_buffer(command_encoder->finish());
            device->wait();
            command_encoder = device->create_command_encoder();
        }
    }
    device->submit_command_buffer(command_encoder->finish());
    device->wait();

    return textures;
}

inline ref<Texture> create_texture_array(
    Device* device,
    Blitter* blitter,
    std::span<SourceImage> source_images,
    std::span<thread::TaskHandle> source_image_tasks,
    const TextureLoader::Options& options,
    ref<Texture> texture,
    uint32_t first_layer,
    uint32_t layer_count
)
{
    SGL_ASSERT(source_images.size() == source_image_tasks.size());
    SGL_ASSERT(source_images.size() > 0);

    bool allocate_mips = options.allocate_mips || options.generate_mips;

    TextureUsage usage = get_effective_texture_usage(device, options);

    ref<CommandEncoder> command_encoder = device->create_command_encoder();

    for (size_t i = 0; i < source_images.size(); ++i) {
        thread::task_wait_and_release(std::exchange(source_image_tasks[i], nullptr));
        SourceImage source_image = std::move(source_images[i]);
        const Bitmap* bitmap = source_image.bitmap;
        if (!bitmap)
            SGL_THROW("Texture array requires all source images to be bitmaps");

        if (!texture) {
            texture = device->create_texture({
                .type = TextureType::texture_2d_array,
                .format = source_image.format,
                .width = bitmap->width(),
                .height = bitmap->height(),
                .array_length = layer_count,
                .mip_count = allocate_mips ? ALL_MIPS : 1u,
                .usage = usage,
            });
        } else {
            if (bitmap->width() != texture->width() || bitmap->height() != texture->height()
                || source_image.format != texture->format())
                SGL_THROW("Texture array requires all bitmaps to have the same dimensions and format");
        }

        if (i && (i % BATCH_SIZE == 0)) {
            device->submit_command_buffer(command_encoder->finish());
            device->wait();
            command_encoder = device->create_command_encoder();
        }

        SubresourceData subresource_data{
            .data = bitmap->data(),
            .size = bitmap->buffer_size(),
            .row_pitch = bitmap->width() * bitmap->bytes_per_pixel(),
        };
        command_encoder->upload_texture_data(texture, first_layer + narrow_cast<uint32_t>(i), 0, subresource_data);
        /// Release bitmap to free CPU memory
        source_image.bitmap = nullptr;

        if (options.generate_mips)
            blitter->generate_mips(command_encoder, texture, first_layer + narrow_cast<uint32_t>(i));
    }
    device->submit_command_buffer(command_encoder->finish());
    device->wait();

    return texture;
}

TextureLoader::TextureLoader(ref<Device> device, uint64_t memory_budget)
    : m_device(std::move(device))
    , m_memory_budget(memory_budget)
{
    SGL_CHECK(memory_budget > 0, "memory_budget must be positive");
    m_blitter = ref(new Blitter(m_device));
}

TextureLoader::~TextureLoader() = default;

ref<Texture> TextureLoader::load_texture(const Bitmap* bitmap, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = convert_bitmap(m_device, ref(const_cast<Bitmap*>(bitmap)), options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, std::move(source_image), options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

ref<Texture> TextureLoader::load_texture(Stream* stream, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = load_and_convert_source_image(m_device.get(), stream, options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, std::move(source_image), options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

ref<Texture> TextureLoader::load_texture(const std::filesystem::path& path, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = load_and_convert_source_image(m_device.get(), path, options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, std::move(source_image), options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const Bitmap*> bitmaps, std::optional<Options> options_)
{
    std::vector<Options> options(bitmaps.size(), options_.value_or(Options{}));
    return load_textures(bitmaps, options);
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const Bitmap*> bitmaps, std::span<const Options> options)
{
    SGL_CHECK(bitmaps.size() == options.size(), "Number of options must be equal to the number of bitmaps");

    std::vector<ref<Texture>> textures(bitmaps.size());
    load_batches<const Bitmap*>(
        m_device,
        bitmaps,
        options,
        m_memory_budget,
        [&](size_t begin, auto images, auto tasks)
        {
            auto batch = create_textures(m_device, m_blitter, images, tasks, options.subspan(begin, images.size()));
            std::move(batch.begin(), batch.end(), textures.begin() + begin);
        }
    );
    return textures;
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const std::filesystem::path> paths, std::optional<Options> options_)
{
    std::vector<Options> options(paths.size(), options_.value_or(Options{}));
    return load_textures(paths, options);
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const std::filesystem::path> paths, std::span<const Options> options)
{
    SGL_CHECK(paths.size() == options.size(), "Number of options must be equal to the number of paths");

    std::vector<ref<Texture>> textures(paths.size());
    load_batches<std::filesystem::path>(
        m_device,
        paths,
        options,
        m_memory_budget,
        [&](size_t begin, auto images, auto tasks)
        {
            auto batch = create_textures(m_device, m_blitter, images, tasks, options.subspan(begin, images.size()));
            std::move(batch.begin(), batch.end(), textures.begin() + begin);
        }
    );
    return textures;
}

ref<Texture> TextureLoader::load_texture_array(std::span<const Bitmap*> bitmaps, std::optional<Options> options_)
{
    if (bitmaps.empty())
        return nullptr;

    Options options = options_.value_or(Options{});

    ref<Texture> texture;
    const std::vector<Options> per_image_options(bitmaps.size(), options);
    load_batches<const Bitmap*>(
        m_device,
        bitmaps,
        per_image_options,
        m_memory_budget,
        [&](size_t begin, auto images, auto tasks)
        {
            texture = create_texture_array(
                m_device,
                m_blitter,
                images,
                tasks,
                options,
                texture,
                narrow_cast<uint32_t>(begin),
                narrow_cast<uint32_t>(bitmaps.size())
            );
        }
    );
    return texture;
}

ref<Texture>
TextureLoader::load_texture_array(std::span<const std::filesystem::path> paths, std::optional<Options> options_)
{
    if (paths.empty())
        return nullptr;

    Options options = options_.value_or(Options{});

    ref<Texture> texture;
    const std::vector<Options> per_image_options(paths.size(), options);
    load_batches<std::filesystem::path>(
        m_device,
        paths,
        per_image_options,
        m_memory_budget,
        [&](size_t begin, auto images, auto tasks)
        {
            texture = create_texture_array(
                m_device,
                m_blitter,
                images,
                tasks,
                options,
                texture,
                narrow_cast<uint32_t>(begin),
                narrow_cast<uint32_t>(paths.size())
            );
        }
    );
    return texture;
}

} // namespace sgl
