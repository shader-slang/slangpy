// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "texture_loader.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"
#include "sgl/device/blit.h"
#include "sgl/device/native_formats.h"

#include "sgl/core/error.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/dds_file.h"
#include "sgl/core/file_stream.h"
#include "sgl/core/timer.h"
#include "sgl/core/thread.h"

#include <map>

namespace sgl {

static constexpr size_t BATCH_SIZE = 32;

/// Holds a source image to be uploaded to a texture.
/// Data can either be a bitmap or a DDS file.
struct SourceImage {
    ref<Bitmap> bitmap;
    ref<DDSFile> dds_file;
    Format format{Format::undefined};
};

/**
 * \brief Determine the texture format given a bitmap.
 *
 * Uses the following option flags to affect the format determination:
 * - \c Options::extend_alpha
 *   RGB bitmap that has no supported format will be determined as RGBA (if a RGBA format exists).
 * - \c Options::load_as_srgb
 *   8-bit RGBA bitmap with sRGB gamma will be determined as \c Format::rgba8_unorm_srgb.
 * - \c Options::load_as_normalized
 *   8/16-bit integer bitmap will be determined as normalized resource format.
 *
 * \param bitmap Bitmap to determine format for.
 * \param options Texture loading options.
 * \return A pair containing the determined format and flag if the bitmap needs to be converted to RGBA to match the format.
 */
inline std::pair<Format, bool> determine_texture_format(const Bitmap* bitmap, const TextureLoader::Options& options)
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

    PixelFormat pixel_format = bitmap->pixel_format();
    if (pixel_format == PixelFormat::y)
        pixel_format = PixelFormat::r;
    ComponentType component_type = bitmap->component_type();
    FormatFlags format_flags = FormatFlags::none;
    if (options.load_as_normalized && DataStruct::is_integer(component_type))
        format_flags = FormatFlags::normalized;

    // Check if bitmap is RGB and we can convert to RGBA.
    bool convert_to_rgba = false;
    if (options.extend_alpha && pixel_format == PixelFormat::rgb) {
        bool rgb_format_supported
            = FORMAT_TABLE.find(make_key(PixelFormat::rgb, component_type, format_flags)) != FORMAT_TABLE.end();
        bool rgba_format_supported
            = FORMAT_TABLE.find(make_key(PixelFormat::rgba, component_type, format_flags)) != FORMAT_TABLE.end();
        if (!rgb_format_supported && rgba_format_supported) {
            convert_to_rgba = true;
            pixel_format = PixelFormat::rgba;
        }
    }

    // Use sRGB format if requested and supported.
    if (options.load_as_srgb && pixel_format == PixelFormat::rgba && component_type == ComponentType::uint8
        && bitmap->srgb_gamma())
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

inline SourceImage convert_bitmap(ref<Bitmap> bitmap, const TextureLoader::Options& options)
{
    auto [format, convert_to_rgba] = determine_texture_format(bitmap, options);
    return SourceImage{
        .bitmap = convert_to_rgba
            ? bitmap->convert(Bitmap::PixelFormat::rgba, bitmap->component_type(), bitmap->srgb_gamma())
            : bitmap,
        .format = format,
    };
}

inline SourceImage load_source_image(const std::filesystem::path& path)
{
    SourceImage source_image;
    FileStream stream(path, FileStream::Mode::read);
    if (DDSFile::detect_dds_file(&stream)) {
        source_image.dds_file = ref(new DDSFile(&stream));
        source_image.format = get_format(DXGI_FORMAT(source_image.dds_file->dxgi_format()));
    } else if (Bitmap::detect_file_format(&stream) != Bitmap::FileFormat::unknown) {
        source_image.bitmap = ref(new Bitmap(&stream));
    }
    return source_image;
}

inline SourceImage
load_and_convert_source_image(const std::filesystem::path& path, const TextureLoader::Options& options)
{
    SourceImage source_image = load_source_image(path);
    if (source_image.bitmap) {
        source_image = convert_bitmap(source_image.bitmap, options);
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

        TextureUsage usage = options.usage;
        if (options.generate_mips)
            usage |= TextureUsage::render_target;

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
        short_vector<SubresourceData, 16> subresource_data;
        for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            for (uint32_t mip_index = 0; mip_index < dds_file->mip_count(); ++mip_index) {
                uint32_t row_pitch;
                uint32_t slice_pitch;
                dds_file->get_subresource_pitch(mip_index, &row_pitch, &slice_pitch);
                subresource_data.push_back({
                    .data = dds_file->get_subresource_data(mip_index, layer_index),
                    .size = dds_file->resource_size(),
                    .row_pitch = row_pitch,
                    .slice_pitch = slice_pitch,
                });
            }
        }

        return device->create_texture({
            .type = texture_type,
            .format = source_image.format,
            .width = dds_file->width(),
            .height = dds_file->height(),
            .depth = dds_file->depth(),
            .array_length = dds_file->array_size(),
            .mip_count = dds_file->mip_count(),
            .usage = options.usage,
            .data = subresource_data,
        });
    } else {
        SGL_THROW("Unsupported source image type");
    }
}


inline std::vector<ref<Texture>> create_textures(
    Device* device,
    Blitter* blitter,
    std::span<std::future<SourceImage>> source_images,
    const TextureLoader::Options& options
)
{
    std::vector<ref<Texture>> textures(source_images.size());
    ref<CommandEncoder> command_encoder = device->create_command_encoder();
    for (size_t i = 0; i < source_images.size(); ++i) {
        textures[i] = create_texture(device, blitter, command_encoder, source_images[i].get(), options);
        if (i && (i % BATCH_SIZE == 0)) {
            device->submit_command_buffer(command_encoder->finish());
            command_encoder = device->create_command_encoder();
        }
    }
    device->submit_command_buffer(command_encoder->finish());

    return textures;
}

inline ref<Texture> create_texture_array(
    Device* device,
    Blitter* blitter,
    std::span<std::future<SourceImage>> source_images,
    const TextureLoader::Options& options
)
{
    SGL_ASSERT(source_images.size() > 0);

    bool allocate_mips = options.allocate_mips || options.generate_mips;

    TextureUsage usage = options.usage;
    if (options.generate_mips)
        usage |= TextureUsage::render_target;

    ref<Texture> texture;
    uint32_t first_width = 0;
    uint32_t first_height = 0;
    Format first_format = Format::undefined;

    ref<CommandEncoder> command_encoder = device->create_command_encoder();

    for (size_t i = 0; i < source_images.size(); ++i) {
        SourceImage source_image = source_images[i].get();
        const Bitmap* bitmap = source_image.bitmap;
        if (!bitmap)
            SGL_THROW("Texture array requires all source images to be bitmaps");

        if (i == 0) {
            texture = device->create_texture({
                .type = TextureType::texture_2d_array,
                .format = source_image.format,
                .width = bitmap->width(),
                .height = bitmap->height(),
                .array_length = narrow_cast<uint32_t>(source_images.size()),
                .mip_count = allocate_mips ? ALL_MIPS : 1u,
                .usage = usage,
            });
            first_width = bitmap->width();
            first_height = bitmap->height();
            first_format = source_image.format;
        } else {
            if (bitmap->width() != first_width || bitmap->height() != first_height
                || source_image.format != first_format)
                SGL_THROW("Texture array requires all bitmaps to have the same dimensions and format");
        }

        if (i && (i % BATCH_SIZE == 0)) {
            device->submit_command_buffer(command_encoder->finish());
            command_encoder = device->create_command_encoder();
        }

        SubresourceData subresource_data{
            .data = bitmap->data(),
            .size = bitmap->buffer_size(),
            .row_pitch = bitmap->width() * bitmap->bytes_per_pixel(),
        };
        command_encoder->upload_texture_data(texture, narrow_cast<uint32_t>(i), 0, subresource_data);

        if (options.generate_mips)
            blitter->generate_mips(command_encoder, texture, narrow_cast<uint32_t>(i));
    }
    device->submit_command_buffer(command_encoder->finish());

    return texture;
}

TextureLoader::TextureLoader(ref<Device> device)
    : m_device(std::move(device))
{
    m_blitter = ref(new Blitter(m_device));
}

TextureLoader::~TextureLoader() = default;

ref<Texture> TextureLoader::load_texture(const Bitmap* bitmap, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = convert_bitmap(ref(const_cast<Bitmap*>(bitmap)), options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, source_image, options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

ref<Texture> TextureLoader::load_texture(const std::filesystem::path& path, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});
    SourceImage source_image = load_and_convert_source_image(path, options);
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    ref<Texture> texture = create_texture(m_device, m_blitter, command_encoder, source_image, options);
    m_device->submit_command_buffer(command_encoder->finish());
    return texture;
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<const Bitmap*> bitmaps, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});

    // Convert bitmaps in parallel.
    std::vector<std::future<SourceImage>> source_images;
    source_images.reserve(bitmaps.size());
    for (const auto& bitmap : bitmaps)
        source_images.push_back(thread::do_async(convert_bitmap, ref(const_cast<Bitmap*>(bitmap)), options));

    return create_textures(m_device, m_blitter, source_images, options);
}

std::vector<ref<Texture>>
TextureLoader::load_textures(std::span<std::filesystem::path> paths, std::optional<Options> options_)
{
    Options options = options_.value_or(Options{});

    // Load & convert source images in parallel.
    std::vector<std::future<SourceImage>> source_images;
    source_images.reserve(paths.size());
    for (const auto& path : paths)
        source_images.push_back(thread::do_async(load_and_convert_source_image, path, options));

    return create_textures(m_device, m_blitter, source_images, options);
}

ref<Texture> TextureLoader::load_texture_array(std::span<const Bitmap*> bitmaps, std::optional<Options> options_)
{
    if (bitmaps.empty())
        return nullptr;

    Options options = options_.value_or(Options{});

    // Convert bitmaps in parallel.
    std::vector<std::future<SourceImage>> source_images;
    source_images.reserve(bitmaps.size());
    for (const auto& bitmap : bitmaps)
        source_images.push_back(thread::do_async(convert_bitmap, ref(const_cast<Bitmap*>(bitmap)), options));

    return create_texture_array(m_device, m_blitter, source_images, options);
}

ref<Texture> TextureLoader::load_texture_array(std::span<std::filesystem::path> paths, std::optional<Options> options_)
{
    if (paths.empty())
        return nullptr;

    Options options = options_.value_or(Options{});

    // Load & convert source images in parallel.
    std::vector<std::future<SourceImage>> source_images;
    source_images.reserve(paths.size());
    for (const auto& path : paths)
        source_images.push_back(thread::do_async(load_and_convert_source_image, path, options));

    return create_texture_array(m_device, m_blitter, source_images, options);
}

} // namespace sgl
