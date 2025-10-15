// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
This is a modified version of the bitmap.cpp file from Mitsuba 3.0 found at
https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/core/bitmap.cpp

Original license below:

Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to the author of this software, without
imposing a separate written license agreement for such Enhancements, then you
hereby grant the following license: a non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.
*/

#include "bitmap.h"

#include "sgl/core/config.h"
#include "sgl/core/macros.h"
#include "sgl/core/error.h"
#include "sgl/core/logger.h"
#include "sgl/core/file_stream.h"
#include "sgl/core/string.h"
#include "sgl/core/thread.h"
#include "sgl/core/type_utils.h"

#include "sgl/math/scalar_types.h"

#include "sgl/stl/bit.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

SGL_DIAGNOSTIC_PUSH
SGL_DISABLE_MSVC_WARNING(4996)
SGL_DISABLE_CLANG_WARNING("-Wdeprecated-declarations")
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
SGL_DIAGNOSTIC_POP

#if SGL_HAS_LIBPNG
#include <png.h>
#endif

#if SGL_HAS_LIBJPEG
#include <jpeglib.h>
#endif

#if SGL_HAS_OPENEXR
#include <ImfInputFile.h>
#include <ImfStandardAttributes.h>
#include <ImfRgbaYca.h>
#include <ImfOutputFile.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfStringAttribute.h>
#include <ImfIntAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfCompressionAttribute.h>
#include <ImfVecAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfVersion.h>
#include <ImfIO.h>
#include <ImathBox.h>
#include <IlmThreadPool.h>
#else
SGL_DIAGNOSTIC_PUSH
SGL_DISABLE_MSVC_WARNING(4245 4706 4702)
#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#include <tinyexr.h>
SGL_DIAGNOSTIC_POP
#endif

#include <algorithm>
#include <map>
#include <mutex>

SGL_DISABLE_MSVC_WARNING(4611)

namespace sgl {

Bitmap::Bitmap(
    PixelFormat pixel_format,
    ComponentType component_type,
    uint32_t width,
    uint32_t height,
    uint32_t channel_count,
    const std::vector<std::string>& channel_names,
    void* data,
    std::optional<bool> srgb_gamma
)
    : m_pixel_format(pixel_format)
    , m_component_type(component_type)
    , m_width(width)
    , m_height(height)
    , m_data(reinterpret_cast<uint8_t*>(data))
    , m_owns_data(false)
{
    SGL_CHECK(
        pixel_format != PixelFormat::multi_channel || channel_count > 0,
        "Expected non-zero channel count for multi-channel pixel format."
    );

    // If not specified, use sRGB gamma by default for 8-bit and 16-bit images.
    if (srgb_gamma.has_value())
        m_srgb_gamma = srgb_gamma.value();
    else
        m_srgb_gamma = (m_component_type == ComponentType::uint8) || (m_component_type == ComponentType::uint16);

    rebuild_pixel_struct(channel_count, channel_names);

    if (!m_data) {
        m_data = std::make_unique<uint8_t[]>(buffer_size());
        m_owns_data = true;
    }
}

Bitmap::Bitmap(const Bitmap& other)
    : Object() // gcc complains otherwise
    , m_pixel_format(other.m_pixel_format)
    , m_component_type(other.m_component_type)
    , m_pixel_struct(new DataStruct(*other.m_pixel_struct))
    , m_width(other.m_width)
    , m_height(other.m_height)
    , m_srgb_gamma(other.m_srgb_gamma)
    , m_data(new uint8_t[other.buffer_size()])
{
    std::memcpy(m_data.get(), other.m_data.get(), other.buffer_size());
}

Bitmap::Bitmap(Bitmap&& other)
    : m_pixel_format(std::exchange(other.m_pixel_format, PixelFormat(0)))
    , m_component_type(std::exchange(other.m_component_type, ComponentType(0)))
    , m_pixel_struct(std::exchange(other.m_pixel_struct, nullptr))
    , m_width(std::exchange(other.m_width, 0))
    , m_height(std::exchange(other.m_height, 0))
    , m_srgb_gamma(std::exchange(other.m_srgb_gamma, false))
    , m_data(std::move(other.m_data))
{
}

Bitmap::Bitmap(Stream* stream, FileFormat format)
{
    read(stream, format);
}

Bitmap::Bitmap(const std::filesystem::path& path, FileFormat format)
{
    FileStream stream(path, FileStream::Mode::read);
    read(&stream, format);
}

Bitmap::~Bitmap()
{
    if (!m_owns_data)
        m_data.release();
}

std::vector<ref<Bitmap>> Bitmap::read_multiple(std::span<std::filesystem::path> paths, FileFormat format)
{
    std::vector<std::future<ref<Bitmap>>> futures;
    futures.reserve(paths.size());
    for (const auto& path : paths)
        futures.push_back(
            thread::do_async(
                [](const std::filesystem::path& path, FileFormat format) { return make_ref<Bitmap>(path, format); },
                path,
                format
            )
        );
    std::vector<ref<Bitmap>> bitmaps;
    bitmaps.reserve(paths.size());
    for (auto& future : futures) {
        future.wait();
        bitmaps.push_back(future.get());
    }
    return bitmaps;
}

void Bitmap::write(Stream* stream, FileFormat format, int quality) const
{
    SGL_UNUSED(quality);

    auto fs = dynamic_cast<FileStream*>(stream);

    if (format == FileFormat::auto_) {
        SGL_CHECK(fs, "Unable to determine image file format without a filename.");
        std::string extension = string::to_lower(fs->path().extension().string());
        if (extension == ".png")
            format = FileFormat::png;
        else if (extension == ".jpg" || extension == ".jpeg")
            format = FileFormat::jpg;
        else if (extension == ".bmp")
            format = FileFormat::bmp;
        else if (extension == ".tga")
            format = FileFormat::tga;
        else if (extension == ".hdr" || extension == ".rgbe")
            format = FileFormat::hdr;
        else if (extension == ".exr")
            format = FileFormat::exr;
        else
            SGL_THROW("Unsupported image file extension \"%s\"", extension);
    }

    log_debug(
        "Writing {} file \"{}\" ({}x{}, {}, {}) ...",
        format,
        fs ? fs->path().string() : "<stream>",
        m_width,
        m_height,
        m_pixel_format,
        m_component_type
    );

    switch (format) {
    case FileFormat::png:
        if (quality == -1)
            quality = 5;
        write_png(stream, quality);
        break;
    case FileFormat::jpg:
        if (quality == -1)
            quality = 100;
        write_jpg(stream, quality);
        break;
    case FileFormat::bmp:
        write_bmp(stream);
        break;
    case FileFormat::tga:
        write_tga(stream);
        break;
    case FileFormat::hdr:
        write_hdr(stream);
        break;
    case FileFormat::exr:
        write_exr(stream, quality);
        break;
    default:
        SGL_THROW("Invalid file format!");
    }
}

void Bitmap::write(const std::filesystem::path& path, FileFormat format, int quality) const
{
    auto stream = make_ref<FileStream>(path, FileStream::Mode::write);
    write(stream, format, quality);
}

void Bitmap::write_async(const std::filesystem::path& path, FileFormat format, int quality) const
{
    // Increment reference count to ensure that the bitmap is not destroyed before written.
    this->inc_ref();
    thread::do_async(
        [=, this]()
        {
            this->write(path, format, quality);
            this->dec_ref();
        }
    );
}

std::vector<std::string> Bitmap::channel_names() const
{
    std::vector<std::string> names(m_pixel_struct->field_count());
    for (size_t i = 0; i < names.size(); ++i)
        names[i] = m_pixel_struct->operator[](i).name;
    return names;
}

void Bitmap::set_srgb_gamma(bool srgb_gamma)
{
    m_srgb_gamma = srgb_gamma;
    // Adjust pixel struct flags.
    if (m_pixel_format != PixelFormat::multi_channel) {
        for (DataStruct::Field& field : *m_pixel_struct) {
            if (field.name != "A") {
                if (m_srgb_gamma)
                    field.flags |= DataStruct::Flags::srgb_gamma;
                else
                    field.flags &= ~DataStruct::Flags::srgb_gamma;
            }
        }
    }
}

void Bitmap::clear()
{
    std::memset(m_data.get(), 0, buffer_size());
}

void Bitmap::vflip()
{
    size_t row_size = buffer_size() / m_height;
    size_t half_height = m_height / 2;
    uint8_t* temp = reinterpret_cast<uint8_t*>(alloca(row_size));
    uint8_t* data = uint8_data();
    for (size_t i = 0, j = m_height - 1; i < half_height; ++i) {
        std::memcpy(temp, data + i * row_size, row_size);
        std::memcpy(data + i * row_size, data + j * row_size, row_size);
        std::memcpy(data + j * row_size, temp, row_size);
        j--;
    }
}

std::vector<std::pair<std::string, ref<Bitmap>>> Bitmap::split() const
{
    if (m_pixel_format != PixelFormat::multi_channel)
        return {{"", ref(const_cast<Bitmap*>(this))}};

    // Split fields by prefix.
    std::multimap<std::string, std::pair<std::string, DataStruct::Field*>> split_fields;
    for (DataStruct::Field& field : *m_pixel_struct) {
        size_t pos = field.name.find_last_of(".");
        std::string prefix;
        std::string suffix;
        if (pos == std::string::npos) {
            suffix = field.name;
        } else {
            prefix = field.name.substr(0, pos);
            suffix = field.name.substr(pos + 1);
        }
        split_fields.emplace(prefix, std::make_pair(suffix, &field));
    }

    std::vector<std::pair<std::string, ref<Bitmap>>> result;

    for (auto it = split_fields.begin(); it != split_fields.end();) {
        std::string prefix = it->first;
        auto range = split_fields.equal_range(prefix);

        std::vector<std::string> field_names;
        std::string field_format;
        for (auto it2 = range.first; it2 != range.second; ++it2) {
            field_names.push_back(it2->second.first);
            if (!field_format.empty())
                field_format += "|";
            field_format += string::to_lower(it2->second.first);
        }

        // Determine common pixel formats.
        PixelFormat pixel_format = PixelFormat::multi_channel;
        if (field_format == "y")
            pixel_format = PixelFormat::y;
        else if (field_format == "y|a")
            pixel_format = PixelFormat::ya;
        else if (field_format == "r")
            pixel_format = PixelFormat::r;
        else if (field_format == "r|g")
            pixel_format = PixelFormat::rg;
        else if (field_format == "r|g|b")
            pixel_format = PixelFormat::rgb;
        else if (field_format == "r|g|b|a")
            pixel_format = PixelFormat::rgba;

        ref<Bitmap> target;
        if (pixel_format == PixelFormat::multi_channel) {
            target = ref(new Bitmap(
                pixel_format,
                m_component_type,
                m_width,
                m_height,
                narrow_cast<uint32_t>(field_names.size()),
                field_names
            ));
        } else {
            target = ref(new Bitmap(pixel_format, m_component_type, m_width, m_height));
        }

        target->set_srgb_gamma(m_srgb_gamma);

        ref<DataStruct> target_struct = ref(new DataStruct(*target->pixel_struct()));
        for (auto it2 = range.first; it2 != range.second; ++it2) {
            std::string field_name
                = pixel_format == PixelFormat::multi_channel ? it2->second.first : string::to_upper(it2->second.first);
            target_struct->field(field_name).name = it2->second.second->name;
        }

        DataStructConverter converter(m_pixel_struct, target_struct);
        converter.convert(data(), target->data(), pixel_count());

        result.push_back({prefix, target});
        it = range.second;
    }

    return result;
}

ref<Bitmap> Bitmap::convert(PixelFormat pixel_format, ComponentType component_type, bool srgb_gamma) const
{
    uint32_t channel_count = 0;
    std::vector<std::string> channel_names;
    if (m_pixel_format == PixelFormat::multi_channel && pixel_format == PixelFormat::multi_channel) {
        channel_count = this->channel_count();
        channel_names = this->channel_names();
    }
    ref<Bitmap> result
        = make_ref<Bitmap>(pixel_format, component_type, m_width, m_height, channel_count, channel_names);
    result->set_srgb_gamma(srgb_gamma);
    convert(result);
    return result;
}

void Bitmap::convert(Bitmap* target) const
{
    if (width() != target->width() || height() != target->height())
        SGL_THROW(
            "Bitmap dimensions do not match (source image is {}x{} vs target image {}x{})!",
            width(),
            height(),
            target->width(),
            target->height()
        );

    bool src_is_rgb = m_pixel_format == PixelFormat::rgb || m_pixel_format == PixelFormat::rgba;
    bool src_is_y = m_pixel_format == PixelFormat::y || m_pixel_format == PixelFormat::ya;

    const DataStruct* src_struct = pixel_struct();
    ref<DataStruct> dst_struct = make_ref<DataStruct>(*target->pixel_struct());

    for (DataStruct::Field& field : *dst_struct) {
        if (src_struct->has_field(field.name)) {
            continue;
        }
        if (field.name == "Y") {
            if (src_is_y) {
                field.name = "Y";
                continue;
            } else if (src_is_rgb) {
                field.blend = {{0.2126, "R"}, {0.7152, "G"}, {0.0722, "B"}};
                continue;
            }
        }
        if (field.name == "R") {
            if (src_is_y) {
                field.name = "Y";
                continue;
            }
        }
        if (field.name == "G") {
            if (src_is_y) {
                field.name = "Y";
                continue;
            }
        }
        if (field.name == "B") {
            if (src_is_y) {
                field.name = "Y";
                continue;
            }
        }
        if (field.name == "A") {
            field.default_value = DataStruct::is_float(field.type) ? 1.0 : DataStruct::type_range(field.type).second;
            field.flags |= DataStruct::Flags::default_;
            continue;
        }
        SGL_THROW("Unable to convert bitmap: cannot determine how to derive field \"{}\" in target image!", field.name);
    }

    ref<DataStructConverter> converter = make_ref<DataStructConverter>(src_struct, dst_struct);
    converter->convert(data(), target->data(), pixel_count());
}

bool Bitmap::operator==(const Bitmap& other) const
{
    return m_pixel_format == other.m_pixel_format && m_component_type == other.m_component_type
        && *m_pixel_struct == *other.m_pixel_struct && m_width == other.m_width && m_height == other.m_height
        && m_srgb_gamma == other.m_srgb_gamma && std::memcmp(m_data.get(), other.m_data.get(), buffer_size()) == 0;
}

std::string Bitmap::to_string() const
{
    return fmt::format(
        "Bitmap(\n"
        "  pixel_format = {},\n"
        "  component_type = {},\n"
        "  width = {},\n"
        "  height = {},\n"
        "  srgb_gamma = {},\n"
        "  pixel_struct = {},\n"
        "  data = {}\n"
        ")",
        m_pixel_format,
        m_component_type,
        m_width,
        m_height,
        m_srgb_gamma,
        string::indent(m_pixel_struct->to_string()),
        string::format_byte_size(buffer_size())
    );
}

Bitmap::FileFormat Bitmap::detect_file_format(Stream* stream)
{
    FileFormat format = FileFormat::unknown;

    size_t pos = stream->tell();
    uint8_t header[8];
    stream->read(header, 8);

    if (header[0] == 'B' && header[1] == 'M') {
        format = FileFormat::bmp;
    } else if (header[0] == '#' && header[1] == '?') {
        format = FileFormat::hdr;
    } else if (header[0] == 0xFF && header[1] == 0xD8) {
        format = FileFormat::jpg;
    } else if (header[0] == 0x89 && header[1] == 0x50 && //
               header[2] == 0x4E && header[3] == 0x47 && //
               header[4] == 0x0D && header[5] == 0x0A && //
               header[6] == 0x1A && header[7] == 0x0A) {
        format = FileFormat::png;
    } else if (header[0] == 0x76 && header[1] == 0x2F && //
               header[2] == 0x31 && header[3] == 0x01) {
        format = FileFormat::exr;
    } else {
        // Check for TGAv1 file
        char spec[10];
        stream->read(spec, 10);
        if ((header[1] == 0 || header[1] == 1)
            && (header[2] == 1 || header[2] == 2 || header[2] == 3 || header[2] == 9 || header[2] == 10
                || header[2] == 11)
            && (spec[8] == 8 || spec[8] == 16 || spec[8] == 24 || spec[8] == 32))
            format = FileFormat::tga;

        // Check for a TGAv2 file
        char footer[18];
        stream->seek(stream->size() - 18);
        stream->read(footer, 18);
        if (footer[17] == 0 && strncmp(footer, "TRUEVISION-XFILE.", 17) == 0)
            format = FileFormat::tga;
    }
    stream->seek(pos);
    return format;
}

void Bitmap::static_init()
{
    // IlmThread::ThreadPool::globalThreadPool().setThreadProvider(new EXRThreadPool());
}

void Bitmap::static_shutdown() { }

void Bitmap::rebuild_pixel_struct(uint32_t channel_count, const std::vector<std::string>& channel_names)
{
    std::vector<std::string> channels;
    switch (m_pixel_format) {
    case PixelFormat::y:
        channels = {"Y"};
        break;
    case PixelFormat::ya:
        channels = {"Y", "A"};
        break;
    case PixelFormat::r:
        channels = {"R"};
        break;
    case PixelFormat::rg:
        channels = {"R", "G"};
        break;
    case PixelFormat::rgb:
        channels = {"R", "G", "B"};
        break;
    case PixelFormat::rgba:
        channels = {"R", "G", "B", "A"};
        break;
    case PixelFormat::multi_channel:
        SGL_ASSERT(channel_count > 0);
        channels = channel_names;
        if (channels.empty()) {
            for (uint32_t i = 0; i < channel_count; ++i)
                channels.push_back(fmt::format("channel{}", i));
        }
        if (channels.size() != channel_count)
            SGL_THROW("Channel names do not match the channel count!");
        std::vector<std::string> sorted_channels{channels};
        std::sort(sorted_channels.begin(), sorted_channels.end());
        if (std::unique(sorted_channels.begin(), sorted_channels.end()) != sorted_channels.end())
            SGL_THROW("Channel names must not contain duplicates!");
        break;
    }

    m_pixel_struct = make_ref<DataStruct>();
    for (const auto& channel : channels) {
        bool is_alpha = m_pixel_format != PixelFormat::multi_channel && channel == "A";
        DataStruct::Flags flags = DataStruct::Flags::none;
        if (DataStruct::is_integer(m_component_type) && DataStruct::type_size(m_component_type) <= 2)
            flags |= DataStruct::Flags::normalized;
        if (m_srgb_gamma && !is_alpha)
            flags |= DataStruct::Flags::srgb_gamma;
        m_pixel_struct->append(channel, m_component_type, flags);
    }
}

void Bitmap::read(Stream* stream, FileFormat format)
{
    if (format == FileFormat::auto_)
        format = detect_file_format(stream);

    switch (format) {
    case FileFormat::png:
        read_png(stream);
        break;
    case FileFormat::jpg:
        read_jpg(stream);
        break;
    case FileFormat::bmp:
        read_bmp(stream);
        break;
    case FileFormat::tga:
        read_tga(stream);
        break;
    case FileFormat::hdr:
        read_hdr(stream);
        break;
    case FileFormat::exr:
        read_exr(stream);
        break;
    default:
        SGL_THROW("Unknown file format!");
    }
}

void Bitmap::check_required_format(
    std::string_view file_format,
    std::vector<Bitmap::PixelFormat> allowed_pixel_formats,
    std::vector<Bitmap::ComponentType> allowed_component_types
) const
{
    if (std::find(allowed_pixel_formats.begin(), allowed_pixel_formats.end(), m_pixel_format)
        == allowed_pixel_formats.end())
        SGL_THROW(
            "Unsupported pixel format {} for writing {} image, expected one of: {}.",
            m_pixel_format,
            file_format,
            fmt::join(allowed_pixel_formats, ", ")
        );
    if (std::find(allowed_component_types.begin(), allowed_component_types.end(), m_component_type)
        == allowed_component_types.end())
        SGL_THROW(
            "Unsupported component type {} for writing {} image, expected one of: {}.",
            m_component_type,
            file_format,
            fmt::join(allowed_component_types, ", ")
        );
}

// ----------------------------------------------------------------------------
// STB I/O
// ----------------------------------------------------------------------------

static void stbi_write_func(void* context, void* data, int size)
{
    static_cast<Stream*>(context)->write(data, size);
}

struct StreamReader {
    Stream* stream;
    size_t initial_pos;
    bool is_eof{false};
    stbi_io_callbacks callbacks;

    StreamReader(Stream* stream)
        : stream(stream)
        , initial_pos(stream->tell())
        , callbacks({.read = &read, .skip = &skip, .eof = &eof})
    {
    }

    void reset() { stream->seek(initial_pos); }

    static int read(void* user, char* data, int size)
    {
        StreamReader* reader = static_cast<StreamReader*>(user);
        try {
            reader->stream->read(data, size);
            return size;
        } catch (const EOFException& e) {
            reader->is_eof = true;
            return static_cast<int>(e.gcount());
        }
    }

    static void skip(void* user, int n)
    {
        StreamReader* reader = static_cast<StreamReader*>(user);
        reader->stream->seek(reader->stream->tell() + n);
    }

    static int eof(void* user)
    {
        StreamReader* reader = static_cast<StreamReader*>(user);
        return reader->is_eof;
    }
};

void Bitmap::read_stb(Stream* stream, const char* format, bool is_srgb, bool is_hdr)
{
    StreamReader reader(stream);

    int w, h, c;
    if (!stbi_info_from_callbacks(&reader.callbacks, &reader, &w, &h, &c))
        SGL_THROW(fmt::format("Failed to read {} file!", format));
    reader.reset();

    m_width = w;
    m_height = h;
    switch (c) {
    case 1:
        m_pixel_format = PixelFormat::y;
        break;
    case 2:
        m_pixel_format = PixelFormat::ya;
        break;
    case 3:
        m_pixel_format = PixelFormat::rgb;
        break;
    case 4:
        m_pixel_format = PixelFormat::rgba;
        break;
    default:
        SGL_THROW("Unsupported number of channels {}!", c);
    }
    m_component_type = is_hdr ? ComponentType::float32 : ComponentType::uint8;
    m_srgb_gamma = is_srgb;

    rebuild_pixel_struct();

    auto fs = dynamic_cast<FileStream*>(stream);
    log_debug(
        "Reading {} file \"{}\" ({}x{}, {}, {}) ...",
        format,
        fs ? fs->path().string() : "<stream>",
        m_width,
        m_height,
        m_pixel_format,
        m_component_type
    );

    void* data = nullptr;
    switch (m_component_type) {
    case ComponentType::uint8:
        data = reinterpret_cast<void*>(stbi_load_from_callbacks(&reader.callbacks, &reader, &w, &h, &c, c));
        break;
    case ComponentType::float32:
        data = reinterpret_cast<void*>(stbi_loadf_from_callbacks(&reader.callbacks, &reader, &w, &h, &c, c));
        break;
    default:
        SGL_THROW("Unsupported component type!", m_component_type);
    }
    if (!data)
        SGL_THROW(fmt::format("Failed to read {} file!", format));

    SGL_ASSERT_EQ(m_width, static_cast<uint32_t>(w));
    SGL_ASSERT_EQ(m_height, static_cast<uint32_t>(h));

    m_data = std::unique_ptr<uint8_t[]>(reinterpret_cast<uint8_t*>(data));
    m_owns_data = true;
}

// ----------------------------------------------------------------------------
// PNG I/O
// ----------------------------------------------------------------------------

#if SGL_HAS_LIBPNG

static void png_flush_data(png_structp png_ptr)
{
    static_cast<Stream*>(png_get_io_ptr(png_ptr))->flush();
}

static void png_read_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    static_cast<Stream*>(png_get_io_ptr(png_ptr))->read(data, length);
}

static void png_write_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    static_cast<Stream*>(png_get_io_ptr(png_ptr))->write(data, length);
}

static void png_error_func(png_structp, png_const_charp msg)
{
    SGL_THROW("libpng error: {}\n", msg);
}

static void png_warn_func(png_structp, png_const_charp msg)
{
    if (std::strstr(msg, "iCCP: known incorrect sRGB profile") != nullptr)
        return;
    log_warn("libpng warning: {}\n", msg);
}

void Bitmap::read_png(Stream* stream)
{
    // Create buffers.
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, &png_error_func, &png_warn_func);
    if (png_ptr == nullptr)
        SGL_THROW("Failed to create PNG data structure!");

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        SGL_THROW("Failed to create PNG information structure!");
    }

    // Setup error handling.
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        SGL_THROW("Error reading the PNG file!");
    }

    // Setup read callback.
    png_set_read_fn(png_ptr, stream, (png_rw_ptr)png_read_data);

    int bit_depth, color_type, interlace_type, compression_type, filter_type;
    png_read_info(png_ptr, info_ptr);
    png_uint_32 width = 0, height = 0;
    png_get_IHDR(
        png_ptr,
        info_ptr,
        &width,
        &height,
        &bit_depth,
        &color_type,
        &interlace_type,
        &compression_type,
        &filter_type
    );

    // Expand 1-, 2- and 4-bit grayscale.
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    // Always expand paletted files.
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    // Expand transparency to a proper alpha channel.
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    // Swap the byte order on little endian machines.
    if constexpr (stdx::endian::native == stdx::endian::little) {
        if (bit_depth == 16)
            png_set_swap(png_ptr);
    }

    // Update the information based on the transformations.
    png_read_update_info(png_ptr, info_ptr);
    png_get_IHDR(
        png_ptr,
        info_ptr,
        &width,
        &height,
        &bit_depth,
        &color_type,
        &interlace_type,
        &compression_type,
        &filter_type
    );
    m_width = width;
    m_height = height;

    switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:
        m_pixel_format = PixelFormat::y;
        break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        m_pixel_format = PixelFormat::ya;
        break;
    case PNG_COLOR_TYPE_RGB:
        m_pixel_format = PixelFormat::rgb;
        break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
        m_pixel_format = PixelFormat::rgba;
        break;
    default:
        SGL_THROW("Unknown color type {}!", color_type);
    }

    switch (bit_depth) {
    case 8:
        m_component_type = ComponentType::uint8;
        break;
    case 16:
        m_component_type = ComponentType::uint16;
        break;
    default:
        SGL_THROW("Unsupported bit depth {}!", bit_depth);
    }

    // TODO should we detect non-srgb pngs?
    m_srgb_gamma = true;

    rebuild_pixel_struct();

#if 0
    m_srgb_gamma = true;
    m_premultiplied_alpha = false;

    rebuild_struct();

    // Load any string-valued metadata
    int text_idx = 0;
    png_textp text_ptr;
    png_get_text(png_ptr, info_ptr, &text_ptr, &text_idx);

    for (int i = 0; i < text_idx; ++i, text_ptr++)
        m_metadata.set_string(text_ptr->key, text_ptr->text);
#endif

    auto fs = dynamic_cast<FileStream*>(stream);
    log_debug(
        "Reading PNG file \"{}\" ({}x{}, {}, {}) ...",
        fs ? fs->path().string() : "<stream>",
        m_width,
        m_height,
        m_pixel_format,
        m_component_type
    );

    size_t size = buffer_size();
    m_data = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
    m_owns_data = true;

    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    SGL_ASSERT(row_bytes == size / m_height);

    png_bytepp rows = static_cast<png_bytepp>(alloca(sizeof(png_bytep) * m_height));
    for (size_t i = 0; i < m_height; i++)
        rows[i] = uint8_data() + i * row_bytes;

    png_read_image(png_ptr, rows);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
}

void Bitmap::write_png(Stream* stream, int compression) const
{
    check_required_format(
        "PNG",
        {PixelFormat::y, PixelFormat::ya, PixelFormat::rgb, PixelFormat::rgba},
        {ComponentType::uint8, ComponentType::uint16}
    );

    int color_type;
    switch (m_pixel_format) {
    case PixelFormat::y:
        color_type = PNG_COLOR_TYPE_GRAY;
        break;
    case PixelFormat::ya:
        color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
        break;
    case PixelFormat::rgb:
        color_type = PNG_COLOR_TYPE_RGB;
        break;
    case PixelFormat::rgba:
        color_type = PNG_COLOR_TYPE_RGBA;
        break;
    default:
        SGL_THROW("Unsupported pixel format!");
    }

    int bit_depth;
    switch (m_component_type) {
    case ComponentType::uint8:
        bit_depth = 8;
        break;
    case ComponentType::uint16:
        bit_depth = 16;
        break;
    default:
        SGL_THROW("Unsupported component type!");
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, &png_error_func, &png_warn_func);
    if (png_ptr == nullptr)
        SGL_THROW("Error while creating PNG data structure");

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        SGL_THROW("Error while creating PNG information structure");
    }

    // Setup error handling.
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        SGL_THROW("Error writing the PNG file");
    }

    png_set_write_fn(png_ptr, stream, &png_write_data, &png_flush_data);
    png_set_compression_level(png_ptr, compression);

#if 0
    png_text* text = nullptr;

    Properties metadata(m_metadata);
    if (!metadata.has_property("generated_by"))
        metadata.set_string("generated_by", "Mitsuba version " MI_VERSION);

    std::vector<std::string> keys = metadata.property_names();
    std::vector<std::string> values(keys.size());

    text = new png_text[keys.size()];
    std::memset(text, 0, sizeof(png_text) * keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        values[i] = metadata.as_string(keys[i]);
        text[i].key = const_cast<char*>(keys[i].c_str());
        text[i].text = const_cast<char*>(values[i].c_str());
        text[i].compression = PNG_TEXT_COMPRESSION_NONE;
    }

    png_set_text(png_ptr, info_ptr, text, (int)keys.size());
#endif

    if (m_srgb_gamma)
        png_set_sRGB_gAMA_and_cHRM(png_ptr, info_ptr, PNG_sRGB_INTENT_ABSOLUTE);

    png_set_IHDR(
        png_ptr,
        info_ptr,
        m_width,
        m_height,
        bit_depth,
        color_type,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE,
        PNG_FILTER_TYPE_BASE
    );

    png_write_info(png_ptr, info_ptr);

    // Swap the byte order on little endian machines.
    if constexpr (stdx::endian::native == stdx::endian::little) {
        if (bit_depth == 16)
            png_set_swap(png_ptr);
    }

    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    SGL_ASSERT(row_bytes == buffer_size() / m_height);

    volatile png_bytepp rows = static_cast<png_bytepp>(alloca(sizeof(png_bytep) * m_height));
    for (size_t i = 0; i < m_height; i++)
        rows[i] = &m_data[row_bytes * i];

    png_write_image(png_ptr, rows);
    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    // delete[] text;
}

#else // SGL_HAS_LIBPNG

void Bitmap::read_png(Stream* stream)
{
    read_stb(stream, "PNG", true, false);
}

void Bitmap::write_png(Stream* stream, int compression) const
{
    check_required_format(
        "PNG",
        {PixelFormat::y, PixelFormat::ya, PixelFormat::rgb, PixelFormat::rgba},
        {ComponentType::uint8}
    );

    // stb_image_write uses global stbi_write_png_compression_level variable,
    // so we need to protect it with a mutex.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    stbi_write_png_compression_level = compression;
    if (!stbi_write_png_to_func(
            &stbi_write_func,
            stream,
            m_width,
            m_height,
            int(channel_count()),
            data(),
            int(m_width * channel_count())
        )) {
        SGL_THROW("Failed to write PNG file!");
    }
}

#endif // SGL_HAS_LIBPNG

// ----------------------------------------------------------------------------
// JPEG I/O
// ----------------------------------------------------------------------------

#if SGL_HAS_LIBJPEG

extern "C" {
static const size_t jpeg_buffer_size = 0x8000;

typedef struct {
    struct jpeg_source_mgr mgr;
    JOCTET* buffer;
    Stream* stream;
} jbuf_in_t;

typedef struct {
    struct jpeg_destination_mgr mgr;
    JOCTET* buffer;
    Stream* stream;
} jbuf_out_t;

static void jpeg_init_source(j_decompress_ptr cinfo)
{
    jbuf_in_t* p = (jbuf_in_t*)cinfo->src;
    p->buffer = new JOCTET[jpeg_buffer_size];
}

static boolean jpeg_fill_input_buffer(j_decompress_ptr cinfo)
{
    jbuf_in_t* p = (jbuf_in_t*)cinfo->src;
    size_t bytes_read = 0;

    try {
        p->stream->read(p->buffer, jpeg_buffer_size);
        bytes_read = jpeg_buffer_size;
    } catch (const EOFException& e) {
        bytes_read = e.gcount();
        if (bytes_read == 0) {
            // Insert a fake EOI marker
            p->buffer[0] = (JOCTET)0xFF;
            p->buffer[1] = (JOCTET)JPEG_EOI;
            bytes_read = 2;
        }
    }

    cinfo->src->bytes_in_buffer = bytes_read;
    cinfo->src->next_input_byte = p->buffer;
    return TRUE;
}

static void jpeg_skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
    if (num_bytes > 0) {
        while (num_bytes > (long)cinfo->src->bytes_in_buffer) {
            num_bytes -= (long)cinfo->src->bytes_in_buffer;
            jpeg_fill_input_buffer(cinfo);
        }
        cinfo->src->next_input_byte += (size_t)num_bytes;
        cinfo->src->bytes_in_buffer -= (size_t)num_bytes;
    }
}

static void jpeg_term_source(j_decompress_ptr cinfo)
{
    jbuf_in_t* p = (jbuf_in_t*)cinfo->src;
    delete[] p->buffer;
}

static void jpeg_init_destination(j_compress_ptr cinfo)
{
    jbuf_out_t* p = (jbuf_out_t*)cinfo->dest;

    p->buffer = new JOCTET[jpeg_buffer_size];
    p->mgr.next_output_byte = p->buffer;
    p->mgr.free_in_buffer = jpeg_buffer_size;
}

static boolean jpeg_empty_output_buffer(j_compress_ptr cinfo)
{
    jbuf_out_t* p = (jbuf_out_t*)cinfo->dest;
    p->stream->write(p->buffer, jpeg_buffer_size);
    p->mgr.next_output_byte = p->buffer;
    p->mgr.free_in_buffer = jpeg_buffer_size;
    return TRUE;
}

static void jpeg_term_destination(j_compress_ptr cinfo)
{
    jbuf_out_t* p = (jbuf_out_t*)cinfo->dest;
    p->stream->write(p->buffer, jpeg_buffer_size - p->mgr.free_in_buffer);
    delete[] p->buffer;
    p->mgr.free_in_buffer = 0;
}

static void jpeg_error_exit(j_common_ptr cinfo)
{
    char msg[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, msg);
    SGL_THROW("libjpeg error: {}", msg);
}

}; // extern "C"

void Bitmap::read_jpg(Stream* stream)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    jbuf_in_t jbuf;

    std::memset(&jbuf, 0, sizeof(jbuf_in_t));

    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = jpeg_error_exit;
    jpeg_create_decompress(&cinfo);
    cinfo.src = (struct jpeg_source_mgr*)&jbuf;
    jbuf.mgr.init_source = jpeg_init_source;
    jbuf.mgr.fill_input_buffer = jpeg_fill_input_buffer;
    jbuf.mgr.skip_input_data = jpeg_skip_input_data;
    jbuf.mgr.term_source = jpeg_term_source;
    jbuf.mgr.resync_to_restart = jpeg_resync_to_restart;
    jbuf.stream = stream;

    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    m_width = cinfo.output_width;
    m_height = cinfo.output_height;
    m_component_type = ComponentType::uint8;
    m_srgb_gamma = true;

    switch (cinfo.output_components) {
    case 1:
        m_pixel_format = PixelFormat::y;
        break;
    case 3:
        m_pixel_format = PixelFormat::rgb;
        break;
    default:
        SGL_THROW("Unsupported number of components!");
    }

    rebuild_pixel_struct();

    auto fs = dynamic_cast<FileStream*>(stream);
    log_debug(
        "Reading JPEG file \"{}\" ({}x{}, {}, {}) ...",
        fs ? fs->path().string() : "<stream>",
        m_width,
        m_height,
        m_pixel_format,
        m_component_type
    );

    size_t row_stride = static_cast<size_t>(cinfo.output_width) * static_cast<size_t>(cinfo.output_components);

    m_data = std::unique_ptr<uint8_t[]>(new uint8_t[buffer_size()]);
    m_owns_data = true;

    JSAMPARRAY scanlines = reinterpret_cast<JSAMPARRAY>(alloca(sizeof(JSAMPROW) * m_height));
    for (size_t i = 0; i < m_height; ++i)
        scanlines[i] = static_cast<JSAMPROW>(uint8_data() + row_stride * i);

    // Process scanline by scanline.
    int counter = 0;
    while (cinfo.output_scanline < cinfo.output_height)
        counter += jpeg_read_scanlines(
            &cinfo,
            &scanlines[counter],
            static_cast<JDIMENSION>(m_height - cinfo.output_scanline)
        );

    // Release the libjpeg data structures.
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
}

void Bitmap::write_jpg(Stream* stream, int quality) const
{
    check_required_format("JPEG", {PixelFormat::y, PixelFormat::rgb}, {ComponentType::uint8});

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    jbuf_out_t jbuf;

    int components = 0;
    switch (m_pixel_format) {
    case PixelFormat::y:
        components = 1;
        break;
    case PixelFormat::rgb:
        components = 3;
        break;
    default:
        SGL_THROW("Unsupported pixel format!");
    }

    if (m_component_type != ComponentType::uint8)
        SGL_THROW("Unsupported component format {}, expected {}.", m_component_type, ComponentType::uint8);

    std::memset(&jbuf, 0, sizeof(jbuf_out_t));
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = jpeg_error_exit;
    jpeg_create_compress(&cinfo);

    cinfo.dest = reinterpret_cast<jpeg_destination_mgr*>(&jbuf);
    jbuf.mgr.init_destination = jpeg_init_destination;
    jbuf.mgr.empty_output_buffer = jpeg_empty_output_buffer;
    jbuf.mgr.term_destination = jpeg_term_destination;
    jbuf.stream = stream;

    cinfo.image_width = static_cast<JDIMENSION>(m_width);
    cinfo.image_height = static_cast<JDIMENSION>(m_height);
    cinfo.input_components = components;
    cinfo.in_color_space = components == 1 ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    if (quality == 100) {
        // Disable chroma subsampling.
        cinfo.comp_info[0].v_samp_factor = 1;
        cinfo.comp_info[0].h_samp_factor = 1;
    }

    jpeg_start_compress(&cinfo, TRUE);

    // Write scanline by scanline
    for (size_t i = 0; i < m_height; ++i) {
        const uint8_t* source = m_data.get() + i * m_width * cinfo.input_components;
        jpeg_write_scanlines(&cinfo, const_cast<JSAMPARRAY>(&source), 1);
    }

    // Release the libjpeg data structures.
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
}

#else // SGL_HAS_LIBJPEG

void Bitmap::read_jpg(Stream* stream)
{
    read_stb(stream, "JPEG", true, false);
}

void Bitmap::write_jpg(Stream* stream, int quality) const
{
    check_required_format("JPEG", {PixelFormat::y, PixelFormat::rgb}, {ComponentType::uint8});

    if (!stbi_write_jpg_to_func(&stbi_write_func, stream, m_width, m_height, int(channel_count()), data(), quality))
        SGL_THROW("Failed to write JPEG file!");
}

#endif // SGL_HAS_LIBJPEG

// ----------------------------------------------------------------------------
// BMP I/O
// ----------------------------------------------------------------------------

void Bitmap::read_bmp(Stream* stream)
{
    read_stb(stream, "BMP", true, false);
}

void Bitmap::write_bmp(Stream* stream) const
{
    check_required_format("BMP", {PixelFormat::y, PixelFormat::rgb, PixelFormat::rgba}, {ComponentType::uint8});

    if (!stbi_write_bmp_to_func(&stbi_write_func, stream, m_width, m_height, int(channel_count()), data()))
        SGL_THROW("Failed to write BMP file!");
}

// ----------------------------------------------------------------------------
// TGA I/O
// ----------------------------------------------------------------------------

void Bitmap::read_tga(Stream* stream)
{
    read_stb(stream, "TGA", true, false);
}

void Bitmap::write_tga(Stream* stream) const
{
    check_required_format("TGA", {PixelFormat::y, PixelFormat::rgb, PixelFormat::rgba}, {ComponentType::uint8});

    if (!stbi_write_tga_to_func(&stbi_write_func, stream, m_width, m_height, int(channel_count()), data()))
        SGL_THROW("Failed to write TGA file!");
}

// ----------------------------------------------------------------------------
// HDR I/O
// ----------------------------------------------------------------------------

void Bitmap::read_hdr(Stream* stream)
{
    read_stb(stream, "HDR", false, true);
}

void Bitmap::write_hdr(Stream* stream) const
{
    check_required_format("HDR", {PixelFormat::rgb}, {ComponentType::float32});

    if (!stbi_write_hdr_to_func(
            &stbi_write_func,
            stream,
            m_width,
            m_height,
            int(channel_count()),
            reinterpret_cast<const float*>(data())
        ))
        SGL_THROW("Failed to write HDR file!");
}

// ----------------------------------------------------------------------------
// OpenEXR I/O
// ----------------------------------------------------------------------------

#if SGL_HAS_OPENEXR

class EXRIStream : public Imf::IStream {
public:
    EXRIStream(Stream* stream)
        : IStream(stream->to_string().c_str())
        , m_stream(stream)
    {
        m_offset = stream->tell();
        m_size = stream->size();
    }

    bool isMemoryMapped() const override
    {
        // TODO add support for memory mapped streams
        return false;
    }

    bool read(char* c, int n) override
    {
        m_stream->read(c, n);
        return m_stream->tell() == m_size;
    }

    char* readMemoryMapped(int n) override
    {
        // TODO add support for memory mapped streams
        SGL_UNUSED(n);
        return nullptr;
    }

    uint64_t tellg() override { return m_stream->tell() - m_offset; }

    void seekg(uint64_t pos) override { m_stream->seek((size_t)pos + m_offset); }

    void clear() override { }

private:
    Stream* m_stream;
    size_t m_offset, m_size;
};

class EXROStream : public Imf::OStream {
public:
    EXROStream(Stream* stream)
        : OStream(stream->to_string().c_str())
        , m_stream(stream)
    {
    }

    void write(const char* c, int n) override { m_stream->write(c, n); }

    uint64_t tellp() override { return m_stream->tell(); }

    void seekp(uint64_t pos) override { m_stream->seek((size_t)pos); }

private:
    Stream* m_stream;
};

void Bitmap::read_exr(Stream* stream)
{
    EXRIStream is(stream);
    Imf::InputFile file(is);

    const Imf::Header& header = file.header();
    const Imf::ChannelList& channels = header.channels();

    if (channels.begin() == channels.end())
        SGL_THROW("EXR image does not contain any channels!");

#if 0
    // Load meta data.
    for (auto it = header.begin(); it != header.end(); ++it) {
        std::string name = it.name();
        const Imf::Attribute* attr = &it.attribute();
        std::string type_name = attr->typeName();

        if (type_name == "string") {
            auto v = static_cast<const Imf::StringAttribute*>(attr);
            m_metadata.set_string(name, v->value());
        } else if (type_name == "int") {
            auto v = static_cast<const Imf::IntAttribute*>(attr);
            m_metadata.set_long(name, v->value());
        } else if (type_name == "float") {
            auto v = static_cast<const Imf::FloatAttribute*>(attr);
            m_metadata.set_float(name, Properties::Float(v->value()));
        } else if (type_name == "double") {
            auto v = static_cast<const Imf::DoubleAttribute*>(attr);
            m_metadata.set_float(name, Properties::Float(v->value()));
        } else if (type_name == "v3f") {
            auto v = static_cast<const Imf::V3fAttribute*>(attr);
            Imath::V3f vec = v->value();
            m_metadata.set_array3f(name, Vector3f(vec.x, vec.y, vec.z));
        } else if (type_name == "m44f") {
            auto v = static_cast<const Imf::M44fAttribute*>(attr);
            Matrix4f M;
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    M(i, j) = v->value().x[i][j];
            m_metadata.set_transform(name, Transform4f(M));
        } else if (type_name == "m33f") {
            auto v = static_cast<const Imf::M33fAttribute*>(attr);
            Matrix3f M;
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    M(i, j) = v->value().x[i][j];
            m_metadata.set_transform3f(name, Transform3f(M));
        }
    }
#endif

    // bool process_colors = false;
    // m_srgb_gamma = false;
    // m_premultiplied_alpha = true;
    // m_pixel_format = PixelFormat::MultiChannel;
    // m_struct = new DataStruct();
    Imf::PixelType pixel_type = channels.begin().channel().type;

    switch (pixel_type) {
    case Imf::HALF:
        m_component_type = ComponentType::float16;
        break;
    case Imf::FLOAT:
        m_component_type = ComponentType::float32;
        break;
    case Imf::UINT:
        m_component_type = ComponentType::uint32;
        break;
    default:
        SGL_THROW("EXR image contains invalid component type (must be float16, float32 or uint32)");
    }

    enum { unknown, R, G, B, X, Y, Z, A, RY, BY, CLASS_COUNT };

    // Classification scheme for color channels.
    auto channel_class = [](std::string name) -> uint8_t
    {
        auto it = name.rfind(".");
        if (it != std::string::npos)
            name = name.substr(it + 1);
        name = string::to_lower(name);
        if (name == "r")
            return R;
        if (name == "g")
            return G;
        if (name == "b")
            return B;
        if (name == "x")
            return X;
        if (name == "y")
            return Y;
        if (name == "z")
            return Z;
        if (name == "ry")
            return RY;
        if (name == "by")
            return BY;
        if (name == "a")
            return A;
        return unknown;
    };

    // Assign a sorting key to color channels.
    auto channel_key = [&](std::string name) -> std::string
    {
        uint8_t class_ = channel_class(name);
        if (class_ == unknown)
            return name;
        auto it = name.rfind(".");
        char suffix('0' + class_);
        if (it != std::string::npos)
            name = name.substr(0, it) + "." + suffix;
        else
            name = suffix;
        return name;
    };

    // Order channels based on their name and suffix.
    bool found[CLASS_COUNT] = {false};
    std::vector<std::string> channels_sorted;
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        std::string name(it.name());
        found[channel_class(name)] = true;
        channels_sorted.push_back(name);
    }
    std::sort(
        channels_sorted.begin(),
        channels_sorted.end(),
        [&](const auto& v0, const auto& v1) { return channel_key(v0) < channel_key(v1); }
    );

    // Create pixel struct.
    m_pixel_struct = ref(new DataStruct());
    for (const auto& name : channels_sorted) {
        m_pixel_struct->append(name, m_component_type);
    }

    // Try to detect common pixel formats.
    m_pixel_format = PixelFormat::multi_channel;
    bool luminance_chroma_format = false;
    if (m_pixel_struct->field_count() == 3 && found[R] && found[G] && found[B]) {
        m_pixel_format = PixelFormat::rgb;
    } else if (m_pixel_struct->field_count() == 4 && found[R] && found[G] && found[B] && found[A]) {
        m_pixel_format = PixelFormat::rgba;
    } else if (m_pixel_struct->field_count() == 3 && found[Y] && found[RY] && found[BY]) {
        m_pixel_format = PixelFormat::rgb;
        luminance_chroma_format = true;
    } else if (m_pixel_struct->field_count() == 4 && found[Y] && found[RY] && found[BY] && found[A]) {
        m_pixel_format = PixelFormat::rgba;
        luminance_chroma_format = true;
    } else if (m_pixel_struct->field_count() == 1 && found[Y]) {
        m_pixel_format = PixelFormat::y;
    } else if (m_pixel_struct->field_count() == 2 && found[Y] && found[A]) {
        m_pixel_format = PixelFormat::ya;
    }

    m_srgb_gamma = false;

    // Check if there is a chromaticity header entry.
    Imf::Chromaticities file_chroma;
    if (Imf::hasChromaticities(file.header()))
        file_chroma = Imf::chromaticities(file.header());

#if 0
    auto chroma_eq = [](const Imf::Chromaticities& a, const Imf::Chromaticities& b)
    {
        return (a.red - b.red).length2() + (a.green - b.green).length2() + (a.blue - b.blue).length2()
            + (a.white - b.white).length2()
            < 1e-6f;
    };
#endif

    auto set_suffix = [](std::string& name, const std::string& suffix)
    {
        auto it = name.rfind(".");
        if (it != std::string::npos)
            name = name.substr(0, it) + "." + suffix;
        else
            name = suffix;
    };

    Imath::Box2i data_window = file.header().dataWindow();
    m_width = data_window.max.x - data_window.min.x + 1;
    m_height = data_window.max.y - data_window.min.y + 1;

    size_t pixel_stride = this->bytes_per_pixel();
    size_t pixel_count = this->pixel_count();
    size_t row_stride = pixel_stride * m_width;

    m_data = std::unique_ptr<uint8_t[]>(new uint8_t[row_stride * m_height]);
    m_owns_data = true;

#if 0
    using ResampleBuffer = std::pair<std::string, ref<Bitmap>>;
    std::vector<ResampleBuffer> resample_buffers;
#endif

    uint8_t* ptr = m_data.get() - (data_window.min.x + data_window.min.y * m_width) * pixel_stride;

    // Tell OpenEXR where the image data should be put.
    Imf::FrameBuffer framebuffer;
    for (const auto& field : *m_pixel_struct) {
        const Imf::Channel& channel = channels[field.name];
        int x_sampling = channel.xSampling;
        int y_sampling = channel.ySampling;
        Imf::Slice slice;

        if (x_sampling == 1 && y_sampling == 1) {
            // Full resolution channel. Load directly.
            slice = Imf::Slice(pixel_type, (char*)(ptr + field.offset), pixel_stride, row_stride);
        } else {
            // Sub-sampled channel. Need to rescale later.
            SGL_THROW("Sub-sampled channels are not supported!");
#if 0
            Vector2u channel_size = m_size / sampling;

            ref<Bitmap> bitmap = new Bitmap(PixelFormat::Y, m_component_format, channel_size);

            uint8_t* ptr_nested = bitmap->uint8_data()
                - (data_window.min.x / sampling.x() + data_window.min.y / sampling.y() * (size_t)channel_size.x())
                    * field.size;

            slice = Imf::Slice(
                pixel_type,
                (char*)ptr_nested,
                field.size,
                field.size * channel_size.x(),
                sampling.x(),
                sampling.y()
            );

            resample_buffers.emplace_back(field.name, std::move(bitmap));
#endif
        }
        framebuffer.insert(field.name, slice);
    }

    auto fs = dynamic_cast<FileStream*>(stream);
    log_debug(
        "Reading OpenEXR file \"{}\" ({}x{}, {}, {}) ...",
        fs ? fs->path().string() : "<stream>",
        m_width,
        m_height,
        m_pixel_format,
        m_component_type
    );

    file.setFrameBuffer(framebuffer);
    file.readPixels(data_window.min.y, data_window.max.y);

#if 0
    for (auto& buf : resample_buffers) {
        Log(Debug,
            "Upsampling layer \"%s\" from %ix%i to %ix%i pixels",
            buf.first,
            buf.second->width(),
            buf.second->height(),
            m_size.x(),
            m_size.y());

        buf.second = buf.second->resample(m_size);
        const DataStruct::Field& field = m_struct->field(buf.first);

        size_t comp_size = field.size;
        uint8_t* dst = uint8_data() + field.offset;
        uint8_t* src = buf.second->uint8_data();

        for (size_t j = 0; j < pixel_count; ++j) {
            std::memcpy(dst, src, comp_size);
            src += comp_size;
            dst += pixel_stride;
        }

        buf.second = nullptr;
    }
#endif

    if (luminance_chroma_format) {
        log_debug("Converting from Luminance-Chroma to RGB format ...");
        Imath::V3f yw = Imf::RgbaYca::computeYw(file_chroma);

        auto convert = [&](auto* data)
        {
            using T = std::decay_t<decltype(*data)>;

            for (size_t j = 0; j < pixel_count; ++j) {
                double y = double(data[0]);
                double ry = double(data[1]);
                double by = double(data[2]);

                if constexpr (!math::floating_point<T>) {
                    double scale = 1.0 / double(std::numeric_limits<T>::max());
                    y *= scale;
                    ry *= scale;
                    by *= scale;
                }

                double r = (ry + 1.0) * y;
                double b = (by + 1.0) * y;
                double g = ((y - r * yw.x - b * yw.z) / yw.y);

                if constexpr (!math::floating_point<T>) {
                    double scale = double(std::numeric_limits<T>::max());
                    r = r * scale + .5f;
                    g = g * scale + .5f;
                    b = b * scale + .5f;
                }

                data[0] = T(R);
                data[1] = T(G);
                data[2] = T(B);
                data += channel_count();
            }
        };

        switch (m_component_type) {
        case ComponentType::float16:
            convert(reinterpret_cast<math::float16_t*>(m_data.get()));
            break;
        case ComponentType::float32:
            convert(reinterpret_cast<float*>(m_data.get()));
            break;
        case ComponentType::uint32:
            convert(reinterpret_cast<uint32_t*>(m_data.get()));
            break;
        default:
            SGL_THROW("Internal error!");
        }

        set_suffix(m_pixel_struct->operator[](0).name, "R");
        set_suffix(m_pixel_struct->operator[](1).name, "G");
        set_suffix(m_pixel_struct->operator[](2).name, "B");
    }

#if 0
    if (Imf::hasChromaticities(file.header())
        && (m_pixel_format == PixelFormat::RGB || m_pixel_format == PixelFormat::RGBA)) {

        Imf::Chromaticities itu_rec_b_709;
        Imf::Chromaticities
            xyz(Imath::V2f(1.f, 0.f), Imath::V2f(0.f, 1.f), Imath::V2f(0.f, 0.f), Imath::V2f(1.f / 3.f, 1.f / 3.f));

        if (chroma_eq(file_chroma, itu_rec_b_709)) {
            // Already in the right space -- do nothing.
        } else if (chroma_eq(file_chroma, xyz)) {
            // This is an XYZ image
            m_pixel_format = m_pixel_format == PixelFormat::RGB ? PixelFormat::XYZ : PixelFormat::XYZA;
            for (int i = 0; i < 3; ++i) {
                std::string& name = m_struct->operator[](i).name;
                name = set_suffix(name, std::string(1, "XYZ"[i]));
            }
        } else {
            // Non-standard chromaticities. Special processing is required..
            process_colors = true;
        }
    }

    if (process_colors) {
        // Convert ITU-R Rec. BT.709 linear RGB
        Imath::M44f M = Imf::RGBtoXYZ(file_chroma, 1) * Imf::XYZtoRGB(Imf::Chromaticities(), 1);

        Log(Debug, "Converting to sRGB color space ..");

        auto convert = [&](auto* data)
        {
            using T = std::decay_t<decltype(*data)>;

            for (size_t j = 0; j < pixel_count; ++j) {
                Float R = (Float)data[0], G = (Float)data[1], B = (Float)data[2];

                if (std::is_integral<T>::value) {
                    Float scale = Float(1) / Float(std::numeric_limits<T>::max());
                    R *= scale;
                    G *= scale;
                    B *= scale;
                }

                Imath::V3f rgb = Imath::V3f(float(R), float(G), float(B)) * M;
                R = Float(rgb[0]);
                G = Float(rgb[1]);
                B = Float(rgb[2]);

                if (std::is_integral<T>::value) {
                    Float scale = Float(std::numeric_limits<T>::max());
                    R *= R * scale + 0.5f;
                    G *= G * scale + 0.5f;
                    B *= B * scale + 0.5f;
                }

                data[0] = T(R);
                data[1] = T(G);
                data[2] = T(B);
                data += channel_count();
            }
        };

        switch (m_component_format) {
        case DataStruct::Type::Float16:
            convert((dr::half*)m_data.get());
            break;
        case DataStruct::Type::Float32:
            convert((float*)m_data.get());
            break;
        case DataStruct::Type::UInt32:
            convert((uint32_t*)m_data.get());
            break;
        default:
            Throw("Internal error!");
        }
    }
#endif
}

void Bitmap::write_exr(Stream* stream, int quality) const
{
    check_required_format(
        "EXR",
        {PixelFormat::y, PixelFormat::ya, PixelFormat::rgb, PixelFormat::rgba, PixelFormat::multi_channel},
        {ComponentType::uint32, ComponentType::float16, ComponentType::float32}
    );

    Imf::Header header(
        static_cast<int>(m_width),                                  // width
        static_cast<int>(m_height),                                 // height,
        1.f,                                                        // pixelAspectRatio
        Imath::V2f(0, 0),                                           // screenWindowCenter,
        1.f,                                                        // screenWindowWidth
        Imf::INCREASING_Y,                                          // lineOrder
        quality <= 0 ? Imf::PIZ_COMPRESSION : Imf::DWAB_COMPRESSION // compression
    );

    if (quality > 0)
        header.dwaCompressionLevel() = static_cast<float>(quality);

#if 0
    Properties metadata(m_metadata);
    if (!metadata.has_property("generatedBy"))
        metadata.set_string("generatedBy", "Mitsuba version " MI_VERSION);

    std::vector<std::string> keys = metadata.property_names();
#endif

#if 0
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        using Type = Properties::Type;

        Type type = metadata.type(*it);
        if (*it == "pixelAspectRatio" || *it == "screenWindowWidth" || *it == "screenWindowCenter")
            continue;

        switch (type) {
        case Type::String:
            header.insert(it->c_str(), Imf::StringAttribute(metadata.string(*it)));
            break;
        case Type::Long:
            header.insert(it->c_str(), Imf::IntAttribute(metadata.get<int>(*it)));
            break;
        case Type::Float:
            header.insert(it->c_str(), Imf::DoubleAttribute(metadata.get<double>(*it)));
            break;
        case Type::Array3f: {
            Vector3f val = metadata.get<Vector3f>(*it);
            header.insert(it->c_str(), Imf::V3fAttribute(Imath::V3f((float)val.x(), (float)val.y(), (float)val.z())));
        } break;
        case Type::Transform3f: {
            Matrix3f val = metadata.get<ScalarTransform3f>(*it).matrix;
            header.insert(
                it->c_str(),
                Imf::M33fAttribute(Imath::M33f(
                    (float)val(0, 0),
                    (float)val(0, 1),
                    (float)val(0, 2),
                    (float)val(1, 0),
                    (float)val(1, 1),
                    (float)val(1, 2),
                    (float)val(2, 0),
                    (float)val(2, 1),
                    (float)val(2, 2)
                ))
            );
        } break;
        case Type::Transform4f: {
            Matrix4f val = metadata.get<ScalarTransform4f>(*it).matrix;
            header.insert(
                it->c_str(),
                Imf::M44fAttribute(Imath::M44f(
                    (float)val(0, 0),
                    (float)val(0, 1),
                    (float)val(0, 2),
                    (float)val(0, 3),
                    (float)val(1, 0),
                    (float)val(1, 1),
                    (float)val(1, 2),
                    (float)val(1, 3),
                    (float)val(2, 0),
                    (float)val(2, 1),
                    (float)val(2, 2),
                    (float)val(2, 3),
                    (float)val(3, 0),
                    (float)val(3, 1),
                    (float)val(3, 2),
                    (float)val(3, 3)
                ))
            );
        } break;
        default:
            header.insert(it->c_str(), Imf::StringAttribute(metadata.as_string(*it)));
            break;
        }
    }
#endif

#if 0
    if (m_pixel_format == PixelFormat::XYZ || m_pixel_format == PixelFormat::XYZA) {
        Imf::addChromaticities(
            header,
            Imf::Chromaticities(
                Imath::V2f(1.f, 0.f),
                Imath::V2f(0.f, 1.f),
                Imath::V2f(0.f, 0.f),
                Imath::V2f(1.f / 3.f, 1.f / 3.f)
            )
        );
    }
#endif

    Imf::PixelType pixel_type;
    switch (m_component_type) {
    case ComponentType::uint32:
        pixel_type = Imf::UINT;
        break;
    case ComponentType::float16:
        pixel_type = Imf::HALF;
        break;
    case ComponentType::float32:
        pixel_type = Imf::FLOAT;
        break;
    default:
        SGL_THROW("Unsupported component type!");
    }

    size_t component_size = DataStruct::type_size(m_component_type);
    size_t pixel_stride = m_pixel_struct->size();
    size_t row_stride = pixel_stride * m_width;

    Imf::ChannelList& channels = header.channels();
    Imf::FrameBuffer framebuffer;
    const uint8_t* ptr = uint8_data();
    for (size_t i = 0; i < m_pixel_struct->field_count(); ++i) {
        const std::string& channel_name = m_pixel_struct->operator[](i).name;
        Imf::Slice slice(pixel_type, (char*)(ptr + i * component_size), pixel_stride, row_stride);
        channels.insert(channel_name, Imf::Channel(pixel_type));
        framebuffer.insert(channel_name, slice);
    }

    EXROStream os(stream);
    Imf::OutputFile file(os, header);
    file.setFrameBuffer(framebuffer);
    file.writePixels(static_cast<int>(m_height));
}

#else // SGL_HAS_OPENEXR

void Bitmap::read_exr(Stream* stream)
{
    size_t size = stream->size();
    std::unique_ptr<uint8_t[]> memory(new uint8_t[size]);
    stream->read(memory.get(), size);

    EXRVersion version;
    if (ParseEXRVersionFromMemory(&version, memory.get(), size) != TINYEXR_SUCCESS)
        SGL_THROW("Failed to parse EXR version!");
    if (version.multipart)
        SGL_THROW("EXR multipart files are not supported yet!");

    EXRHeader header;
    InitEXRHeader(&header);
    const char* err = nullptr;
    if (ParseEXRHeaderFromMemory(&header, &version, memory.get(), size, &err) != TINYEXR_SUCCESS) {
        SGL_THROW(fmt::format("Failed to parse EXR header!\n{}", err));
        // FreeEXRErrorMessage(err);
    }

    switch (header.pixel_types[0]) {
    case TINYEXR_PIXELTYPE_UINT:
        m_component_type = ComponentType::uint32;
        break;
    case TINYEXR_PIXELTYPE_HALF:
        m_component_type = ComponentType::float16;
        break;
    case TINYEXR_PIXELTYPE_FLOAT:
        m_component_type = ComponentType::float32;
        break;
    default:
        SGL_THROW("EXR image contains invalid component type (must be float16, float32 or uint32)");
    }

    enum { unknown, R, G, B, X, Y, Z, A, RY, BY, CLASS_COUNT };

    // Classification scheme for color channels.
    auto channel_class = [](std::string name) -> uint8_t
    {
        auto it = name.rfind(".");
        if (it != std::string::npos)
            name = name.substr(it + 1);
        name = string::to_lower(name);
        if (name == "r")
            return R;
        if (name == "g")
            return G;
        if (name == "b")
            return B;
        if (name == "x")
            return X;
        if (name == "y")
            return Y;
        if (name == "z")
            return Z;
        if (name == "ry")
            return RY;
        if (name == "by")
            return BY;
        if (name == "a")
            return A;
        return unknown;
    };

    // Assign a sorting key to color channels.
    auto channel_key = [&](std::string name) -> std::string
    {
        uint8_t class_ = channel_class(name);
        if (class_ == unknown)
            return name;
        auto it = name.rfind(".");
        char suffix('0' + class_);
        if (it != std::string::npos)
            name = name.substr(0, it) + "." + suffix;
        else
            name = suffix;
        return name;
    };

    // Order channels based on their name and suffix.
    bool found[CLASS_COUNT] = {false};
    std::vector<std::string> channels_sorted;
    for (int i = 0; i < header.num_channels; ++i) {
        std::string name(header.channels[i].name);
        found[channel_class(name)] = true;
        channels_sorted.push_back(name);
    }
    std::sort(
        channels_sorted.begin(),
        channels_sorted.end(),
        [&](const auto& v0, const auto& v1) { return channel_key(v0) < channel_key(v1); }
    );

    // Create pixel struct.
    m_pixel_struct = ref(new DataStruct());
    for (const auto& name : channels_sorted) {
        m_pixel_struct->append(name, m_component_type);
    }

    // Try to detect common pixel formats.
    m_pixel_format = PixelFormat::multi_channel;
    bool luminance_chroma_format = false;
    if (m_pixel_struct->field_count() == 3 && found[R] && found[G] && found[B]) {
        m_pixel_format = PixelFormat::rgb;
    } else if (m_pixel_struct->field_count() == 4 && found[R] && found[G] && found[B] && found[A]) {
        m_pixel_format = PixelFormat::rgba;
    } else if (m_pixel_struct->field_count() == 3 && found[Y] && found[RY] && found[BY]) {
        m_pixel_format = PixelFormat::rgb;
        luminance_chroma_format = true;
    } else if (m_pixel_struct->field_count() == 4 && found[Y] && found[RY] && found[BY] && found[A]) {
        m_pixel_format = PixelFormat::rgba;
        luminance_chroma_format = true;
    } else if (m_pixel_struct->field_count() == 1 && found[Y]) {
        m_pixel_format = PixelFormat::y;
    } else if (m_pixel_struct->field_count() == 2 && found[Y] && found[A]) {
        m_pixel_format = PixelFormat::ya;
    }

    m_srgb_gamma = false;

    auto fs = dynamic_cast<FileStream*>(stream);
    log_debug(
        "Reading OpenEXR file \"{}\" ({}x{}, {}, {}) ...",
        fs ? fs->path().string() : "<stream>",
        m_width,
        m_height,
        m_pixel_format,
        m_component_type
    );

    EXRImage image;
    InitEXRImage(&image);

    if (LoadEXRImageFromMemory(&image, &header, memory.get(), size, &err) != TINYEXR_SUCCESS) {
        FreeEXRErrorMessage(err);
        FreeEXRHeader(&header);
        SGL_THROW(fmt::format("Failed to load EXR image!\n{}", err));
    }

    auto find_channel_index = [&](const std::string& name) -> int
    {
        for (int i = 0; i < header.num_channels; ++i)
            if (header.channels[i].name == name)
                return i;
        SGL_THROW(fmt::format("EXR image does not contain channel \"{}\"", name));
    };

    auto set_suffix = [](std::string& name, const std::string& suffix)
    {
        auto it = name.rfind(".");
        if (it != std::string::npos)
            name = name.substr(0, it) + "." + suffix;
        else
            name = suffix;
    };

    m_width = header.data_window.max_x - header.data_window.min_x + 1;
    m_height = header.data_window.max_y - header.data_window.min_y + 1;

    size_t component_size = DataStruct::type_size(m_component_type);
    size_t pixel_stride = this->bytes_per_pixel();
    size_t pixel_count = this->pixel_count();
    size_t row_stride = pixel_stride * m_width;

    m_data = std::unique_ptr<uint8_t[]>(new uint8_t[row_stride * m_height]);
    m_owns_data = true;

    for (const auto& field : *m_pixel_struct) {
        int channel_index = find_channel_index(field.name);
        const uint8_t* src = image.images[channel_index];
        uint8_t* dst = uint8_data() + field.offset;
        if (component_size == 2) {
            for (size_t j = 0; j < m_width * m_height; ++j) {
                std::memcpy(dst, src, 2);
                src += 2;
                dst += pixel_stride;
            }
        } else if (component_size == 4) {
            for (size_t j = 0; j < m_width * m_height; ++j) {
                std::memcpy(dst, src, 4);
                src += 4;
                dst += pixel_stride;
            }
        } else {
            SGL_THROW("Unsupported component size!");
        }
    }

    FreeEXRImage(&image);
    FreeEXRHeader(&header);
}

void Bitmap::write_exr(Stream* stream, int quality) const
{
    SGL_UNUSED(quality);

    check_required_format(
        "EXR",
        {PixelFormat::y, PixelFormat::ya, PixelFormat::rgb, PixelFormat::rgba, PixelFormat::multi_channel},
        {ComponentType::uint32, ComponentType::float16, ComponentType::float32}
    );

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    int pixel_type = 0;
    switch (m_component_type) {
    case ComponentType::uint32:
        pixel_type = TINYEXR_PIXELTYPE_UINT;
        break;
    case ComponentType::float16:
        pixel_type = TINYEXR_PIXELTYPE_HALF;
        break;
    case ComponentType::float32:
        pixel_type = TINYEXR_PIXELTYPE_FLOAT;
        break;
    default:
        SGL_THROW("Unsupported component type!");
    }

    std::vector<EXRChannelInfo> channels(channel_count());
    for (size_t i = 0; i < channel_count(); ++i) {
        EXRChannelInfo& channel = channels[i];
        channel.pixel_type = pixel_type;
        string::copy_to_cstr(channel.name, sizeof(EXRChannelInfo::name), m_pixel_struct->operator[](i).name);
    }

    std::vector<int> pixel_types(channel_count(), pixel_type);

    header.num_channels = channel_count();
    header.channels = channels.data();
    header.pixel_types = pixel_types.data();
    header.requested_pixel_types = pixel_types.data();

    // Convert interleaved data to planar format.
    size_t component_size = DataStruct::type_size(m_component_type);
    size_t pixel_stride = m_pixel_struct->size();
    size_t row_stride = pixel_stride * m_width;
    size_t plane_size = row_stride * m_height;

    std::vector<std::unique_ptr<uint8_t[]>> images(channel_count());
    std::vector<uint8_t*> image_ptrs(channel_count());
    for (size_t i = 0; i < channel_count(); ++i) {
        images[i] = std::unique_ptr<uint8_t[]>(new uint8_t[plane_size]);
        image_ptrs[i] = images[i].get();
        const uint8_t* src = uint8_data() + i * component_size;
        uint8_t* dst = images[i].get();
        if (component_size == 2) {
            for (size_t j = 0; j < m_width * m_height; ++j) {
                std::memcpy(dst, src, 2);
                src += pixel_stride;
                dst += 2;
            }
        } else if (component_size == 4) {
            for (size_t j = 0; j < m_width * m_height; ++j) {
                std::memcpy(dst, src, 4);
                src += pixel_stride;
                dst += 4;
            }
        } else {
            SGL_THROW("Unsupported component size!");
        }
    }

    image.width = m_width;
    image.height = m_height;
    image.num_channels = channel_count();
    image.images = image_ptrs.data();

    const char* err = nullptr;
    uint8_t* memory = nullptr;
    size_t size = SaveEXRImageToMemory(&image, &header, &memory, &err);
    if (size == 0) {
        FreeEXRErrorMessage(err);
        SGL_THROW(fmt::format("Failed to save EXR image!\n{}", err));
    }
    stream->write(memory, size);
}

#endif // SGL_HAS_OPENEXR

} // namespace sgl
