// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
This is a modified version of the bitmap.h file from Mitsuba 3.0 found at
https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/bitmap.h

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

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/core/stream.h"
#include "sgl/core/data_struct.h"

#include <filesystem>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sgl {

class SGL_API Bitmap : public Object {
    SGL_OBJECT(Bitmap)
public:
    enum class FileFormat {
        unknown,
        auto_,
        png,
        jpg,
        bmp,
        tga,
        hdr,
        exr,
    };

    SGL_ENUM_INFO(
        FileFormat,
        {
            {FileFormat::unknown, "unknown"},
            {FileFormat::auto_, "auto"},
            {FileFormat::png, "png"},
            {FileFormat::jpg, "jpg"},
            {FileFormat::bmp, "bmp"},
            {FileFormat::tga, "tga"},
            {FileFormat::hdr, "hdr"},
            {FileFormat::exr, "exr"},
        }
    );

    enum class PixelFormat {
        /// Luminance only.
        y,
        /// Luminance + alpha.
        ya,
        /// Red.
        r,
        /// Red + green.
        rg,
        /// RGB.
        rgb,
        /// RGB + alpha.
        rgba,
        /// Arbitrary multi-channel.
        multi_channel,
    };

    static constexpr uint32_t PIXEL_FORMAT_COUNT = uint32_t(PixelFormat::multi_channel) + 1;

    SGL_ENUM_INFO(
        PixelFormat,
        {
            {PixelFormat::y, "y"},
            {PixelFormat::ya, "ya"},
            {PixelFormat::r, "r"},
            {PixelFormat::rg, "rg"},
            {PixelFormat::rgb, "rgb"},
            {PixelFormat::rgba, "rgba"},
            {PixelFormat::multi_channel, "multi_channel"},
        }
    );

    using ComponentType = DataStruct::Type;

    Bitmap(
        PixelFormat pixel_format,
        ComponentType component_type,
        uint32_t width,
        uint32_t height,
        uint32_t channel_count = 0,
        const std::vector<std::string>& channel_names = {},
        void* data = nullptr
    );

    Bitmap(Stream* stream, FileFormat format = FileFormat::auto_);

    Bitmap(const std::filesystem::path& path, FileFormat format = FileFormat::auto_);

    /// Copy constructor.
    Bitmap(const Bitmap& other);

    /// Move constructor.
    Bitmap(Bitmap&& other);

    /// Destructor.
    ~Bitmap();

    /// Load a list of bitmaps from multiple paths. Uses multi-threading to load bitmaps in parallel.
    static std::vector<ref<Bitmap>>
    read_multiple(std::span<std::filesystem::path> paths, FileFormat format = FileFormat::auto_);

    void write(Stream* stream, FileFormat format = FileFormat::auto_, int quality = -1) const;
    void write(const std::filesystem::path& path, FileFormat format = FileFormat::auto_, int quality = -1) const;

    void write_async(const std::filesystem::path& path, FileFormat format = FileFormat::auto_, int quality = -1) const;

    /// The pixel format.
    PixelFormat pixel_format() const { return m_pixel_format; }

    /// The component type.
    ComponentType component_type() const { return m_component_type; }

    /// DataStruct describing the pixel layout.
    const DataStruct* pixel_struct() const { return m_pixel_struct; }

    /// The width of the bitmap in pixels.
    uint32_t width() const { return m_width; }

    /// The height of the bitmap in pixels.
    uint32_t height() const { return m_height; }

    /// The total number of pixels in the bitmap.
    size_t pixel_count() const { return static_cast<size_t>(m_width) * m_height; }

    /// The number of channels in the bitmap.
    uint32_t channel_count() const { return static_cast<uint32_t>(m_pixel_struct->field_count()); }

    /// The names of the channels in the bitmap.
    std::vector<std::string> channel_names() const;

    /// True if the bitmap is in sRGB gamma space.
    bool srgb_gamma() const { return m_srgb_gamma; }

    /// Set the sRGB gamma flag.
    /// Note that this does not convert the pixel values, it only sets the flag and adjusts the pixel struct.
    void set_srgb_gamma(bool srgb_gamma);

    /// Returns true if the bitmap has an alpha channel.
    bool has_alpha() const { return m_pixel_format == PixelFormat::ya || m_pixel_format == PixelFormat::rgba; }

    /// The number of bytes per pixel.
    size_t bytes_per_pixel() const { return m_pixel_struct->size(); }

    /// The total size of the bitmap in bytes.
    size_t buffer_size() const { return pixel_count() * bytes_per_pixel(); }

    /// The raw image data.
    void* data() { return m_data.get(); }
    const void* data() const { return m_data.get(); }

    /// The raw image data as uint8_t.
    uint8_t* uint8_data() { return m_data.get(); }
    const uint8_t* uint8_data() const { return m_data.get(); }

    template<typename T>
    T* data_as()
    {
        return reinterpret_cast<T*>(m_data.get());
    }
    template<typename T>
    const T* data_as() const
    {
        return reinterpret_cast<const T*>(m_data.get());
    }

    /// True if bitmap is empty.
    bool empty() const { return m_width == 0 || m_height == 0; }

    /// Clears the bitmap to zeros.
    void clear();

    /// Vertically flip the bitmap.
    void vflip();

    /**
     * \brief Split bitmap into multiple bitmaps, each containing the channels with the same prefix.
     *
     * For example, if the bitmap has channels `albedo.R`, `albedo.G`, `albedo.B`, `normal.R`, `normal.G`, `normal.B`,
     * this function will return two bitmaps, one containing the channels `albedo.R`, `albedo.G`, `albedo.B` and the
     * other containing the channels `normal.R`, `normal.G`, `normal.B`.
     *
     * Common pixel formats (e.g. `y`, `rgb`, `rgba`) are automatically detected and used for the split bitmaps.
     *
     * Any channels that do not have a prefix will be returned in the bitmap with the empty prefix.
     *
     * \return Returns a list of (prefix, bitmap) pairs.
     */
    std::vector<std::pair<std::string, ref<Bitmap>>> split() const;

    ref<Bitmap> convert(PixelFormat pixel_format, ComponentType component_type, bool srgb_gamma) const;

    void convert(Bitmap* target) const;

    /// Equality operator.
    bool operator==(const Bitmap& other) const;

    /// Inequality operator.
    bool operator!=(const Bitmap& other) const { return !operator==(other); }

    std::string to_string() const override;

    static FileFormat detect_file_format(Stream* stream);

    static void static_init();
    static void static_shutdown();

private:
    void rebuild_pixel_struct(uint32_t channel_count = 0, const std::vector<std::string>& channel_names = {});

    void read(Stream* stream, FileFormat format);

    void check_required_format(
        std::string_view file_format,
        std::vector<Bitmap::PixelFormat> allowed_pixel_formats,
        std::vector<Bitmap::ComponentType> allowed_component_types
    ) const;

    void read_stb(Stream* stream, const char* format, bool is_srgb, bool is_hdr);

    void read_png(Stream* stream);
    void write_png(Stream* stream, int compression) const;

    void read_jpg(Stream* stream);
    void write_jpg(Stream* stream, int quality) const;

    void read_bmp(Stream* stream);
    void write_bmp(Stream* stream) const;

    void read_tga(Stream* stream);
    void write_tga(Stream* stream) const;

    void read_hdr(Stream* stream);
    void write_hdr(Stream* stream) const;

    void read_exr(Stream* stream);
    void write_exr(Stream* stream, int quality) const;

    PixelFormat m_pixel_format;
    ComponentType m_component_type;
    ref<DataStruct> m_pixel_struct;
    uint32_t m_width;
    uint32_t m_height;
    bool m_srgb_gamma;
    std::unique_ptr<uint8_t[]> m_data;
    bool m_owns_data;
};

SGL_ENUM_REGISTER(Bitmap::FileFormat);
SGL_ENUM_REGISTER(Bitmap::PixelFormat);

} // namespace sgl
