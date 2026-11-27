// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/core/stream.h"

#include <filesystem>
#include <vector>

namespace sgl {

/**
 * \brief Helper class for loading DDS files.
 */
class SGL_API DDSFile : public Object {
    SGL_OBJECT(DDSFile)
public:
    SGL_NON_COPYABLE_AND_MOVABLE(DDSFile);

    enum class TextureType {
        texture_1d,
        texture_2d,
        texture_3d,
        texture_cube,
    };

    SGL_ENUM_INFO(
        TextureType,
        {
            {TextureType::texture_1d, "texture_1d"},
            {TextureType::texture_2d, "texture_2d"},
            {TextureType::texture_3d, "texture_3d"},
            {TextureType::texture_cube, "texture_cube"},
        }
    );

    /// DDS properties decoded from the file header.
    struct Info {
        uint32_t dxgi_format;
        TextureType type;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        uint32_t mip_count;
        uint32_t array_size;
        bool compressed;
        bool srgb;
    };

    /// Read a DDS file extending from the current stream position to the end of the stream.
    explicit DDSFile(Stream* stream);
    explicit DDSFile(const std::filesystem::path& path);

    /// Read DDS properties from the header without loading the resource data.
    /// Validates the header and the file size like the constructor. The stream position is restored.
    static Info read_info(Stream* stream);

    /// Properties of this file.
    Info info() const;

    const uint8_t* data() const { return m_data.data(); }
    size_t size() const { return m_data.size(); }

    const uint8_t* resource_data() const { return m_data.data() + m_header_size; }
    size_t resource_size() const { return m_data.size() - m_header_size; }

    uint32_t dxgi_format() const { return m_dxgi_format; }
    TextureType type() const { return m_type; }

    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    uint32_t depth() const { return m_depth; }
    uint32_t mip_count() const { return m_mip_count; }
    uint32_t array_size() const { return m_array_size; }

    uint32_t row_pitch() const { return m_row_pitch; }
    uint32_t slice_pitch() const { return m_slice_pitch; }
    uint32_t bits_per_pixel_or_block() const { return m_bits_per_pixel_or_block; }
    uint32_t block_width() const { return m_block_width; }
    uint32_t block_height() const { return m_block_height; }
    bool compressed() const { return m_compressed; }
    bool srgb() const { return m_srgb; }

    /**
     * \brief Get a pointer to the start of the data for the specified mip and slice.
     *
     * \param mip Mip level.
     * \param slice Slice index (array index, face index  or volume slice index).
     * \return Pointer to the start of the data.
     */
    const uint8_t* get_subresource_data(uint32_t mip, uint32_t slice) const;

    void get_subresource_pitch(uint32_t mip, uint32_t* row_pitch, uint32_t* slice_pitch) const;

    virtual std::string to_string() const override;

    static bool detect_dds_file(Stream* stream);

    /// Write a DDS file to a stream.
    static void write_dds(
        Stream* stream,
        uint32_t dxgi_format,
        TextureType type,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint32_t mip_count,
        uint32_t array_size,
        const void* resource_data,
        size_t resource_size
    );

    /// Write a DDS file to a file path.
    static void write_dds(
        const std::filesystem::path& path,
        uint32_t dxgi_format,
        TextureType type,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint32_t mip_count,
        uint32_t array_size,
        const void* resource_data,
        size_t resource_size
    );

private:
    DDSFile() = default;

    /// Decode the header from the first data_size bytes and validate it against the total file_size.
    bool decode_header(const uint8_t* data, size_t data_size, size_t file_size);
    bool get_subresource_offset(uint32_t mip, uint32_t slice, size_t* offset) const;
    bool get_required_resource_size(size_t* size) const;

    std::vector<uint8_t> m_data;

    uint32_t m_dxgi_format{0};
    TextureType m_type{TextureType::texture_1d};
    uint32_t m_width{0};
    uint32_t m_height{0};
    uint32_t m_depth{0};
    uint32_t m_mip_count{0};
    uint32_t m_array_size{0};
    uint32_t m_row_pitch{0};
    uint32_t m_slice_pitch{0};
    uint32_t m_bits_per_pixel_or_block{0};
    uint32_t m_block_width{0};
    uint32_t m_block_height{0};
    bool m_compressed{false};
    bool m_srgb{false};
    size_t m_header_size{0};
};

SGL_ENUM_REGISTER(DDSFile::TextureType);

} // namespace sgl
