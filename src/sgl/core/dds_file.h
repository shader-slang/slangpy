// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"
#include "sgl/core/enum.h"
#include "sgl/core/stream.h"

#include <filesystem>

namespace sgl {

/**
 * \brief Helper class for loading DDS files.
 */
class SGL_API DDSFile : public Object {
    SGL_OBJECT(DDSFile)
public:
    SGL_NON_COPYABLE_AND_MOVABLE(DDSFile);

    explicit DDSFile(Stream* stream);
    explicit DDSFile(const std::filesystem::path& path);
    ~DDSFile();

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

    const uint8_t* data() const { return m_data; }
    size_t size() const { return m_size; }

    const uint8_t* resource_data() const { return m_data + m_header_size; }
    size_t resource_size() const { return m_size - m_header_size; }

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

private:
    bool decode_header(const uint8_t* data, size_t size);

    uint8_t* m_data{nullptr};
    size_t m_size{0};

    uint32_t m_dxgi_format;
    TextureType m_type;
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_depth;
    uint32_t m_mip_count;
    uint32_t m_array_size;
    uint32_t m_row_pitch;
    uint32_t m_slice_pitch;
    uint32_t m_bits_per_pixel_or_block;
    uint32_t m_block_width;
    uint32_t m_block_height;
    bool m_compressed;
    bool m_srgb;
    size_t m_header_size;
};

SGL_ENUM_REGISTER(DDSFile::TextureType);

} // namespace sgl
