// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bc_dds.h"

#include "sgl/core/error.h"
#include "sgl/core/file_stream.h"
#include "sgl/device/native_formats.h"

#include <algorithm>
#include <cstring>

namespace sgl {

BCCompressedImage bc_compressed_image_from_dds(const DDSFile& dds)
{
    // Validate: must be a 2D texture, single array element, depth 1.
    SGL_CHECK(
        dds.type() == DDSFile::TextureType::texture_2d,
        "bc_compressed_image_from_dds: only 2D textures are supported (got {})",
        dds.type()
    );
    SGL_CHECK(dds.array_size() == 1, "bc_compressed_image_from_dds: array textures are not supported");
    SGL_CHECK(dds.depth() == 1, "bc_compressed_image_from_dds: 3D textures are not supported");

    // Convert DXGI format to sgl::Format, then to BCFormat.
    Format format = get_format(DXGI_FORMAT(dds.dxgi_format()));
    SGL_CHECK(
        format != Format::undefined,
        "bc_compressed_image_from_dds: unsupported DXGI format {}",
        dds.dxgi_format()
    );

    std::optional<BCFormat> bc_format = format_to_bc_format(format);
    SGL_CHECK(
        bc_format.has_value(),
        "bc_compressed_image_from_dds: DDS format is not a BC format (DXGI {})",
        dds.dxgi_format()
    );

    BCCompressedImage result;
    result.format = *bc_format;
    result.mip_levels.resize(dds.mip_count());

    // Walk resource data contiguously rather than using get_subresource_data(),
    // which uses an approximate offset calculation (>> 2*m) that is inaccurate
    // for textures with non-power-of-2 dimensions.
    const uint8_t* ptr = dds.resource_data();
    for (uint32_t mip = 0; mip < dds.mip_count(); ++mip) {
        uint32_t mip_width = std::max(1u, dds.width() >> mip);
        uint32_t mip_height = std::max(1u, dds.height() >> mip);

        uint32_t row_pitch, slice_pitch;
        dds.get_subresource_pitch(mip, &row_pitch, &slice_pitch);

        BCCompressedMip& mip_level = result.mip_levels[mip];
        mip_level.width = mip_width;
        mip_level.height = mip_height;
        mip_level.data.resize(slice_pitch);
        std::memcpy(mip_level.data.data(), ptr, slice_pitch);
        ptr += slice_pitch;
    }

    return result;
}

void bc_compressed_image_to_dds(const BCCompressedImage& image, Stream* stream)
{
    SGL_CHECK(!image.mip_levels.empty(), "bc_compressed_image_to_dds: image has no mip levels");

    // Convert BCFormat -> sgl::Format -> DXGI format.
    Format format = bc_format_to_format(image.format);
    SGL_CHECK(format != Format::undefined, "bc_compressed_image_to_dds: unsupported BC format");

    uint32_t dxgi_format = get_format_info(format).dxgi_format;

    uint32_t width = image.mip_levels[0].width;
    uint32_t height = image.mip_levels[0].height;
    uint32_t mip_count = static_cast<uint32_t>(image.mip_levels.size());

    // Concatenate all mip level data.
    size_t total_size = 0;
    for (const auto& mip : image.mip_levels)
        total_size += mip.data.size();

    std::vector<uint8_t> resource_data(total_size);
    size_t offset = 0;
    for (const auto& mip : image.mip_levels) {
        std::memcpy(resource_data.data() + offset, mip.data.data(), mip.data.size());
        offset += mip.data.size();
    }

    DDSFile::write_dds(
        stream,
        dxgi_format,
        DDSFile::TextureType::texture_2d,
        width,
        height,
        1, // depth
        mip_count,
        1, // array_size
        resource_data.data(),
        resource_data.size()
    );
}

void bc_compressed_image_to_dds(const BCCompressedImage& image, const std::filesystem::path& path)
{
    FileStream stream(path, FileStream::Mode::write);
    bc_compressed_image_to_dds(image, &stream);
}

} // namespace sgl
