// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bc_dds.h"

#include "sgl/core/error.h"
#include "sgl/core/file_stream.h"
#include "sgl/device/native_formats.h"

#include <algorithm>
#include <cstring>
#include <limits>

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

    Format format = get_format(DXGI_FORMAT(dds.dxgi_format()));
    SGL_CHECK(bc_format_bytes_per_block(format) != 0, "bc_compressed_image_from_dds: DDS format is not a BC format");
    std::vector<BCCompressedMip> mips;
    mips.reserve(dds.mip_count());

    for (uint32_t mip = 0; mip < dds.mip_count(); ++mip) {
        uint32_t mip_width = mip >= 32 ? 1u : std::max(1u, dds.width() >> mip);
        uint32_t mip_height = mip >= 32 ? 1u : std::max(1u, dds.height() >> mip);

        uint32_t row_pitch, slice_pitch;
        dds.get_subresource_pitch(mip, &row_pitch, &slice_pitch);

        std::vector<uint8_t> data(slice_pitch);
        std::memcpy(data.data(), dds.get_subresource_data(mip, 0), slice_pitch);
        mips.emplace_back(mip_width, mip_height, std::move(data));
    }

    return BCCompressedImage(format, std::move(mips));
}

void bc_compressed_image_to_dds(const BCCompressedImage& image, Stream* stream)
{
    uint32_t dxgi_format = get_format_info(image.format()).dxgi_format;

    // Concatenate all mip level data.
    size_t total_size = 0;
    for (const auto& mip : image.mip_levels()) {
        SGL_CHECK(
            mip.data().size() <= std::numeric_limits<size_t>::max() - total_size,
            "bc_compressed_image_to_dds: total data size overflow"
        );
        total_size += mip.data().size();
    }

    std::vector<uint8_t> resource_data(total_size);
    size_t offset = 0;
    for (const auto& mip : image.mip_levels()) {
        std::memcpy(resource_data.data() + offset, mip.data().data(), mip.data().size());
        offset += mip.data().size();
    }

    DDSFile::write_dds(
        stream,
        dxgi_format,
        DDSFile::TextureType::texture_2d,
        image.width(),
        image.height(),
        1, // depth
        image.mip_count(),
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

BCCompressedImage::BCCompressedImage(const std::filesystem::path& path)
    : BCCompressedImage(bc_compressed_image_from_dds(DDSFile(path)))
{
}

BCCompressedImage::BCCompressedImage(Stream* stream)
    : BCCompressedImage(bc_compressed_image_from_dds(DDSFile(stream)))
{
}

void BCCompressedImage::write_dds(const std::filesystem::path& path) const
{
    bc_compressed_image_to_dds(*this, path);
}

void BCCompressedImage::write_dds(Stream* stream) const
{
    bc_compressed_image_to_dds(*this, stream);
}

} // namespace sgl
