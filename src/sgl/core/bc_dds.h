// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/bc_types.h"
#include "sgl/core/dds_file.h"

#include <filesystem>

namespace sgl {

/// Extract a BCCompressedImage from a DDS file.
/// The DDS file must contain a BC-compressed 2D texture with array_size == 1 and depth == 1.
SGL_API BCCompressedImage bc_compressed_image_from_dds(const DDSFile& dds);

/// Write a BCCompressedImage as a DDS file to a stream.
SGL_API void bc_compressed_image_to_dds(const BCCompressedImage& image, Stream* stream);

/// Write a BCCompressedImage as a DDS file to a file path.
SGL_API void bc_compressed_image_to_dds(const BCCompressedImage& image, const std::filesystem::path& path);

} // namespace sgl
