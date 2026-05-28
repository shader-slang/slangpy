// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/memory_stream.h"
#include "sgl/core/memory_mapped_file.h"

#include <filesystem>
#include <memory>

namespace sgl {

class SGL_API MemoryMappedFileStream : public MemoryStream {
    SGL_OBJECT(MemoryMappedFileStream)
public:
    MemoryMappedFileStream(
        const std::filesystem::path& path,
        size_t mapped_size = MemoryMappedFile::WHOLE_FILE,
        MemoryMappedFile::AccessHint access_hint = MemoryMappedFile::AccessHint::normal
    );
    ~MemoryMappedFileStream();

    const std::filesystem::path& path() const { return m_path; }

    std::string to_string() const override;

public:
    std::filesystem::path m_path;
    std::unique_ptr<MemoryMappedFile> m_file;
};

} // namespace sgl
