// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <cstdint>
#include <filesystem>
#include <limits>

namespace sgl {

/**
 * Utility class for reading memory-mapped files.
 */
class SGL_API MemoryMappedFile {
public:
    enum class AccessHint {
        /// Good overall performance.
        normal,
        /// Read file once with few seeks.
        sequential,
        /// Good for random access.
        random
    };

    static constexpr size_t WHOLE_FILE = std::numeric_limits<size_t>::max();

    /**
     * Default constructor. Use open() for opening a file.
     */
    MemoryMappedFile() = default;

    /**
     * Constructor opening a file. Use is_open() to check if successful.
     * \param path Path to open.
     * \param mapped_size Number of bytes to map into memory (automatically clamped to the file size).
     * \param access_hint Hint on how memory is accessed.
     */
    MemoryMappedFile(
        const std::filesystem::path& path,
        size_t mapped_size = WHOLE_FILE,
        AccessHint access_hint = AccessHint::normal
    );

    /// Destructor. Closes the file.
    ~MemoryMappedFile();

    /**
     * Open a file.
     * \param path Path to open.
     * \param mapped_size Number of bytes to map into memory (automatically clamped to the file size).
     * \param access_hint Hint on how memory is accessed.
     * \return True if file was successfully opened.
     */
    bool open(
        const std::filesystem::path& path,
        size_t mapped_size = WHOLE_FILE,
        AccessHint access_hint = AccessHint::normal
    );

    /// Close the file.
    void close();

    /// True, if file successfully opened.
    bool is_open() const { return m_mapped_data != nullptr; }

    /// Get the file size in bytes.
    size_t size() const { return m_size; }

    /// Get the mapped data.
    const void* data() const { return m_mapped_data; };

    /// Get the mapped memory size in bytes.
    size_t mapped_size() const { return m_mapped_size; };

    /// Get the OS page size (for remap).
    static size_t page_size();

private:
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile(MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&&) = delete;

    /**
     * Replace mapping by a new one of the same file.
     * \param offset Offset from start of the file in bytes (must be multiple of page size).
     * \param mapped_size Size of mapping in bytes.
     * \return True if successful.
     */
    bool remap(uint64_t offset, size_t mapped_size);

    std::filesystem::path m_path;
    AccessHint m_access_hint = AccessHint::normal;
    size_t m_size = 0;

#if SGL_WINDOWS
    using FileHandle = void*;
    void* m_mapped_file{nullptr};
#elif SGL_LINUX || SGL_MACOS
    using FileHandle = int;
#else
#error "Unknown OS"
#endif

    FileHandle m_file = 0;
    void* m_mapped_data = 0;
    size_t m_mapped_size = 0;
};

} // namespace sgl
