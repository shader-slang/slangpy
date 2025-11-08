// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "memory_mapped_file.h"

#include <cstdio>
#include <stdexcept>

#if SGL_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif SGL_LINUX
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif SGL_MACOS
#define _DARWIN_USE_64_BIT_INODE
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#error "Unknown OS"
#endif

namespace sgl {

MemoryMappedFile::MemoryMappedFile(const std::filesystem::path& path, size_t mapped_size, AccessHint access_hint)
{
    open(path, mapped_size, access_hint);
}

MemoryMappedFile::~MemoryMappedFile()
{
    close();
}

bool MemoryMappedFile::open(const std::filesystem::path& path, size_t mapped_size, AccessHint access_hint)
{
    if (is_open())
        return false;

    m_path = path;
    m_access_hint = access_hint;

#if SGL_WINDOWS
    // Handle access hint.
    DWORD flags = 0;
    switch (m_access_hint) {
    case AccessHint::normal:
        flags = FILE_ATTRIBUTE_NORMAL;
        break;
    case AccessHint::sequential:
        flags = FILE_FLAG_SEQUENTIAL_SCAN;
        break;
    case AccessHint::random:
        flags = FILE_FLAG_RANDOM_ACCESS;
        break;
    default:
        break;
    }

    // Open file.
    m_file = ::CreateFileW(m_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
    if (!m_file)
        return false;

    // Get file size.
    LARGE_INTEGER size;
    if (!::GetFileSizeEx(m_file, &size)) {
        close();
        return false;
    }
    m_size = static_cast<size_t>(size.QuadPart);

    // Create file mapping.
    m_mapped_file = ::CreateFileMapping(m_file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m_mapped_file) {
        close();
        return false;
    }

#elif SGL_LINUX

    // Open file.
    m_file = ::open(path.c_str(), O_RDONLY | O_LARGEFILE);
    if (m_file == -1) {
        m_file = 0;
        return false;
    }

    // Get file size.
    struct stat64 stat_info;
    if (fstat64(m_file, &stat_info) < 0) {
        close();
        return false;
    }
    m_size = stat_info.st_size;

#elif SGL_MACOS

    // Open file.
    m_file = ::open(path.c_str(), O_RDONLY);
    if (m_file == -1) {
        m_file = 0;
        return false;
    }

    // Get file size.
    struct stat stat_info;
    if (fstat(m_file, &stat_info) < 0) {
        close();
        return false;
    }
    m_size = stat_info.st_size;

#endif

    // Initial mapping.
    if (!remap(0, mapped_size)) {
        close();
        return false;
    }

    return true;
}

void MemoryMappedFile::close()
{
    // Unmap memory.
    if (m_mapped_data) {
#if SGL_WINDOWS
        ::UnmapViewOfFile(m_mapped_data);
#elif SGL_LINUX || SGL_MACOS
        ::munmap(m_mapped_data, m_mapped_size);
#endif
        m_mapped_data = nullptr;
    }

#if SGL_WINDOWS
    if (m_mapped_file) {
        ::CloseHandle(m_mapped_file);
        m_mapped_file = nullptr;
    }
#endif

    // Close file.
    if (m_file) {
#if SGL_WINDOWS
        ::CloseHandle(m_file);
#elif SGL_LINUX || SGL_MACOS
        ::close(m_file);
#endif
        m_file = 0;
    }

    m_size = 0;
}

size_t MemoryMappedFile::page_size()
{
#if SGL_WINDOWS
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwAllocationGranularity;
#elif SGL_LINUX || SGL_MACOS
    return sysconf(_SC_PAGESIZE);
#endif
}

bool MemoryMappedFile::remap(uint64_t offset, size_t mapped_size)
{
    if (!m_file)
        return false;
    if (offset >= m_size)
        return false;

    // Close previous mapping.
    if (m_mapped_data) {
#if SGL_WINDOWS
        ::UnmapViewOfFile(m_mapped_data);
#elif SGL_LINUX || SGL_MACOS
        ::munmap(m_mapped_data, m_mapped_size);
#endif
        m_mapped_data = nullptr;
        m_mapped_size = 0;
    }

    // Clamp mapped range.
    if (offset + mapped_size > m_size)
        mapped_size = size_t(m_size - offset);

#if SGL_WINDOWS
    DWORD offsetLow = DWORD(offset & 0xFFFFFFFF);
    DWORD offsetHigh = DWORD(offset >> 32);

    // Create new mapping.
    m_mapped_data = ::MapViewOfFile(m_mapped_file, FILE_MAP_READ, offsetHigh, offsetLow, mapped_size);
    if (!m_mapped_data)
        m_mapped_size = 0;
    m_mapped_size = mapped_size;
#elif SGL_LINUX || SGL_MACOS
    // Create new mapping.
#if SGL_LINUX
    m_mapped_data = ::mmap64(NULL, mapped_size, PROT_READ, MAP_SHARED, m_file, offset);
#elif SGL_MACOS
    m_mapped_data = ::mmap(NULL, mapped_size, PROT_READ, MAP_SHARED, m_file, offset);
#endif
    if (m_mapped_data == MAP_FAILED) {
        m_mapped_data = nullptr;
        return false;
    }
    m_mapped_size = mapped_size;

    // Handle access hint.
    int advice = 0;
    switch (m_access_hint) {
    case AccessHint::normal:
        advice = MADV_NORMAL;
        break;
    case AccessHint::sequential:
        advice = MADV_SEQUENTIAL;
        break;
    case AccessHint::random:
        advice = MADV_RANDOM;
        break;
    default:
        break;
    }
    ::madvise(m_mapped_data, m_mapped_size, advice);
#endif

    return true;
}

} // namespace sgl
