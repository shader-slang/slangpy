// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/stream.h"
#include "sgl/core/enum.h"

#include <exception>
#include <filesystem>
#include <iosfwd>
#include <memory>

namespace sgl {


class SGL_API FileStream : public Stream {
    SGL_OBJECT(FileStream)
public:
    enum class Mode {
        read,
        write,
        read_write,
    };

    SGL_ENUM_INFO(
        Mode,
        {
            {Mode::read, "read"},
            {Mode::write, "write"},
            {Mode::read_write, "read_write"},
        }
    );

    FileStream(const std::filesystem::path& path, Mode mode);
    virtual ~FileStream();

    const std::filesystem::path& path() const { return m_path; }
    Mode mode() const { return m_mode; }

    bool is_open() const override;
    bool is_readable() const override { return m_mode == Mode::read || m_mode == Mode::read_write; }
    bool is_writable() const override { return m_mode == Mode::write || m_mode == Mode::read_write; }

    void close() override;

    void read(void* p, size_t size) override;
    void write(const void* p, size_t size) override;
    void seek(size_t pos) override;
    void truncate(size_t size) override;
    size_t tell() const override;
    size_t size() const override;
    void flush() override;

    std::string to_string() const override;

private:
    std::filesystem::path m_path;
    Mode m_mode;
    std::unique_ptr<std::fstream> m_stream;
};

SGL_ENUM_REGISTER(FileStream::Mode);

} // namespace sgl
