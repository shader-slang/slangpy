// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/short_vector.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace sgl {

/// Stack-allocated signature buffer.
/// Used in hot paths that need cheap, append-only signature construction.
class SignatureBuffer {
public:
    SignatureBuffer() = default;

    SignatureBuffer(const SignatureBuffer&) = delete;
    SignatureBuffer& operator=(const SignatureBuffer&) = delete;

    void add(const std::string& value) { add_bytes(reinterpret_cast<const uint8_t*>(value.data()), value.length()); }
    void add(const char* value) { add_bytes(reinterpret_cast<const uint8_t*>(value), std::strlen(value)); }

    void add(uint32_t value)
    {
        static constexpr char hex[] = "0123456789abcdef";
        uint8_t buf[8];
        for (int i = 0; i < 8; ++i)
            buf[7 - i] = static_cast<uint8_t>(hex[(value >> (i * 4)) & 0xF]);
        add_bytes(buf, 8);
    }

    void add(uint64_t value)
    {
        static constexpr char hex[] = "0123456789abcdef";
        uint8_t buf[16];
        for (int i = 0; i < 16; ++i)
            buf[15 - i] = static_cast<uint8_t>(hex[(value >> (i * 4)) & 0xF]);
        add_bytes(buf, 16);
    }

    template<typename T>
    SignatureBuffer& operator<<(const T& v)
    {
        add(v);
        return *this;
    }

    std::string_view view() const { return {reinterpret_cast<const char*>(m_buf.data()), m_buf.size()}; }

private:
    short_vector<uint8_t, 1024> m_buf;

    void add_bytes(const uint8_t* data, size_t sz)
    {
        size_t old_size = m_buf.size();
        m_buf.resize(old_size + sz);
        std::memcpy(m_buf.data() + old_size, data, sz);
    }
};

} // namespace sgl
