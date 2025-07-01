// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "print.h"

#include "sgl/core/format.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"

#include "sgl/math/vector.h"
#include "sgl/math/matrix.h"

#include <fmt/args.h>

namespace sgl {
namespace print_buffer {

    /// Argument value kind.
    /// This needs to be in sync with the enum in print.slang.
    enum class Kind {
        scalar,
        vector,
        matrix,
    };
    static_assert(int(Kind::matrix) <= 16, "Kind is encoded in 4 bits");

    /// Argument value type.
    /// This needs to be in sync with the enum in print.slang.
    enum class Type {
        boolean,
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float16,
        float32,
        float64,
        count,
    };
    static_assert(int(Type::count) <= 16, "Type is encoded in 4 bits");

    /// Argument value layout.
    struct Layout {
        Kind kind;
        Type type;
        uint32_t rows;
        uint32_t cols;
    };

    /// Storage for a single argument value.
    template<typename T>
    struct Value {
        static constexpr size_t MAX_ELEMENT_COUNT = 16;
        Layout layout;
        T elements[MAX_ELEMENT_COUNT];
        uint32_t element_count;
    };
} // namespace print_buffer
} // namespace sgl

/// Custom formatter for argument values.
/// Supports printing scalars, vectors and matrices.
/// Takes into account the formatting specifiers of the underlying element type.
template<typename T>
struct fmt::formatter<sgl::print_buffer::Value<T>> : formatter<T> {
    template<typename FormatContext>
    auto format(const sgl::print_buffer::Value<T>& v, FormatContext& ctx) const
    {
        auto out = ctx.out();
        switch (v.layout.kind) {
        case sgl::print_buffer::Kind::scalar:
            out = formatter<T>::format(v.elements[0], ctx);
            break;
        case sgl::print_buffer::Kind::vector:
            for (uint32_t i = 0; i < v.element_count; ++i) {
                out = fmt::format_to(out, "{}", (i == 0) ? "{" : ", ");
                out = formatter<T>::format(v.elements[i], ctx);
            }
            out = fmt::format_to(out, "}}");
            break;
        case sgl::print_buffer::Kind::matrix:
            for (uint32_t r = 0; r < v.layout.rows; ++r) {
                out = fmt::format_to(out, "{}", (r == 0) ? "{" : ", ");
                for (uint32_t c = 0; c < v.layout.cols; ++c) {
                    out = fmt::format_to(out, "{}", (c == 0) ? "{" : ", ");
                    out = formatter<T>::format(v.elements[r * v.layout.cols + c], ctx);
                }
                out = fmt::format_to(out, "}}");
            }
            out = fmt::format_to(out, "}}");
            break;
        }
        return out;
    }
};

namespace sgl {
namespace print_buffer {

    template<typename T>
    inline Value<T> decode_value(std::span<const uint8_t> data, const Layout layout)
    {
        Value<T> value;
        value.layout = layout;

        switch (layout.kind) {
        case Kind::scalar:
            value.element_count = 1;
            break;
        case Kind::vector:
            value.element_count = layout.rows;
            break;
        case Kind::matrix:
            value.element_count = layout.rows * layout.cols;
            break;
        default:
            SGL_THROW("Invalid argument kind in print()");
        }

        SGL_ASSERT(value.element_count <= Value<T>::MAX_ELEMENT_COUNT);

        // Elements are aligned to 4 bytes.
        uint32_t element_size = ((uint32_t(sizeof(T)) + 3) / 4) * 4;
        SGL_ASSERT(data.size() >= element_size * value.element_count);

        for (uint32_t i = 0; i < value.element_count; ++i)
            std::memcpy(&value.elements[i], data.data() + i * element_size, element_size);

        return value;
    }

    inline void decode_arg(std::span<const uint8_t> data, fmt::dynamic_format_arg_store<fmt::format_context>& arg_store)
    {
        SGL_ASSERT(data.size() >= 4);
        uint32_t header = *reinterpret_cast<const uint32_t*>(data.data());
        uint32_t arg_size = header & 0xffff;
        SGL_ASSERT(data.size() == arg_size);
        Layout layout;
        layout.kind = Kind((header >> 28) & 0xf);
        layout.type = Type((header >> 24) & 0xf);
        layout.rows = (header >> 20) & 0xf;
        layout.cols = (header >> 16) & 0xf;

        switch (layout.type) {
        case Type::boolean:
            arg_store.push_back(decode_value<bool>(data.subspan(4), layout));
            break;
        case Type::int8:
            arg_store.push_back(decode_value<int8_t>(data.subspan(4), layout));
            break;
        case Type::int16:
            arg_store.push_back(decode_value<int16_t>(data.subspan(4), layout));
            break;
        case Type::int32:
            arg_store.push_back(decode_value<int32_t>(data.subspan(4), layout));
            break;
        case Type::int64:
            arg_store.push_back(decode_value<int64_t>(data.subspan(4), layout));
            break;
        case Type::uint8:
            arg_store.push_back(decode_value<uint8_t>(data.subspan(4), layout));
            break;
        case Type::uint16:
            arg_store.push_back(decode_value<uint16_t>(data.subspan(4), layout));
            break;
        case Type::uint32:
            arg_store.push_back(decode_value<uint32_t>(data.subspan(4), layout));
            break;
        case Type::uint64:
            arg_store.push_back(decode_value<uint64_t>(data.subspan(4), layout));
            break;
        case Type::float16:
            arg_store.push_back(decode_value<float16_t>(data.subspan(4), layout));
            break;
        case Type::float32:
            arg_store.push_back(decode_value<float>(data.subspan(4), layout));
            break;
        case Type::float64:
            arg_store.push_back(decode_value<double>(data.subspan(4), layout));
            break;
        default:
            SGL_THROW("Invalid argument type in print()");
        }
    }

    template<typename Output>
    inline void
    decode_msg(std::span<const uint8_t> data, const std::map<uint32_t, std::string>& hashed_strings, Output output)
    {
        const uint8_t* ptr = data.data();

        // Decode message header.
        SGL_ASSERT(data.size() >= 12);
        uint32_t msg_size = *reinterpret_cast<const uint32_t*>(ptr);
        SGL_ASSERT(data.size() == msg_size);
        uint32_t fmt_hash = *reinterpret_cast<const uint32_t*>(ptr + 4);
        uint32_t arg_count = *reinterpret_cast<const uint32_t*>(ptr + 8);
        const uint8_t* end = ptr + msg_size;
        ptr += 12;

        // Decode arguments and append to argument store.
        fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
        for (uint32_t i = 0; i < arg_count; ++i) {
            uint32_t arg_size = *reinterpret_cast<const uint32_t*>(ptr) & 0xffff;
            SGL_ASSERT(ptr + arg_size <= end);
            decode_arg(std::span(ptr, arg_size), arg_store);
            ptr += arg_size;
        }

        /// Lookup format string.
        std::string_view fmt;
        if (auto it = hashed_strings.find(fmt_hash); it != hashed_strings.end())
            fmt = it->second;

        /// Output formatted string.
        try {
            output(fmt::vformat(fmt, arg_store));
        } catch (const fmt::format_error& e) {
            SGL_THROW("Invalid shader print() formatting: {}\nFormat string:\n{}", e.what(), fmt);
        }
    }

    template<typename Output>
    inline void
    decode_buffer(const void* data, size_t size, const std::map<uint32_t, std::string>& hashed_strings, Output output)
    {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
        uint32_t buffer_size = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;
        // Clamp buffer size to the actual size of the buffer (in case the device has overflown the buffer).
        buffer_size = std::min(buffer_size, uint32_t(size) - 4);
        const uint8_t* end = ptr + buffer_size;

        size_t count = 0;
        while (ptr < end) {
            uint32_t msg_size = *reinterpret_cast<const uint32_t*>(ptr);
            if (msg_size == 0xffffffff) {
                log_warn("Print buffer overflow!");
                break;
            }
            SGL_ASSERT(ptr + msg_size <= end)
            decode_msg(std::span(ptr, msg_size), hashed_strings, output);
            ++count;
            ptr += msg_size;
        }
    }

} // namespace print_buffer


DebugPrinter::DebugPrinter(Device* device, size_t buffer_size)
    : m_device(device)
{
    m_buffer = m_device->create_buffer({
        .size = buffer_size,
        .usage = BufferUsage::unordered_access | BufferUsage::copy_source,
        .label = "debug_printer_buffer",
    });

    m_buffer_size = m_device->create_buffer({
        .size = 4,
        .usage = BufferUsage::unordered_access | BufferUsage::copy_source,
        .label = "debug_printer_buffer_size",
    });

    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    command_encoder->clear_buffer(m_buffer_size);
    m_device->submit_command_buffer(command_encoder->finish());

    m_readback_buffer = m_device->create_buffer({
        .size = buffer_size + 4,
        .memory_type = MemoryType::read_back,
        .usage = BufferUsage::copy_destination,
        .label = "debug_printer_readback_buffer",
    });
}

void DebugPrinter::add_hashed_strings(const std::map<uint32_t, std::string>& hashed_strings)
{
    m_hashed_strings.insert(hashed_strings.begin(), hashed_strings.end());
}

void DebugPrinter::flush()
{
    flush_device(true);
    const void* data = m_readback_buffer->map();
    print_buffer::decode_buffer(
        data,
        m_readback_buffer->size(),
        m_hashed_strings,
        [](std::string_view str) { Logger::get().log(LogLevel::none, str); }
    );
    m_readback_buffer->unmap();
}

std::string DebugPrinter::flush_to_string()
{
    flush_device(true);
    std::string result;
    const void* data = m_readback_buffer->map();
    print_buffer::decode_buffer(
        data,
        m_readback_buffer->size(),
        m_hashed_strings,
        [&result](std::string_view str)
        {
            result += str;
            result += "\n";
        }
    );
    m_readback_buffer->unmap();
    return result;
}

void DebugPrinter::bind(ShaderCursor cursor)
{
    if (cursor.is_valid())
        cursor = cursor.find_field("g_debug_printer");
    if (cursor.is_valid()) {
        cursor["buffer"] = m_buffer;
        cursor["buffer_capacity"] = static_cast<uint32_t>(m_buffer->size());
        cursor["buffer_size"] = m_buffer_size;
    }
}

void DebugPrinter::flush_device(bool wait)
{
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    command_encoder->copy_buffer(m_readback_buffer, 0, m_buffer_size, 0, 4);
    command_encoder->copy_buffer(m_readback_buffer, 4, m_buffer, 0, m_buffer->size());
    command_encoder->clear_buffer(m_buffer_size);
    m_device->submit_command_buffer(command_encoder->finish());
    if (wait)
        m_device->wait_for_idle();
}

} // namespace sgl
