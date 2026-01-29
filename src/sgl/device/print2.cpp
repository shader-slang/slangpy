// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "print2.h"

#include "sgl/core/format.h"
#include "sgl/core/logger.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"

#include "sgl/math/vector.h"
#include "sgl/math/matrix.h"

#include <fmt/args.h>

#include <cstring>
#include <map>
#include <span>
#include <vector>

namespace sgl {

namespace {

    /// Constants for the chunked stream buffer format.
    constexpr uint32_t BUFFER_HEADER_SIZE = 8;        // capacity + write_pos
    constexpr uint32_t CHUNK_SIZE = 256;              // Each chunk is 256 bytes
    constexpr uint32_t CHUNK_HEADER = 16;             // Chunk header size
    constexpr uint32_t CHUNK_PAYLOAD = 240;           // Payload size per chunk (60 uints)
    constexpr uint32_t OVERFLOW_CHUNK = 8;            // Reserved overflow chunk offset
    constexpr uint32_t FIRST_DATA_CHUNK = 264;        // First actual data chunk (after header + overflow)
    constexpr uint32_t MAGIC_FIRST = 0xFFFFFFFF;      // First chunk of stream marker
    constexpr uint32_t MAGIC_CONTINUE = 0x00000000;   // Continuation chunk marker
    constexpr uint32_t MAGIC_OVERFLOW = 0xFFFFFFFE;   // Overflow chunk marker

    /// Record kind values (must match print2.slang).
    enum class RecordKind : uint8_t {
        format          = 0,
        scalar          = 1,
        vector          = 2,
        matrix          = 3,
        begin_composite = 4,
        end_composite   = 5,
    };

    /// Scalar type values (must match print2.slang).
    enum class ScalarType : uint8_t {
        bool_   = 0,
        int8    = 1,
        int16   = 2,
        int32   = 3,
        int64   = 4,
        uint8   = 5,
        uint16  = 6,
        uint32  = 7,
        uint64  = 8,
        float16 = 9,
        float32 = 10,
        float64 = 11,
        string  = 12,
    };

    /// Returns number of uints per element for a given scalar type.
    inline uint32_t uints_per_element(ScalarType type)
    {
        switch (type) {
        case ScalarType::int64:
        case ScalarType::uint64:
        case ScalarType::float64:
            return 2;
        default:
            return 1;
        }
    }

    /// Decode a stream chain starting from a first chunk.
    /// Returns the collected payload values from all chunks in the chain.
    /// Returns empty vector if the stream has overflowed (contains MAGIC_OVERFLOW).
    std::vector<uint32_t> decode_stream_chain(const uint32_t* data, uint32_t first_chunk, uint32_t buffer_size, bool& overflowed)
    {
        std::vector<uint32_t> payload;
        overflowed = false;

        uint32_t chunk = first_chunk;
        while (chunk != 0) {
            // Bounds check
            if (chunk + CHUNK_SIZE > buffer_size) {
                log_warn("Stream chunk out of bounds: {} + {} > {}", chunk, CHUNK_SIZE, buffer_size);
                break;
            }

            // Read chunk header: [magic:32][next:32][used:32][reserved:32]
            uint32_t magic = data[chunk / 4 + 0];
            uint32_t next = data[chunk / 4 + 1];
            uint32_t used = data[chunk / 4 + 2];

            // Check for overflow
            if (magic == MAGIC_OVERFLOW) {
                overflowed = true;
                return {}; // Discard this stream
            }

            // Validate magic (first chunk should be MAGIC_FIRST, others MAGIC_CONTINUE)
            if (chunk == first_chunk && magic != MAGIC_FIRST) {
                log_warn("Expected first chunk magic at offset {}", chunk);
                break;
            }

            // Clamp used to payload size
            used = std::min(used, CHUNK_PAYLOAD);

            // Read payload (used bytes, in 4-byte increments)
            uint32_t payload_offset = chunk + CHUNK_HEADER;
            for (uint32_t i = 0; i < used; i += 4) {
                payload.push_back(data[(payload_offset + i) / 4]);
            }

            chunk = next;
        }

        return payload;
    }

    /// Decode all streams from the buffer.
    /// Returns a vector of streams, where each stream is a vector of uint32_t payloads.
    std::vector<std::vector<uint32_t>> decode_all_streams(const void* buffer_data, size_t buffer_size, bool& had_overflow)
    {
        std::vector<std::vector<uint32_t>> streams;
        had_overflow = false;

        const uint32_t* data = reinterpret_cast<const uint32_t*>(buffer_data);

        // Read header
        uint32_t capacity = data[0];
        uint32_t write_pos = data[1];

        // Clamp write_pos to buffer size (in case of overflow)
        write_pos = std::min(write_pos, static_cast<uint32_t>(buffer_size));

        if (write_pos > capacity) {
            had_overflow = true;
        }

        // Scan for first chunks at 256-byte intervals starting from offset 264 (skip overflow chunk)
        for (uint32_t offset = FIRST_DATA_CHUNK; offset + CHUNK_SIZE <= write_pos; offset += CHUNK_SIZE) {
            uint32_t magic = data[offset / 4];

            // Check for first chunk marker
            if (magic == MAGIC_FIRST) {
                bool stream_overflowed = false;
                auto stream = decode_stream_chain(data, offset, write_pos, stream_overflowed);
                if (stream_overflowed) {
                    had_overflow = true;
                } else if (!stream.empty()) {
                    streams.push_back(std::move(stream));
                }
            }
        }

        return streams;
    }

    // -------------------------------------------------------------------------
    // Record Decoder
    // -------------------------------------------------------------------------

    /// Reader for consuming uint32_t values from a stream.
    class StreamReader {
    public:
        StreamReader(std::span<const uint32_t> data)
            : m_data(data)
            , m_pos(0)
        {
        }

        bool has_more() const { return m_pos < m_data.size(); }
        size_t remaining() const { return m_data.size() - m_pos; }
        size_t position() const { return m_pos; }

        uint32_t read()
        {
            SGL_CHECK(m_pos < m_data.size(), "StreamReader: read past end");
            return m_data[m_pos++];
        }

        uint32_t peek() const
        {
            SGL_CHECK(m_pos < m_data.size(), "StreamReader: peek past end");
            return m_data[m_pos];
        }

    private:
        std::span<const uint32_t> m_data;
        size_t m_pos;
    };

    /// Storage for scalar/vector/matrix values.
    template<typename T>
    struct Value {
        static constexpr size_t MAX_ELEMENT_COUNT = 16;
        T elements[MAX_ELEMENT_COUNT];
        uint32_t rows;
        uint32_t cols;
        uint32_t element_count;
        RecordKind kind;
    };

    /// Decode scalar value from stream.
    template<typename T>
    Value<T> decode_scalar(StreamReader& reader, ScalarType type)
    {
        Value<T> value;
        value.kind = RecordKind::scalar;
        value.rows = 1;
        value.cols = 1;
        value.element_count = 1;

        uint32_t uints = uints_per_element(type);
        if constexpr (sizeof(T) <= 4) {
            uint32_t raw = reader.read();
            std::memcpy(&value.elements[0], &raw, sizeof(T));
        } else {
            uint32_t lo = reader.read();
            uint32_t hi = reader.read();
            uint64_t raw = (uint64_t(hi) << 32) | lo;
            std::memcpy(&value.elements[0], &raw, sizeof(T));
        }
        return value;
    }

    /// Decode vector value from stream.
    template<typename T>
    Value<T> decode_vector(StreamReader& reader, ScalarType type, uint32_t count)
    {
        Value<T> value;
        value.kind = RecordKind::vector;
        value.rows = count;
        value.cols = 1;
        value.element_count = count;

        uint32_t uints = uints_per_element(type);
        for (uint32_t i = 0; i < count; ++i) {
            if constexpr (sizeof(T) <= 4) {
                uint32_t raw = reader.read();
                std::memcpy(&value.elements[i], &raw, sizeof(T));
            } else {
                uint32_t lo = reader.read();
                uint32_t hi = reader.read();
                uint64_t raw = (uint64_t(hi) << 32) | lo;
                std::memcpy(&value.elements[i], &raw, sizeof(T));
            }
        }
        return value;
    }

    /// Decode matrix value from stream.
    template<typename T>
    Value<T> decode_matrix(StreamReader& reader, ScalarType type, uint32_t rows, uint32_t cols)
    {
        Value<T> value;
        value.kind = RecordKind::matrix;
        value.rows = rows;
        value.cols = cols;
        value.element_count = rows * cols;

        SGL_CHECK(value.element_count <= Value<T>::MAX_ELEMENT_COUNT, "Matrix too large");

        uint32_t uints = uints_per_element(type);
        for (uint32_t i = 0; i < value.element_count; ++i) {
            if constexpr (sizeof(T) <= 4) {
                uint32_t raw = reader.read();
                std::memcpy(&value.elements[i], &raw, sizeof(T));
            } else {
                uint32_t lo = reader.read();
                uint32_t hi = reader.read();
                uint64_t raw = (uint64_t(hi) << 32) | lo;
                std::memcpy(&value.elements[i], &raw, sizeof(T));
            }
        }
        return value;
    }

} // anonymous namespace
} // namespace sgl

/// Custom formatter for Value<T>.
template<typename T>
struct fmt::formatter<sgl::Value<T>> : formatter<T> {
    template<typename FormatContext>
    auto format(const sgl::Value<T>& v, FormatContext& ctx) const
    {
        auto out = ctx.out();
        switch (v.kind) {
        case sgl::RecordKind::scalar:
            out = formatter<T>::format(v.elements[0], ctx);
            break;
        case sgl::RecordKind::vector:
            for (uint32_t i = 0; i < v.element_count; ++i) {
                out = fmt::format_to(out, "{}", (i == 0) ? "{" : ", ");
                out = formatter<T>::format(v.elements[i], ctx);
            }
            out = fmt::format_to(out, "}}");
            break;
        case sgl::RecordKind::matrix:
            for (uint32_t r = 0; r < v.rows; ++r) {
                out = fmt::format_to(out, "{}", (r == 0) ? "{" : ", ");
                for (uint32_t c = 0; c < v.cols; ++c) {
                    out = fmt::format_to(out, "{}", (c == 0) ? "{" : ", ");
                    out = formatter<T>::format(v.elements[r * v.cols + c], ctx);
                }
                out = fmt::format_to(out, "}}");
            }
            out = fmt::format_to(out, "}}");
            break;
        default:
            break;
        }
        return out;
    }
};

namespace sgl {
namespace {

    // Forward declaration
    std::string decode_record(StreamReader& reader, const std::map<uint32_t, std::string>& hashed_strings);

    /// Decode a data record (scalar, vector, or matrix) and add to arg store.
    void decode_data_record(
        StreamReader& reader,
        RecordKind kind,
        uint32_t header,
        fmt::dynamic_format_arg_store<fmt::format_context>& arg_store,
        const std::map<uint32_t, std::string>& hashed_strings
    )
    {
        ScalarType type = ScalarType((header >> 16) & 0xFF);

        if (kind == RecordKind::scalar) {
            switch (type) {
            case ScalarType::bool_:
                arg_store.push_back(decode_scalar<bool>(reader, type));
                break;
            case ScalarType::int8:
                arg_store.push_back(decode_scalar<int8_t>(reader, type));
                break;
            case ScalarType::int16:
                arg_store.push_back(decode_scalar<int16_t>(reader, type));
                break;
            case ScalarType::int32:
                arg_store.push_back(decode_scalar<int32_t>(reader, type));
                break;
            case ScalarType::int64:
                arg_store.push_back(decode_scalar<int64_t>(reader, type));
                break;
            case ScalarType::uint8:
                arg_store.push_back(decode_scalar<uint8_t>(reader, type));
                break;
            case ScalarType::uint16:
                arg_store.push_back(decode_scalar<uint16_t>(reader, type));
                break;
            case ScalarType::uint32:
                arg_store.push_back(decode_scalar<uint32_t>(reader, type));
                break;
            case ScalarType::uint64:
                arg_store.push_back(decode_scalar<uint64_t>(reader, type));
                break;
            case ScalarType::float16:
                arg_store.push_back(decode_scalar<float16_t>(reader, type));
                break;
            case ScalarType::float32:
                arg_store.push_back(decode_scalar<float>(reader, type));
                break;
            case ScalarType::float64:
                arg_store.push_back(decode_scalar<double>(reader, type));
                break;
            case ScalarType::string: {
                uint32_t hash = reader.read();
                auto it = hashed_strings.find(hash);
                if (it != hashed_strings.end()) {
                    arg_store.push_back(std::string_view(it->second));
                } else {
                    arg_store.push_back(std::string_view("<unknown string>"));
                }
                break;
            }
            default:
                SGL_THROW("Invalid scalar type: {}", int(type));
            }
        } else if (kind == RecordKind::vector) {
            uint32_t count = (header >> 8) & 0xFF;
            switch (type) {
            case ScalarType::bool_:
                arg_store.push_back(decode_vector<bool>(reader, type, count));
                break;
            case ScalarType::int8:
                arg_store.push_back(decode_vector<int8_t>(reader, type, count));
                break;
            case ScalarType::int16:
                arg_store.push_back(decode_vector<int16_t>(reader, type, count));
                break;
            case ScalarType::int32:
                arg_store.push_back(decode_vector<int32_t>(reader, type, count));
                break;
            case ScalarType::int64:
                arg_store.push_back(decode_vector<int64_t>(reader, type, count));
                break;
            case ScalarType::uint8:
                arg_store.push_back(decode_vector<uint8_t>(reader, type, count));
                break;
            case ScalarType::uint16:
                arg_store.push_back(decode_vector<uint16_t>(reader, type, count));
                break;
            case ScalarType::uint32:
                arg_store.push_back(decode_vector<uint32_t>(reader, type, count));
                break;
            case ScalarType::uint64:
                arg_store.push_back(decode_vector<uint64_t>(reader, type, count));
                break;
            case ScalarType::float16:
                arg_store.push_back(decode_vector<float16_t>(reader, type, count));
                break;
            case ScalarType::float32:
                arg_store.push_back(decode_vector<float>(reader, type, count));
                break;
            case ScalarType::float64:
                arg_store.push_back(decode_vector<double>(reader, type, count));
                break;
            default:
                SGL_THROW("Invalid vector element type: {}", int(type));
            }
        } else if (kind == RecordKind::matrix) {
            uint32_t rows = (header >> 8) & 0xFF;
            uint32_t cols = header & 0xFF;
            switch (type) {
            case ScalarType::bool_:
                arg_store.push_back(decode_matrix<bool>(reader, type, rows, cols));
                break;
            case ScalarType::int8:
                arg_store.push_back(decode_matrix<int8_t>(reader, type, rows, cols));
                break;
            case ScalarType::int16:
                arg_store.push_back(decode_matrix<int16_t>(reader, type, rows, cols));
                break;
            case ScalarType::int32:
                arg_store.push_back(decode_matrix<int32_t>(reader, type, rows, cols));
                break;
            case ScalarType::int64:
                arg_store.push_back(decode_matrix<int64_t>(reader, type, rows, cols));
                break;
            case ScalarType::uint8:
                arg_store.push_back(decode_matrix<uint8_t>(reader, type, rows, cols));
                break;
            case ScalarType::uint16:
                arg_store.push_back(decode_matrix<uint16_t>(reader, type, rows, cols));
                break;
            case ScalarType::uint32:
                arg_store.push_back(decode_matrix<uint32_t>(reader, type, rows, cols));
                break;
            case ScalarType::uint64:
                arg_store.push_back(decode_matrix<uint64_t>(reader, type, rows, cols));
                break;
            case ScalarType::float16:
                arg_store.push_back(decode_matrix<float16_t>(reader, type, rows, cols));
                break;
            case ScalarType::float32:
                arg_store.push_back(decode_matrix<float>(reader, type, rows, cols));
                break;
            case ScalarType::float64:
                arg_store.push_back(decode_matrix<double>(reader, type, rows, cols));
                break;
            default:
                SGL_THROW("Invalid matrix element type: {}", int(type));
            }
        }
    }

    /// Decode a composite argument (between begin_composite and end_composite).
    /// Returns the formatted string result.
    std::string decode_composite(StreamReader& reader, const std::map<uint32_t, std::string>& hashed_strings)
    {
        std::string result;

        while (reader.has_more()) {
            uint32_t header = reader.peek();
            RecordKind kind = RecordKind((header >> 24) & 0xFF);

            if (kind == RecordKind::end_composite) {
                reader.read(); // Consume the end marker
                break;
            }

            result += decode_record(reader, hashed_strings);
        }

        return result;
    }

    /// Decode a single record and return its string representation.
    std::string decode_record(StreamReader& reader, const std::map<uint32_t, std::string>& hashed_strings)
    {
        uint32_t header = reader.read();
        RecordKind kind = RecordKind((header >> 24) & 0xFF);

        switch (kind) {
        case RecordKind::format: {
            uint32_t arg_count = header & 0x00FFFFFF;
            uint32_t fmt_hash = reader.read();

            // Lookup format string
            std::string_view fmt_str;
            auto it = hashed_strings.find(fmt_hash);
            if (it != hashed_strings.end()) {
                fmt_str = it->second;
            } else {
                return fmt::format("<unknown format string: {:08x}>", fmt_hash);
            }

            // Decode arguments
            fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
            for (uint32_t i = 0; i < arg_count; ++i) {
                if (!reader.has_more()) {
                    return fmt::format("<truncated format: expected {} args, got {}>", arg_count, i);
                }

                uint32_t arg_header = reader.peek();
                RecordKind arg_kind = RecordKind((arg_header >> 24) & 0xFF);

                if (arg_kind == RecordKind::begin_composite) {
                    reader.read(); // Consume begin marker
                    std::string composite_result = decode_composite(reader, hashed_strings);
                    arg_store.push_back(composite_result);
                } else if (arg_kind == RecordKind::scalar || arg_kind == RecordKind::vector
                           || arg_kind == RecordKind::matrix) {
                    reader.read(); // Consume header
                    decode_data_record(reader, arg_kind, arg_header, arg_store, hashed_strings);
                } else if (arg_kind == RecordKind::format) {
                    // Nested format without composite wrapper (shouldn't happen with current design)
                    std::string nested = decode_record(reader, hashed_strings);
                    arg_store.push_back(nested);
                } else {
                    return fmt::format("<unexpected record kind in format args: {}>", int(arg_kind));
                }
            }

            // Format the string
            try {
                return fmt::vformat(fmt_str, arg_store);
            } catch (const fmt::format_error& e) {
                return fmt::format("<format error: {}>", e.what());
            }
        }

        case RecordKind::scalar:
        case RecordKind::vector:
        case RecordKind::matrix: {
            // Standalone data record (shouldn't typically happen at top level)
            fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
            decode_data_record(reader, kind, header, arg_store, hashed_strings);
            return fmt::vformat("{}", arg_store);
        }

        case RecordKind::begin_composite:
            // Unexpected begin_composite at top level
            return decode_composite(reader, hashed_strings);

        case RecordKind::end_composite:
            // Unexpected end_composite
            return "<unexpected end_composite>";

        default:
            return fmt::format("<unknown record kind: {}>", int(kind));
        }
    }

    /// Decode a complete stream and return all formatted messages.
    std::vector<std::string>
    decode_stream_records(const std::vector<uint32_t>& stream, const std::map<uint32_t, std::string>& hashed_strings)
    {
        std::vector<std::string> messages;
        StreamReader reader(stream);

        while (reader.has_more()) {
            messages.push_back(decode_record(reader, hashed_strings));
        }

        return messages;
    }

} // anonymous namespace

DebugPrinter::DebugPrinter(Device* device, size_t buffer_size)
    : m_device(device)
{
    m_buffer = m_device->create_buffer({
        .size = buffer_size,
        .usage = BufferUsage::unordered_access | BufferUsage::copy_source,
        .default_state = ResourceState::unordered_access,
        .label = "debug_printer_buffer",
    });

    m_readback_buffer = m_device->create_buffer({
        .size = buffer_size,
        .memory_type = MemoryType::read_back,
        .usage = BufferUsage::copy_destination,
        .label = "debug_printer_readback_buffer",
    });

    // Write buffer header.
    // Buffer layout: [capacity:4][write_pos:4][overflow_chunk:256][data_chunks...]
    // write_pos starts at 264 (after header + reserved overflow chunk)
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    uint32_t header[2] = {uint32_t(buffer_size), FIRST_DATA_CHUNK};
    command_encoder->upload_buffer_data(m_buffer, 0, sizeof(header), header);
    m_device->submit_command_buffer(command_encoder->finish());
}

void DebugPrinter::add_hashed_strings(const std::map<uint32_t, std::string>& hashed_strings)
{
    m_hashed_strings.insert(hashed_strings.begin(), hashed_strings.end());
}

void DebugPrinter::flush()
{
    flush_device(true);

    const void* data = m_readback_buffer->map();
    bool had_overflow = false;
    auto streams = decode_all_streams(data, m_readback_buffer->size(), had_overflow);
    m_readback_buffer->unmap();

    if (had_overflow) {
        log_warn("Print buffer overflow: some messages were lost");
    }

    // Decode and log each stream's messages
    for (const auto& stream : streams) {
        auto messages = decode_stream_records(stream, m_hashed_strings);
        for (const auto& msg : messages) {
            log_info("{}", msg);
        }
    }
}

std::string DebugPrinter::flush_to_string()
{
    flush_device(true);

    const void* data = m_readback_buffer->map();
    bool had_overflow = false;
    auto streams = decode_all_streams(data, m_readback_buffer->size(), had_overflow);
    m_readback_buffer->unmap();

    // Decode all streams and concatenate messages
    std::string result;
    if (had_overflow) {
        result = "[WARNING: Print buffer overflow, some messages lost]\n";
    }
    for (const auto& stream : streams) {
        auto messages = decode_stream_records(stream, m_hashed_strings);
        for (const auto& msg : messages) {
            if (!result.empty() && result.back() != '\n')
                result += "\n";
            result += msg;
        }
    }
    return result;
}

void DebugPrinter::bind(ShaderCursor cursor)
{
    if (cursor.is_valid())
        cursor = cursor.find_field("g_debug_printer");
    if (cursor.is_valid())
        cursor["buffer"] = m_buffer;
}

void DebugPrinter::flush_device(bool wait)
{
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    command_encoder->copy_buffer(m_readback_buffer, 0, m_buffer, 0, m_buffer->size());
    // Reset write position to 264 (after header + overflow chunk)
    uint32_t reset_write_pos = FIRST_DATA_CHUNK;
    command_encoder->upload_buffer_data(m_buffer, 4, sizeof(reset_write_pos), &reset_write_pos);
    m_device->submit_command_buffer(command_encoder->finish());
    if (wait)
        m_device->wait_for_idle();
}

} // namespace sgl
