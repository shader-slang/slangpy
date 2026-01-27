// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "print2.h"

#include "sgl/core/format.h"
#include "sgl/core/logger.h"

#include "sgl/device/device.h"
#include "sgl/device/command.h"

#include "sgl/math/vector.h"
#include "sgl/math/matrix.h"

#include <vector>

namespace sgl {

namespace {

    /// Constants for the stream buffer format.
    constexpr uint32_t BUFFER_HEADER_SIZE = 8;        // capacity + write_pos
    constexpr uint32_t STREAM_SENTINEL = 0xFFFFFFFF;  // Sentinel value marking stream start
    constexpr uint32_t ENTRY_SIZE = 8;                // Each entry is 8 bytes (link + payload)
    constexpr uint32_t STREAM_HEADER_SIZE = 16;       // Sentinel pair + dummy entry

    /// Decode a single stream starting at the given offset.
    /// Returns the collected payload values.
    std::vector<uint32_t> decode_stream(const uint32_t* data, uint32_t dummy_entry_offset, uint32_t buffer_size)
    {
        std::vector<uint32_t> payload;

        // Read the link from the dummy entry's header
        uint32_t link = data[dummy_entry_offset / 4];

        // Follow the chain
        while (link != 0) {
            // Bounds check
            if (link + ENTRY_SIZE > buffer_size) {
                log_warn("Stream link out of bounds: {} > {}", link, buffer_size);
                break;
            }

            // Read entry: [link:32][payload:32]
            uint32_t next_link = data[link / 4];
            uint32_t value = data[link / 4 + 1];

            payload.push_back(value);
            link = next_link;
        }

        return payload;
    }

    /// Decode all streams from the buffer.
    /// Returns a vector of streams, where each stream is a vector of uint32_t payloads.
    std::vector<std::vector<uint32_t>> decode_all_streams(const void* buffer_data, size_t buffer_size)
    {
        std::vector<std::vector<uint32_t>> streams;

        const uint32_t* data = reinterpret_cast<const uint32_t*>(buffer_data);

        // Read header
        uint32_t capacity = data[0];
        uint32_t write_pos = data[1];

        // Clamp write_pos to buffer size (in case of overflow)
        write_pos = std::min(write_pos, static_cast<uint32_t>(buffer_size));

        if (write_pos > capacity) {
            log_warn("Print buffer overflow detected: write_pos={} > capacity={}", write_pos, capacity);
        }

        // Scan for sentinel pairs
        // We scan in 4-byte increments since entries are 8-byte aligned but we need to find the sentinel pair
        for (uint32_t offset = BUFFER_HEADER_SIZE; offset + STREAM_HEADER_SIZE <= write_pos; offset += ENTRY_SIZE) {
            uint32_t word0 = data[offset / 4];
            uint32_t word1 = data[offset / 4 + 1];

            // Check for sentinel pair
            if (word0 == STREAM_SENTINEL && word1 == STREAM_SENTINEL) {
                // Found a stream start
                // The dummy entry is at offset + 8
                uint32_t dummy_entry_offset = offset + 8;

                auto stream = decode_stream(data, dummy_entry_offset, write_pos);
                if (!stream.empty()) {
                    streams.push_back(std::move(stream));
                }

                // Skip the rest of the stream header (we've consumed 16 bytes total)
                offset += ENTRY_SIZE; // Will be incremented by another ENTRY_SIZE in the loop
            }
        }

        return streams;
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
    // Buffer layout: [capacity:4][write_pos:4][entries...]
    // write_pos starts at 8 (right after the header)
    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();
    uint32_t header[2] = {uint32_t(buffer_size), 8};
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
    auto streams = decode_all_streams(data, m_readback_buffer->size());
    m_readback_buffer->unmap();

    // For now, just log the raw stream data
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto& stream = streams[i];
        std::string hex;
        for (uint32_t value : stream) {
            if (!hex.empty())
                hex += " ";
            hex += fmt::format("{:08x}", value);
        }
        log_info("Stream {}: [{}]", i, hex);
    }
}

std::string DebugPrinter::flush_to_string()
{
    flush_device(true);

    const void* data = m_readback_buffer->map();
    auto streams = decode_all_streams(data, m_readback_buffer->size());
    m_readback_buffer->unmap();

    // For now, just format the raw stream data
    std::string result;
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto& stream = streams[i];
        result += fmt::format("Stream {}: [", i);
        for (size_t j = 0; j < stream.size(); ++j) {
            if (j > 0)
                result += " ";
            result += fmt::format("{:08x}", stream[j]);
        }
        result += "]\n";
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
    // Reset write position to 8 (right after header)
    uint32_t reset_write_pos = 8;
    command_encoder->upload_buffer_data(m_buffer, 4, sizeof(reset_write_pos), &reset_write_pos);
    m_device->submit_command_buffer(command_encoder->finish());
    if (wait)
        m_device->wait_for_idle();
}

} // namespace sgl
