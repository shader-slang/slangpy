// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/shader_cursor.h"

#include <map>
#include <string>

namespace sgl {

/**
 * \brief Debug printer.
 *
 * This class implements host-side support for shader debug printing.
 */
class DebugPrinter {
public:
    DebugPrinter(Device* device, size_t buffer_size = 4 * 1024 * 1024);

    /// Add a map of hashed strings to the printer.
    /// This needs to be called for any shader that uses debug printing.
    void add_hashed_strings(const std::map<uint32_t, std::string>& hashed_strings);

    /// Flush the print buffer and output any messages to stdout.
    void flush();

    /// Flush the print buffer and output any messages as a string.
    std::string flush_to_string();

    void bind(ShaderCursor cursor);

private:
    void flush_device(bool wait);

    Device* m_device;

    ref<Buffer> m_buffer;
    ref<Buffer> m_readback_buffer;

    std::map<uint32_t, std::string> m_hashed_strings;
};

} // namespace sgl
