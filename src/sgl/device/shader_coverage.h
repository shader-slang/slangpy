// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/device_child.h"
#include <string>
#include <vector>

namespace sgl {

struct ShaderCoverageCapture;

/// Host-owned data for one compiled program generation. Branch IDs are local to this record.
struct ShaderCoverageProgramSnapshot {
    uint64_t generation_id{0};
    std::string label;
    std::string manifest;
    uint32_t counter_width{0};
    std::vector<uint64_t> counters;
};

/// A blocking capture of one device's registered coverage buffers, independent of device lifetime.
struct ShaderCoverageSnapshot {
    std::string collection_id;
    uint64_t capture_id{0};
    uint64_t interval_id{0};
    bool reset_after{false};
    std::vector<ShaderCoverageProgramSnapshot> programs;
};

struct ShaderCoverageCapabilities {
    bool supported{false};
    /// Widths supported for exact execution counts.
    std::vector<uint32_t> counter_widths;
    /// Widths supported for boolean hit/miss recording.
    std::vector<uint32_t> boolean_counter_widths;
    std::string reason;
};

/// Device-wide collection. Reset affects all registered programs, irrespective of report filters.
/// Native callers must serialize capture submission with other device mutation, including close.
/// Python bindings serialize submission with the GIL and release it for completion.
class SGL_API ShaderCoverageCollector : public DeviceChild {
    SGL_OBJECT(ShaderCoverageCollector)
public:
    explicit ShaderCoverageCollector(ref<Device> device);
    void _release_rhi_resources() override { }
    ShaderCoverageCapabilities capabilities() const;
    ShaderCoverageSnapshot snapshot(bool reset = false);
    void reset();

    /// Internal split used by bindings to release the GIL only after submission.
    ShaderCoverageCapture _begin_capture(bool read, bool reset);
};

} // namespace sgl
