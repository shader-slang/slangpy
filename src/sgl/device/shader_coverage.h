// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/device_child.h"
#include <slang-rhi.h>
#include <mutex>
#include <string>
#include <vector>

namespace sgl {

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
    std::vector<uint32_t> counter_widths;
    std::string reason;
};

/// Internal registry: retain RHI buffers, never SGL DeviceChild objects that retain the Device.
struct ShaderCoverageState {
    struct Program {
        uint64_t generation_id;
        Slang::ComPtr<rhi::IBuffer> buffer;
        std::string label;
        std::string manifest;
        uint32_t counter_width;
    };
    std::string collection_id;
    uint64_t next_generation_id{1};
    uint64_t next_capture_id{1};
    uint64_t interval_id{0};
    size_t retained_bytes{0};
    std::vector<Program> programs;
};

/// Device-wide collection. Reset affects all registered programs, irrespective of report filters.
class SGL_API ShaderCoverageCollector : public DeviceChild {
    SGL_OBJECT(ShaderCoverageCollector)
public:
    explicit ShaderCoverageCollector(ref<Device> device);
    void _release_rhi_resources() override { }
    ShaderCoverageCapabilities capabilities() const;
    ShaderCoverageSnapshot snapshot(bool reset = false);
    void reset();
};

} // namespace sgl
