// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/shader_coverage.h"
#include "sgl/device/resource.h"
#include <slang-rhi.h>
#include <mutex>

namespace sgl {

struct ShaderProgramData;
struct ShaderCoverageOptions;

/// Allocated only for instrumented programs. Ordinary program data holds a null pointer.
struct ShaderCoverageProgramData {
    ref<Buffer> buffer;
    std::string manifest;
    uint32_t resource_id{0};
    uint32_t counter_width{0};
    uint64_t generation_id{0};

    /// Creates the RHI program with the compiler's hidden coverage resource attached.
    static SlangResult create_program(
        Device* device,
        const ShaderCoverageOptions& options,
        const rhi::ShaderProgramDesc& desc,
        ShaderProgramData& data,
        ISlangBlob** diagnostics
    );
    void bind(rhi::IShaderProgram* program, rhi::IShaderObject* root_object) const;
};

/// Internal registry: retain RHI buffers, never SGL DeviceChild objects that retain the Device.
struct ShaderCoverageState {
    // Recursive for submission callbacks and hot reload that can register programs.
    std::recursive_mutex mutex;
    std::recursive_mutex capture_mutex;
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

} // namespace sgl
