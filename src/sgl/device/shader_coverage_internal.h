// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/shader_coverage.h"
#include "sgl/device/resource.h"
#include <slang-rhi.h>

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

/// Owns one submitted capture independently of the device's mutable resource handles and registry.
/// Submission is serialized with device mutation; completion only accesses these resources.
struct SGL_API ShaderCoverageCapture {
    ShaderCoverageCapture() = default;
    ShaderCoverageCapture(const ShaderCoverageCapture&) = delete;
    ShaderCoverageCapture& operator=(const ShaderCoverageCapture&) = delete;
    ShaderCoverageCapture(ShaderCoverageCapture&&) = default;
    ShaderCoverageCapture& operator=(ShaderCoverageCapture&&) = delete;

    /// Waits for this capture and returns its host data. Call once per capture.
    ShaderCoverageSnapshot finish();

    // The RHI device borrows its diagnostic callback from the SGL device. Keep
    // that owner alive too, and destroy the RHI references before releasing it.
    ref<Device> owner;
    Slang::ComPtr<rhi::IDevice> device;
    Slang::ComPtr<rhi::ICommandBuffer> command_buffer;
    Slang::ComPtr<rhi::IFence> fence;
    std::vector<Slang::ComPtr<rhi::IBuffer>> staging;
    ShaderCoverageSnapshot result;
};

} // namespace sgl
