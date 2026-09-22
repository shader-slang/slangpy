// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/device.h"
#include "sgl/core/error.h"

#include <algorithm>
#include <span>

namespace sgl::detail {

// Internal selection helper, shared with CPU-only tests for sparse compiler/GPU lists.
inline uint32_t select_cuda_architecture(
    const CUDACompilerInfo& compiler,
    std::span<const std::string> gpu_capabilities,
    std::optional<uint32_t> requested
)
{
    auto gpu_supports = [&](uint32_t architecture)
    {
        auto capability = fmt::format("_cuda_sm_{}_{}", architecture / 10, architecture % 10);
        return std::find(gpu_capabilities.begin(), gpu_capabilities.end(), capability) != gpu_capabilities.end();
    };
    if (requested) {
        SGL_CHECK(*requested > 0, "Explicit CUDA architecture must be greater than zero.");
        SGL_CHECK(
            std::find(compiler.supported_architectures.begin(), compiler.supported_architectures.end(), *requested)
                != compiler.supported_architectures.end(),
            "NVRTC {}.{} does not support compute_{}.",
            compiler.version_major,
            compiler.version_minor,
            *requested
        );
        return *requested;
    }
    uint32_t selected = 0;
    for (uint32_t architecture : compiler.supported_architectures)
        if (gpu_supports(architecture))
            selected = std::max(selected, architecture);
    SGL_CHECK(
        selected,
        "NVRTC {}.{} has no target supported by the reported GPU capabilities.",
        compiler.version_major,
        compiler.version_minor
    );
    return selected;
}

// An empty result denotes a non-CUDA backend.
std::optional<uint32_t> resolve_cuda_architecture(const Device* device, const SlangCompilerOptions& options);

} // namespace sgl::detail
