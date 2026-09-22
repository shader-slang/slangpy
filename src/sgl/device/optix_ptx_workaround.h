// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "cuda_architecture.h"
#include "sgl/core/platform.h"

#include <charconv>
#include <memory>
#include <slang.h>

namespace sgl::detail {

// CUDA's cuDriverGetVersion reports CUDA compatibility, not the display driver
// release needed for this workaround. NVML is loaded optionally, without an SDK dependency.
inline uint32_t optix_driver_branch()
{
#if SGL_WINDOWS
    const char* library_name = "nvml.dll";
#elif SGL_LINUX
    const char* library_name = "libnvidia-ml.so.1";
#else
    return 0;
#endif
#if SGL_WINDOWS || SGL_LINUX
    std::unique_ptr<void, decltype(&platform::release_shared_library)> library(
        platform::load_shared_library(library_name),
        platform::release_shared_library
    );
    if (!library)
        return 0;
    auto init = reinterpret_cast<int (*)()>(platform::get_proc_address(library.get(), "nvmlInit_v2"));
    auto shutdown = reinterpret_cast<int (*)()>(platform::get_proc_address(library.get(), "nvmlShutdown"));
    auto version = reinterpret_cast<int (*)(char*, unsigned int)>(
        platform::get_proc_address(library.get(), "nvmlSystemGetDriverVersion")
    );
    if (!init || !shutdown || !version || init() != 0)
        return 0;
    char text[80] = {};
    int result = version(text, sizeof(text));
    shutdown();
    uint32_t branch = 0;
    if (result != 0 || std::from_chars(text, text + sizeof(text), branch).ec != std::errc{})
        return 0;
    return branch;
#endif
}

// Remove when the minimum supported OptiX driver includes the R595 PTX fix.
// Older drivers reject NVRTC 13's .b32 float arguments for sm_100+ trace intrinsics.
// OptiX recommends compute_75 input and retargets it for the physical GPU.
// Reproduction: https://github.com/shader-slang/slangpy/pull/1180
// Delete this header and its call in ShaderProgram::link when no longer needed.
inline std::optional<uint32_t>
optix_ptx_target_workaround(const Device* device, const SlangCompilerOptions& options, slang::IComponentType* program)
{
    if (device->type() != DeviceType::cuda || !device->info().optix_version || options.cuda_architecture
        || device->cuda_compiler_info()->version_major < 13 || *resolve_cuda_architecture(device, options) < 100)
        return std::nullopt;
    auto layout = program->getLayout();
    if (!layout)
        return std::nullopt;
    bool ray_program = false;
    for (SlangUInt i = 0; i < layout->getEntryPointCount(); ++i) {
        auto stage = layout->getEntryPointByIndex(i)->getStage();
        ray_program |= stage >= SLANG_STAGE_RAY_GENERATION && stage <= SLANG_STAGE_CALLABLE;
    }
    if (!ray_program)
        return std::nullopt;
    static const uint32_t branch = []
    {
        uint32_t result = optix_driver_branch();
        if (result > 0 && result < 595)
            log_warn(
                "Using compute_75 for automatic OptiX compilation on pre-R595 drivers. "
                "An explicit cuda_architecture overrides this compatibility workaround."
            );
        return result;
    }();
    return branch > 0 && branch < 595 ? std::optional<uint32_t>(75) : std::nullopt;
}

} // namespace sgl::detail
