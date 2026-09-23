// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/shader.h"
#include "sgl/device/device.h"

namespace sgl {

/// Resolve inputs without modifying the requested options or device reports.
SlangTargetInfo resolve_compiler_target(
    DeviceType device_type,
    std::span<const std::string> detected,
    const SlangCompilerOptions& options,
    slang::IGlobalSession* compiler
);

/// Reject downstream flags that would independently override the resolved target.
void validate_compiler_target_args(DeviceType device_type, std::span<const std::string> args);

/// Validate explicit CUDA architecture requests against generated entry-point PTX before RHI compilation.
void validate_cuda_program(slang::IComponentType* program, const SlangTargetInfo& target_info, PersistentCache* cache);

} // namespace sgl
