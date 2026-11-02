// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"

#include <optional>
#include <string>
#include <vector>

namespace sgl {

// Internal inputs used to construct the Slang target descriptor.
struct ResolvedCompilerTarget {
    std::optional<std::string> profile;
    std::vector<std::string> capabilities;
    std::vector<std::string> downstream_args;
};

/// Resolve inputs without modifying the requested options or device reports.
ResolvedCompilerTarget resolve_compiler_target(const Device& device, const SlangCompilerOptions& options);

// Query supported architectures using NVRTC's numeric encoding (e.g. 86).
std::vector<int> query_nvrtc_architectures(const Device& device);

} // namespace sgl
