// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"

#include <cstdint>
#include <compare>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sgl {

enum class DeviceType : uint32_t;

enum class TargetProfileFamily {
    shader_model,
    spirv,
    metallib,
};

struct TargetProfile {
    TargetProfileFamily family;
    uint32_t major;
    uint32_t minor;

    bool operator==(const TargetProfile&) const = default;
    auto operator<=>(const TargetProfile&) const = default;

    std::string to_string() const;
};

SGL_API std::optional<TargetProfile> parse_target_profile(std::string_view profile);
SGL_API std::optional<TargetProfile> target_profile_from_capability(std::string_view capability);
SGL_API std::vector<std::string> build_target_profile_inventory(
    std::span<const std::string> candidates,
    DeviceType device_type,
    const std::function<bool(std::string_view)>& is_known_profile
);
SGL_API std::string highest_target_profile(std::span<const std::string> profiles, TargetProfileFamily family);

} // namespace sgl
