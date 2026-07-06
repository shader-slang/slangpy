// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "detail/profile.h"

#include "sgl/core/format.h"
#include "sgl/core/error.h"
#include "sgl/device/device.h"

#include <algorithm>
#include <charconv>
#include <set>

namespace sgl {

namespace {

    std::optional<uint32_t> parse_version_part(std::string_view value)
    {
        uint32_t result{0};
        const char* first = value.data();
        const char* last = first + value.size();
        auto [end, error] = std::from_chars(first, last, result);
        if (error != std::errc{} || end != last)
            return {};
        return result;
    }

    std::optional<TargetProfile>
    parse_target_profile(std::string_view profile, std::string_view prefix, TargetProfileFamily family)
    {
        if (!profile.starts_with(prefix))
            return {};

        std::string_view version = profile.substr(prefix.size());
        size_t separator = version.find('_');
        if (separator == std::string_view::npos || version.find('_', separator + 1) != std::string_view::npos)
            return {};

        auto major = parse_version_part(version.substr(0, separator));
        auto minor = parse_version_part(version.substr(separator + 1));
        if (!major || !minor)
            return {};

        return TargetProfile{family, *major, *minor};
    }

} // namespace

std::string TargetProfile::to_string() const
{
    const char* prefix{nullptr};
    switch (family) {
    case TargetProfileFamily::shader_model:
        prefix = "sm";
        break;
    case TargetProfileFamily::spirv:
        prefix = "spirv";
        break;
    case TargetProfileFamily::metallib:
        prefix = "metallib";
        break;
    }
    SGL_ASSERT(prefix);
    return fmt::format("{}_{}_{}", prefix, major, minor);
}

std::optional<TargetProfile> parse_target_profile(std::string_view profile)
{
    if (auto result = parse_target_profile(profile, "sm_", TargetProfileFamily::shader_model))
        return result;
    if (auto result = parse_target_profile(profile, "spirv_", TargetProfileFamily::spirv))
        return result;
    return parse_target_profile(profile, "metallib_", TargetProfileFamily::metallib);
}

std::optional<TargetProfile> target_profile_from_capability(std::string_view capability)
{
    if (capability.starts_with("_sm_") || capability.starts_with("_spirv_"))
        capability.remove_prefix(1);
    return parse_target_profile(capability);
}

std::vector<std::string> build_target_profile_inventory(
    std::span<const std::string> candidates,
    DeviceType device_type,
    const std::function<bool(std::string_view)>& is_known_profile
)
{
    std::set<TargetProfile> canonical_profiles;
    for (const std::string& candidate : candidates) {
        auto profile = target_profile_from_capability(candidate);
        if (!profile)
            continue;

        bool valid_for_backend{false};
        switch (device_type) {
        case DeviceType::d3d12:
            valid_for_backend = profile->family == TargetProfileFamily::shader_model && profile->major == 6;
            break;
        case DeviceType::vulkan:
            valid_for_backend = profile->family == TargetProfileFamily::spirv
                || (profile->family == TargetProfileFamily::shader_model && profile->major == 6);
            break;
        case DeviceType::metal:
            valid_for_backend = profile->family == TargetProfileFamily::metallib;
            break;
        default:
            break;
        }

        if (valid_for_backend && is_known_profile(profile->to_string()))
            canonical_profiles.insert(*profile);
    }

    std::vector<std::string> result;
    result.reserve(canonical_profiles.size());
    for (const TargetProfile& profile : canonical_profiles)
        result.push_back(profile.to_string());
    return result;
}

std::string highest_target_profile(std::span<const std::string> profiles, TargetProfileFamily family)
{
    std::optional<TargetProfile> highest;
    for (const std::string& profile : profiles) {
        auto parsed = parse_target_profile(profile);
        if (parsed && parsed->family == family && (!highest || *parsed > *highest))
            highest = *parsed;
    }
    return highest ? highest->to_string() : std::string{};
}

} // namespace sgl
