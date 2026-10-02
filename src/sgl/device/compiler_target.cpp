// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler_target.h"
#include "sgl/core/error.h"
#include "sgl/core/logger.h"
#include "sgl/core/platform.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"

#include <algorithm>
#include <charconv>
#include <memory>
#include <regex>
#include <span>
#include <string_view>

namespace sgl {
namespace {

    struct Version {
        int major{0};
        int minor{0};
        auto operator<=>(const Version&) const = default;
    };

    enum class Family { unknown, dx, spirv, metal, cuda, glsl };

    struct VersionName {
        Family family{Family::unknown};
        Version version;
    };

    VersionName version_name(const std::string& name)
    {
        // Only recognize numeric versions, never guess CUDA architecture suffixes or '*_latest'.
        static const std::regex pattern(
            R"(^(_?sm|cs|ps|vs|gs|hs|ds|lib|ms|as|_?spirv|metallib|_?cuda_sm)_(\d+)_(\d+)$)"
        );
        std::smatch match;
        if (std::regex_match(name, match, pattern)) {
            auto prefix = match[1].str();
            Family family = Family::dx;
            if (prefix == "spirv" || prefix == "_spirv")
                family = Family::spirv;
            else if (prefix == "metallib")
                family = Family::metal;
            else if (prefix == "cuda_sm" || prefix == "_cuda_sm")
                family = Family::cuda;
            auto number = [](const std::string& text)
            {
                int value = 0;
                auto result = std::from_chars(text.data(), text.data() + text.size(), value);
                SGL_CHECK(result.ec == std::errc{}, "Invalid target version: {}", text);
                return value;
            };
            return {family, {number(match[2].str()), number(match[3].str())}};
        }
        if (name.starts_with("glsl_"))
            return {Family::glsl, {}};
        return {};
    }

    Family native_family(DeviceType type)
    {
        switch (type) {
        case DeviceType::d3d12:
            return Family::dx;
        case DeviceType::vulkan:
            return Family::spirv;
        case DeviceType::metal:
            return Family::metal;
        case DeviceType::cuda:
            return Family::cuda;
        default:
            return Family::unknown;
        }
    }

    std::string baseline(DeviceType type)
    {
        switch (type) {
        case DeviceType::d3d12:
            return "hlsl";
        case DeviceType::vulkan:
            return "spirv";
        case DeviceType::metal:
            return "metal";
        case DeviceType::cuda:
            return "cuda";
        case DeviceType::cpu:
            return "cpp";
        case DeviceType::wgpu:
            return "wgsl";
        default:
            SGL_UNREACHABLE();
        }
    }

    bool has_dxc_option(std::span<const std::string> args, std::string_view name)
    {
        return std::ranges::any_of(
            args,
            [name](const std::string& arg)
            {
                return arg.size() > 1 && (arg.front() == '-' || arg.front() == '/')
                    && std::string_view(arg).substr(1) == name;
            }
        );
    }

    // Architectures use NVRTC's major * 10 + minor encoding (e.g. 120 for compute capability 12.0).
    // Automatic selection takes the highest entry in NVRTC's supported list that does not exceed
    // the device's compute capability or the environment ceiling. Search the actual list: toolkit
    // support can have gaps or drop older architectures, so clamping to its maximum is insufficient. The list
    // need not be sorted. Explicit requests must be supported by both NVRTC and the device;
    // never round or clamp them. Fail if the device reports no capability or no target is usable.
    int select_cuda_architecture(int device_arch, std::span<const int> supported, std::optional<int> requested)
    {
        SGL_CHECK(device_arch > 0, "CUDA device did not report a compute capability");
        constexpr const char* variable = "SGL_MAX_CUDA_COMPUTE_CAPABILITY";
        std::optional<int> ceiling;
        if (auto value = platform::get_environment_variable(variable); value && !value->empty()) {
            int arch = 0;
            const auto parsed = std::from_chars(value->data(), value->data() + value->size(), arch);
            SGL_CHECK(
                parsed.ec == std::errc{} && parsed.ptr == value->data() + value->size() && arch >= 10
                    && value->front() != '0',
                "Invalid {}='{}' (expected major * 10 + minor, e.g. 90)",
                variable,
                *value
            );
            ceiling = arch;
        }
        if (requested) {
            if (ceiling)
                SGL_CHECK(
                    *requested <= *ceiling,
                    "CUDA profile compute_{} exceeds {}={}",
                    *requested,
                    variable,
                    *ceiling
                );
            SGL_CHECK(
                *requested <= device_arch,
                "CUDA profile compute_{} exceeds detected device target compute_{}",
                *requested,
                device_arch
            );
            SGL_CHECK(
                std::ranges::find(supported, *requested) != supported.end(),
                "CUDA profile compute_{} is not supported by Slang's NVRTC library",
                *requested
            );
            return *requested;
        }
        int selected = 0;
        for (int arch : supported)
            if (arch <= device_arch)
                selected = std::max(selected, arch);
        SGL_CHECK(
            selected > 0,
            "NVRTC supports no CUDA architecture compatible with device target compute_{}",
            device_arch
        );
        if (ceiling) {
            int capped = 0;
            for (int arch : supported)
                if (arch <= device_arch && arch <= *ceiling)
                    capped = std::max(capped, arch);
            SGL_CHECK(
                capped > 0,
                "NVRTC supports no CUDA architecture compatible with device target compute_{} and {}={}",
                device_arch,
                variable,
                *ceiling
            );
            if (capped < selected)
                log_info(
                    "{}={} lowers automatic CUDA target from compute_{} to compute_{}",
                    variable,
                    *ceiling,
                    selected,
                    capped
                );
            selected = capped;
        }
        return selected;
    }

} // namespace

ResolvedCompilerTarget resolve_compiler_target(const Device& device, const SlangCompilerOptions& options)
{
    const DeviceType type = device.type();
    auto compiler = device.global_session();
    const auto& detected = device.capabilities();

    ResolvedCompilerTarget result;
    result.profile = options.profile;
    const auto family = native_family(type);
    Version hardware_maximum;
    for (const auto& name : detected) {
        auto version = version_name(name);
        if (version.family == family)
            hardware_maximum = std::max(hardware_maximum, version.version);
    }

    VersionName selected;
    if (type == DeviceType::cuda && (!options.profile || options.profile->starts_with("compute_"))) {
        std::optional<int> requested;
        if (options.profile) {
            const auto text = std::string_view(*options.profile).substr(8);
            int arch = 0;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), arch);
            SGL_CHECK(
                parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && arch >= 10
                    && text.front() != '0',
                "Invalid CUDA profile '{}' (expected compute_<major * 10 + minor>, e.g. compute_120)",
                *options.profile
            );
            requested = arch;
        }
        const int arch = select_cuda_architecture(
            hardware_maximum.major * 10 + hardware_maximum.minor,
            device._nvrtc_supported_architectures(),
            requested
        );
        const Version version{arch / 10, arch % 10};
        result.profile.reset();
        // Slang's CUDA version vocabulary can lag behind NVRTC. Supply only recognized
        // version assumptions up to the selected architecture; NVRTC receives the exact target.
        for (const auto& name : detected) {
            const auto candidate = version_name(name);
            if (candidate.family == Family::cuda && candidate.version <= version
                && compiler->findCapability(name.c_str()) != SLANG_CAPABILITY_UNKNOWN)
                result.capabilities.push_back(name);
        }
        if (version >= Version{9, 0} && device.has_capability("optix_coopvec")
            && compiler->findCapability("optix_coopvec") != SLANG_CAPABILITY_UNKNOWN)
            result.capabilities.push_back("optix_coopvec");
        result.downstream_args.push_back(fmt::format("--gpu-architecture=compute_{}", arch));
    } else if (options.profile) {
        selected = version_name(*options.profile);
        SGL_CHECK(
            !options.profile->starts_with("compute_"),
            "Profile '{}' is not supported for the {} backend",
            *options.profile,
            type
        );
        SGL_CHECK(
            compiler->findProfile(options.profile->c_str()) != SLANG_PROFILE_UNKNOWN,
            "Unknown Slang profile '{}'",
            *options.profile
        );
        const bool native = family != Family::unknown && selected.family == family;
        const bool cross
            = type == DeviceType::vulkan && (selected.family == Family::dx || selected.family == Family::glsl);
        SGL_CHECK(native || cross, "Profile '{}' is not supported for the {} backend", *options.profile, type);
        if (selected.family == family && hardware_maximum != Version{})
            SGL_CHECK(
                selected.version <= hardware_maximum,
                "Profile '{}' exceeds detected {} device version {}.{}",
                *options.profile,
                type,
                hardware_maximum.major,
                hardware_maximum.minor
            );
    } else {
        // Automatic sessions use recognized device inputs. Explicit profiles start with
        // only their own requirements, rather than a device list whose optional features
        // can silently raise the selected version.
        for (const auto& name : detected) {
            if (compiler->findCapability(name.c_str()) == SLANG_CAPABILITY_UNKNOWN)
                continue;
            // Handle D3D NVAPI/SER capabilities below.
            if (type == DeviceType::d3d12 && (name == "hlsl_nvapi" || name.starts_with("ser_")))
                continue;
            result.capabilities.push_back(name);
            auto version = version_name(name);
            if (version.family == family && version.version > selected.version)
                selected = version;
        }
        if (type == DeviceType::d3d12) {
            // Slang capabilities alone do not select DXC's shader model.
            selected = {family, std::max(selected.version, Version{6, 0})};
            result.profile = fmt::format("sm_{}_{}", selected.version.major, selected.version.minor);
        } else if (type == DeviceType::vulkan) {
            // Avoid the implicit SPIR-V 1.5 feature bundle; device inputs supply the
            // available version and extensions for automatic sessions.
            result.profile = "spirv_1_0";
        }
        // Metal versions are supplied by the capabilities above. Do not synthesize a
        // profile: Slang recognizes capabilities such as metallib_3_2 without a matching profile.
    }

    // Slang can omit [raypayload] annotations on separately compiled miss/hit shaders.
    // Disable DXC's payload-qualifier checks while preserving the selected shader model.
    // An explicit session-level option takes precedence.
    if (type == DeviceType::d3d12 && selected.version >= Version{6, 7}
        && !has_dxc_option(options.downstream_args, "enable-payload-qualifiers")
        && !has_dxc_option(options.downstream_args, "disable-payload-qualifiers"))
        result.downstream_args.emplace_back("-disable-payload-qualifiers");

    // NVAPI and native HLSL SER expose different Slang APIs. Preserve the NVAPI path
    // across profiles until callers can explicitly control capability selection.
    if (type == DeviceType::d3d12 && device.has_capability("hlsl_nvapi")
        && compiler->findCapability("hlsl_nvapi") != SLANG_CAPABILITY_UNKNOWN)
        result.capabilities.emplace_back("hlsl_nvapi");
    if (result.profile)
        SGL_CHECK(
            compiler->findProfile(result.profile->c_str()) != SLANG_PROFILE_UNKNOWN,
            "Unknown resolved Slang profile '{}'",
            *result.profile
        );

    // An explicit baseline also keeps Slang capability checking active with minimal inputs.
    result.capabilities.push_back(baseline(type));
    std::ranges::sort(result.capabilities);
    auto duplicates = std::ranges::unique(result.capabilities);
    result.capabilities.erase(duplicates.begin(), duplicates.end());
    return result;
}

std::vector<int> query_nvrtc_architectures(const Device& device)
{
    Slang::ComPtr<ISlangBlob> path_blob;
    SGL_CHECK(
        SLANG_SUCCEEDED(
            device.global_session()->getDownstreamCompilerPath(SLANG_PASS_THROUGH_NVRTC, path_blob.writeRef())
        ),
        "Slang could not resolve the NVRTC library path required for CUDA target selection"
    );
    const std::string_view path(static_cast<const char*>(path_blob->getBufferPointer()), path_blob->getBufferSize());
    std::unique_ptr<void, decltype(&platform::release_shared_library)> library(
        platform::load_shared_library(path),
        &platform::release_shared_library
    );
    SGL_CHECK(library, "Failed to load Slang's NVRTC library '{}'", path);
    using NvrtcGetNumSupportedArchsFn = int (*)(int*);
    using NvrtcGetSupportedArchsFn = int (*)(int*);
    const auto get_count = reinterpret_cast<NvrtcGetNumSupportedArchsFn>(
        platform::get_proc_address(library.get(), "nvrtcGetNumSupportedArchs")
    );
    const auto get_archs = reinterpret_cast<NvrtcGetSupportedArchsFn>(
        platform::get_proc_address(library.get(), "nvrtcGetSupportedArchs")
    );
    SGL_CHECK(get_count && get_archs, "NVRTC library '{}' does not support architecture queries", path);
    int count = 0;
    SGL_CHECK(
        get_count(&count) == 0 && count > 0,
        "Failed to query supported architecture count from NVRTC '{}'",
        path
    );
    std::vector<int> architectures(count);
    SGL_CHECK(get_archs(architectures.data()) == 0, "Failed to query supported architectures from NVRTC '{}'", path);
    return architectures;
}

} // namespace sgl
