// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler_target.h"
#include "sgl/core/error.h"
#include "sgl/core/logger.h"
#include "sgl/device/slang_utils.h"
#include "sgl/device/persistent_cache.h"

#include <algorithm>
#include <charconv>
#include <regex>

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

    std::string canonical_name(DeviceType type, const std::string& name)
    {
        // SLANG-W005: these aliases are equivalent only on their native backend. SPIR-V public aliases
        // contain feature bundles and must never be normalized to raw version atoms.
        auto version = version_name(name);
        if ((type == DeviceType::d3d12 && name.starts_with("sm_"))
            || (type == DeviceType::cuda && name.starts_with("cuda_sm_"))) {
            if (version.family == native_family(type))
                return "_" + name;
        }
        return name;
    }

    VersionName required_version(const std::string& name)
    {
        auto direct = version_name(name);
        if (direct.family != Family::unknown)
            return direct;
        // SLANG-W005: bounded facts from slang-capabilities.capdef, not a capability graph.
        // See plan/slang-compiler-workarounds.md for upstream queries and removal criteria.
        static const std::map<std::string, VersionName> requirements{
            {"ser_hlsl_native", {Family::dx, {6, 9}}},
            {"ser_dxr", {Family::dx, {6, 9}}},
            {"ser_dxr_raygen", {Family::dx, {6, 9}}},
            {"ser_dxr_raygen_closesthit_miss", {Family::dx, {6, 9}}},
            {"SPV_EXT_physical_storage_buffer", {Family::spirv, {1, 3}}},
            {"SPV_KHR_ray_tracing", {Family::spirv, {1, 4}}},
            {"spvRayTracingKHR", {Family::spirv, {1, 4}}},
            {"SPV_KHR_cooperative_matrix", {Family::spirv, {1, 6}}},
            {"spvCooperativeMatrixKHR", {Family::spirv, {1, 6}}},
            {"SPV_NV_cooperative_matrix2", {Family::spirv, {1, 6}}},
            {"spvCooperativeMatrix2NV", {Family::spirv, {1, 6}}},
            {"SPV_NV_cooperative_vector", {Family::spirv, {1, 6}}},
            {"spvCooperativeVectorNV", {Family::spirv, {1, 6}}},
            {"optix_coopvec", {Family::cuda, {9, 0}}},
        };
        auto it = requirements.find(name);
        return it != requirements.end() ? it->second : VersionName{};
    }

} // namespace

bool uses_compiler_target_options(const SlangCompilerOptions& options)
{
    return options.profile.has_value() || options.capabilities.has_value() || !options.capability_overrides.empty();
}

void validate_compiler_target_args(DeviceType type, std::span<const std::string> args)
{
    for (const auto& arg : args) {
        // Also reject identical flags: NVRTC rejects duplicate architecture options. See the
        // rejected CUDA bridge in the workaround ledger. Slang owns both downstream target flags.
        bool conflict = type == DeviceType::cuda
            ? (arg.starts_with("--gpu-architecture") || arg.starts_with("-arch"))
            : type == DeviceType::d3d12 && (arg.starts_with("-T") || arg.starts_with("/T"));
        SGL_CHECK(
            !conflict,
            "Downstream argument '{}' conflicts with compiler target selection; use profile/capabilities instead",
            arg
        );
    }
}

SlangTargetInfo resolve_compiler_target(
    DeviceType type,
    std::span<const std::string> detected,
    const SlangCompilerOptions& options,
    slang::IGlobalSession* compiler
)
{
    SGL_CHECK(
        options.shader_model == ShaderModel::unknown,
        "shader_model cannot be combined with profile, capabilities, or capability_overrides"
    );
    validate_compiler_target_args(type, options.downstream_args);
    SlangTargetInfo result;
    result.legacy = false;
    result.requested_profile = options.profile;
    result.profile_automatic = !options.profile.has_value();
    result.profile = options.profile;
    result.input_capabilities
        = options.capabilities.value_or(std::vector<std::string>(detected.begin(), detected.end()));
    result.notes.push_back(
        "Capability inputs and known dependency checks are not a complete implication closure or emitted-version "
        "ceiling."
    );
    if (type == DeviceType::cuda)
        result.notes.push_back(
            "CUDA architecture also depends on shader code and the downstream toolkit. An explicitly selected highest "
            "numeric CUDA tier is checked against entry-point PTX at link time and requires a fully specialized "
            "program. "
            "Device-derived tiers remain compiler assumptions, not exact architecture requests."
        );

    auto family = native_family(type);
    VersionName profile;
    if (options.profile) {
        SGL_CHECK(
            compiler->findProfile(options.profile->c_str()) != SLANG_PROFILE_UNKNOWN,
            "Unknown Slang profile '{}'",
            *options.profile
        );
        profile = version_name(*options.profile);
        bool native = family != Family::unknown && family != Family::cuda && profile.family == family;
        bool cross = type == DeviceType::vulkan && (profile.family == Family::dx || profile.family == Family::glsl);
        SGL_CHECK(native || cross, "Profile '{}' is not supported for the {} backend", *options.profile, type);
        if (cross) {
            SGL_CHECK(
                options.capabilities.has_value(),
                "Vulkan cross-family profile '{}' requires an explicit capabilities list (which may be empty)",
                *options.profile
            );
            result.notes.push_back(
                "Cross-family Vulkan profile uses Slang's mapping; no native-version reconciliation is applied."
            );
        }
    }

    std::map<std::string, std::string> inputs;
    for (const auto& name : result.input_capabilities) {
        if (options.capabilities)
            SGL_CHECK(
                compiler->findCapability(name.c_str()) != SLANG_CAPABILITY_UNKNOWN,
                "Unknown explicit capability '{}'",
                name
            );
        inputs[canonical_name(type, name)] = options.capabilities ? "explicit" : "device";
    }
    std::map<std::string, bool> overrides;
    for (const auto& [name, enabled] : options.capability_overrides) {
        auto canonical = canonical_name(type, name);
        auto id = compiler->findCapability(name.c_str());
        SGL_CHECK(
            id != SLANG_CAPABILITY_UNKNOWN || (!enabled && inputs.contains(canonical)),
            "Unknown capability override '{}'",
            name
        );
        auto [it, inserted] = overrides.emplace(canonical, enabled);
        SGL_CHECK(inserted || it->second == enabled, "Conflicting overrides for equivalent capability '{}'", canonical);
    }
    for (const auto& [name, enabled] : overrides) {
        if (enabled)
            inputs[name] = "override";
        else {
            inputs.erase(name);
            result.removed_capabilities[name]
                = "Removed by capability_overrides (profile and implied requirements remain).";
        }
    }

    Version maximum;
    Version hardware_maximum;
    for (const auto& name : detected) {
        auto version = version_name(name);
        if (version.family == family)
            hardware_maximum = std::max(hardware_maximum, version.version);
    }
    for (auto it = inputs.begin(); it != inputs.end();) {
        const auto& [name, origin] = *it;
        auto required = required_version(name);
        if (profile.family == family && family != Family::unknown && required.family == family
            && required.version > profile.version) {
            if (origin == "device" && version_name(name).family == family) {
                result.removed_capabilities[name]
                    = fmt::format("Detected version exceeds explicit profile '{}'.", *options.profile);
                it = inputs.erase(it);
                continue;
            }
            SGL_THROW(
                "Capability '{}' ({}) requires version {}.{}, exceeding profile '{}'; remove the input, supply an "
                "authoritative capabilities list, or choose a suitable profile",
                name,
                origin,
                required.version.major,
                required.version.minor,
                *options.profile
            );
        }
        if (compiler->findCapability(name.c_str()) == SLANG_CAPABILITY_UNKNOWN) {
            SGL_CHECK(origin == "device", "Unknown explicit capability '{}'", name);
            result.ignored_capabilities.push_back(name);
            it = inputs.erase(it);
            continue;
        }
        if (required.family == family)
            maximum = std::max(maximum, required.version);
        ++it;
    }
    if (profile.family == family)
        maximum = std::max(maximum, profile.version);
    if (hardware_maximum != Version{} && maximum > hardware_maximum)
        SGL_THROW(
            "Selected target version {}.{} exceeds detected {} device version {}.{}",
            maximum.major,
            maximum.minor,
            type,
            hardware_maximum.major,
            hardware_maximum.minor
        );

    if (!options.profile && type == DeviceType::d3d12) {
        // SLANG-W001: capabilities do not currently select DXC's shader model.
        maximum = std::max(maximum, Version{6, 0});
        result.profile = fmt::format("sm_{}_{}", maximum.major, maximum.minor);
        result.notes.push_back("SLANG-W001: derived DX profile from known selected model requirements.");
    } else if (!options.profile && type == DeviceType::vulkan) {
        // SLANG-W002: avoid Slang's implicit SPIR-V 1.5 feature bundle.
        result.profile = "spirv_1_0";
        result.notes.push_back("SLANG-W002: minimal SPIR-V profile preserves raw version/feature selection.");
    }
    if (result.profile)
        SGL_CHECK(
            compiler->findProfile(result.profile->c_str()) != SLANG_PROFILE_UNKNOWN,
            "Unknown resolved Slang profile '{}'",
            *result.profile
        );

    // SLANG-W006: a fixed target baseline prevents empty inputs bypassing ordinary capability checking.
    auto base = baseline(type);
    inputs.try_emplace(base, "baseline");
    if (overrides.contains(base) && !overrides.at(base))
        result.notes.push_back(fmt::format("Mandatory backend baseline '{}' remains despite input removal.", base));
    if (options.profile && profile.family == Family::spirv && profile.version >= Version{1, 5}
        && overrides.contains("SPV_EXT_physical_storage_buffer") && !overrides.at("SPV_EXT_physical_storage_buffer"))
        result.notes.push_back(
            "The explicit SPIR-V profile still supplies SPV_EXT_physical_storage_buffer after input removal."
        );

    result.capability_origins = inputs;
    for (const auto& [name, origin] : inputs)
        result.capabilities.push_back(name);
    return result;
}

void validate_cuda_program(slang::IComponentType* program, const SlangTargetInfo& target_info, PersistentCache* cache)
{
    if (target_info.legacy || target_info.target != "ptx")
        return;

    Version selected;
    bool explicit_version = false;
    for (const auto& [name, origin] : target_info.capability_origins) {
        auto version = version_name(name);
        if (version.family == Family::cuda && version.version > selected) {
            selected = version.version;
            explicit_version = origin == "explicit" || origin == "override";
        }
    }
    if (!explicit_version)
        return;

    // SLANG-W007: RHI owns deferred/specialized compilation and persistent cache reads. Its public
    // API has no generated-code validation hook. Validate eagerly through the actual linked Slang
    // component; RHI reuses its generated code, and this check cannot be bypassed by an RHI cache hit.
    // Do not advertise exact validation for code that only becomes known at dispatch time.
    SGL_CHECK(
        program->getSpecializationParamCount() == 0,
        "Exact CUDA target {}.{} requires a fully specialized program at link time; specialize the shader before "
        "linking or use device-derived capabilities for runtime specialization",
        selected.major,
        selected.minor
    );
    slang::ProgramLayout* layout = nullptr;
    SGL_CATCH_INTERNAL_SLANG_ERROR(layout = program->getLayout());
    SGL_CHECK(layout, "Failed to get program layout for CUDA target validation");
    for (SlangUInt index = 0; index < layout->getEntryPointCount(); ++index) {
        Slang::ComPtr<ISlangBlob> code;
        Slang::ComPtr<ISlangBlob> diagnostics;
        SlangResult status = SLANG_FAIL;
        SGL_CATCH_INTERNAL_SLANG_ERROR(
            status = program->getEntryPointCode(index, 0, code.writeRef(), diagnostics.writeRef())
        );
        auto diagnostic_text = diagnostics
            ? std::string(static_cast<const char*>(diagnostics->getBufferPointer()), diagnostics->getBufferSize())
            : std::string();
        SGL_CHECK(
            SLANG_SUCCEEDED(status) && code,
            "Failed to compile entry point '{}' for exact CUDA target {}.{}; the shader and loaded NVRTC toolkit "
            "must support this architecture. {}",
            layout->getEntryPointByIndex(index)->getName(),
            selected.major,
            selected.minor,
            diagnostic_text
        );
        if (!diagnostic_text.empty())
            log_warn("Slang compiler warnings:\n{}", diagnostic_text);
        std::string ptx(static_cast<const char*>(code->getBufferPointer()), code->getBufferSize());
        std::smatch target_match;
        std::smatch version_match;
        // Match directives at line starts, not occurrences in generated comments or identifiers.
        const std::regex target_pattern(R"((?:^|\n)[ \t]*\.target[ \t]+(sm_[0-9]+[a-z]*)(?:[ \t,\r\n]|$))");
        const std::regex version_pattern(R"((?:^|\n)[ \t]*\.version[ \t]+([0-9]+\.[0-9]+)(?:[ \t\r\n]|$))");
        SGL_CHECK(
            std::regex_search(ptx, target_match, target_pattern)
                && std::regex_search(ptx, version_match, version_pattern),
            "Cannot validate exact CUDA target {}.{}: generated code has no recognized PTX target/version header",
            selected.major,
            selected.minor
        );
        auto expected = fmt::format("sm_{}{}", selected.major, selected.minor);
        SGL_CHECK(
            target_match[1].str() == expected,
            "Exact CUDA target {}.{} requested {}, but entry point '{}' emitted {} (PTX {}). Shader requirements "
            "or the NVRTC toolkit changed the architecture; choose a matching supported capability or use "
            "device-derived capabilities",
            selected.major,
            selected.minor,
            expected,
            layout->getEntryPointByIndex(index)->getName(),
            target_match[1].str(),
            version_match[1].str()
        );
        if (cache) {
            Slang::ComPtr<ISlangBlob> key;
            SGL_CATCH_INTERNAL_SLANG_ERROR(program->getEntryPointHash(index, 0, key.writeRef()));
            cache->expect_entry(key, code);
        }
    }
}

} // namespace sgl
