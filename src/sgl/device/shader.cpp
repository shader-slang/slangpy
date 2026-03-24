// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shader.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"
#include "sgl/device/reflection.h"
#include "sgl/device/kernel.h"
#include "sgl/device/print.h"
#include "sgl/device/slang_utils.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/hot_reload.h"

#include "sgl/core/type_utils.h"
#include "sgl/core/platform.h"
#include "sgl/core/string.h"
#include "sgl/core/crypto.h"
#include "sgl/core/timer.h"
#include "sgl/core/file_stream.h"

#include <slang.h>

#include <atomic>
#include <random>
#include <regex>

namespace sgl {

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Filter specific messages from slang diagnostics.
/// This is mainly a workaround for the annoying warnings from NVAPI.
/// Ideally to solve that case, slang should have diagnostic pragmas.
inline std::string filter_diagnostics(const char* diagnostics)
{
    static std::regex re("^.*nvHLSLExtns.h.*\\n.*\\n.*\\^~+\\n", std::regex::ECMAScript);
    std::string result = diagnostics;
    result = std::regex_replace(result, re, "");
    return result;
}

/// Append slang diagnostics to a string if not null.
inline std::string append_diagnostics(std::string msg, ISlangBlob* diagnostics)
{
    if (diagnostics)
        msg += fmt::format("\n{}", filter_diagnostics(static_cast<const char*>(diagnostics->getBufferPointer())));
    return msg;
}

/// Report slang diagnostics to log if not null.
inline void report_diagnostics(ISlangBlob* diagnostics)
{
    if (diagnostics) {
        std::string filtered = filter_diagnostics(static_cast<const char*>(diagnostics->getBufferPointer()));
        if (!filtered.empty())
            log_warn("Slang compiler warnings:\n{}", filtered);
    }
}

/// Single point for calling slang's createCompositeComponentType.
/// Both SlangSession::create_composite_component_type() and ShaderProgram::link()
/// route through this helper so there is exactly one raw API call site.
static Slang::ComPtr<slang::IComponentType>
create_slang_composite(slang::ISession* session, std::span<slang::IComponentType* const> components)
{
    Slang::ComPtr<slang::IComponentType> composite;
    Slang::ComPtr<ISlangBlob> diagnostics;
    SGL_CATCH_INTERNAL_SLANG_ERROR(session->createCompositeComponentType(
        components.data(),
        narrow_cast<SlangInt>(components.size()),
        composite.writeRef(),
        diagnostics.writeRef()
    ));
    report_diagnostics(diagnostics);
    SGL_CHECK(composite, "Failed to create composite component type");
    return composite;
}

// ----------------------------------------------------------------------------
// TypeConformance
// ----------------------------------------------------------------------------

std::string TypeConformance::to_string() const
{
    return fmt::format(
        "TypeConformance(interface_name=\"{}\", type_name=\"{}\", id={})",
        interface_name,
        type_name,
        id
    );
}

// ----------------------------------------------------------------------------
// SpecializationArg
// ----------------------------------------------------------------------------

std::string SpecializationArg::to_string() const
{
    return fmt::format("SpecializationArg(kind={}, value=\"{}\")", kind, value);
}

// ----------------------------------------------------------------------------
// CompilerOptionEntries
// ----------------------------------------------------------------------------

/// Helper class for creating slang::CompilerOptionEntry entries.
class CompilerOptionEntries {
public:
    void add(slang::CompilerOptionName name, int value)
    {
        m_entries.push_back({
            .name = name,
            .value = {
                .kind = slang::CompilerOptionValueKind::Int,
                .int0 = value,
            },
        });
    }

    void add(slang::CompilerOptionName name, bool value)
    {
        m_entries.push_back({
            .name = name,
            .value = {
                .kind = slang::CompilerOptionValueKind::Int,
                .int0 = value ? 1 : 0,
            },
        });
    }

    void add(slang::CompilerOptionName name, std::string_view value)
    {
        m_entries.push_back({
            .name = name,
            .value = {
                .kind = slang::CompilerOptionValueKind::String,
                .string0 = std::string{value},
            },
        });
    }

    void add(slang::CompilerOptionName name, std::string_view value0, std::string_view value1)
    {
        m_entries.push_back({
            .name = name,
            .value = {
                .kind = slang::CompilerOptionValueKind::String,
                .string0 = std::string{value0},
                .string1 = std::string{value1},
            },
        });
    }

    void add_macro_define(std::string_view name, std::string_view value)
    {
        add(slang::CompilerOptionName::MacroDefine, name, value);
    }

    void add_include(const std::filesystem::path& path) { add(slang::CompilerOptionName::Include, path.string()); }

    void insert_cache_include(const std::filesystem::path& path, size_t index)
    {
        SGL_ASSERT(index <= m_entries.size());
        m_entries.insert(
            m_entries.begin() + index,
            {
                .name = slang::CompilerOptionName::Include,
                .value = {
                    .kind = slang::CompilerOptionValueKind::String,
                    .string0 = path.string(),
                },
            }
        );
    }

    std::vector<slang::CompilerOptionEntry> slang_entries() const
    {
        std::vector<slang::CompilerOptionEntry> result{m_entries.size()};
        for (size_t i = 0; i < m_entries.size(); ++i) {
            result[i].name = m_entries[i].name;
            result[i].value.kind = m_entries[i].value.kind;
            switch (m_entries[i].value.kind) {
            case slang::CompilerOptionValueKind::Int:
                result[i].value.intValue0 = m_entries[i].value.int0;
                result[i].value.intValue1 = m_entries[i].value.int1;
                break;
            case slang::CompilerOptionValueKind::String:
                result[i].value.stringValue0 = m_entries[i].value.string0.c_str();
                result[i].value.stringValue1 = m_entries[i].value.string1.c_str();
                break;
            }
        }
        return result;
    }

private:
    struct Entry {
        slang::CompilerOptionName name;
        struct {
            slang::CompilerOptionValueKind kind;
            int int0{0};
            int int1{0};
            std::string string0;
            std::string string1;
        } value;
    };

    std::vector<Entry> m_entries;
};

// ----------------------------------------------------------------------------
// SlangSession
// ----------------------------------------------------------------------------

SlangSession::SlangSession(ref<Device> device, SlangSessionDesc desc)
    : m_device(std::move(device))
    , m_desc(std::move(desc))
{
    ConstructorRefGuard ref_guard(this);

    // Register with hot load reload system if enabled.
    if (m_device->_hot_reload())
        m_device->_hot_reload()->_register_slang_session(this);

    // Create (but don't compile yet) the NVAPI module
    // We link this to all programs because slang uses NVAPI features while not including NVAPI itself.
    if (SGL_HAS_NVAPI && m_device->type() == DeviceType::d3d12) {
        m_nvapi_module = make_ref<SlangModule>(
            ref(this),
            SlangModuleDesc{
                .module_name = "sgl/device/nvapi.slang",
            }
        );
        m_nvapi_module->break_strong_reference_to_session();
    }

    recreate_session();
}

SlangSession::~SlangSession()
{
    // Ensure nvapi module gets released before destructor completes, to
    // avoid issues with list of registered modules getting corrupted
    // due to member variable destructor ordering.
    m_nvapi_module.reset();

    // Unregister with hot load reload system if enabled.
    if (m_device->_hot_reload())
        m_device->_hot_reload()->_unregister_slang_session(this);
}

void SlangSession::recreate_session()
{
    SGL_CHECK_NOT_NULL(m_device);

    // Invalidate all standalone component types (from specialize/link/compose).
    for (auto ct : m_standalone_component_types) {
        ct->_invalidate();
    }

    SlangSessionBuild build;

    // Build everything first.
    create_session(build);
    for (auto module : m_registered_modules) {
        module->load(build);
    }
    for (auto tc : m_registered_type_conformances) {
        tc->rebuild(build);
    }
    for (auto program : m_registered_programs) {
        program->link(build);
    }

    // On success, store it all.
    m_data = build.session;
    for (auto module : m_registered_modules) {
        module->store_built_data(build);
    }
    for (auto tc : m_registered_type_conformances) {
        tc->store_built_data(build);
    }
    for (auto program : m_registered_programs) {
        program->store_built_data(build);
    }

    // Update cache of loaded modules.
    update_module_cache_and_dependencies();
}

void SlangSession::create_session(SlangSessionBuild& build)
{
    SGL_CHECK_NOT_NULL(m_device);

    DeviceType device_type = m_device->type();

    CompilerOptionEntries session_options;
    CompilerOptionEntries target_options;

    const SlangCompilerOptions& options = m_desc.compiler_options;

    // Use device's highest supported shader model if none is provided explicitly.
    ShaderModel supported_shader_model = m_device->supported_shader_model();
    ShaderModel default_shader_model = supported_shader_model;
    // TODO: Slang generates invalid HLSL for SM 6.7 when using ray payloads.
    if (default_shader_model == ShaderModel::sm_6_7)
        default_shader_model = ShaderModel::sm_6_6;
    ShaderModel shader_model = options.shader_model;
    if (options.shader_model == ShaderModel::unknown)
        shader_model = default_shader_model;

    // Check that requested shader model is supported.
    SGL_CHECK(
        shader_model <= supported_shader_model,
        "Shader model {} is not supported (max shader model is {})",
        shader_model,
        supported_shader_model
    );

    // Set matrix layout.
    if (options.matrix_layout == SlangMatrixLayout::row_major)
        session_options.add(slang::CompilerOptionName::MatrixLayoutRow, true);
    else if (options.matrix_layout == SlangMatrixLayout::column_major)
        session_options.add(slang::CompilerOptionName::MatrixLayoutColumn, true);

    // TODO: Globally disable warning 30856.
    // This is a workaround for an issue in the Slang compiler:
    // https://github.com/shader-slang/slang/issues/8166
    session_options.add(slang::CompilerOptionName::DisableWarning, std::string_view("30856"));
    // TODO: Globally disable warning 41012.
    // Example: entry point 'foo' uses additional capabilities that are not part of the specified profile 'unknown'.
    // This warning happens on CUDA because we're not properly setting the target profile (i.e. "cuda_sm_x_x").
    session_options.add(slang::CompilerOptionName::DisableWarning, std::string_view("41012"));

    // Set warnings.
    for (const auto& warning : options.enable_warnings)
        session_options.add(slang::CompilerOptionName::EnableWarning, warning);
    for (const auto& warning : options.disable_warnings)
        session_options.add(slang::CompilerOptionName::DisableWarning, warning);
    for (const auto& warning : options.warnings_as_errors)
        session_options.add(slang::CompilerOptionName::WarningsAsErrors, warning);

    // Set diagnostic options.
    session_options.add(slang::CompilerOptionName::ReportDownstreamTime, options.report_downstream_time);
    session_options.add(slang::CompilerOptionName::ReportPerfBenchmark, options.report_perf_benchmark);
    session_options.add(slang::CompilerOptionName::SkipSPIRVValidation, options.skip_spirv_validation);

    // Set optimization and debug options.
    session_options.add(slang::CompilerOptionName::DebugInformation, int(options.debug_info));
    session_options.add(slang::CompilerOptionName::Optimization, int(options.optimization));

    // Set downstream arguments.
    if (device_type == DeviceType::d3d12) {
        for (const auto& arg : options.downstream_args)
            session_options.add(slang::CompilerOptionName::DownstreamArgs, "dxc", arg);
    }

    // Set downstream argument for optix include path.
    if (device_type == DeviceType::cuda) {
        uint32_t optix_version = m_device->info().optix_version;
        if (optix_version > 0) {
            std::string version_tag = fmt::format("{}_{}", optix_version / 10000, (optix_version % 10000) / 100);
            auto optix_path = platform::runtime_directory() / "optix" / version_tag;
            session_options.add(slang::CompilerOptionName::DownstreamArgs, "nvrtc", "-I" + optix_path.string());
        }
    }

    // Set intermediate dump options.
    session_options.add(slang::CompilerOptionName::DumpIntermediates, options.dump_intermediates);
    session_options.add(slang::CompilerOptionName::DumpIntermediatePrefix, options.dump_intermediates_prefix);

    // Enable experimental features (required for neural module).
    if (options.enable_experimental_features)
        session_options.add(slang::CompilerOptionName::ExperimentalFeature, true);

    // Add hlsl_nvapi capability.
    session_options.add(
        slang::CompilerOptionName::Capability,
        int(m_device->global_session()->findCapability("hlsl_nvapi"))
    );
    // TODO: Pass all detected capabilities to the session.
    // This currently leads to slang compilation errors and needs more investigation.
    // for (SlangCapabilityID capability : m_device->_slang_capabilities()) {
    //     session_options.add(slang::CompilerOptionName::Capability, int(capability));
    // }

    // TODO: We enable loop inversion as it was the default in older versions of Slang,
    //       and leads to artifacts in one project using sgl.
    session_options.add(slang::CompilerOptionName::LoopInversion, true);

    // Only use up to date binary modules (required for caching to work properly).
    if (m_desc.cache_path)
        session_options.add(slang::CompilerOptionName::UseUpToDateBinaryModule, true);

    slang::SessionDesc session_desc{};

    auto data = make_ref<SlangSessionData>();

    // Set include paths.
    // Slang will search for files in the order they are specified.
    // Use provided include paths first, followed by default include paths.
    // Keep a copy of the include paths for local resolution of module names.
    data->include_paths = options.include_paths;
    if (m_desc.add_default_include_paths)
        data->include_paths.push_back(platform::runtime_directory() / "shaders");
    for (const auto& path : data->include_paths)
        session_options.add_include(path);

    // Set macro defines.
    for (const auto& define : options.defines)
        session_options.add_macro_define(define.first, define.second);

    // Select target profile.
    slang::TargetDesc target_desc;
    target_desc.format = SLANG_TARGET_UNKNOWN;
    uint32_t shader_model_major = get_shader_model_major_version(shader_model);
    uint32_t shader_model_minor = get_shader_model_minor_version(shader_model);
    std::string profile_str = fmt::format("sm_{}_{}", shader_model_major, shader_model_minor);

    // TODO: CUDA doesn't support shader model profiles like Vulkan or D3D12.
    if (device_type == DeviceType::d3d12 || device_type == DeviceType::vulkan) {
        target_desc.profile = m_device->global_session()->findProfile(profile_str.c_str());
        SGL_CHECK(target_desc.profile != SLANG_PROFILE_UNKNOWN, "Unsupported target profile: {}", profile_str);
    }

    // Set floating point mode.
    target_desc.floatingPointMode = static_cast<::SlangFloatingPointMode>(options.floating_point_mode);
    target_options.add(slang::CompilerOptionName::FloatingPointMode, int(options.floating_point_mode));

    // Force GLSL scalar buffer layout.
    target_desc.forceGLSLScalarBufferLayout = true;
    target_options.add(slang::CompilerOptionName::GLSLForceScalarLayout, true);

    // Select target format based on device type.
    const char* target_define{nullptr};
    switch (device_type) {
    case DeviceType::d3d12:
        target_desc.format = SLANG_DXIL;
        target_define = "__TARGET_D3D12__";
        break;
    case DeviceType::vulkan:
        target_desc.format = SLANG_SPIRV;
        target_desc.flags |= SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
        target_define = "__TARGET_VULKAN__";
        break;
    case DeviceType::metal:
        target_desc.format = SLANG_METAL_LIB;
        target_define = "__TARGET_METAL__";
        break;
    case DeviceType::wgpu:
        target_desc.format = SLANG_WGSL;
        target_define = "__TARGET_WGPU__";
        break;
    case DeviceType::cpu:
        target_desc.format = SLANG_SHADER_HOST_CALLABLE;
        target_define = "__TARGET_CPU__";
        break;
    case DeviceType::cuda:
        target_desc.format = SLANG_PTX;
        target_define = "__TARGET_CUDA__";
        break;
    default:
        SGL_UNREACHABLE();
    }
    SGL_ASSERT(target_define);

    // Add target define.
    session_options.add_macro_define(target_define, "1");

    // Add shader model defines.
    session_options.add_macro_define("__SHADER_TARGET_MAJOR", fmt::format("{}", shader_model_major));
    session_options.add_macro_define("__SHADER_TARGET_MINOR", fmt::format("{}", shader_model_minor));

    // Add NVAPI defines.
    session_options.add_macro_define(
        "SGL_ENABLE_NVAPI",
        (SGL_HAS_NVAPI && m_device->type() == DeviceType::d3d12) ? "1" : "0"
    );
#if SGL_HAS_NVAPI
    session_options.add_macro_define("NV_SHADER_EXTN_SLOT", "u999");
    session_options.add(
        slang::CompilerOptionName::DownstreamArgs,
        "dxc",
        fmt::format("-I{}", (platform::runtime_directory() / "shaders/nvapi").string())
    );
#endif

    // Add device print enable flag.
    session_options.add_macro_define("SGL_ENABLE_PRINT", m_device->desc().enable_print ? "1" : "0");

    auto slang_target_option_entries = target_options.slang_entries();
    target_desc.compilerOptionEntries = slang_target_option_entries.data();
    target_desc.compilerOptionEntryCount = narrow_cast<uint32_t>(slang_target_option_entries.size());

    session_desc.targets = &target_desc;
    session_desc.targetCount = 1;

    auto slang_session_option_entries = session_options.slang_entries();
    session_desc.compilerOptionEntries = slang_session_option_entries.data();
    session_desc.compilerOptionEntryCount = narrow_cast<uint32_t>(slang_session_option_entries.size());

    // Use the session descriptor to generate a hash that uniquely identifies the session.
    Slang::ComPtr<ISlangBlob> session_digest;
    SLANG_CALL(m_device->global_session()->getSessionDescDigest(&session_desc, session_digest.writeRef()));
    data->uid = string::hexlify(session_digest->getBufferPointer(), session_digest->getBufferSize());

    // Setup session cache.
    // The session cache relies on Slang's ability to serialize shader modules.
    // We use the session digest to create a unique cache directory for the session.
    // When loading a shader module, we store a serialized version ('.slang-module' file) to the cache.
    // Next time the module is loaded, Slang can detect the cached file and load it directly.
    // Cached modules are stored in a directory structure that mirrors the include paths.
    if (m_desc.cache_path) {
        data->cache_enabled = true;
        data->cache_path = *m_desc.cache_path;
        SGL_CHECK(data->cache_path.is_absolute(), "Cache path must be an absolute path");
        data->cache_path /= data->uid;
        data->cache_include_paths.resize(data->include_paths.size());
        for (size_t i = 0; i < data->include_paths.size(); ++i) {
            data->cache_include_paths[i] = data->cache_path / fmt::format("{}", i);
            std::filesystem::create_directories(data->cache_include_paths[i]);
            session_options.insert_cache_include(data->cache_include_paths[i], i);
        }

        // Update session descriptor with the patched include paths.
        slang_session_option_entries = session_options.slang_entries();
        session_desc.compilerOptionEntries = slang_session_option_entries.data();
        session_desc.compilerOptionEntryCount = narrow_cast<uint32_t>(slang_session_option_entries.size());
    }

    SLANG_CALL(m_device->global_session()->createSession(session_desc, data->slang_session.writeRef()));

    // Store session.
    build.session = std::move(data);
}

ref<SlangModule> SlangSession::load_module(std::string_view module_name)
{
    SlangModuleDesc desc;
    desc.module_name = module_name;

    ref<SlangModule> module = make_ref<SlangModule>(ref(this), desc);

    // Setup build info with just this session in and load/store the module.
    SlangSessionBuild build;
    build.session = m_data;
    module->load(build);
    module->store_built_data(build);

    // Update cache of loaded modules.
    update_module_cache_and_dependencies();

    return module;
}

ref<SlangModule> SlangSession::load_module_from_source(
    std::string_view module_name,
    std::string_view source,
    std::optional<std::filesystem::path> path
)
{
    SlangModuleDesc desc;
    desc.module_name = module_name;
    desc.source = source;
    desc.path = path;

    ref<SlangModule> module = make_ref<SlangModule>(ref(this), desc);

    // Setup build info with just this session in and load/store the module.
    SlangSessionBuild build;
    build.session = m_data;
    module->load(build);
    module->store_built_data(build);

    // Update cache of loaded modules.
    update_module_cache_and_dependencies();

    return module;
}

ref<ShaderProgram> SlangSession::link_program(
    std::vector<ref<SlangModule>> modules,
    std::vector<ref<SlangEntryPoint>> entry_points,
    std::optional<SlangLinkOptions> link_options,
    std::span<const TypeConformance> type_conformances
)
{
    for (const auto& module : modules)
        SGL_CHECK(module->session() == this, "All modules must belong to this session.");
    for (const auto& entry_point : entry_points)
        SGL_CHECK(entry_point->module()->session() == this, "All entry points must belong to this session.");

    // Validate type conformance entries.
    if (!type_conformances.empty()) {
        std::set<std::pair<std::string_view, std::string_view>> type_conformance_set;
        std::multimap<std::string_view, int32_t> type_conformance_ids;
        for (const TypeConformance& c : type_conformances) {
            if (type_conformance_set.contains({c.interface_name, c.type_name}))
                SGL_THROW(
                    "Duplicate type conformance entry for interface type \"{}\" and type \"{}\"",
                    c.interface_name,
                    c.type_name
                );
            type_conformance_set.insert({c.interface_name, c.type_name});
            if (c.id >= 0) {
                auto range = type_conformance_ids.equal_range(c.interface_name);
                if (std::any_of(
                        range.first,
                        range.second,
                        [&c](const auto& pair)
                        {
                            return pair.second == c.id;
                        }
                    ))
                    SGL_THROW("Duplicate type id {} for interface type \"{}\"", c.id, c.interface_name);
                type_conformance_ids.insert({c.interface_name, c.id});
            }
        }
    }

    // Create SlangTypeConformance objects for each type conformance descriptor.
    std::vector<ref<SlangTypeConformance>> tc_objects;
    tc_objects.reserve(type_conformances.size());
    for (const TypeConformance& tc : type_conformances) {
        tc_objects.push_back(create_type_conformance(tc.type_name, tc.interface_name, tc.id));
    }

    // Link NVAPI module if available.
    if (SGL_HAS_NVAPI && m_device->type() == DeviceType::d3d12)
        modules.push_back(m_nvapi_module);

    // Generate label
    std::string label;
    for (const auto& entry_point : entry_points) {
        label += (label.empty() ? "" : ", ") + entry_point->module()->name() + ":" + entry_point->name();
    }

    ShaderProgramDesc desc{
        .modules = modules,
        .entry_points = entry_points,
        .type_conformances = std::move(tc_objects),
        .link_options = link_options,
        .label = label,
    };

    auto program = make_ref<ShaderProgram>(ref(device()), ref(this), desc);

    // Setup build with this session and populate with all relevant
    // modules (and consequentially their entry points), then link and
    // store the program.
    SlangSessionBuild build;
    build.session = m_data;
    for (auto module : desc.modules) {
        module->populate_build_data(build);
    }
    for (const auto& tc : desc.type_conformances) {
        tc->populate_build_data(build);
    }
    program->link(build);
    program->store_built_data(build);

    // Update cache of loaded modules, as it may have changed after program link.
    update_module_cache_and_dependencies();

    return program;
}

ref<ShaderProgram> SlangSession::load_program(
    std::string_view module_name,
    std::vector<std::string_view> entry_point_names,
    std::optional<std::string_view> additional_source,
    std::optional<SlangLinkOptions> link_options
)
{
    ref<SlangModule> module = load_module(module_name);
    std::vector<ref<SlangModule>> modules{module};
    // TODO improve the way we generate unique names for additional sources
    static std::atomic<uint32_t> id{0};
    if (additional_source)
        modules.push_back(load_module_from_source(fmt::format("additional_source_{}", id++), *additional_source));
    std::vector<ref<SlangEntryPoint>> entry_points;
    for (const auto& entry_point_name : entry_point_names)
        entry_points.push_back(module->entry_point(entry_point_name));
    return link_program(std::move(modules), std::move(entry_points), link_options);
}

std::string SlangSession::load_source(std::string_view module_name)
{
    std::string resolved_name = m_data->resolve_module_name(module_name);

    for (const std::filesystem::path& include_path : m_data->include_paths) {
        std::filesystem::path full_path = include_path / resolved_name;
        if (std::filesystem::exists(full_path)) {
            FileStream stream(full_path, FileStream::Mode::read);
            std::string result;
            result.resize(stream.size());
            stream.read(result.data(), result.size());
            return result;
        }
    }
    SGL_THROW("Failed to load source for module \"{}\"", module_name);
}

ref<SlangTypeConformance>
SlangSession::create_type_conformance(std::string_view type_name, std::string_view interface_name, int32_t id_override)
{
    slang::ISession* slang_session = m_data->slang_session;

    slang::TypeReflection* type = find_type_by_name(type_name);
    SGL_CHECK(type, "Type \"{}\" not found", type_name);
    slang::TypeReflection* interface_type = find_type_by_name(interface_name);
    SGL_CHECK(interface_type, "Interface type \"{}\" not found", interface_name);

    Slang::ComPtr<slang::ITypeConformance> slang_conformance;
    Slang::ComPtr<ISlangBlob> diagnostics;
    SGL_CATCH_INTERNAL_SLANG_ERROR(slang_session->createTypeConformanceComponentType(
        type,
        interface_type,
        slang_conformance.writeRef(),
        id_override,
        diagnostics.writeRef()
    ));
    report_diagnostics(diagnostics);
    SGL_CHECK(
        slang_conformance,
        "Failed to create type conformance for type \"{}\" and interface \"{}\"",
        type_name,
        interface_name
    );

    return make_ref<SlangTypeConformance>(
        ref(this),
        Slang::ComPtr<slang::IComponentType>(slang_conformance.get()),
        TypeConformance(interface_name, type_name, id_override)
    );
}

ref<SlangComponentType>
SlangSession::create_composite_component_type(std::span<const ref<SlangComponentType>> components)
{
    std::vector<slang::IComponentType*> slang_components;
    slang_components.reserve(components.size());
    for (const auto& c : components)
        slang_components.push_back(c->slang_component_type());

    auto composite = create_slang_composite(m_data->slang_session, slang_components);

    auto result = make_ref<SlangComponentType>(ref<SlangSession>(this), std::move(composite));
    _register_standalone_component_type(result.get());
    return result;
}

slang::TypeReflection* SlangSession::find_type_by_name(std::string_view name) const
{
    SGL_CHECK(!m_registered_modules.empty(), "No modules loaded; cannot resolve type name \"{}\"", name);

    std::string name_str(name);
    for (auto* module : m_registered_modules) {
        slang::TypeReflection* type = module->find_type_by_name(name);
        if (type)
            return type;
    }
    return nullptr;
}

std::vector<std::filesystem::path> SlangSession::all_module_dependency_paths() const
{
    std::vector<std::filesystem::path> result;
    slang::ISession* slang_session = m_data->slang_session;
    SlangInt module_count = slang_session->getLoadedModuleCount();
    for (SlangInt module_index = 0; module_index < module_count; module_index++) {
        slang::IModule* slang_module = slang_session->getLoadedModule(module_index);
        SlangInt32 dependency_count = slang_module->getDependencyFileCount();
        for (SlangInt32 dependency_index = 0; dependency_index < dependency_count; dependency_index++) {
            const char* path = slang_module->getDependencyFilePath(dependency_index);
            if (path)
                result.emplace_back(path);
        }
    }
    return result;
}

void SlangSession::_register_program(ShaderProgram* program)
{
    SGL_ASSERT(m_registered_programs.count(program) == 0);
    m_registered_programs.insert(program);
}

void SlangSession::_unregister_program(ShaderProgram* program)
{
    SGL_ASSERT(m_registered_programs.count(program) == 1);
    m_registered_programs.erase(program);
}

void SlangSession::_register_module(SlangModule* module)
{
    SGL_ASSERT(
        std::find(m_registered_modules.begin(), m_registered_modules.end(), module) == m_registered_modules.end()
    );
    m_registered_modules.push_back(module);
}

void SlangSession::_unregister_module(SlangModule* module)
{
    auto existing = std::find(m_registered_modules.begin(), m_registered_modules.end(), module);
    SGL_ASSERT(existing != m_registered_modules.end());
    m_registered_modules.erase(existing);
}

void SlangSession::_register_type_conformance(SlangTypeConformance* tc)
{
    SGL_ASSERT(m_registered_type_conformances.count(tc) == 0);
    m_registered_type_conformances.insert(tc);
}

void SlangSession::_unregister_type_conformance(SlangTypeConformance* tc)
{
    SGL_ASSERT(m_registered_type_conformances.count(tc) == 1);
    m_registered_type_conformances.erase(tc);
}

void SlangSession::_register_standalone_component_type(SlangComponentType* ct)
{
    m_standalone_component_types.insert(ct);
}

void SlangSession::_unregister_standalone_component_type(SlangComponentType* ct)
{
    m_standalone_component_types.erase(ct);
}

std::string SlangSession::to_string() const
{
    return fmt::format(
        "SlangSession(\n"
        "  device = {},\n"
        "  uid = {}\n"
        ")",
        m_device,
        m_data->uid
    );
}

void SlangSession::update_module_cache_and_dependencies()
{
    // Notify hot reload system of potentially changed modules,
    // thus potentially changed dependencies.
    if (m_device->_hot_reload())
        m_device->_hot_reload()->_on_session_modules_changed(this);

    // Cache newly loaded modules if enabled.
    if (m_data->cache_enabled) {
        for (int i = 0; i < m_data->slang_session->getLoadedModuleCount(); ++i) {
            slang::IModule* slang_module = m_data->slang_session->getLoadedModule(i);
            if (m_data->loaded_modules.contains(slang_module))
                continue;
            write_module_to_cache(slang_module);
            m_data->loaded_modules.insert(slang_module);
        }
    }
}

bool SlangSession::write_module_to_cache(slang::IModule* module)
{
    std::filesystem::path path{module->getFilePath()};

    // Do not cache module if it was loaded from the cache.
    if (path.extension() == ".slang-module")
        return false;

    // Check if target path is within the specified root path.
    auto is_sub_path = [](const std::filesystem::path root, const std::filesystem::path& target)
    {
        std::string relative = std::filesystem::relative(target, root).string();
        return relative.size() == 1 || (relative.size() > 1 && relative[0] != '.' && relative[1] != '.');
    };

    // Find the include path that contains the module.
    SGL_ASSERT(m_data->include_paths.size() == m_data->cache_include_paths.size());
    size_t include_index = size_t(-1);
    for (size_t i = 0; i < m_data->include_paths.size(); ++i) {
        if (is_sub_path(m_data->include_paths[i], path)) {
            include_index = i;
            break;
        }
    }
    if (include_index == size_t(-1))
        return false;

    // Determine path of the cached module.
    std::filesystem::path relative = std::filesystem::relative(path, m_data->include_paths[include_index]);
    std::filesystem::path cache_path = m_data->cache_include_paths[include_index] / relative;
    cache_path.replace_extension(".slang-module");

    // Create directories to cache path.
    std::error_code ec;
    std::filesystem::create_directories(cache_path.parent_path(), ec);
    if (ec) {
        log_warn(
            "Failed to create directory \"{}\" for slang module cache ({})",
            cache_path.parent_path(),
            ec.message()
        );
        return false;
    }

    // Write module to a temporary file.
    std::filesystem::path tmp_path = cache_path;
    std::random_device rd;
    uint64_t uid = rd();
    tmp_path.replace_extension(".slang-module-" + string::hexlify(&uid, sizeof(uid)));
    if (std::filesystem::exists(tmp_path))
        return false;
    if (!SLANG_SUCCEEDED(module->writeToFile(tmp_path.string().c_str()))) {
        log_warn("Failed to write cached slang module \"{}\" to \"{}\"", module->getName(), cache_path);
        return false;
    }

    // Rename temporary file to cache path.
    std::filesystem::rename(tmp_path, cache_path, ec);
    if (ec) {
        log_warn("Failed to rename cached slang module \"{}\" to \"{}\" ({})", tmp_path, cache_path, ec.message());
        std::filesystem::remove(tmp_path, ec);
        return false;
    }

    log_debug("Cached slang module \"{}\" to \"{}\"", module->getName(), cache_path);

    return true;
}

std::string SlangSessionData::resolve_module_name(std::string_view module_name) const
{
    // Return if module name is an absolute file path.
    std::filesystem::path path = module_name;
    if (path.is_absolute())
        return path.string();

    // Return relative path if module name is a relative path within the include paths.
    for (const std::filesystem::path& include_path : include_paths) {
        std::filesystem::path full_path = include_path / path;
        if (std::filesystem::exists(full_path))
            return path.string();
    }

    // Assume we have a module name in the form "a.b.c" which we interpret as "a/b/c.slang".
    std::string str{module_name};
    std::replace(str.begin(), str.end(), '.', '/');
    str += ".slang";
    path = str;
    for (const std::filesystem::path& include_path : include_paths) {
        std::filesystem::path full_path = include_path / path;
        if (std::filesystem::exists(full_path))
            return path.string();
    }

    // Failed to resolve, return the module name as is.
    return std::string{module_name};
}

// ----------------------------------------------------------------------------
// SlangComponentType
// ----------------------------------------------------------------------------

SlangComponentType::SlangComponentType(
    breakable_ref<SlangSession> session,
    Slang::ComPtr<slang::IComponentType> component_type
)
    : m_session(std::move(session))
    , m_component_type(std::move(component_type))
{
}

SlangComponentType::~SlangComponentType()
{
    // If this was a standalone component type (from specialize/link/compose),
    // unregister from the session's tracking set.
    m_session->_unregister_standalone_component_type(this);
}

ref<const ProgramLayout> SlangComponentType::layout(int target_index) const
{
    slang::ProgramLayout* slang_layout;
    SGL_CATCH_INTERNAL_SLANG_ERROR(slang_layout = m_component_type->getLayout(target_index));
    return ProgramLayout::from_slang(ref(this), slang_layout);
}

ref<SlangComponentType> SlangComponentType::specialize(std::span<const SpecializationArg> specialization_args) const
{
    // Use this component type's own layout for type resolution.
    slang::ProgramLayout* component_layout;
    SGL_CATCH_INTERNAL_SLANG_ERROR(component_layout = m_component_type->getLayout());

    // Build slang specialization args from our descriptors.
    std::vector<slang::SpecializationArg> slang_spec_args;
    slang_spec_args.reserve(specialization_args.size());

    for (const SpecializationArg& arg : specialization_args) {
        slang::SpecializationArg slang_arg;
        switch (arg.kind) {
        case SpecializationArgKind::type: {
            slang::TypeReflection* type = component_layout->findTypeByName(arg.value.c_str());
            SGL_CHECK(type, "Specialization type \"{}\" not found", arg.value);
            slang_arg = slang::SpecializationArg::fromType(type);
            break;
        }
        case SpecializationArgKind::expr:
            slang_arg = slang::SpecializationArg::fromExpr(arg.value.c_str());
            break;
        case SpecializationArgKind::unknown:
            SGL_THROW("Invalid specialization argument: kind is 'unknown'");
        }
        slang_spec_args.push_back(slang_arg);
    }

    Slang::ComPtr<slang::IComponentType> specialized;
    Slang::ComPtr<ISlangBlob> diagnostics;
    SlangResult result;
    SGL_CATCH_INTERNAL_SLANG_ERROR(
        result = m_component_type->specialize(
            slang_spec_args.data(),
            narrow_cast<SlangInt>(slang_spec_args.size()),
            specialized.writeRef(),
            diagnostics.writeRef()
        )
    );
    report_diagnostics(diagnostics);
    SGL_CHECK(SLANG_SUCCEEDED(result), "Failed to specialize component type");
    SGL_CHECK(specialized, "Failed to specialize component type");

    auto ct = make_ref<SlangComponentType>(m_session, std::move(specialized));
    m_session->_register_standalone_component_type(ct.get());
    return ct;
}

ref<SlangComponentType> SlangComponentType::link() const
{
    Slang::ComPtr<slang::IComponentType> linked;
    Slang::ComPtr<ISlangBlob> diagnostics;
    SGL_CATCH_INTERNAL_SLANG_ERROR(m_component_type->link(linked.writeRef(), diagnostics.writeRef()));
    report_diagnostics(diagnostics);
    SGL_CHECK(linked, "Failed to link component type");

    auto ct = make_ref<SlangComponentType>(m_session, std::move(linked));
    m_session->_register_standalone_component_type(ct.get());
    return ct;
}

ref<SlangComponentType> SlangComponentType::link_with_options(const SlangLinkOptions& link_options) const
{
    CompilerOptionEntries link_option_entries;

    if (link_options.floating_point_mode)
        link_option_entries.add(slang::CompilerOptionName::FloatingPointMode, int(*link_options.floating_point_mode));
    if (link_options.debug_info)
        link_option_entries.add(slang::CompilerOptionName::DebugInformation, int(*link_options.debug_info));
    if (link_options.optimization)
        link_option_entries.add(slang::CompilerOptionName::Optimization, int(*link_options.optimization));
    // TODO: Downstream args are hardcoded to "dxc" here because SlangComponentType
    // doesn't have access to the device type. This is incorrect for non-D3D12 backends.
    // See ShaderProgram::link() for the correct device-type-aware handling.
    if (link_options.downstream_args) {
        for (const auto& arg : *link_options.downstream_args)
            link_option_entries.add(slang::CompilerOptionName::DownstreamArgs, "dxc", arg);
    }
    if (link_options.dump_intermediates)
        link_option_entries.add(slang::CompilerOptionName::DumpIntermediates, *link_options.dump_intermediates);
    if (link_options.dump_intermediates_prefix)
        link_option_entries.add(
            slang::CompilerOptionName::DumpIntermediatePrefix,
            *link_options.dump_intermediates_prefix
        );

    Slang::ComPtr<slang::IComponentType> linked;
    Slang::ComPtr<ISlangBlob> diagnostics;
    auto slang_link_option_entries = link_option_entries.slang_entries();
    SGL_CATCH_INTERNAL_SLANG_ERROR(m_component_type->linkWithOptions(
        linked.writeRef(),
        narrow_cast<uint32_t>(slang_link_option_entries.size()),
        slang_link_option_entries.data(),
        diagnostics.writeRef()
    ));
    report_diagnostics(diagnostics);
    SGL_CHECK(linked, "Failed to link component type");

    auto ct = make_ref<SlangComponentType>(m_session, std::move(linked));
    m_session->_register_standalone_component_type(ct.get());
    return ct;
}

uint32_t SlangComponentType::specialization_param_count() const
{
    return narrow_cast<uint32_t>(m_component_type->getSpecializationParamCount());
}

uint32_t SlangComponentType::entry_point_count() const
{
    slang::ProgramLayout* slang_layout;
    SGL_CATCH_INTERNAL_SLANG_ERROR(slang_layout = m_component_type->getLayout());
    return narrow_cast<uint32_t>(slang_layout->getEntryPointCount());
}

std::string SlangComponentType::to_string() const
{
    return fmt::format("SlangComponentType(entry_point_count={})", entry_point_count());
}

SlangModule::SlangModule(ref<SlangSession> session, const SlangModuleDesc& desc)
    : SlangComponentType(std::move(session), nullptr)
    , m_desc(desc)
{
    m_session->_register_module(this);
}

SlangModule::~SlangModule()
{
    m_session->_unregister_module(this);
}

void SlangModule::load(SlangSessionBuild& build_data) const
{
    Timer timer;
    Slang::ComPtr<ISlangBlob> diagnostics;
    slang::IModule* slang_module;

    const SlangModuleDesc& desc = m_desc;
    const SlangSessionData* session_data = build_data.session.get();

    // Load module either from resolved name or source depending on whether source specified
    if (!desc.source.has_value()) {
        std::string resolved_name = session_data->resolve_module_name(desc.module_name);
        SGL_CATCH_INTERNAL_SLANG_ERROR(
            slang_module = session_data->slang_session->loadModule(resolved_name.c_str(), diagnostics.writeRef());
        );
        if (!slang_module) {
            std::string msg
                = append_diagnostics(fmt::format("Failed to load slang module \"{}\"", desc.module_name), diagnostics);
            throw SlangCompileError(msg);
        }
    } else {
        // TODO workaround: slang doesn't like loading the same source twice
        static std::atomic<uint32_t> id{0};
        std::string source_str = fmt::format("// {}\n{}", id++, desc.source);

        SGL_CATCH_INTERNAL_SLANG_ERROR(
            slang_module = session_data->slang_session->loadModuleFromSourceString(
                std::string{desc.module_name}.c_str(),
                desc.path ? desc.path->string().c_str() : nullptr,
                source_str.c_str(),
                diagnostics.writeRef()
            )
        );
        if (!slang_module) {
            std::string msg = append_diagnostics(
                fmt::format("Failed to load slang module \"{}\" from source", desc.module_name),
                diagnostics
            );
            throw SlangCompileError(msg);
        }
    }

    report_diagnostics(diagnostics);
    log_debug("Loading slang module \"{}\" took {}", desc.module_name, string::format_duration(timer.elapsed_s()));


    auto data = make_ref<SlangModuleData>();

    // Store initialized module info.
    data->slang_module = slang_module;
    data->name = slang_module->getName();
    data->path = slang_module->getFilePath() ? slang_module->getFilePath() : "";

    // Output the built module.
    build_data.modules[this] = std::move(data);

    // Build all registered entry points.
    for (auto entry_point : m_registered_entry_points) {
        entry_point->init(build_data);
    }
}

void SlangModule::store_built_data(SlangSessionBuild& build_data)
{
    m_data = build_data.modules[this];
    m_component_type = m_data->slang_module;
    for (auto ep : m_registered_entry_points)
        ep->store_built_data(build_data);
}

void SlangModule::populate_build_data(SlangSessionBuild& build_data)
{
    build_data.modules[this] = m_data;
    for (auto ep : m_registered_entry_points)
        ep->populate_build_data(build_data);
}

slang::TypeReflection* SlangModule::find_type_by_name(std::string_view name) const
{
    slang::ProgramLayout* layout;
    SGL_CATCH_INTERNAL_SLANG_ERROR(layout = m_data->slang_module->getLayout());
    return layout->findTypeByName(std::string(name).c_str());
}

std::vector<std::filesystem::path> SlangModule::dependency_file_paths() const
{
    std::vector<std::filesystem::path> result;
    SlangInt32 count = m_data->slang_module->getDependencyFileCount();
    for (SlangInt32 i = 0; i < count; i++) {
        const char* path = m_data->slang_module->getDependencyFilePath(i);
        if (path)
            result.emplace_back(path);
    }
    return result;
}

std::vector<ref<SlangEntryPoint>> SlangModule::entry_points() const
{
    std::vector<ref<SlangEntryPoint>> entry_points;
    for (SlangInt32 i = 0; i < m_data->slang_module->getDefinedEntryPointCount(); ++i) {
        Slang::ComPtr<slang::IEntryPoint> slang_entry_point;
        SGL_CATCH_INTERNAL_SLANG_ERROR(m_data->slang_module->getDefinedEntryPoint(i, slang_entry_point.writeRef()););

        slang::EntryPointLayout* layout;
        SGL_CATCH_INTERNAL_SLANG_ERROR(layout = slang_entry_point->getLayout()->getEntryPointByIndex(0););
        std::string name = layout->getNameOverride() ? layout->getNameOverride() : layout->getName();

        entry_points.push_back(entry_point(name));
    }
    return entry_points;
}

ref<SlangEntryPoint> SlangModule::entry_point(std::string_view name) const
{
    SlangEntryPointDesc desc;
    desc.name = name;

    auto entry_point = make_ref<SlangEntryPoint>(ref(const_cast<SlangModule*>(this)), desc);

    // Setup build containing just the session and this module, then build and store the entry point.
    SlangSessionBuild build;
    build.session = session()->_data();
    build.modules[this] = m_data;
    entry_point->init(build);
    entry_point->store_built_data(build);

    return entry_point;
}

bool SlangModule::has_entry_point(std::string_view name) const
{
    Slang::ComPtr<slang::IEntryPoint> slang_entry_point;
    SGL_CATCH_INTERNAL_SLANG_ERROR(
        m_data->slang_module->findEntryPointByName(std::string{name}.c_str(), slang_entry_point.writeRef());
    );
    return slang_entry_point != nullptr;
}

ref<const DeclReflection> SlangModule::module_decl() const
{
    return detail::from_slang(ref(this), m_data->slang_module->getModuleReflection());
}

void SlangModule::_register_entry_point(SlangEntryPoint* entry_point) const
{
    m_registered_entry_points.insert(entry_point);
}

void SlangModule::_unregister_entry_point(SlangEntryPoint* entry_point) const
{
    m_registered_entry_points.erase(entry_point);
}

std::string SlangModule::to_string() const
{
    return fmt::format(
        "SlangModule(\n"
        "  name = {},\n"
        "  path = {},\n"
        "  entry_points = {}\n"
        ")",
        m_data->name,
        m_data->path,
        string::indent(string::list_to_string(entry_points()))
    );
}


SlangEntryPoint::SlangEntryPoint(ref<SlangModule> module, const SlangEntryPointDesc& desc)
    : SlangComponentType(ref<SlangSession>(module->session()), nullptr)
    , m_module(module)
    , m_desc(desc)
{
    m_module->_register_entry_point(this);
}

SlangEntryPoint::~SlangEntryPoint()
{
    m_module->_unregister_entry_point(this);
}

void SlangEntryPoint::init(SlangSessionBuild& build_data) const
{
    const SlangEntryPointDesc& desc = m_desc;

    SlangModuleData* module_data = build_data.modules[m_module.get()];
    slang::IModule* slang_module = module_data->slang_module;

    auto data = make_ref<SlangEntryPointData>();

    // Find the entry point from its module.
    Slang::ComPtr<slang::IEntryPoint> slang_entry_point;
    SGL_CATCH_INTERNAL_SLANG_ERROR(
        slang_module->findEntryPointByName(std::string{desc.name}.c_str(), slang_entry_point.writeRef());
    );
    if (!slang_entry_point)
        SGL_THROW("Entry point \"{}\" not found", desc.name);
    data->slang_entry_point = std::move(slang_entry_point);

    // If we have specialization arguments, specialize the entry point.
    if (!desc.specialization_args.empty()) {
        slang::ProgramLayout* layout;
        SGL_CATCH_INTERNAL_SLANG_ERROR(layout = slang_module->getLayout());

        // Build the slang specialization args from our descriptors.
        std::vector<slang::SpecializationArg> slang_spec_args;
        slang_spec_args.reserve(desc.specialization_args.size());

        for (const SpecializationArg& arg : desc.specialization_args) {
            slang::SpecializationArg slang_arg;
            switch (arg.kind) {
            case SpecializationArgKind::type: {
                slang::TypeReflection* type = layout->findTypeByName(arg.value.c_str());
                SGL_CHECK(type, "Specialization type \"{}\" not found", arg.value);
                slang_arg = slang::SpecializationArg::fromType(type);
                break;
            }
            case SpecializationArgKind::expr:
                slang_arg = slang::SpecializationArg::fromExpr(arg.value.c_str());
                break;
            case SpecializationArgKind::unknown:
                SGL_THROW("Invalid specialization argument: kind is 'unknown'");
            }
            slang_spec_args.push_back(slang_arg);
        }

        // Specialize the entry point.
        Slang::ComPtr<slang::IComponentType> specialized_entry_point;
        Slang::ComPtr<ISlangBlob> diagnostics;
        SlangResult result;
        SGL_CATCH_INTERNAL_SLANG_ERROR(
            result = data->slang_entry_point->specialize(
                slang_spec_args.data(),
                narrow_cast<SlangInt>(slang_spec_args.size()),
                specialized_entry_point.writeRef(),
                diagnostics.writeRef()
            )
        );
        report_diagnostics(diagnostics);
        SGL_CHECK(SLANG_SUCCEEDED(result), "Failed to specialize generic entry point \"{}\"", desc.name);
        SGL_CHECK(specialized_entry_point, "Failed to specialize generic entry point \"{}\"", desc.name);

        data->slang_entry_point = std::move(specialized_entry_point);
    }

    // Read name and stage from the entry point.
    slang::EntryPointLayout* layout;
    SGL_CATCH_INTERNAL_SLANG_ERROR(layout = data->slang_entry_point->getLayout()->getEntryPointByIndex(0););
    data->name = layout->getNameOverride() ? layout->getNameOverride() : layout->getName();
    data->stage = static_cast<ShaderStage>(layout->getStage());

    // Output the built entry point.
    build_data.entry_points[this] = data;
}

void SlangEntryPoint::store_built_data(SlangSessionBuild& build_data)
{
    m_data = build_data.entry_points[this];
    m_component_type = m_data->slang_entry_point;
}

void SlangEntryPoint::populate_build_data(SlangSessionBuild& build_data)
{
    build_data.entry_points[this] = m_data;
}

ref<SlangEntryPoint> SlangEntryPoint::rename(const std::string& new_name)
{
    return with_name(new_name);
}

ref<SlangEntryPoint> SlangEntryPoint::with_name(const std::string& name) const
{
    Slang::ComPtr<slang::IComponentType> renamed_entry_point;
    SLANG_CALL(m_data->slang_entry_point->renameEntryPoint(name.c_str(), renamed_entry_point.writeRef()));

    SlangEntryPointDesc desc = m_desc;
    desc.name = name;
    auto ep = make_ref<SlangEntryPoint>(m_module, desc);

    // Setup build containing just the session and this module, then build and store the entry point.
    SlangSessionBuild build_data;
    build_data.session = m_module->session()->_data();
    build_data.modules[m_module.get()] = m_module->_data();
    ep->init(build_data);
    ep->store_built_data(build_data);

    return ep;
}

ref<SlangEntryPoint> SlangEntryPoint::specialize(std::span<const SpecializationArg> specialization_args) const
{
    SlangEntryPointDesc desc = m_desc;
    desc.specialization_args.assign(specialization_args.begin(), specialization_args.end());

    auto ep = make_ref<SlangEntryPoint>(m_module, desc);

    // Setup build containing just the session and this module, then build and store the entry point.
    SlangSessionBuild build_data;
    build_data.session = m_module->session()->_data();
    build_data.modules[m_module.get()] = m_module->_data();
    ep->init(build_data);
    ep->store_built_data(build_data);

    return ep;
}

ref<const EntryPointLayout> SlangEntryPoint::layout() const
{
    slang::EntryPointLayout* ep_layout;
    SGL_CATCH_INTERNAL_SLANG_ERROR(ep_layout = m_data->slang_entry_point->getLayout()->getEntryPointByIndex(0););
    return EntryPointLayout::from_slang(ref(this), ep_layout);
}

std::string SlangEntryPoint::to_string() const
{
    return fmt::format("SlangEntryPoint(name=\"{}\", stage={})", m_data->name, m_data->stage);
}

// ----------------------------------------------------------------------------
// SlangTypeConformance
// ----------------------------------------------------------------------------

SlangTypeConformance::SlangTypeConformance(
    ref<SlangSession> session,
    Slang::ComPtr<slang::IComponentType> component_type,
    TypeConformance conformance
)
    : SlangComponentType(std::move(session), std::move(component_type))
    , m_conformance(std::move(conformance))
{
    m_session->_register_type_conformance(this);
}

SlangTypeConformance::~SlangTypeConformance()
{
    m_session->_unregister_type_conformance(this);
}

void SlangTypeConformance::rebuild(SlangSessionBuild& build) const
{
    slang::ISession* slang_session = build.session->slang_session;

    // Look up types by searching all loaded modules in the build session.
    slang::TypeReflection* type = nullptr;
    slang::TypeReflection* interface_type = nullptr;
    for (const auto& [module_ptr, module_data] : build.modules) {
        slang::ProgramLayout* layout;
        SGL_CATCH_INTERNAL_SLANG_ERROR(layout = module_data->slang_module->getLayout());
        if (!type)
            type = layout->findTypeByName(m_conformance.type_name.c_str());
        if (!interface_type)
            interface_type = layout->findTypeByName(m_conformance.interface_name.c_str());
        if (type && interface_type)
            break;
    }
    SGL_CHECK(type, "Type \"{}\" not found during hot reload", m_conformance.type_name);
    SGL_CHECK(interface_type, "Interface type \"{}\" not found during hot reload", m_conformance.interface_name);

    Slang::ComPtr<slang::ITypeConformance> slang_conformance;
    Slang::ComPtr<ISlangBlob> diagnostics;
    SGL_CATCH_INTERNAL_SLANG_ERROR(slang_session->createTypeConformanceComponentType(
        type,
        interface_type,
        slang_conformance.writeRef(),
        m_conformance.id,
        diagnostics.writeRef()
    ));
    report_diagnostics(diagnostics);
    SGL_CHECK(
        slang_conformance,
        "Failed to rebuild type conformance for type \"{}\" and interface \"{}\"",
        m_conformance.type_name,
        m_conformance.interface_name
    );

    build.type_conformances[this] = Slang::ComPtr<slang::IComponentType>(slang_conformance.get());
}

void SlangTypeConformance::store_built_data(SlangSessionBuild& build)
{
    m_component_type = build.type_conformances[this];
    m_valid = true;
}

void SlangTypeConformance::populate_build_data(SlangSessionBuild& build) const
{
    build.type_conformances[this] = m_component_type;
}

std::string SlangTypeConformance::to_string() const
{
    return fmt::format(
        "SlangTypeConformance(interface=\"{}\", type=\"{}\")",
        m_conformance.interface_name,
        m_conformance.type_name
    );
}

ShaderProgram::ShaderProgram(ref<Device> device, ref<SlangSession> session, const ShaderProgramDesc& desc)
    : DeviceChild(std::move(device))
    , m_session(std::move(session))
    , m_desc(desc)
{
    m_session->_register_program(this);
}

ShaderProgram::~ShaderProgram()
{
    m_session->_unregister_program(this);
}

void ShaderProgram::link(SlangSessionBuild& build_data) const
{
    Device* device = m_device;
    const ShaderProgramDesc& desc = m_desc;
    slang::ISession* session = build_data.session->slang_session;

    Timer timer;

    // Compose the program from its components: modules + entry points + type conformances.
    Slang::ComPtr<slang::IComponentType> composed_program;
    {
        std::vector<slang::IComponentType*> component_types;
        for (const auto& module : desc.modules) {
            const SlangModuleData* module_data = build_data.modules[module.get()];
            component_types.push_back(module_data->slang_module);
        }

        for (const auto& entry_point : desc.entry_points) {
            const SlangEntryPointData* entry_point_data = build_data.entry_points[entry_point.get()];
            component_types.push_back(entry_point_data->slang_entry_point);
        }

        for (const auto& tc : desc.type_conformances) {
            auto it = build_data.type_conformances.find(tc.get());
            if (it != build_data.type_conformances.end())
                component_types.push_back(it->second.get());
            else
                component_types.push_back(tc->slang_component_type());
        }

        composed_program = create_slang_composite(session, component_types);
    }

    // Setup link options.
    CompilerOptionEntries link_option_entries;
    if (desc.link_options) {
        const SlangLinkOptions& link_options = desc.link_options.value();

        if (link_options.floating_point_mode)
            link_option_entries.add(
                slang::CompilerOptionName::FloatingPointMode,
                int(*link_options.floating_point_mode)
            );
        if (link_options.debug_info)
            link_option_entries.add(slang::CompilerOptionName::DebugInformation, int(*link_options.debug_info));
        if (link_options.optimization)
            link_option_entries.add(slang::CompilerOptionName::Optimization, int(*link_options.optimization));
        if (link_options.downstream_args) {
            if (device->type() == DeviceType::d3d12) {
                for (const auto& arg : *link_options.downstream_args)
                    link_option_entries.add(slang::CompilerOptionName::DownstreamArgs, "dxc", arg);
            }
        }
        if (link_options.dump_intermediates)
            link_option_entries.add(slang::CompilerOptionName::DumpIntermediates, *link_options.dump_intermediates);
        if (link_options.dump_intermediates_prefix)
            link_option_entries.add(
                slang::CompilerOptionName::DumpIntermediatePrefix,
                *link_options.dump_intermediates_prefix
            );
    }

    // Link the composed program.
    Slang::ComPtr<slang::IComponentType> linked_program;
    {
        Slang::ComPtr<ISlangBlob> diagnostics;
        auto slang_link_option_entries = link_option_entries.slang_entries();
        SGL_CATCH_INTERNAL_SLANG_ERROR(composed_program->linkWithOptions(
            linked_program.writeRef(),
            narrow_cast<uint32_t>(slang_link_option_entries.size()),
            slang_link_option_entries.data(),
            diagnostics.writeRef()
        ));
        if (!linked_program) {
            std::string msg = append_diagnostics("Failed to link program", diagnostics);
            throw SlangCompileError(msg);
        }
        report_diagnostics(diagnostics);
    }

    // Create shader program.
    Slang::ComPtr<rhi::IShaderProgram> rhi_shader_program;
    {
        rhi::ShaderProgramDesc rhi_desc{
            .slangGlobalScope = linked_program,
            .label = desc.label.empty() ? nullptr : desc.label.c_str(),
        };

        Slang::ComPtr<ISlangBlob> diagnostics;
        if (device->rhi_device()->createShaderProgram(rhi_desc, rhi_shader_program.writeRef(), diagnostics.writeRef())
            != SLANG_OK) {
            std::string msg = append_diagnostics("Failed to create shader program", diagnostics);
            SGL_THROW(msg);
        }
        report_diagnostics(diagnostics);
    }

    // Report link time.
    log_debug("Linking shader program \"{}\" took {}", desc.label, string::format_duration(timer.elapsed_s()));

    auto data = make_ref<ShaderProgramData>();

    // Store program info.
    data->linked_program = linked_program;
    data->rhi_shader_program = rhi_shader_program;

    // Store resulting program.
    build_data.programs[this] = std::move(data);
}

void ShaderProgram::store_built_data(SlangSessionBuild& build_data)
{
    // Store built program data
    m_data = build_data.programs[this];

    // Notify all registered pipelines that this program has rebuilt.
    for (auto pipeline : m_registered_pipelines)
        pipeline->notify_program_reloaded();

    // Add all hashed strings from this program to debug printer.
    if (m_device->debug_printer()) {
        m_device->debug_printer()->add_hashed_strings(layout()->hashed_strings_map());
    }
}

void ShaderProgram::_register_pipeline(Pipeline* pipeline)
{
    m_registered_pipelines.insert(pipeline);
}

void ShaderProgram::_unregister_pipeline(Pipeline* pipeline)
{
    m_registered_pipelines.erase(pipeline);
}

ref<SlangComponentType> ShaderProgram::linked_component_type() const
{
    auto ct = make_ref<SlangComponentType>(ref<SlangSession>(m_session.get()), m_data->linked_program);
    m_session->_register_standalone_component_type(ct.get());
    return ct;
}

std::string ShaderProgram::to_string() const
{
    return fmt::format(
        "ShaderProgram(\n"
        "  modules = {},\n"
        "  entry_points = {},\n"
        "  label = {}\n"
        ")",
        string::indent(string::list_to_string(m_desc.modules)),
        string::indent(string::list_to_string(m_desc.entry_points)),
        m_desc.label
    );
}

} // namespace sgl
