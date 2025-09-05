// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"
#include "sgl/device/reflection.h"
#include "sgl/device/device_child.h"

#include "sgl/core/object.h"
#include "sgl/core/enum.h"

#include <exception>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <slang.h>

namespace sgl {

/// \brief Type conformance entry.
/// Type conformances are used to narrow the set of types supported by a slang interface.
/// They can be specified on an entry point to omit generating code for types that do not conform.
struct SGL_API TypeConformance {
    /// Name of the interface.
    std::string interface_name;
    /// Name of the concrete type.
    std::string type_name;
    /// Unique id per type for an interface (optional).
    int32_t id{-1};

    TypeConformance() = default;
    TypeConformance(std::string_view interface_name, std::string_view type_name, int32_t id = -1)
        : interface_name(interface_name)
        , type_name(type_name)
        , id(id)
    {
    }

    std::string to_string() const;
};

/// Exception thrown on compilation errors.
class SlangCompileError : public std::runtime_error {
public:
    SlangCompileError(const std::string& what)
        : std::runtime_error(what)
    {
    }
};

/// Slang matrix layout modes.
enum class SlangMatrixLayout {
    row_major = SLANG_MATRIX_LAYOUT_ROW_MAJOR,
    column_major = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
};

SGL_ENUM_INFO(
    SlangMatrixLayout,
    {
        {SlangMatrixLayout::row_major, "row_major"},
        {SlangMatrixLayout::column_major, "column_major"},
    }
);
SGL_ENUM_REGISTER(SlangMatrixLayout);

/// Slang floating point modes.
enum class SlangFloatingPointMode {
    default_ = SLANG_FLOATING_POINT_MODE_DEFAULT,
    fast = SLANG_FLOATING_POINT_MODE_FAST,
    precise = SLANG_FLOATING_POINT_MODE_PRECISE,
};

SGL_ENUM_INFO(
    SlangFloatingPointMode,
    {
        {SlangFloatingPointMode::default_, "default"},
        {SlangFloatingPointMode::fast, "fast"},
        {SlangFloatingPointMode::precise, "precise"},
    }
);
SGL_ENUM_REGISTER(SlangFloatingPointMode);

/// Slang debug info levels.
enum class SlangDebugInfoLevel {
    /// No debug information.
    none = SLANG_DEBUG_INFO_LEVEL_NONE,
    /// Emit as little debug information as possible, while still supporting stack trackes.
    minimal = SLANG_DEBUG_INFO_LEVEL_MINIMAL,
    /// Emit whatever is the standard level of debug information for each target.
    standard = SLANG_DEBUG_INFO_LEVEL_STANDARD,
    /// Emit as much debug infromation as possible for each target.
    maximal = SLANG_DEBUG_INFO_LEVEL_MAXIMAL,
};

SGL_ENUM_INFO(
    SlangDebugInfoLevel,
    {
        {SlangDebugInfoLevel::none, "none"},
        {SlangDebugInfoLevel::minimal, "minimal"},
        {SlangDebugInfoLevel::standard, "standard"},
        {SlangDebugInfoLevel::maximal, "maximal"},
    }
);
SGL_ENUM_REGISTER(SlangDebugInfoLevel);

/// Slang optimization levels.
enum class SlangOptimizationLevel {
    /// No optimizations.
    none = SLANG_OPTIMIZATION_LEVEL_NONE,
    /// Default optimization level: balance code quality and compilation time.
    default_ = SLANG_OPTIMIZATION_LEVEL_DEFAULT,
    /// Optimize aggressively.
    high = SLANG_OPTIMIZATION_LEVEL_HIGH,
    /// Include optimizations that may take a very long time, or may involve severe space-vs-speed tradeoffs.
    maximal = SLANG_OPTIMIZATION_LEVEL_MAXIMAL,
};

SGL_ENUM_INFO(
    SlangOptimizationLevel,
    {
        {SlangOptimizationLevel::none, "none"},
        {SlangOptimizationLevel::default_, "default"},
        {SlangOptimizationLevel::high, "high"},
        {SlangOptimizationLevel::maximal, "maximal"},
    }
);
SGL_ENUM_REGISTER(SlangOptimizationLevel);

/// Slang compiler options.
/// Can be set when creating a Slang session.
struct SlangCompilerOptions {
    /// Specifies a list of include paths to be used when resolving module/include paths.
    std::vector<std::filesystem::path> include_paths;

    /// Specifies a list of preprocessor defines.
    std::map<std::string, std::string> defines;

    /// Specifies the shader model to use.
    /// Defaults to latest available on the device.
    ShaderModel shader_model{ShaderModel::unknown};

    /// Specifies the matrix layout.
    /// Defaults to row-major.
    SlangMatrixLayout matrix_layout{SlangMatrixLayout::row_major};

    /// Specifies a list of warnings to disable (warning codes or names).
    std::vector<std::string> disable_warnings;

    /// Specifies a list of warnings to enable (warning codes or names).
    std::vector<std::string> enable_warnings;

    /// Specifies a list of warnings to be treated as errors
    /// (warning codes or names, or "all" to indicate all warnings).
    std::vector<std::string> warnings_as_errors;

    /// Turn on/off downstream compilation time report.
    bool report_downstream_time{false};

    /// Turn on/off reporting of time spend in different parts of the compiler.
    bool report_perf_benchmark{false};

    /// Specifies whether or not to skip the validation step after emitting SPIRV.
    bool skip_spirv_validation{false};

    //
    // Target options.
    // These can be extended/overriden at link time.
    //

    /// Specifies the floating point mode.
    SlangFloatingPointMode floating_point_mode{SlangFloatingPointMode::default_};

    /// Specifies the level of debug information to include in the generated code.
    SlangDebugInfoLevel debug_info{SlangDebugInfoLevel::none};

    /// Specifies the optimization level.
    SlangOptimizationLevel optimization{SlangOptimizationLevel::default_};

    /// Specifies a list of additional arguments to be passed to the downstream compiler.
    std::vector<std::string> downstream_args;

    /// When set will dump the intermediate source output.
    bool dump_intermediates{false};

    /// The file name prefix for the intermediate source output.
    std::string dump_intermediates_prefix;
};

/// Slang link options.
/// These can optionally be set when linking a shader program.
struct SlangLinkOptions {
    /// Specifies the floating point mode.
    std::optional<SlangFloatingPointMode> floating_point_mode;

    /// Specifies the level of debug information to include in the generated code.
    std::optional<SlangDebugInfoLevel> debug_info;

    /// Specifies the optimization level.
    std::optional<SlangOptimizationLevel> optimization;

    /// Specifies a list of additional arguments to be passed to the downstream compiler.
    std::optional<std::vector<std::string>> downstream_args;

    /// When set will dump the intermediate source output.
    std::optional<bool> dump_intermediates;

    /// The file name prefix for the intermediate source output.
    std::optional<std::string> dump_intermediates_prefix;
};

struct SlangSessionData;
struct SlangModuleData;
struct SlangEntryPointData;
struct ShaderProgramData;

/// Intermediate structure used during a build that stores new session, module,
/// program and entry point information. This is populated during a build, then
/// applied to all built modules/entrypoints/programs at once on success.
struct SlangSessionBuild {
    ref<SlangSessionData> session;
    std::map<const SlangModule*, ref<SlangModuleData>> modules;
    std::map<const ShaderProgram*, ref<ShaderProgramData>> programs;
    std::map<const SlangEntryPoint*, ref<SlangEntryPointData>> entry_points;
};

/// Descriptor for slang session initialization.
struct SlangSessionDesc {
    SlangCompilerOptions compiler_options;
    bool add_default_include_paths{true};
    std::optional<std::filesystem::path> cache_path;
};

/// Internal data stored once the slang session has been created.
struct SlangSessionData : Object {
    SGL_OBJECT(SlangSessionData)

    /// Pointer to internal slang session.
    Slang::ComPtr<slang::ISession> slang_session;

    /// Set of all currently loaded slang modules.
    std::set<slang::IModule*> loaded_modules;

    /// Unique session hash.
    std::string uid;

    /// List of include paths used for resolving module/include paths.
    std::vector<std::filesystem::path> include_paths;

    /// True if session cache is enabled.
    bool cache_enabled{false};

    /// Cache root path.
    std::filesystem::path cache_path;

    /// One cache path for each include path under the root cache path.
    std::vector<std::filesystem::path> cache_include_paths;

    /// Finds fully qualified module name by scanning the cache and include paths.
    std::string resolve_module_name(std::string_view module_name) const;
};

/// A slang session, used to load modules and link programs.
class SGL_API SlangSession : public Object {
    SGL_OBJECT(SlangSession)
public:
    SlangSession(ref<Device> device, SlangSessionDesc desc);
    ~SlangSession();

    /// Fully recreates this session and any loaded modules or linked programs.
    void recreate_session();

    Device* device() const { return m_device; }
    const SlangSessionDesc& desc() const { return m_desc; }

    /// Load a module by name.
    ref<SlangModule> load_module(std::string_view module_name);

    /// Load a module from string source code.
    ref<SlangModule> load_module_from_source(
        std::string_view module_name,
        std::string_view source,
        std::optional<std::filesystem::path> path = {}
    );

    /// Link a program with a set of modules and entry points.
    ref<ShaderProgram> link_program(
        std::vector<ref<SlangModule>> modules,
        std::vector<ref<SlangEntryPoint>> entry_points,
        std::optional<SlangLinkOptions> link_options = {}
    );

    /// Load a program from a given module with a set of entry
    /// points. Internally this simply wraps link_program without
    /// requiring the user to explicitly load modules.
    ref<ShaderProgram> load_program(
        std::string_view module_name,
        std::vector<std::string_view> entry_point_names,
        std::optional<std::string_view> additional_source = {},
        std::optional<SlangLinkOptions> link_options = {}
    );

    /// Load the source code for a given module.
    std::string load_source(std::string_view module_name);

    slang::ISession* get_slang_session() const { return m_data->slang_session; }

    std::string to_string() const override;

    // Internal functions to link programs+modules to their owning session
    void _register_program(ShaderProgram* program);
    void _unregister_program(ShaderProgram* program);
    void _register_module(SlangModule* module);
    void _unregister_module(SlangModule* module);

    // Internal access to the built session data.
    ref<SlangSessionData> _data() { return m_data; }

private:
    ref<Device> m_device;

    /// Descriptor containing all info required to build the session.
    SlangSessionDesc m_desc;

    /// Data pointer contains all slang pointers + data resulting from build.
    ref<SlangSessionData> m_data;

    /// Global NVAPI module linked to all programs.
    ref<SlangModule> m_nvapi_module;

    /// All loaded sgl modules (wrappers around IModule returned from load_module).
    /// Note: this is a vector, as order of creation matters.
    std::vector<SlangModule*> m_registered_modules;

    /// All created sgl programs (via link_program)
    std::set<ShaderProgram*> m_registered_programs;

    void update_module_cache_and_dependencies();
    bool write_module_to_cache(slang::IModule* module);
    void create_session(SlangSessionBuild& build);
};

struct SlangModuleDesc {
    /// Required module name
    std::string module_name;

    /// Optional module source. If not specified slang module resolution is used.
    std::optional<std::string> source;

    /// If source specified, additional path for compilation.
    std::optional<std::filesystem::path> path;
};

struct SlangModuleData : Object {
    SGL_OBJECT(SlangModuleData)

    slang::IModule* slang_module = nullptr;
    std::string name;
    std::filesystem::path path;
};

class SGL_API SlangModule : public Object {
    SGL_OBJECT(SlangModule)
public:
    SlangModule(ref<SlangSession> session, const SlangModuleDesc& desc);
    ~SlangModule();

    // Deleted because the hot reload mechanism in SlangSession assumes only objects created by it exist.
    SGL_NON_COPYABLE_AND_MOVABLE(SlangModule);

    /// Loads slang module and outputs the resulting SlangModuleData in current build info.
    void load(SlangSessionBuild& build) const;

    /// Finds this module in current build and updates internal m_data to point at it.
    void store_built_data(SlangSessionBuild& build_data);

    /// Repopulates a build structure with reference to internal m_data ptr.
    void populate_build_data(SlangSessionBuild& build_data);

    /// The session from which this module was built.
    SlangSession* session() const { return m_session; }

    /// Descriptor that holds all data required to create this module.
    const SlangModuleDesc& desc() const { return m_desc; }

    /// Module name.
    const std::string& name() const { return m_data->name; }

    /// Module source path. This can be empty if the module was generated from a string.
    const std::filesystem::path& path() const { return m_data->path; }
    ref<const ProgramLayout> layout() const
    {
        return ProgramLayout::from_slang(ref(this), m_data->slang_module->getLayout());
    }

    /// Build and return vector of all current entry points in the module.
    std::vector<ref<SlangEntryPoint>> entry_points() const;

    /// Get an entry point, optionally applying type conformances to it.
    ref<SlangEntryPoint> entry_point(
        std::string_view name,
        std::span<TypeConformance> type_conformances = std::span<TypeConformance>()
    ) const;
    bool has_entry_point(std::string_view name) const;

    /// Get root decl ref for this module
    ref<const DeclReflection> module_decl() const;

    /// Internal slang module.
    slang::IModule* slang_module() const { return m_data->slang_module; }

    /// Text summary of the module.
    std::string to_string() const override;

    /// Unlinks the session reference for modules that are referred to by the session to avoid ref loops.
    void break_strong_reference_to_session() { m_session.break_strong_reference(); }


    void _register_entry_point(SlangEntryPoint* entry_point) const;
    void _unregister_entry_point(SlangEntryPoint* entry_point) const;
    ref<SlangModuleData> _data() { return m_data; }

private:
    breakable_ref<SlangSession> m_session;
    SlangModuleDesc m_desc;
    ref<SlangModuleData> m_data;

    mutable std::set<SlangEntryPoint*> m_registered_entry_points;
};

struct SlangEntryPointDesc {
    std::string name;
    std::vector<TypeConformance> type_conformances;
};
struct SlangEntryPointData : Object {
    SGL_OBJECT(SlangEntryPointData)

    Slang::ComPtr<slang::IComponentType> slang_entry_point;
    std::string name;
    ShaderStage stage;
};
class SGL_API SlangEntryPoint : public Object {
    SGL_OBJECT(SlangEntryPoint)
public:
    SlangEntryPoint(ref<SlangModule> module, const SlangEntryPointDesc& desc);
    ~SlangEntryPoint();

    /// Inits slang entry point and outputs the resulting SlangEntryPointData in current build info.
    void init(SlangSessionBuild& build) const;

    /// Finds this entry point in current build and updates internal m_data to point at it.
    void store_built_data(SlangSessionBuild& build);

    /// Repopulates a build structure with reference to internal m_data ptr.
    void populate_build_data(SlangSessionBuild& build);

    SlangModule* module() const { return m_module; }

    const std::string& name() const { return m_data->name; }
    const SlangEntryPointDesc& desc() const { return m_desc; }
    ShaderStage stage() const { return m_data->stage; }
    ref<const EntryPointLayout> layout() const;

    ref<SlangEntryPoint> rename(const std::string& new_name);

    /// Returns a copy of the entry point with a new name.
    ref<SlangEntryPoint> with_name(const std::string& name) const;

    slang::IComponentType* slang_entry_point() const { return m_data->slang_entry_point.get(); }

    virtual std::string to_string() const override;

private:
    ref<SlangModule> m_module;
    SlangEntryPointDesc m_desc;
    ref<SlangEntryPointData> m_data;
};

struct ShaderProgramDesc {
    std::vector<ref<SlangModule>> modules;
    std::vector<ref<SlangEntryPoint>> entry_points;
    std::optional<SlangLinkOptions> link_options;
    std::string label;
};
struct ShaderProgramData : Object {
    SGL_OBJECT(ShaderProgramData)
    Slang::ComPtr<slang::IComponentType> linked_program;
    Slang::ComPtr<rhi::IShaderProgram> rhi_shader_program;
};
class SGL_API ShaderProgram : public DeviceChild {
    SGL_OBJECT(ShaderProgram)
public:
    ShaderProgram(ref<Device> device, ref<SlangSession> session, const ShaderProgramDesc& desc);
    ~ShaderProgram();

    // Deleted because the hot reload mechanism in SlangSession assumes only objects created by it exist.
    SGL_NON_COPYABLE_AND_MOVABLE(ShaderProgram);

    virtual void _release_rhi_resources() override
    {
        if (m_data)
            m_data->rhi_shader_program.setNull();
    }

    /// Links program and outputs the resulting ShaderProgramData to current build info.
    void link(SlangSessionBuild& build) const;

    /// Finds this program in current build and updates internal m_data to point at it.
    void store_built_data(SlangSessionBuild& build);

    const ShaderProgramDesc& desc() const { return m_desc; }

    ref<const ProgramLayout> layout() const
    {
        return ProgramLayout::from_slang(ref(this), m_data->linked_program->getLayout());
    }

    ReflectionCursor reflection() const { return ReflectionCursor(this); }

    rhi::IShaderProgram* rhi_shader_program() const { return m_data->rhi_shader_program; }

    virtual std::string to_string() const override;

    void _register_pipeline(Pipeline* pipeline);
    void _unregister_pipeline(Pipeline* pipeline);

private:
    ref<SlangSession> m_session;
    ShaderProgramDesc m_desc;
    ref<ShaderProgramData> m_data;
    std::set<Pipeline*> m_registered_pipelines;
};

} // namespace sgl
