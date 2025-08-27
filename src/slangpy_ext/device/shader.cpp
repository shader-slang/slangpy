// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/reflection.h"
#include "sgl/device/kernel.h"

namespace sgl {
using DefineList = std::map<std::string, std::string>;
SGL_DICT_TO_DESC_BEGIN(SlangCompilerOptions)
SGL_DICT_TO_DESC_FIELD_LIST(include_paths, std::filesystem::path)
SGL_DICT_TO_DESC_FIELD(defines, DefineList)
SGL_DICT_TO_DESC_FIELD(shader_model, ShaderModel)
SGL_DICT_TO_DESC_FIELD(matrix_layout, SlangMatrixLayout)
SGL_DICT_TO_DESC_FIELD_LIST(enable_warnings, std::string)
SGL_DICT_TO_DESC_FIELD_LIST(disable_warnings, std::string)
SGL_DICT_TO_DESC_FIELD_LIST(warnings_as_errors, std::string)
SGL_DICT_TO_DESC_FIELD(report_downstream_time, bool)
SGL_DICT_TO_DESC_FIELD(report_perf_benchmark, bool)
SGL_DICT_TO_DESC_FIELD(skip_spirv_validation, bool)
SGL_DICT_TO_DESC_FIELD(floating_point_mode, SlangFloatingPointMode)
SGL_DICT_TO_DESC_FIELD(debug_info, SlangDebugInfoLevel)
SGL_DICT_TO_DESC_FIELD(optimization, SlangOptimizationLevel)
SGL_DICT_TO_DESC_FIELD_LIST(downstream_args, std::string)
SGL_DICT_TO_DESC_FIELD(dump_intermediates, bool)
SGL_DICT_TO_DESC_FIELD(dump_intermediates_prefix, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(SlangLinkOptions)
SGL_DICT_TO_DESC_FIELD(floating_point_mode, SlangFloatingPointMode)
SGL_DICT_TO_DESC_FIELD(debug_info, SlangDebugInfoLevel)
SGL_DICT_TO_DESC_FIELD(optimization, SlangOptimizationLevel)
// TODO(slang-rhi) cannot use SGL_DICT_TO_DESC_FIELD because downstream_args is std::optional<std::vector<std::string>>
SGL_DICT_TO_DESC_FIELD(downstream_args, std::vector<std::string>)
SGL_DICT_TO_DESC_FIELD(dump_intermediates, bool)
SGL_DICT_TO_DESC_FIELD(dump_intermediates_prefix, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(SlangSessionDesc)
SGL_DICT_TO_DESC_FIELD(compiler_options, SlangCompilerOptions)
SGL_DICT_TO_DESC_FIELD(add_default_include_paths, bool)
SGL_DICT_TO_DESC_FIELD(cache_path, std::filesystem::path)
SGL_DICT_TO_DESC_END()

} // namespace sgl

NB_MAKE_OPAQUE(std::map<std::string, std::string, std::less<>>);
NB_MAKE_OPAQUE(std::vector<sgl::TypeConformance>);

SGL_PY_EXPORT(device_shader)
{
    using namespace sgl;

    nb::class_<TypeConformance>(m, "TypeConformance", D(TypeConformance))
        .def(nb::init<>())
        .def(nb::init<std::string, std::string, int32_t>(), "interface_name"_a, "type_name"_a, "id"_a = -1)
        .def(
            "__init__",
            [](TypeConformance* self, nb::tuple tuple)
            {
                new (self) TypeConformance{
                    nb::cast<std::string>(tuple[0]),
                    nb::cast<std::string>(tuple[1]),
                    tuple.size() > 2 ? nb::cast<int32_t>(tuple[2]) : -1};
            }
        )
        .def_rw("interface_name", &TypeConformance::interface_name, D(TypeConformance, interface_name))
        .def_rw("type_name", &TypeConformance::type_name, D(TypeConformance, type_name))
        .def_rw("id", &TypeConformance::id, D(TypeConformance, id))
        .def("__repr__", &TypeConformance::to_string);
    nb::implicitly_convertible<nb::tuple, TypeConformance>();

    nb::exception<SlangCompileError>(m, "SlangCompileError");

    nb::sgl_enum<SlangMatrixLayout>(m, "SlangMatrixLayout");
    nb::sgl_enum<sgl::SlangFloatingPointMode>(m, "SlangFloatingPointMode");
    nb::sgl_enum<sgl::SlangDebugInfoLevel>(m, "SlangDebugInfoLevel");
    nb::sgl_enum<sgl::SlangOptimizationLevel>(m, "SlangOptimizationLevel");

    nb::class_<SlangCompilerOptions>(m, "SlangCompilerOptions", D(SlangCompilerOptions))
        .def(nb::init<>())
        .def(
            "__init__",
            [](SlangCompilerOptions* self, nb::dict dict)
            { new (self) SlangCompilerOptions(dict_to_SlangCompilerOptions(dict)); }
        )
        .def_rw("include_paths", &SlangCompilerOptions::include_paths, D(SlangCompilerOptions, include_paths))
        .def_rw("defines", &SlangCompilerOptions::defines, D(SlangCompilerOptions, defines))
        .def_rw("shader_model", &SlangCompilerOptions::shader_model, D(SlangCompilerOptions, shader_model))
        .def_rw("matrix_layout", &SlangCompilerOptions::matrix_layout, D(SlangCompilerOptions, matrix_layout))
        .def_rw("enable_warnings", &SlangCompilerOptions::enable_warnings, D(SlangCompilerOptions, enable_warnings))
        .def_rw("disable_warnings", &SlangCompilerOptions::disable_warnings, D(SlangCompilerOptions, disable_warnings))
        .def_rw(
            "warnings_as_errors",
            &SlangCompilerOptions::warnings_as_errors,
            D(SlangCompilerOptions, warnings_as_errors)
        )
        .def_rw(
            "report_downstream_time",
            &SlangCompilerOptions::report_downstream_time,
            D(SlangCompilerOptions, report_downstream_time)
        )
        .def_rw(
            "report_perf_benchmark",
            &SlangCompilerOptions::report_perf_benchmark,
            D(SlangCompilerOptions, report_perf_benchmark)
        )
        .def_rw(
            "skip_spirv_validation",
            &SlangCompilerOptions::skip_spirv_validation,
            D(SlangCompilerOptions, skip_spirv_validation)
        )
        .def_rw(
            "floating_point_mode",
            &SlangCompilerOptions::floating_point_mode,
            D(SlangCompilerOptions, floating_point_mode)
        )
        .def_rw("debug_info", &SlangCompilerOptions::debug_info, D(SlangCompilerOptions, debug_info))
        .def_rw("optimization", &SlangCompilerOptions::optimization, D(SlangCompilerOptions, optimization))
        .def_rw("downstream_args", &SlangCompilerOptions::downstream_args, D(SlangCompilerOptions, downstream_args))
        .def_rw(
            "dump_intermediates",
            &SlangCompilerOptions::dump_intermediates,
            D(SlangCompilerOptions, dump_intermediates)
        )
        .def_rw(
            "dump_intermediates_prefix",
            &SlangCompilerOptions::dump_intermediates_prefix,
            D(SlangCompilerOptions, dump_intermediates_prefix)
        );
    nb::implicitly_convertible<nb::dict, SlangCompilerOptions>();

    nb::class_<SlangLinkOptions>(m, "SlangLinkOptions", D(SlangLinkOptions))
        .def(nb::init<>())
        .def(
            "__init__",
            [](SlangLinkOptions* self, nb::dict dict) { new (self) SlangLinkOptions(dict_to_SlangLinkOptions(dict)); }
        )
        .def_rw(
            "floating_point_mode",
            &SlangLinkOptions::floating_point_mode,
            nb::none(),
            D(SlangLinkOptions, floating_point_mode)
        )
        .def_rw("debug_info", &SlangLinkOptions::debug_info, nb::none(), D(SlangLinkOptions, debug_info))
        .def_rw("optimization", &SlangLinkOptions::optimization, nb::none(), D(SlangLinkOptions, optimization))
        .def_rw("downstream_args", &SlangLinkOptions::downstream_args, nb::none(), D(SlangLinkOptions, downstream_args))
        .def_rw(
            "dump_intermediates",
            &SlangLinkOptions::dump_intermediates,
            nb::none(),
            D(SlangLinkOptions, dump_intermediates)
        )
        .def_rw(
            "dump_intermediates_prefix",
            &SlangLinkOptions::dump_intermediates_prefix,
            nb::none(),
            D(SlangLinkOptions, dump_intermediates_prefix)
        );
    nb::implicitly_convertible<nb::dict, SlangLinkOptions>();

    nb::class_<SlangSessionDesc>(m, "SlangSessionDesc", D(SlangSessionDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](SlangSessionDesc* self, nb::dict dict) { new (self) SlangSessionDesc(dict_to_SlangSessionDesc(dict)); }
        )
        .def_rw("compiler_options", &SlangSessionDesc::compiler_options, D(SlangSessionDesc, compiler_options))
        .def_rw(
            "add_default_include_paths",
            &SlangSessionDesc::add_default_include_paths,
            D(SlangSessionDesc, add_default_include_paths)
        )
        .def_rw("cache_path", &SlangSessionDesc::cache_path, nb::none(), D(SlangSessionDesc, cache_path));
    nb::implicitly_convertible<nb::dict, SlangSessionDesc>();

    // Disambiguate from the types in slang.h
    using sgl::SlangSession;
    using sgl::SlangEntryPoint;

    nb::class_<SlangSession, Object>(m, "SlangSession", D(SlangSession))
        .def_prop_ro("device", &SlangSession::device, D(SlangSession, device))
        .def_prop_ro("desc", &SlangSession::desc, D(SlangSession, desc))
        .def("load_module", &SlangSession::load_module, "module_name"_a, D(SlangSession, load_module))
        .def(
            "load_module_from_source",
            &SlangSession::load_module_from_source,
            "module_name"_a,
            "source"_a,
            "path"_a.none() = nb::none(),
            D(SlangSession, load_module_from_source)
        )
        .def(
            "link_program",
            &SlangSession::link_program,
            "modules"_a,
            "entry_points"_a,
            "link_options"_a.none() = nb::none(),
            D(SlangSession, link_program)
        )
        .def(
            "load_program",
            &SlangSession::load_program,
            "module_name"_a,
            "entry_point_names"_a,
            "additional_source"_a.none() = nb::none(),
            "link_options"_a.none() = nb::none(),
            D(SlangSession, load_program)
        )
        .def("load_source", &SlangSession::load_source, "module_name"_a, D(SlangSession, load_source));

    nb::class_<SlangModule, Object>(m, "SlangModule", D(SlangModule))
        .def_prop_ro("session", &SlangModule::session, D(SlangModule, session))
        .def_prop_ro("name", &SlangModule::name, D(SlangModule, name))
        .def_prop_ro("path", &SlangModule::path, D(SlangModule, path))
        .def_prop_ro("layout", &SlangModule::layout, D(SlangModule, layout))
        .def_prop_ro("entry_points", &SlangModule::entry_points, D(SlangModule, entry_points))
        .def_prop_ro("module_decl", &SlangModule::module_decl, D(SlangModule, module_decl))
        .def(
            "entry_point",
            &SlangModule::entry_point,
            "name"_a,
            "type_conformances"_a = std::span<TypeConformance>(),
            D(SlangModule, entry_point)
        );

    nb::class_<SlangEntryPoint, Object>(m, "SlangEntryPoint", D(SlangEntryPoint))
        .def_prop_ro("name", &SlangEntryPoint::name, D(SlangEntryPoint, name))
        .def_prop_ro("stage", &SlangEntryPoint::stage, D(SlangEntryPoint, stage))
        .def_prop_ro("layout", &SlangEntryPoint::layout, D(SlangEntryPoint, layout))
        .def("rename", &SlangEntryPoint::rename, "new_name"_a, D(SlangEntryPoint, rename))
        .def("with_name", &SlangEntryPoint::with_name, "new_name"_a, D(SlangEntryPoint, with_name));

    nb::class_<ShaderProgram, DeviceChild>(m, "ShaderProgram", D(ShaderProgram))
        .def_prop_ro("layout", &ShaderProgram::layout, D(ShaderProgram, layout))
        .def_prop_ro("reflection", &ShaderProgram::reflection, D(ShaderProgram, reflection));
}
