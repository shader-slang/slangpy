// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"
#include "sgl/device/shader.h"

#include <algorithm>
#include <cstddef>

namespace sgl::detail {

using PyCompilationReport = nb::typed<nb::dict, nb::str, nb::any>;
using PyCompilationReportList = nb::typed<nb::list, PyCompilationReport>;

template<size_t N>
nb::str fixed_string_to_py_str(const char (&value)[N])
{
    const char* end = std::find(value, value + N, '\0');
    return nb::str(value, static_cast<size_t>(end - value));
}

inline const char* compilation_pipeline_type_to_string(rhi::CompilationReport::PipelineType type)
{
    switch (type) {
    case rhi::CompilationReport::PipelineType::Render:
        return "render";
    case rhi::CompilationReport::PipelineType::Compute:
        return "compute";
    case rhi::CompilationReport::PipelineType::RayTracing:
        return "ray_tracing";
    default:
        return "unknown";
    }
}

inline nb::dict
compilation_entry_point_report_to_dict(const rhi::CompilationReport::EntryPointReport& entry_point_report)
{
    nb::dict result;
    result["name"] = fixed_string_to_py_str(entry_point_report.name);
    result["start_time"] = entry_point_report.startTime;
    result["end_time"] = entry_point_report.endTime;
    result["create_time"] = entry_point_report.createTime;
    result["compile_time"] = entry_point_report.compileTime;
    result["compile_slang_time"] = entry_point_report.compileSlangTime;
    result["compile_downstream_time"] = entry_point_report.compileDownstreamTime;
    result["is_cached"] = entry_point_report.isCached;
    result["cache_size"] = entry_point_report.cacheSize;
    return result;
}

inline nb::dict compilation_pipeline_report_to_dict(const rhi::CompilationReport::PipelineReport& pipeline_report)
{
    nb::dict result;
    result["type"] = compilation_pipeline_type_to_string(pipeline_report.type);
    result["start_time"] = pipeline_report.startTime;
    result["end_time"] = pipeline_report.endTime;
    result["create_time"] = pipeline_report.createTime;
    result["is_cached"] = pipeline_report.isCached;
    result["cache_size"] = pipeline_report.cacheSize;
    return result;
}

inline PyCompilationReport compilation_report_to_dict(const rhi::CompilationReport& report)
{
    SGL_CHECK(
        report.entryPointReportCount == 0 || report.entryPointReports != nullptr,
        "Invalid compilation report entry point data"
    );
    SGL_CHECK(
        report.pipelineReportCount == 0 || report.pipelineReports != nullptr,
        "Invalid compilation report pipeline data"
    );

    nb::list entry_point_reports;
    for (uint32_t i = 0; i < report.entryPointReportCount; ++i)
        entry_point_reports.append(compilation_entry_point_report_to_dict(report.entryPointReports[i]));

    nb::list pipeline_reports;
    for (uint32_t i = 0; i < report.pipelineReportCount; ++i)
        pipeline_reports.append(compilation_pipeline_report_to_dict(report.pipelineReports[i]));

    nb::dict result;
    result["label"] = fixed_string_to_py_str(report.label);
    result["alive"] = report.alive;
    result["create_time"] = report.createTime;
    result["compile_time"] = report.compileTime;
    result["compile_slang_time"] = report.compileSlangTime;
    result["compile_downstream_time"] = report.compileDownstreamTime;
    result["create_pipeline_time"] = report.createPipelineTime;
    result["entry_point_reports"] = std::move(entry_point_reports);
    result["pipeline_reports"] = std::move(pipeline_reports);
    return PyCompilationReport(std::move(result));
}

inline void validate_compilation_report_blob(ISlangBlob* report_blob)
{
    SGL_CHECK_NOT_NULL(report_blob);
    SGL_CHECK(report_blob->getBufferSize() >= sizeof(rhi::CompilationReport), "Compilation report blob is too small");

    const auto* report = static_cast<const rhi::CompilationReport*>(report_blob->getBufferPointer());
    size_t expected_size = sizeof(rhi::CompilationReport);
    expected_size += report->entryPointReportCount * sizeof(rhi::CompilationReport::EntryPointReport);
    expected_size += report->pipelineReportCount * sizeof(rhi::CompilationReport::PipelineReport);
    SGL_CHECK(
        report_blob->getBufferSize() == expected_size,
        "Compilation report blob has unexpected size (expected {}, got {})",
        expected_size,
        report_blob->getBufferSize()
    );
}

inline void validate_compilation_report_list_blob(ISlangBlob* report_list_blob)
{
    SGL_CHECK_NOT_NULL(report_list_blob);
    SGL_CHECK(
        report_list_blob->getBufferSize() >= sizeof(rhi::CompilationReportList),
        "Compilation report list blob is too small"
    );

    const auto* report_list = static_cast<const rhi::CompilationReportList*>(report_list_blob->getBufferPointer());
    SGL_CHECK(report_list->reportCount == 0 || report_list->reports != nullptr, "Invalid compilation report list data");

    size_t expected_size = sizeof(rhi::CompilationReportList);
    for (uint32_t i = 0; i < report_list->reportCount; ++i) {
        const rhi::CompilationReport& report = report_list->reports[i];
        expected_size += sizeof(rhi::CompilationReport);
        expected_size += report.entryPointReportCount * sizeof(rhi::CompilationReport::EntryPointReport);
        expected_size += report.pipelineReportCount * sizeof(rhi::CompilationReport::PipelineReport);
    }
    SGL_CHECK(
        report_list_blob->getBufferSize() == expected_size,
        "Compilation report list blob has unexpected size (expected {}, got {})",
        expected_size,
        report_list_blob->getBufferSize()
    );
}

inline PyCompilationReport get_compilation_report(ShaderProgram* program)
{
    SGL_ASSERT(program);
    SGL_ASSERT(program->rhi_shader_program());

    Slang::ComPtr<ISlangBlob> report_blob;
    SLANG_RHI_CALL(program->rhi_shader_program()->getCompilationReport(report_blob.writeRef()), program->device());
    validate_compilation_report_blob(report_blob);
    return compilation_report_to_dict(*static_cast<const rhi::CompilationReport*>(report_blob->getBufferPointer()));
}

inline PyCompilationReportList get_compilation_reports(Device* device)
{
    SGL_ASSERT(device);
    SGL_ASSERT(device->rhi_device());

    Slang::ComPtr<ISlangBlob> report_list_blob;
    SLANG_RHI_CALL(device->rhi_device()->getCompilationReportList(report_list_blob.writeRef()), device);
    validate_compilation_report_list_blob(report_list_blob);

    const auto* report_list = static_cast<const rhi::CompilationReportList*>(report_list_blob->getBufferPointer());
    nb::list result;
    for (uint32_t i = 0; i < report_list->reportCount; ++i)
        result.append(compilation_report_to_dict(report_list->reports[i]));
    return PyCompilationReportList(std::move(result));
}

} // namespace sgl::detail
