// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/pipeline.h"
#include "sgl/device/shader.h"
#include "sgl/device/input_layout.h"

namespace sgl {
SGL_DICT_TO_DESC_BEGIN(ComputePipelineDesc)
SGL_DICT_TO_DESC_FIELD(program, ref<ShaderProgram>)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(RenderPipelineDesc)
SGL_DICT_TO_DESC_FIELD(program, ref<ShaderProgram>)
SGL_DICT_TO_DESC_FIELD(input_layout, ref<InputLayout>)
SGL_DICT_TO_DESC_FIELD(primitive_topology, PrimitiveTopology)
SGL_DICT_TO_DESC_FIELD(targets, std::vector<ColorTargetDesc>)
SGL_DICT_TO_DESC_FIELD(depth_stencil, DepthStencilDesc)
SGL_DICT_TO_DESC_FIELD(rasterizer, RasterizerDesc)
SGL_DICT_TO_DESC_FIELD(multisample, MultisampleDesc)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(HitGroupDesc)
SGL_DICT_TO_DESC_FIELD(hit_group_name, std::string)
SGL_DICT_TO_DESC_FIELD(closest_hit_entry_point, std::string)
SGL_DICT_TO_DESC_FIELD(any_hit_entry_point, std::string)
SGL_DICT_TO_DESC_FIELD(intersection_entry_point, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(RayTracingPipelineDesc)
SGL_DICT_TO_DESC_FIELD(program, ref<ShaderProgram>)
SGL_DICT_TO_DESC_FIELD_LIST(hit_groups, HitGroupDesc)
SGL_DICT_TO_DESC_FIELD(max_recursion, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_ray_payload_size, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_attribute_size, uint32_t)
SGL_DICT_TO_DESC_FIELD(flags, RayTracingPipelineFlags)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()
} // namespace sgl

SGL_PY_EXPORT(device_pipeline)
{
    using namespace sgl;

    nb::class_<Pipeline, DeviceChild>(m, "Pipeline", D(Pipeline));

    nb::class_<ComputePipelineDesc>(m, "ComputePipelineDesc", D(ComputePipelineDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](ComputePipelineDesc* self, nb::dict dict)
            { new (self) ComputePipelineDesc(dict_to_ComputePipelineDesc(dict)); }
        )
        .def_rw("program", &ComputePipelineDesc::program, D(ComputePipelineDesc, program))
        .def_rw("label", &ComputePipelineDesc::label, D(ComputePipelineDesc, label));
    nb::implicitly_convertible<nb::dict, ComputePipelineDesc>();

    nb::class_<ComputePipeline, Pipeline>(m, "ComputePipeline", D(ComputePipeline))
        .def_prop_ro("thread_group_size", &ComputePipeline::thread_group_size, D(ComputePipeline, thread_group_size))
        .def_prop_ro("native_handle", &ComputePipeline::native_handle, D(ComputePipeline, native_handle));

    nb::class_<RenderPipelineDesc>(m, "RenderPipelineDesc", D(RenderPipelineDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](RenderPipelineDesc* self, nb::dict dict)
            { new (self) RenderPipelineDesc(dict_to_RenderPipelineDesc(dict)); }
        )
        .def_rw("program", &RenderPipelineDesc::program, D(RenderPipelineDesc, program))
        .def_rw("input_layout", &RenderPipelineDesc::input_layout, D(RenderPipelineDesc, input_layout))
        .def_rw(
            "primitive_topology",
            &RenderPipelineDesc::primitive_topology,
            D(RenderPipelineDesc, primitive_topology)
        )
        .def_rw("targets", &RenderPipelineDesc::targets, D(RenderPipelineDesc, targets))
        .def_rw("depth_stencil", &RenderPipelineDesc::depth_stencil, D(RenderPipelineDesc, depth_stencil))
        .def_rw("rasterizer", &RenderPipelineDesc::rasterizer, D(RenderPipelineDesc, rasterizer))
        .def_rw("multisample", &RenderPipelineDesc::multisample, D(RenderPipelineDesc, multisample))
        .def_rw("label", &RenderPipelineDesc::label, D(RenderPipelineDesc, label));

    nb::class_<RenderPipeline, Pipeline>(m, "RenderPipeline", D(RenderPipeline)) //
        .def_prop_ro("native_handle", &RenderPipeline::native_handle, D(RenderPipeline, native_handle));

    nb::class_<HitGroupDesc>(m, "HitGroupDesc", D(HitGroupDesc))
        .def(nb::init<>())
        .def("__init__", [](HitGroupDesc* self, nb::dict dict) { new (self) HitGroupDesc(dict_to_HitGroupDesc(dict)); })
        .def(
            nb::init<std::string, std::string, std::string, std::string>(),
            "hit_group_name"_a,
            "closest_hit_entry_point"_a = "",
            "any_hit_entry_point"_a = "",
            "intersection_entry_point"_a = ""
        )
        .def_rw("hit_group_name", &HitGroupDesc::hit_group_name, D(HitGroupDesc, hit_group_name))
        .def_rw(
            "closest_hit_entry_point",
            &HitGroupDesc::closest_hit_entry_point,
            D(HitGroupDesc, closest_hit_entry_point)
        )
        .def_rw("any_hit_entry_point", &HitGroupDesc::any_hit_entry_point, D(HitGroupDesc, any_hit_entry_point))
        .def_rw(
            "intersection_entry_point",
            &HitGroupDesc::intersection_entry_point,
            D(HitGroupDesc, intersection_entry_point)
        );
    nb::implicitly_convertible<nb::dict, HitGroupDesc>();

    nb::class_<RayTracingPipelineDesc>(m, "RayTracingPipelineDesc", D(RayTracingPipelineDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](RayTracingPipelineDesc* self, nb::dict dict)
            { new (self) RayTracingPipelineDesc(dict_to_RayTracingPipelineDesc(dict)); }
        )
        .def_rw("program", &RayTracingPipelineDesc::program, D(RayTracingPipelineDesc, program))
        .def_ro(
            "hit_groups",
            &RayTracingPipelineDesc::hit_groups,
            nb::rv_policy::reference_internal,
            D(RayTracingPipelineDesc, hit_groups)
        )
        .def_rw("max_recursion", &RayTracingPipelineDesc::max_recursion, D(RayTracingPipelineDesc, max_recursion))
        .def_rw(
            "max_ray_payload_size",
            &RayTracingPipelineDesc::max_ray_payload_size,
            D(RayTracingPipelineDesc, max_ray_payload_size)
        )
        .def_rw(
            "max_attribute_size",
            &RayTracingPipelineDesc::max_attribute_size,
            D(RayTracingPipelineDesc, max_attribute_size)
        )
        .def_rw("flags", &RayTracingPipelineDesc::flags, D(RayTracingPipelineDesc, flags))
        .def_rw("label", &RayTracingPipelineDesc::label, D(RayTracingPipelineDesc, label));
    nb::implicitly_convertible<nb::dict, RayTracingPipelineDesc>();

    nb::class_<RayTracingPipeline, Pipeline>(m, "RayTracingPipeline", D(RayTracingPipeline)) //
        .def_prop_ro("native_handle", &RayTracingPipeline::native_handle, D(RayTracingPipeline, native_handle));
}
