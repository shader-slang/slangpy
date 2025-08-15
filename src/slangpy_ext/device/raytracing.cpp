// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/raytracing.h"
#include "sgl/device/query.h"
#include "sgl/device/shader.h"

namespace sgl {

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureInstanceDesc)
SGL_DICT_TO_DESC_FIELD(transform, float3x4)
SGL_DICT_TO_DESC_FIELD(instance_id, uint32_t)
SGL_DICT_TO_DESC_FIELD(instance_mask, uint32_t)
SGL_DICT_TO_DESC_FIELD(instance_contribution_to_hit_group_index, uint32_t)
SGL_DICT_TO_DESC_FIELD(flags, AccelerationStructureInstanceFlags)
SGL_DICT_TO_DESC_FIELD(acceleration_structure, AccelerationStructureHandle)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildInputInstances)
SGL_DICT_TO_DESC_FIELD(instance_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(instance_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD(instance_count, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildInputTriangles)
SGL_DICT_TO_DESC_FIELD_LIST(vertex_buffers, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(vertex_format, Format)
SGL_DICT_TO_DESC_FIELD(vertex_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(vertex_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD(index_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(index_format, IndexFormat)
SGL_DICT_TO_DESC_FIELD(index_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(pre_transform_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(flags, AccelerationStructureGeometryFlags)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildInputProceduralPrimitives)
SGL_DICT_TO_DESC_FIELD_LIST(aabb_buffers, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(aabb_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD(primitive_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(flags, AccelerationStructureGeometryFlags)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildInputMotionOptions)
SGL_DICT_TO_DESC_FIELD(key_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(time_start, float)
SGL_DICT_TO_DESC_FIELD(time_end, float)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildDesc)
SGL_DICT_TO_DESC_FIELD_LIST(inputs, AccelerationStructureBuildInput)
SGL_DICT_TO_DESC_FIELD(motion_options, AccelerationStructureBuildInputMotionOptions)
SGL_DICT_TO_DESC_FIELD(mode, AccelerationStructureBuildMode)
SGL_DICT_TO_DESC_FIELD(flags, AccelerationStructureBuildFlags)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureQueryDesc)
SGL_DICT_TO_DESC_FIELD(query_type, QueryType)
SGL_DICT_TO_DESC_FIELD(query_pool, ref<QueryPool>)
SGL_DICT_TO_DESC_FIELD(first_query_index, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureDesc)
SGL_DICT_TO_DESC_FIELD(size, DeviceSize)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(ShaderTableDesc)
SGL_DICT_TO_DESC_FIELD(program, ref<ShaderProgram>)
SGL_DICT_TO_DESC_FIELD_LIST(ray_gen_entry_points, std::string)
SGL_DICT_TO_DESC_FIELD_LIST(miss_entry_points, std::string)
SGL_DICT_TO_DESC_FIELD_LIST(hit_group_names, std::string)
SGL_DICT_TO_DESC_FIELD_LIST(callable_entry_points, std::string)
SGL_DICT_TO_DESC_END()

} // namespace sgl

SGL_PY_EXPORT(device_raytracing)
{
    using namespace sgl;

    nb::class_<AccelerationStructureHandle>(m, "AccelerationStructureHandle", "Acceleration structure handle.")
        .def(nb::init<>());

    nb::sgl_enum_flags<AccelerationStructureGeometryFlags>(m, "AccelerationStructureGeometryFlags");
    nb::sgl_enum_flags<AccelerationStructureInstanceFlags>(m, "AccelerationStructureInstanceFlags");

    nb::class_<AccelerationStructureInstanceDesc>(
        m,
        "AccelerationStructureInstanceDesc",
        D(AccelerationStructureInstanceDesc)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureInstanceDesc* self, nb::dict dict)
            { new (self) AccelerationStructureInstanceDesc(dict_to_AccelerationStructureInstanceDesc(dict)); }
        )
        .def_rw(
            "transform",
            &AccelerationStructureInstanceDesc::transform,
            D(AccelerationStructureInstanceDesc, transform)
        )
        .def_prop_rw(
            "instance_id",
            [](AccelerationStructureInstanceDesc& self) { return self.instance_id; },
            [](AccelerationStructureInstanceDesc& self, uint32_t value) { self.instance_id = value; },
            D(AccelerationStructureInstanceDesc, instance_id)
        )
        .def_prop_rw(
            "instance_mask",
            [](AccelerationStructureInstanceDesc& self) { return self.instance_mask; },
            [](AccelerationStructureInstanceDesc& self, uint32_t value) { self.instance_mask = value; },
            D(AccelerationStructureInstanceDesc, instance_mask)
        )
        .def_prop_rw(
            "instance_contribution_to_hit_group_index",
            [](AccelerationStructureInstanceDesc& self) { return self.instance_contribution_to_hit_group_index; },
            [](AccelerationStructureInstanceDesc& self, uint32_t value)
            { self.instance_contribution_to_hit_group_index = value; },
            D(AccelerationStructureInstanceDesc, instance_contribution_to_hit_group_index)
        )
        .def_prop_rw(
            "flags",
            [](AccelerationStructureInstanceDesc& self) { return self.flags; },
            [](AccelerationStructureInstanceDesc& self, AccelerationStructureInstanceFlags value)
            { self.flags = value; },
            D(AccelerationStructureInstanceDesc, flags)
        )
        .def_rw(
            "acceleration_structure",
            &AccelerationStructureInstanceDesc::acceleration_structure,
            D(AccelerationStructureInstanceDesc, acceleration_structure)
        )
        .def(
            "to_numpy",
            [](AccelerationStructureInstanceDesc& self)
            {
                size_t shape[1] = {64};
                return nb::ndarray<nb::numpy, const uint8_t, nb::shape<64>>(&self, 1, shape, nb::handle());
            }
        );
    nb::implicitly_convertible<nb::dict, AccelerationStructureInstanceDesc>();

    nb::class_<AccelerationStructureBuildInputInstances>(
        m,
        "AccelerationStructureBuildInputInstances",
        D(AccelerationStructureBuildInputInstances)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildInputInstances* self, nb::dict dict) {
                new (self)
                    AccelerationStructureBuildInputInstances(dict_to_AccelerationStructureBuildInputInstances(dict));
            }
        )
        .def_rw("instance_buffer", &AccelerationStructureBuildInputInstances::instance_buffer)
        .def_rw("instance_stride", &AccelerationStructureBuildInputInstances::instance_stride)
        .def_rw("instance_count", &AccelerationStructureBuildInputInstances::instance_count);
    nb::implicitly_convertible<nb::dict, AccelerationStructureBuildInputInstances>();

    nb::class_<AccelerationStructureBuildInputTriangles>(
        m,
        "AccelerationStructureBuildInputTriangles",
        D(AccelerationStructureBuildInputTriangles)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildInputTriangles* self, nb::dict dict) {
                new (self)
                    AccelerationStructureBuildInputTriangles(dict_to_AccelerationStructureBuildInputTriangles(dict));
            }
        )
        .def_rw("vertex_buffers", &AccelerationStructureBuildInputTriangles::vertex_buffers)
        .def_rw("vertex_format", &AccelerationStructureBuildInputTriangles::vertex_format)
        .def_rw("vertex_count", &AccelerationStructureBuildInputTriangles::vertex_count)
        .def_rw("vertex_stride", &AccelerationStructureBuildInputTriangles::vertex_stride)
        .def_rw("index_buffer", &AccelerationStructureBuildInputTriangles::index_buffer)
        .def_rw("index_format", &AccelerationStructureBuildInputTriangles::index_format)
        .def_rw("index_count", &AccelerationStructureBuildInputTriangles::index_count)
        .def_rw("pre_transform_buffer", &AccelerationStructureBuildInputTriangles::pre_transform_buffer)
        .def_rw("flags", &AccelerationStructureBuildInputTriangles::flags);
    nb::implicitly_convertible<nb::dict, AccelerationStructureBuildInputTriangles>();

    nb::class_<AccelerationStructureBuildInputProceduralPrimitives>(
        m,
        "AccelerationStructureBuildInputProceduralPrimitives",
        D(AccelerationStructureBuildInputProceduralPrimitives)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildInputProceduralPrimitives* self, nb::dict dict)
            {
                new (self) AccelerationStructureBuildInputProceduralPrimitives(
                    dict_to_AccelerationStructureBuildInputProceduralPrimitives(dict)
                );
            }
        )
        .def_rw("aabb_buffers", &AccelerationStructureBuildInputProceduralPrimitives::aabb_buffers)
        .def_rw("aabb_stride", &AccelerationStructureBuildInputProceduralPrimitives::aabb_stride)
        .def_rw("primitive_count", &AccelerationStructureBuildInputProceduralPrimitives::primitive_count)
        .def_rw("flags", &AccelerationStructureBuildInputProceduralPrimitives::flags);
    nb::implicitly_convertible<nb::dict, AccelerationStructureBuildInputProceduralPrimitives>();

    // nb::class_<AccelerationStructureBuildInput>(
    //     m,
    //     "AccelerationStructureBuildInput",
    //     D(AccelerationStructureBuildInput)
    // );

    nb::class_<AccelerationStructureBuildInputMotionOptions>(
        m,
        "AccelerationStructureBuildInputMotionOptions",
        D(AccelerationStructureBuildInputMotionOptions)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildInputMotionOptions* self, nb::dict dict) {
                new (self) AccelerationStructureBuildInputMotionOptions(
                    dict_to_AccelerationStructureBuildInputMotionOptions(dict)
                );
            }
        )
        .def_rw("key_count", &AccelerationStructureBuildInputMotionOptions::key_count)
        .def_rw("time_start", &AccelerationStructureBuildInputMotionOptions::time_start)
        .def_rw("time_end", &AccelerationStructureBuildInputMotionOptions::time_end);
    nb::implicitly_convertible<nb::dict, AccelerationStructureBuildInputMotionOptions>();

    nb::sgl_enum<AccelerationStructureBuildMode>(m, "AccelerationStructureBuildMode");
    nb::sgl_enum_flags<AccelerationStructureBuildFlags>(m, "AccelerationStructureBuildFlags");

    nb::class_<AccelerationStructureBuildDesc>(m, "AccelerationStructureBuildDesc", D(AccelerationStructureBuildDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildDesc* self, nb::dict dict)
            { new (self) AccelerationStructureBuildDesc(dict_to_AccelerationStructureBuildDesc(dict)); }
        )
        .def_rw("inputs", &AccelerationStructureBuildDesc::inputs, D(AccelerationStructureBuildDesc, inputs))
        .def_rw(
            "motion_options",
            &AccelerationStructureBuildDesc::motion_options,
            D(AccelerationStructureBuildDesc, motion_options)
        )
        .def_rw("mode", &AccelerationStructureBuildDesc::mode, D(AccelerationStructureBuildDesc, mode))
        .def_rw("flags", &AccelerationStructureBuildDesc::flags, D(AccelerationStructureBuildDesc, flags));
    nb::implicitly_convertible<nb::dict, AccelerationStructureBuildDesc>();

    nb::sgl_enum<AccelerationStructureCopyMode>(m, "AccelerationStructureCopyMode");

    nb::class_<AccelerationStructureSizes>(m, "AccelerationStructureSizes", D(AccelerationStructureSizes))
        .def_rw(
            "acceleration_structure_size",
            &AccelerationStructureSizes::acceleration_structure_size,
            D(AccelerationStructureSizes, acceleration_structure_size)
        )
        .def_rw("scratch_size", &AccelerationStructureSizes::scratch_size, D(AccelerationStructureSizes, scratch_size))
        .def_rw(
            "update_scratch_size",
            &AccelerationStructureSizes::update_scratch_size,
            D(AccelerationStructureSizes, update_scratch_size)
        );

    nb::class_<AccelerationStructureQueryDesc>(m, "AccelerationStructureQueryDesc", D(AccelerationStructureQueryDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureQueryDesc* self, nb::dict dict)
            { new (self) AccelerationStructureQueryDesc(dict_to_AccelerationStructureQueryDesc(dict)); }
        )
        .def_rw(
            "query_type",
            &AccelerationStructureQueryDesc::query_type,
            D(AccelerationStructureQueryDesc, query_type)
        )
        .def_rw(
            "query_pool",
            &AccelerationStructureQueryDesc::query_pool,
            D(AccelerationStructureQueryDesc, query_pool)
        )
        .def_rw(
            "first_query_index",
            &AccelerationStructureQueryDesc::first_query_index,
            D(AccelerationStructureQueryDesc, first_query_index)
        );
    nb::implicitly_convertible<nb::dict, AccelerationStructureQueryDesc>();

    nb::class_<AccelerationStructureDesc>(m, "AccelerationStructureDesc", D(AccelerationStructureDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureDesc* self, nb::dict dict)
            { new (self) AccelerationStructureDesc(dict_to_AccelerationStructureDesc(dict)); }
        )
        .def_rw("size", &AccelerationStructureDesc::size, D(AccelerationStructureDesc, size))
        .def_rw("label", &AccelerationStructureDesc::label, D(AccelerationStructureDesc, label));
    nb::implicitly_convertible<nb::dict, AccelerationStructureDesc>();

    nb::class_<AccelerationStructure, DeviceChild>(m, "AccelerationStructure", D(AccelerationStructure))
        .def_prop_ro("desc", &AccelerationStructure::desc, D(AccelerationStructure, desc))
        .def_prop_ro("handle", &AccelerationStructure::handle, D(AccelerationStructure, handle));

    nb::class_<AccelerationStructureInstanceList, DeviceChild>(
        m,
        "AccelerationStructureInstanceList",
        D(AccelerationStructureInstanceList)
    )
        .def_prop_ro("size", &AccelerationStructureInstanceList::size, D(AccelerationStructureInstanceList, size))
        .def_prop_ro(
            "instance_stride",
            &AccelerationStructureInstanceList::instance_stride,
            D(AccelerationStructureInstanceList, instance_stride)
        )
        .def(
            "resize",
            &AccelerationStructureInstanceList::resize,
            "size"_a,
            D(AccelerationStructureInstanceList, resize)
        )
        .def(
            "write",
            nb::overload_cast<size_t, const AccelerationStructureInstanceDesc&>(
                &AccelerationStructureInstanceList::write
            ),
            "index"_a,
            "instance"_a,
            D(AccelerationStructureInstanceList, write)
        )
        .def(
            "write",
            nb::overload_cast<size_t, std::span<AccelerationStructureInstanceDesc>>(
                &AccelerationStructureInstanceList::write
            ),
            "index"_a,
            "instances"_a,
            D(AccelerationStructureInstanceList, write, 2)
        )
        .def("buffer", &AccelerationStructureInstanceList::buffer, D(AccelerationStructureInstanceList, buffer))
        .def(
            "build_input_instances",
            &AccelerationStructureInstanceList::build_input_instances,
            D(AccelerationStructureInstanceList, build_input_instances)
        );

    nb::class_<ShaderTableDesc>(m, "ShaderTableDesc", D(ShaderTableDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](ShaderTableDesc* self, nb::dict dict) { new (self) ShaderTableDesc(dict_to_ShaderTableDesc(dict)); }
        )
        .def_rw("program", &ShaderTableDesc::program, D(ShaderTableDesc, program))
        .def_rw(
            "ray_gen_entry_points",
            &ShaderTableDesc::ray_gen_entry_points,
            D(ShaderTableDesc, ray_gen_entry_points)
        )
        .def_rw("miss_entry_points", &ShaderTableDesc::miss_entry_points, D(ShaderTableDesc, miss_entry_points))
        .def_rw("hit_group_names", &ShaderTableDesc::hit_group_names, D(ShaderTableDesc, hit_group_names))
        .def_rw(
            "callable_entry_points",
            &ShaderTableDesc::callable_entry_points,
            D(ShaderTableDesc, callable_entry_points)
        );
    nb::implicitly_convertible<nb::dict, ShaderTableDesc>();

    nb::class_<ShaderTable, DeviceChild>(m, "ShaderTable", D(ShaderTable));
}
