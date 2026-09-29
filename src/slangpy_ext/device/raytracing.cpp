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

SGL_DICT_TO_DESC_BEGIN(MicromapUsageCount)
SGL_DICT_TO_DESC_FIELD(count, uint32_t)
SGL_DICT_TO_DESC_FIELD(subdivision_level, uint32_t)
SGL_DICT_TO_DESC_FIELD(format, OpacityMicromapFormat)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(MicromapBuildDesc)
SGL_DICT_TO_DESC_FIELD(type, MicromapType)
SGL_DICT_TO_DESC_FIELD(flags, MicromapBuildFlags)
SGL_DICT_TO_DESC_FIELD(data_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(descriptor_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(descriptor_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD_LIST(histogram, MicromapUsageCount)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(MicromapDesc)
SGL_DICT_TO_DESC_FIELD(type, MicromapType)
SGL_DICT_TO_DESC_FIELD(size, DeviceSize)
SGL_DICT_TO_DESC_FIELD(flags, MicromapBuildFlags)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureOpacityMicromapDesc)
SGL_DICT_TO_DESC_FIELD(micromap, ref<Micromap>)
SGL_DICT_TO_DESC_FIELD(indexing_mode, MicromapIndexingMode)
SGL_DICT_TO_DESC_FIELD(index_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(index_format, MicromapIndexFormat)
SGL_DICT_TO_DESC_FIELD(index_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD(base_micromap_index, uint32_t)
SGL_DICT_TO_DESC_FIELD_LIST(usage_counts, MicromapUsageCount)
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
SGL_DICT_TO_DESC_FIELD(opacity_micromap, std::optional<AccelerationStructureOpacityMicromapDesc>)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildInputProceduralPrimitives)
SGL_DICT_TO_DESC_FIELD_LIST(aabb_buffers, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(aabb_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD(primitive_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(flags, AccelerationStructureGeometryFlags)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildInputSpheres)
SGL_DICT_TO_DESC_FIELD(vertex_count, uint32_t)
SGL_DICT_TO_DESC_FIELD_LIST(vertex_position_buffers, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(vertex_position_format, Format)
SGL_DICT_TO_DESC_FIELD(vertex_position_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD_LIST(vertex_radius_buffers, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(vertex_radius_format, Format)
SGL_DICT_TO_DESC_FIELD(vertex_radius_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD(index_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(index_format, IndexFormat)
SGL_DICT_TO_DESC_FIELD(index_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(flags, AccelerationStructureGeometryFlags)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(AccelerationStructureBuildInputLinearSweptSpheres)
SGL_DICT_TO_DESC_FIELD(vertex_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(primitive_count, uint32_t)
SGL_DICT_TO_DESC_FIELD_LIST(vertex_position_buffers, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(vertex_position_format, Format)
SGL_DICT_TO_DESC_FIELD(vertex_position_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD_LIST(vertex_radius_buffers, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(vertex_radius_format, Format)
SGL_DICT_TO_DESC_FIELD(vertex_radius_stride, uint32_t)
SGL_DICT_TO_DESC_FIELD(index_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(index_format, IndexFormat)
SGL_DICT_TO_DESC_FIELD(index_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(indexing_mode, LinearSweptSpheresIndexingMode)
SGL_DICT_TO_DESC_FIELD(end_caps_mode, LinearSweptSpheresEndCapsMode)
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
SGL_DICT_TO_DESC_FIELD(kind, AccelerationStructureKind)
SGL_DICT_TO_DESC_FIELD(size, DeviceSize)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(ClusterOperationMoveParams)
SGL_DICT_TO_DESC_FIELD(type, ClusterOperationMoveType)
SGL_DICT_TO_DESC_FIELD(max_size, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(ClusterOperationClasBuildParams)
SGL_DICT_TO_DESC_FIELD(vertex_format, Format)
SGL_DICT_TO_DESC_FIELD(max_geometry_index, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_unique_geometry_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_triangle_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_vertex_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_total_triangle_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_total_vertex_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(min_position_truncate_bit_count, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(ClusterOperationBlasBuildParams)
SGL_DICT_TO_DESC_FIELD(max_clas_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(max_total_clas_count, uint32_t)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(ClusterOperationParams)
SGL_DICT_TO_DESC_FIELD(max_arg_count, uint32_t)
SGL_DICT_TO_DESC_FIELD(type, ClusterOperationType)
SGL_DICT_TO_DESC_FIELD(mode, ClusterOperationMode)
SGL_DICT_TO_DESC_FIELD(flags, ClusterOperationFlags)
SGL_DICT_TO_DESC_FIELD(move, ClusterOperationMoveParams)
SGL_DICT_TO_DESC_FIELD(clas, ClusterOperationClasBuildParams)
SGL_DICT_TO_DESC_FIELD(blas, ClusterOperationBlasBuildParams)
SGL_DICT_TO_DESC_END()

SGL_DICT_TO_DESC_BEGIN(ClusterOperationDesc)
SGL_DICT_TO_DESC_FIELD(params, ClusterOperationParams)
SGL_DICT_TO_DESC_FIELD(arg_count_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(args_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(args_buffer_stride, uint64_t)
SGL_DICT_TO_DESC_FIELD(scratch_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(addresses_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(addresses_buffer_stride, size_t)
SGL_DICT_TO_DESC_FIELD(result_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(sizes_buffer, BufferOffsetPair)
SGL_DICT_TO_DESC_FIELD(sizes_buffer_stride, size_t)
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
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureHandle* self, uint64_t value)
            {
                new (self) AccelerationStructureHandle{value};
            },
            "value"_a
        )
        .def_rw("value", &AccelerationStructureHandle::value);
    nb::implicitly_convertible<uint64_t, AccelerationStructureHandle>();

    nb::sgl_enum_flags<AccelerationStructureGeometryFlags>(m, "AccelerationStructureGeometryFlags");
    nb::sgl_enum_flags<AccelerationStructureInstanceFlags>(m, "AccelerationStructureInstanceFlags");

    nb::sgl_enum<MicromapType>(m, "MicromapType");
    nb::sgl_enum<OpacityMicromapFormat>(m, "OpacityMicromapFormat");
    nb::sgl_enum<OpacityMicromapSpecialIndex>(m, "OpacityMicromapSpecialIndex");
    nb::sgl_enum<MicromapIndexingMode>(m, "MicromapIndexingMode");
    nb::sgl_enum<MicromapIndexFormat>(m, "MicromapIndexFormat");
    nb::sgl_enum_flags<MicromapBuildFlags>(m, "MicromapBuildFlags");

    nb::class_<MicromapUsageCount>(m, "MicromapUsageCount", D(MicromapUsageCount))
        .def(nb::init<>())
        .def(
            "__init__",
            [](MicromapUsageCount* self, nb::dict dict)
            {
                new (self) MicromapUsageCount(dict_to_MicromapUsageCount(dict));
            }
        )
        .def_rw("count", &MicromapUsageCount::count, D(MicromapUsageCount, count))
        .def_rw("subdivision_level", &MicromapUsageCount::subdivision_level, D(MicromapUsageCount, subdivision_level))
        .def_rw("format", &MicromapUsageCount::format, D(MicromapUsageCount, format));
    nb::implicitly_convertible<nb::dict, MicromapUsageCount>();

    nb::class_<MicromapBuildDesc>(m, "MicromapBuildDesc", D(MicromapBuildDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](MicromapBuildDesc* self, nb::dict dict)
            {
                new (self) MicromapBuildDesc(dict_to_MicromapBuildDesc(dict));
            }
        )
        .def_rw("type", &MicromapBuildDesc::type, D(MicromapBuildDesc, type))
        .def_rw("flags", &MicromapBuildDesc::flags, D(MicromapBuildDesc, flags))
        .def_rw("data_buffer", &MicromapBuildDesc::data_buffer, D(MicromapBuildDesc, data_buffer))
        .def_rw("descriptor_buffer", &MicromapBuildDesc::descriptor_buffer, D(MicromapBuildDesc, descriptor_buffer))
        .def_rw("descriptor_stride", &MicromapBuildDesc::descriptor_stride, D(MicromapBuildDesc, descriptor_stride))
        .def_rw("histogram", &MicromapBuildDesc::histogram, D(MicromapBuildDesc, histogram));
    nb::implicitly_convertible<nb::dict, MicromapBuildDesc>();

    nb::class_<MicromapSizes>(m, "MicromapSizes", D(MicromapSizes))
        .def_ro("micromap_size", &MicromapSizes::micromap_size, D(MicromapSizes, micromap_size))
        .def_ro("scratch_size", &MicromapSizes::scratch_size, D(MicromapSizes, scratch_size));

    nb::class_<MicromapDesc>(m, "MicromapDesc", D(MicromapDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](MicromapDesc* self, nb::dict dict)
            {
                new (self) MicromapDesc(dict_to_MicromapDesc(dict));
            }
        )
        .def_rw("type", &MicromapDesc::type, D(MicromapDesc, type))
        .def_rw("size", &MicromapDesc::size, D(MicromapDesc, size))
        .def_rw("flags", &MicromapDesc::flags, D(MicromapDesc, flags))
        .def_rw("label", &MicromapDesc::label, D(MicromapDesc, label));
    nb::implicitly_convertible<nb::dict, MicromapDesc>();

    nb::class_<Micromap, Resource>(m, "Micromap", D(Micromap))
        .def_prop_ro("desc", &Micromap::desc, D(Micromap, desc))
        .def_prop_ro("device_address", &Micromap::device_address, D(Micromap, device_address));

    nb::class_<AccelerationStructureOpacityMicromapDesc>(
        m,
        "AccelerationStructureOpacityMicromapDesc",
        D(AccelerationStructureOpacityMicromapDesc)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureOpacityMicromapDesc* self, nb::dict dict)
            {
                new (self)
                    AccelerationStructureOpacityMicromapDesc(dict_to_AccelerationStructureOpacityMicromapDesc(dict));
            }
        )
        .def_rw(
            "micromap",
            &AccelerationStructureOpacityMicromapDesc::micromap,
            D(AccelerationStructureOpacityMicromapDesc, micromap)
        )
        .def_rw(
            "indexing_mode",
            &AccelerationStructureOpacityMicromapDesc::indexing_mode,
            D(AccelerationStructureOpacityMicromapDesc, indexing_mode)
        )
        .def_rw(
            "index_buffer",
            &AccelerationStructureOpacityMicromapDesc::index_buffer,
            D(AccelerationStructureOpacityMicromapDesc, index_buffer)
        )
        .def_rw(
            "index_format",
            &AccelerationStructureOpacityMicromapDesc::index_format,
            D(AccelerationStructureOpacityMicromapDesc, index_format)
        )
        .def_rw(
            "index_stride",
            &AccelerationStructureOpacityMicromapDesc::index_stride,
            D(AccelerationStructureOpacityMicromapDesc, index_stride)
        )
        .def_rw(
            "base_micromap_index",
            &AccelerationStructureOpacityMicromapDesc::base_micromap_index,
            D(AccelerationStructureOpacityMicromapDesc, base_micromap_index)
        )
        .def_rw(
            "usage_counts",
            &AccelerationStructureOpacityMicromapDesc::usage_counts,
            D(AccelerationStructureOpacityMicromapDesc, usage_counts)
        );
    nb::implicitly_convertible<nb::dict, AccelerationStructureOpacityMicromapDesc>();

    nb::class_<AccelerationStructureInstanceDesc>(
        m,
        "AccelerationStructureInstanceDesc",
        D(AccelerationStructureInstanceDesc)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureInstanceDesc* self, nb::dict dict)
            {
                new (self) AccelerationStructureInstanceDesc(dict_to_AccelerationStructureInstanceDesc(dict));
            }
        )
        .def_rw(
            "transform",
            &AccelerationStructureInstanceDesc::transform,
            D(AccelerationStructureInstanceDesc, transform)
        )
        .def_prop_rw(
            "instance_id",
            [](AccelerationStructureInstanceDesc& self)
            {
                return self.instance_id;
            },
            [](AccelerationStructureInstanceDesc& self, uint32_t value)
            {
                self.instance_id = value;
            },
            D(AccelerationStructureInstanceDesc, instance_id)
        )
        .def_prop_rw(
            "instance_mask",
            [](AccelerationStructureInstanceDesc& self)
            {
                return self.instance_mask;
            },
            [](AccelerationStructureInstanceDesc& self, uint32_t value)
            {
                self.instance_mask = value;
            },
            D(AccelerationStructureInstanceDesc, instance_mask)
        )
        .def_prop_rw(
            "instance_contribution_to_hit_group_index",
            [](AccelerationStructureInstanceDesc& self)
            {
                return self.instance_contribution_to_hit_group_index;
            },
            [](AccelerationStructureInstanceDesc& self, uint32_t value)
            {
                self.instance_contribution_to_hit_group_index = value;
            },
            D(AccelerationStructureInstanceDesc, instance_contribution_to_hit_group_index)
        )
        .def_prop_rw(
            "flags",
            [](AccelerationStructureInstanceDesc& self)
            {
                return self.flags;
            },
            [](AccelerationStructureInstanceDesc& self, AccelerationStructureInstanceFlags value)
            {
                self.flags = value;
            },
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
            [](AccelerationStructureBuildInputInstances* self, nb::dict dict)
            {
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
            [](AccelerationStructureBuildInputTriangles* self, nb::dict dict)
            {
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
        .def_rw("flags", &AccelerationStructureBuildInputTriangles::flags)
        .def_rw(
            "opacity_micromap",
            &AccelerationStructureBuildInputTriangles::opacity_micromap,
            nb::arg().none(),
            D(AccelerationStructureBuildInputTriangles, opacity_micromap)
        );
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

    nb::class_<AccelerationStructureBuildInputSpheres>(
        m,
        "AccelerationStructureBuildInputSpheres",
        D(AccelerationStructureBuildInputSpheres)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildInputSpheres* self, nb::dict dict)
            {
                new (self) AccelerationStructureBuildInputSpheres(dict_to_AccelerationStructureBuildInputSpheres(dict));
            }
        )
        .def_rw("vertex_count", &AccelerationStructureBuildInputSpheres::vertex_count)
        .def_rw("vertex_position_buffers", &AccelerationStructureBuildInputSpheres::vertex_position_buffers)
        .def_rw("vertex_position_format", &AccelerationStructureBuildInputSpheres::vertex_position_format)
        .def_rw("vertex_position_stride", &AccelerationStructureBuildInputSpheres::vertex_position_stride)
        .def_rw("vertex_radius_buffers", &AccelerationStructureBuildInputSpheres::vertex_radius_buffers)
        .def_rw("vertex_radius_format", &AccelerationStructureBuildInputSpheres::vertex_radius_format)
        .def_rw("vertex_radius_stride", &AccelerationStructureBuildInputSpheres::vertex_radius_stride)
        .def_rw("index_buffer", &AccelerationStructureBuildInputSpheres::index_buffer)
        .def_rw("index_format", &AccelerationStructureBuildInputSpheres::index_format)
        .def_rw("index_count", &AccelerationStructureBuildInputSpheres::index_count)
        .def_rw("flags", &AccelerationStructureBuildInputSpheres::flags);
    nb::implicitly_convertible<nb::dict, AccelerationStructureBuildInputSpheres>();

    nb::sgl_enum<LinearSweptSpheresIndexingMode>(m, "LinearSweptSpheresIndexingMode");
    nb::sgl_enum<LinearSweptSpheresEndCapsMode>(m, "LinearSweptSpheresEndCapsMode");

    nb::class_<AccelerationStructureBuildInputLinearSweptSpheres>(
        m,
        "AccelerationStructureBuildInputLinearSweptSpheres",
        D(AccelerationStructureBuildInputLinearSweptSpheres)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildInputLinearSweptSpheres* self, nb::dict dict)
            {
                new (self) AccelerationStructureBuildInputLinearSweptSpheres(
                    dict_to_AccelerationStructureBuildInputLinearSweptSpheres(dict)
                );
            }
        )
        .def_rw("vertex_count", &AccelerationStructureBuildInputLinearSweptSpheres::vertex_count)
        .def_rw("primitive_count", &AccelerationStructureBuildInputLinearSweptSpheres::primitive_count)
        .def_rw("vertex_position_buffers", &AccelerationStructureBuildInputLinearSweptSpheres::vertex_position_buffers)
        .def_rw("vertex_position_format", &AccelerationStructureBuildInputLinearSweptSpheres::vertex_position_format)
        .def_rw("vertex_position_stride", &AccelerationStructureBuildInputLinearSweptSpheres::vertex_position_stride)
        .def_rw("vertex_radius_buffers", &AccelerationStructureBuildInputLinearSweptSpheres::vertex_radius_buffers)
        .def_rw("vertex_radius_format", &AccelerationStructureBuildInputLinearSweptSpheres::vertex_radius_format)
        .def_rw("vertex_radius_stride", &AccelerationStructureBuildInputLinearSweptSpheres::vertex_radius_stride)
        .def_rw("index_buffer", &AccelerationStructureBuildInputLinearSweptSpheres::index_buffer)
        .def_rw("index_format", &AccelerationStructureBuildInputLinearSweptSpheres::index_format)
        .def_rw("index_count", &AccelerationStructureBuildInputLinearSweptSpheres::index_count)
        .def_rw("indexing_mode", &AccelerationStructureBuildInputLinearSweptSpheres::indexing_mode)
        .def_rw("end_caps_mode", &AccelerationStructureBuildInputLinearSweptSpheres::end_caps_mode)
        .def_rw("flags", &AccelerationStructureBuildInputLinearSweptSpheres::flags);
    nb::implicitly_convertible<nb::dict, AccelerationStructureBuildInputLinearSweptSpheres>();

    nb::class_<AccelerationStructureBuildInputMotionOptions>(
        m,
        "AccelerationStructureBuildInputMotionOptions",
        D(AccelerationStructureBuildInputMotionOptions)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureBuildInputMotionOptions* self, nb::dict dict)
            {
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
            {
                new (self) AccelerationStructureBuildDesc(dict_to_AccelerationStructureBuildDesc(dict));
            }
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
            {
                new (self) AccelerationStructureQueryDesc(dict_to_AccelerationStructureQueryDesc(dict));
            }
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

    nb::sgl_enum<AccelerationStructureKind>(m, "AccelerationStructureKind");

    nb::class_<AccelerationStructureDesc>(m, "AccelerationStructureDesc", D(AccelerationStructureDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](AccelerationStructureDesc* self, nb::dict dict)
            {
                new (self) AccelerationStructureDesc(dict_to_AccelerationStructureDesc(dict));
            }
        )
        .def_rw("kind", &AccelerationStructureDesc::kind, D(AccelerationStructureDesc, kind))
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

    nb::sgl_enum<ClusterOperationType>(m, "ClusterOperationType");
    nb::sgl_enum<ClusterOperationMode>(m, "ClusterOperationMode");
    nb::sgl_enum_flags<ClusterOperationFlags>(m, "ClusterOperationFlags");
    nb::sgl_enum<ClusterOperationMoveType>(m, "ClusterOperationMoveType");

    nb::class_<ClusterOperationMoveParams>(m, "ClusterOperationMoveParams", D(ClusterOperationMoveParams))
        .def(nb::init<>())
        .def(
            "__init__",
            [](ClusterOperationMoveParams* self, nb::dict dict)
            {
                new (self) ClusterOperationMoveParams(dict_to_ClusterOperationMoveParams(dict));
            }
        )
        .def_rw("type", &ClusterOperationMoveParams::type, D(ClusterOperationMoveParams, type))
        .def_rw("max_size", &ClusterOperationMoveParams::max_size, D(ClusterOperationMoveParams, max_size));
    nb::implicitly_convertible<nb::dict, ClusterOperationMoveParams>();

    nb::class_<ClusterOperationClasBuildParams>(
        m,
        "ClusterOperationClasBuildParams",
        D(ClusterOperationClasBuildParams)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](ClusterOperationClasBuildParams* self, nb::dict dict)
            {
                new (self) ClusterOperationClasBuildParams(dict_to_ClusterOperationClasBuildParams(dict));
            }
        )
        .def_rw(
            "vertex_format",
            &ClusterOperationClasBuildParams::vertex_format,
            D(ClusterOperationClasBuildParams, vertex_format)
        )
        .def_rw(
            "max_geometry_index",
            &ClusterOperationClasBuildParams::max_geometry_index,
            D(ClusterOperationClasBuildParams, max_geometry_index)
        )
        .def_rw(
            "max_unique_geometry_count",
            &ClusterOperationClasBuildParams::max_unique_geometry_count,
            D(ClusterOperationClasBuildParams, max_unique_geometry_count)
        )
        .def_rw(
            "max_triangle_count",
            &ClusterOperationClasBuildParams::max_triangle_count,
            D(ClusterOperationClasBuildParams, max_triangle_count)
        )
        .def_rw(
            "max_vertex_count",
            &ClusterOperationClasBuildParams::max_vertex_count,
            D(ClusterOperationClasBuildParams, max_vertex_count)
        )
        .def_rw(
            "max_total_triangle_count",
            &ClusterOperationClasBuildParams::max_total_triangle_count,
            D(ClusterOperationClasBuildParams, max_total_triangle_count)
        )
        .def_rw(
            "max_total_vertex_count",
            &ClusterOperationClasBuildParams::max_total_vertex_count,
            D(ClusterOperationClasBuildParams, max_total_vertex_count)
        )
        .def_rw(
            "min_position_truncate_bit_count",
            &ClusterOperationClasBuildParams::min_position_truncate_bit_count,
            D(ClusterOperationClasBuildParams, min_position_truncate_bit_count)
        );
    nb::implicitly_convertible<nb::dict, ClusterOperationClasBuildParams>();

    nb::class_<ClusterOperationBlasBuildParams>(
        m,
        "ClusterOperationBlasBuildParams",
        D(ClusterOperationBlasBuildParams)
    )
        .def(nb::init<>())
        .def(
            "__init__",
            [](ClusterOperationBlasBuildParams* self, nb::dict dict)
            {
                new (self) ClusterOperationBlasBuildParams(dict_to_ClusterOperationBlasBuildParams(dict));
            }
        )
        .def_rw(
            "max_clas_count",
            &ClusterOperationBlasBuildParams::max_clas_count,
            D(ClusterOperationBlasBuildParams, max_clas_count)
        )
        .def_rw(
            "max_total_clas_count",
            &ClusterOperationBlasBuildParams::max_total_clas_count,
            D(ClusterOperationBlasBuildParams, max_total_clas_count)
        );
    nb::implicitly_convertible<nb::dict, ClusterOperationBlasBuildParams>();

    nb::class_<ClusterOperationParams>(m, "ClusterOperationParams", D(ClusterOperationParams))
        .def(nb::init<>())
        .def(
            "__init__",
            [](ClusterOperationParams* self, nb::dict dict)
            {
                new (self) ClusterOperationParams(dict_to_ClusterOperationParams(dict));
            }
        )
        .def_rw("max_arg_count", &ClusterOperationParams::max_arg_count, D(ClusterOperationParams, max_arg_count))
        .def_rw("type", &ClusterOperationParams::type, D(ClusterOperationParams, type))
        .def_rw("mode", &ClusterOperationParams::mode, D(ClusterOperationParams, mode))
        .def_rw("flags", &ClusterOperationParams::flags, D(ClusterOperationParams, flags))
        .def_rw("move", &ClusterOperationParams::move, D(ClusterOperationParams, move))
        .def_rw("clas", &ClusterOperationParams::clas, D(ClusterOperationParams, clas))
        .def_rw("blas", &ClusterOperationParams::blas, D(ClusterOperationParams, blas));
    nb::implicitly_convertible<nb::dict, ClusterOperationParams>();

    nb::class_<ClusterOperationDesc>(m, "ClusterOperationDesc", D(ClusterOperationDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](ClusterOperationDesc* self, nb::dict dict)
            {
                new (self) ClusterOperationDesc(dict_to_ClusterOperationDesc(dict));
            }
        )
        .def_rw("params", &ClusterOperationDesc::params, D(ClusterOperationDesc, params))
        .def_rw("arg_count_buffer", &ClusterOperationDesc::arg_count_buffer, D(ClusterOperationDesc, arg_count_buffer))
        .def_rw("args_buffer", &ClusterOperationDesc::args_buffer, D(ClusterOperationDesc, args_buffer))
        .def_rw(
            "args_buffer_stride",
            &ClusterOperationDesc::args_buffer_stride,
            D(ClusterOperationDesc, args_buffer_stride)
        )
        .def_rw("scratch_buffer", &ClusterOperationDesc::scratch_buffer, D(ClusterOperationDesc, scratch_buffer))
        .def_rw("addresses_buffer", &ClusterOperationDesc::addresses_buffer, D(ClusterOperationDesc, addresses_buffer))
        .def_rw(
            "addresses_buffer_stride",
            &ClusterOperationDesc::addresses_buffer_stride,
            D(ClusterOperationDesc, addresses_buffer_stride)
        )
        .def_rw("result_buffer", &ClusterOperationDesc::result_buffer, D(ClusterOperationDesc, result_buffer))
        .def_rw("sizes_buffer", &ClusterOperationDesc::sizes_buffer, D(ClusterOperationDesc, sizes_buffer))
        .def_rw(
            "sizes_buffer_stride",
            &ClusterOperationDesc::sizes_buffer_stride,
            D(ClusterOperationDesc, sizes_buffer_stride)
        );
    nb::implicitly_convertible<nb::dict, ClusterOperationDesc>();

    nb::class_<ClusterOperationSizes>(m, "ClusterOperationSizes", D(ClusterOperationSizes))
        .def_ro("result_size", &ClusterOperationSizes::result_size, D(ClusterOperationSizes, result_size))
        .def_ro("scratch_size", &ClusterOperationSizes::scratch_size, D(ClusterOperationSizes, scratch_size));

    m.attr("CLUSTER_MAX_TRIANGLE_COUNT") = CLUSTER_MAX_TRIANGLE_COUNT;
    m.attr("CLUSTER_MAX_VERTEX_COUNT") = CLUSTER_MAX_VERTEX_COUNT;
    m.attr("CLUSTER_MAX_GEOMETRY_INDEX") = CLUSTER_MAX_GEOMETRY_INDEX;
    m.attr("CLUSTER_DEFAULT_HANDLE_STRIDE") = CLUSTER_DEFAULT_HANDLE_STRIDE;
    m.attr("CLUSTER_OUTPUT_ALIGNMENT") = CLUSTER_OUTPUT_ALIGNMENT;

    nb::class_<ShaderTableDesc>(m, "ShaderTableDesc", D(ShaderTableDesc))
        .def(nb::init<>())
        .def(
            "__init__",
            [](ShaderTableDesc* self, nb::dict dict)
            {
                new (self) ShaderTableDesc(dict_to_ShaderTableDesc(dict));
            }
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
