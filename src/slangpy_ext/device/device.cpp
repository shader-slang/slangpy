// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/device/device.h"
#include "sgl/device/reflection.h"
#include "sgl/device/sampler.h"
#include "sgl/device/fence.h"
#include "sgl/device/resource.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/kernel.h"
#include "sgl/device/raytracing.h"
#include "sgl/device/query.h"
#include "sgl/device/input_layout.h"
#include "sgl/device/surface.h"
#include "sgl/device/shader.h"
#include "sgl/device/command.h"
#include "sgl/device/hot_reload.h"

#include "sgl/core/window.h"

namespace sgl {
SGL_DICT_TO_DESC_BEGIN(DeviceDesc)
SGL_DICT_TO_DESC_FIELD(type, DeviceType)
SGL_DICT_TO_DESC_FIELD(enable_debug_layers, bool)
SGL_DICT_TO_DESC_FIELD(enable_cuda_interop, bool)
SGL_DICT_TO_DESC_FIELD(enable_print, bool)
SGL_DICT_TO_DESC_FIELD(enable_hot_reload, bool)
SGL_DICT_TO_DESC_FIELD(enable_compilation_reports, bool)
SGL_DICT_TO_DESC_FIELD(adapter_luid, AdapterLUID)
SGL_DICT_TO_DESC_FIELD(compiler_options, SlangCompilerOptions)
SGL_DICT_TO_DESC_FIELD(shader_cache_path, std::filesystem::path)
SGL_DICT_TO_DESC_FIELD(label, std::string)
SGL_DICT_TO_DESC_END()

// Utility functions for doing CoopVec conversions between ndarrays
template<typename... Args>
static bool ndarray_is_row_maj(const nb::ndarray<Args...>& array)
{
    return array.ndim() == 2 && (array.stride(0) == int64_t(array.shape(1)) && array.stride(1) == 1);
}
template<typename... Args>
static bool ndarray_is_col_maj(const nb::ndarray<Args...>& array)
{
    return array.ndim() == 2 && (array.stride(0) == 1 && array.stride(1) == int64_t(array.shape(0)));
}

template<typename... Args>
static void coopvec_check_ndarray(const nb::ndarray<Args...>& array)
{
    DataType dtype = dtype_to_data_type(array.dtype());
    SGL_CHECK(dtype != DataType::void_ && dtype != DataType::bool_, "Invalid CoopVec element type \"%d\"", dtype);
    if (array.ndim() == 1)
        return; // 1D arrays are always ok
    SGL_CHECK(array.ndim() == 2, "Expected 2-dimensional array (received ndim = %d)", array.ndim());
    SGL_CHECK(ndarray_is_row_maj(array) || ndarray_is_col_maj(array), "2D arrays must be row or column major");
}
template<typename... Args>
static CoopVecMatrixLayout
coopvec_get_ndarray_matrix_layout(const nb::ndarray<Args...>& array, std::optional<CoopVecMatrixLayout> hint)
{
    if (hint)
        return hint.value();
    if (ndarray_is_row_maj(array))
        return CoopVecMatrixLayout::row_major;
    if (ndarray_is_col_maj(array))
        return CoopVecMatrixLayout::column_major;
    SGL_THROW("Require a 2D row major/column major array when no explicit matrix layout is given");
}
template<typename... Args>
static CoopVecMatrixDesc
coopvec_get_ndarray_matrix_desc(const nb::ndarray<Args...>& array, std::optional<CoopVecMatrixLayout> layout_hint)
{
    coopvec_check_ndarray(array);

    CoopVecMatrixDesc desc;
    desc.size = array.nbytes();
    desc.layout = coopvec_get_ndarray_matrix_layout(array, layout_hint);
    desc.element_type = dtype_to_data_type(array.dtype());
    desc.rows = array.ndim() > 0 ? uint32_t(array.shape(0)) : 0;
    desc.cols = array.ndim() > 1 ? uint32_t(array.shape(1)) : 0;
    return desc;
}

inline size_t coopvec_convert_matrix_host_to_host(
    Device* self,
    nb::ndarray<nb::device::cpu> src,
    nb::ndarray<nb::device::cpu> dst,
    std::optional<CoopVecMatrixLayout> src_layout,
    std::optional<CoopVecMatrixLayout> dst_layout
)
{
    CoopVecMatrixDesc src_desc = coopvec_get_ndarray_matrix_desc(src, src_layout);
    CoopVecMatrixDesc dst_desc = coopvec_get_ndarray_matrix_desc(dst, dst_layout);

    if (src.ndim() == 2 && dst.ndim() == 2) {
        SGL_CHECK(
            src_desc.rows == dst_desc.rows && src_desc.cols == dst_desc.cols,
            "Array shapes of src and dst do not match ((%d, %d) != (%d, %d))",
            src_desc.rows,
            src_desc.cols,
            dst_desc.rows,
            dst_desc.cols
        );
    } else if (src.ndim() == 2) {
        dst_desc.rows = src_desc.rows;
        dst_desc.cols = src_desc.cols;
    } else if (dst.ndim() == 2) {
        src_desc.rows = dst_desc.rows;
        src_desc.cols = dst_desc.cols;
    } else {
        SGL_THROW("At least one of src or dst must be a 2D array");
    }

    return self->get_or_create_coop_vec()->convert_matrix_host(src.data(), src_desc, dst.data(), dst_desc);
}

} // namespace sgl

SGL_PY_EXPORT(device_device)
{
    using namespace sgl;

    nb::class_<AdapterInfo>(m, "AdapterInfo", D(AdapterInfo))
        .def_ro("name", &AdapterInfo::name, D(AdapterInfo, name))
        .def_ro("vendor_id", &AdapterInfo::vendor_id, D(AdapterInfo, vendor_id))
        .def_ro("device_id", &AdapterInfo::device_id, D(AdapterInfo, device_id))
        .def_ro("luid", &AdapterInfo::luid, D(AdapterInfo, luid))
        .def("__repr__", &AdapterInfo::to_string);

    nb::sgl_enum<DeviceType>(m, "DeviceType");

    nb::class_<DeviceDesc>(m, "DeviceDesc", D(DeviceDesc))
        .def(nb::init<>())
        .def("__init__", [](DeviceDesc* self, nb::dict dict) { new (self) DeviceDesc(dict_to_DeviceDesc(dict)); })
        .def_rw("type", &DeviceDesc::type, D(DeviceDesc, type))
        .def_rw("enable_debug_layers", &DeviceDesc::enable_debug_layers, D(DeviceDesc, enable_debug_layers))
        .def_rw("enable_cuda_interop", &DeviceDesc::enable_cuda_interop, D(DeviceDesc, enable_cuda_interop))
        .def_rw("enable_print", &DeviceDesc::enable_print, D(DeviceDesc, enable_print))
        .def_rw("enable_hot_reload", &DeviceDesc::enable_hot_reload, D(DeviceDesc, adapter_luid))
        .def_rw(
            "enable_compilation_reports",
            &DeviceDesc::enable_compilation_reports,
            D(DeviceDesc, enable_compilation_reports)
        )
        .def_rw("adapter_luid", &DeviceDesc::adapter_luid, D(DeviceDesc, adapter_luid))
        .def_rw("compiler_options", &DeviceDesc::compiler_options, D(DeviceDesc, compiler_options))
        .def_rw("shader_cache_path", &DeviceDesc::shader_cache_path, D(DeviceDesc, shader_cache_path))
        .def_rw(
            "existing_device_handles",
            &DeviceDesc::existing_device_handles,
            D_NA(DeviceDesc, existing_device_handles)
        )
        .def_rw("label", &DeviceDesc::label, D_NA(DeviceDesc, label));

    nb::implicitly_convertible<nb::dict, DeviceDesc>();

    nb::class_<DeviceLimits>(m, "DeviceLimits", D(DeviceLimits))
        .def_ro(
            "max_texture_dimension_1d",
            &DeviceLimits::max_texture_dimension_1d,
            D(DeviceLimits, max_texture_dimension_1d)
        )
        .def_ro(
            "max_texture_dimension_2d",
            &DeviceLimits::max_texture_dimension_2d,
            D(DeviceLimits, max_texture_dimension_2d)
        )
        .def_ro(
            "max_texture_dimension_3d",
            &DeviceLimits::max_texture_dimension_3d,
            D(DeviceLimits, max_texture_dimension_3d)
        )
        .def_ro(
            "max_texture_dimension_cube",
            &DeviceLimits::max_texture_dimension_cube,
            D(DeviceLimits, max_texture_dimension_cube)
        )
        .def_ro("max_texture_layers", &DeviceLimits::max_texture_layers, D(DeviceLimits, max_texture_layers))
        .def_ro(
            "max_vertex_input_elements",
            &DeviceLimits::max_vertex_input_elements,
            D(DeviceLimits, max_vertex_input_elements)
        )
        .def_ro(
            "max_vertex_input_element_offset",
            &DeviceLimits::max_vertex_input_element_offset,
            D(DeviceLimits, max_vertex_input_element_offset)
        )
        .def_ro("max_vertex_streams", &DeviceLimits::max_vertex_streams, D(DeviceLimits, max_vertex_streams))
        .def_ro(
            "max_vertex_stream_stride",
            &DeviceLimits::max_vertex_stream_stride,
            D(DeviceLimits, max_vertex_stream_stride)
        )
        .def_ro(
            "max_compute_threads_per_group",
            &DeviceLimits::max_compute_threads_per_group,
            D(DeviceLimits, max_compute_threads_per_group)
        )
        .def_ro(
            "max_compute_thread_group_size",
            &DeviceLimits::max_compute_thread_group_size,
            D(DeviceLimits, max_compute_thread_group_size)
        )
        .def_ro(
            "max_compute_dispatch_thread_groups",
            &DeviceLimits::max_compute_dispatch_thread_groups,
            D(DeviceLimits, max_compute_dispatch_thread_groups)
        )
        .def_ro("max_viewports", &DeviceLimits::max_viewports, D(DeviceLimits, max_viewports))
        .def_ro(
            "max_viewport_dimensions",
            &DeviceLimits::max_viewport_dimensions,
            D(DeviceLimits, max_viewport_dimensions)
        )
        .def_ro(
            "max_framebuffer_dimensions",
            &DeviceLimits::max_framebuffer_dimensions,
            D(DeviceLimits, max_framebuffer_dimensions)
        )
        .def_ro(
            "max_shader_visible_samplers",
            &DeviceLimits::max_shader_visible_samplers,
            D(DeviceLimits, max_shader_visible_samplers)
        );

    nb::class_<DeviceInfo>(m, "DeviceInfo", D(DeviceInfo))
        .def_ro("type", &DeviceInfo::type, D(DeviceInfo, type))
        .def_ro("api_name", &DeviceInfo::api_name, D(DeviceInfo, api_name))
        .def_ro("adapter_name", &DeviceInfo::adapter_name, D(DeviceInfo, adapter_name))
        .def_ro("timestamp_frequency", &DeviceInfo::timestamp_frequency, D(DeviceInfo, timestamp_frequency))
        .def_ro("limits", &DeviceInfo::limits, D(DeviceInfo, limits));

    nb::class_<ShaderCacheStats>(m, "ShaderCacheStats", D(ShaderCacheStats))
        .def_ro("entry_count", &ShaderCacheStats::entry_count, D(ShaderCacheStats, entry_count))
        .def_ro("hit_count", &ShaderCacheStats::hit_count, D(ShaderCacheStats, hit_count))
        .def_ro("miss_count", &ShaderCacheStats::miss_count, D(ShaderCacheStats, miss_count));

    nb::class_<ShaderHotReloadEvent>(m, "ShaderHotReloadEvent", D(ShaderHotReloadEvent));

    nb::class_<HeapReport>(m, "HeapReport", D_NA(HeapReport))
        .def_rw("name", &HeapReport::name, D_NA(HeapReport, name))
        .def_rw("num_pages", &HeapReport::num_pages, D_NA(HeapReport, num_pages))
        .def_rw("total_allocated", &HeapReport::total_allocated, D_NA(HeapReport, total_allocated))
        .def_rw("total_mem_usage", &HeapReport::total_mem_usage, D_NA(HeapReport, total_mem_usage))
        .def_rw("num_allocations", &HeapReport::num_allocations, D_NA(HeapReport, num_allocations));

    nb::class_<Device, Object> device(m, "Device", nb::is_weak_referenceable(), D(Device));
    device.def(
        "__init__",
        [](Device* self,
           DeviceType type,
           bool enable_debug_layers,
           bool enable_cuda_interop,
           bool enable_print,
           bool enable_hot_reload,
           bool enable_compilation_reports,
           std::optional<AdapterLUID> adapter_luid,
           std::optional<SlangCompilerOptions> compiler_options,
           std::optional<std::filesystem::path> shader_cache_path,
           std::optional<std::array<NativeHandle, 3>> existing_device_handles,
           std::string label = "")
        {
            new (self) Device(
                {.type = type,
                 .enable_debug_layers = enable_debug_layers,
                 .enable_cuda_interop = enable_cuda_interop,
                 .enable_print = enable_print,
                 .enable_hot_reload = enable_hot_reload,
                 .enable_compilation_reports = enable_compilation_reports,
                 .adapter_luid = adapter_luid,
                 .compiler_options = compiler_options.value_or(SlangCompilerOptions{}),
                 .shader_cache_path = shader_cache_path,
                 .existing_device_handles = existing_device_handles.value_or(std::array<NativeHandle, 3>()),
                 .label = label}
            );
        },
        "type"_a = DeviceDesc().type,
        "enable_debug_layers"_a = DeviceDesc().enable_debug_layers,
        "enable_cuda_interop"_a = DeviceDesc().enable_cuda_interop,
        "enable_print"_a = DeviceDesc().enable_print,
        "enable_hot_reload"_a = DeviceDesc().enable_hot_reload,
        "enable_compilation_reports"_a = DeviceDesc().enable_compilation_reports,
        "adapter_luid"_a.none() = nb::none(),
        "compiler_options"_a.none() = nb::none(),
        "shader_cache_path"_a.none() = nb::none(),
        "existing_device_handles"_a.none() = nb::none(),
        "label"_a = DeviceDesc().label,
        D(Device, Device)
    );
    device.def(nb::init<DeviceDesc>(), "desc"_a, D(Device, Device));
    device.def_prop_ro("desc", &Device::desc, D(Device, desc));
    device.def_prop_ro("info", &Device::info, D(Device, info));
    device.def_prop_ro("shader_cache_stats", &Device::shader_cache_stats, D(Device, shader_cache_stats));
    device.def_prop_ro("supported_shader_model", &Device::supported_shader_model, D(Device, supported_shader_model));
    device.def_prop_ro("features", &Device::features, D(Device, features));
    device.def_prop_ro("supports_cuda_interop", &Device::supports_cuda_interop, D(Device, supports_cuda_interop));
    device.def_prop_ro("native_handles", &Device::native_handles, D(Device, native_handles));
    device.def("has_feature", &Device::has_feature, "feature"_a, D(Device, has_feature));
    device.def("get_format_support", &Device::get_format_support, "format"_a, D(Device, get_format_support));

    device.def_prop_ro("slang_session", &Device::slang_session, D(Device, slang_session));
    device.def("close", &Device::close, D(Device, close));
    device.def(
        "create_surface",
        [](Device* self, ref<Window> window) { return self->create_surface(window); },
        "window"_a,
        D(Device, create_surface)
    );
    device.def(
        "create_surface",
        [](Device* self, WindowHandle window_handle) { return self->create_surface(window_handle); },
        "window_handle"_a,
        D(Device, create_surface, 2)
    );
    device.def(
        "create_buffer",
        [](Device* self,
           size_t size,
           size_t element_count,
           size_t struct_size,
           nb::object resource_type_layout,
           Format format,
           MemoryType memory_type,
           BufferUsage usage,
           ResourceState default_state,
           std::string label,
           std::optional<nb::ndarray<nb::numpy>> data)
        {
            if (data) {
                SGL_CHECK(is_ndarray_contiguous(*data), "Data is not contiguous.");
            }

            ref<const TypeLayoutReflection> resolved_resource_type_layout;
            if (!resource_type_layout.is_none()) {
                // Note: nanobind can't try cast to a ref counted pointer, however the reflection
                // cursor code below needs to ensure it maintains a reference to the type layout if
                // returned, so we attempt to convert to the raw ptr here, and then immediately
                // store it in a local ref counted ptr.
                if (const TypeLayoutReflection * resolved_struct_type_ptr;
                    nb::try_cast(resource_type_layout, resolved_struct_type_ptr)) {
                    resolved_resource_type_layout = ref<const TypeLayoutReflection>(resolved_struct_type_ptr);
                }
                // If this is a reflection cursor, get type layout from it
                else if (ReflectionCursor reflection_cursor; nb::try_cast(resource_type_layout, reflection_cursor)) {

                    resolved_resource_type_layout = reflection_cursor.type_layout();
                }
                // Otherwise we got an invalid type
                else {
                    throw nb::type_error(
                        "Expected a TypeLayoutReflection or ReflectionCursor for 'resource_type_layout'"
                    );
                }
                if (resolved_resource_type_layout->kind() != TypeReflection::Kind::resource
                    || resolved_resource_type_layout->type()->resource_shape()
                        != TypeReflection::ResourceShape::structured_buffer) {
                    throw nb::type_error(
                        "Expected a TypeLayoutReflection of a structured buffer for 'resource_type_layout'"
                    );
                }
            }

            return self->create_buffer({
                .size = size,
                .element_count = element_count,
                .struct_size = struct_size,
                .resource_type_layout = resolved_resource_type_layout,
                .format = format,
                .memory_type = memory_type,
                .usage = usage,
                .default_state = default_state,
                .label = std::move(label),
                .data = data ? data->data() : nullptr,
                .data_size = data ? data->nbytes() : 0,
            });
        },
        "size"_a = BufferDesc().size,
        "element_count"_a = BufferDesc().element_count,
        "struct_size"_a = BufferDesc().struct_size,
        "resource_type_layout"_a.none() = nb::none(),
        "format"_a = BufferDesc().format,
        "memory_type"_a = BufferDesc().memory_type,
        "usage"_a = BufferDesc().usage,
        "default_state"_a = BufferDesc().default_state,
        "label"_a = BufferDesc().label,
        "data"_a.none() = nb::none(),
        D(Device, create_buffer)
    );
    device.def(
        "create_buffer",
        [](Device* self, const BufferDesc& desc) { return self->create_buffer(desc); },
        "desc"_a,
        D(Device, create_buffer)
    );

    device.def(
        "create_texture",
        [](Device* self,
           TextureType type,
           Format format,
           uint32_t width,
           uint32_t height,
           uint32_t depth,
           uint32_t array_length,
           uint32_t mip_count,
           uint32_t sample_count,
           uint32_t sample_quality,
           MemoryType memory_type,
           TextureUsage usage,
           ResourceState default_state,
           std::string label,
           std::optional<nb::ndarray<nb::numpy>> data)
        {
            SubresourceData subresourceData[1];
            if (data) {
                SGL_CHECK(
                    array_length == 1,
                    "Texture arrays cannot be populated on construction - use upload_texture_data"
                );
                SGL_CHECK(
                    type != TextureType::texture_cube && type != TextureType::texture_cube_array,
                    "Cube textures cannot be populated on construction - use upload_texture_data"
                );
                SGL_CHECK(is_ndarray_contiguous(*data), "Data is not contiguous.");
                subresourceData[0].data = data->data();
                subresourceData[0].size = data->nbytes();
                subresourceData[0].slice_pitch = subresourceData[0].size / depth;
                subresourceData[0].row_pitch = subresourceData[0].slice_pitch / height;
            }
            return self->create_texture({
                .type = type,
                .format = format,
                .width = width,
                .height = height,
                .depth = depth,
                .array_length = array_length,
                .mip_count = mip_count,
                .sample_count = sample_count,
                .sample_quality = sample_quality,
                .memory_type = memory_type,
                .usage = usage,
                .default_state = default_state,
                .label = std::move(label),
                .data = data ? subresourceData : std::span<SubresourceData>(),
            });
        },
        "type"_a = TextureDesc().type,
        "format"_a = TextureDesc().format,
        "width"_a = TextureDesc().width,
        "height"_a = TextureDesc().height,
        "depth"_a = TextureDesc().depth,
        "array_length"_a = TextureDesc().array_length,
        "mip_count"_a = TextureDesc().mip_count,
        "sample_count"_a = TextureDesc().sample_count,
        "sample_quality"_a = TextureDesc().sample_quality,
        "memory_type"_a = TextureDesc().memory_type,
        "usage"_a = TextureDesc().usage,
        "default_state"_a = TextureDesc().default_state,
        "label"_a = TextureDesc().label,
        "data"_a.none() = nb::none(),
        D(Device, create_texture)
    );
    device.def(
        "create_texture",
        [](Device* self, const TextureDesc& desc) { return self->create_texture(desc); },
        "desc"_a,
        D(Device, create_texture)
    );

    device.def(
        "create_sampler",
        [](Device* self,
           TextureFilteringMode min_filter,
           TextureFilteringMode mag_filter,
           TextureFilteringMode mip_filter,
           TextureReductionOp reduction_op,
           TextureAddressingMode address_u,
           TextureAddressingMode address_v,
           TextureAddressingMode address_w,
           float mip_lod_bias,
           uint32_t max_anisotropy,
           ComparisonFunc comparison_func,
           float4 border_color,
           float min_lod,
           float max_lod,
           std::string label)
        {
            return self->create_sampler({
                .min_filter = min_filter,
                .mag_filter = mag_filter,
                .mip_filter = mip_filter,
                .reduction_op = reduction_op,
                .address_u = address_u,
                .address_v = address_v,
                .address_w = address_w,
                .mip_lod_bias = mip_lod_bias,
                .max_anisotropy = max_anisotropy,
                .comparison_func = comparison_func,
                .border_color = border_color,
                .min_lod = min_lod,
                .max_lod = max_lod,
                .label = std::move(label),
            });
        },
        "min_filter"_a = SamplerDesc().min_filter,
        "mag_filter"_a = SamplerDesc().mag_filter,
        "mip_filter"_a = SamplerDesc().mip_filter,
        "reduction_op"_a = SamplerDesc().reduction_op,
        "address_u"_a = SamplerDesc().address_u,
        "address_v"_a = SamplerDesc().address_v,
        "address_w"_a = SamplerDesc().address_w,
        "mip_lod_bias"_a = SamplerDesc().mip_lod_bias,
        "max_anisotropy"_a = SamplerDesc().max_anisotropy,
        "comparison_func"_a = SamplerDesc().comparison_func,
        "border_color"_a = SamplerDesc().border_color,
        "min_lod"_a = SamplerDesc().min_lod,
        "max_lod"_a = SamplerDesc().max_lod,
        "label"_a = SamplerDesc().label,
        D(Device, create_sampler)
    );
    device.def("create_sampler", &Device::create_sampler, "desc"_a, D(Device, create_sampler));

    device.def(
        "create_fence",
        [](Device* self, uint64_t initial_value, bool shared)
        { return self->create_fence({.initial_value = initial_value, .shared = shared}); },
        "initial_value"_a = FenceDesc().initial_value,
        "shared"_a = FenceDesc().shared,
        D(Device, create_fence)
    );
    device.def("create_fence", &Device::create_fence, "desc"_a, D(Device, create_fence));

    device.def(
        "create_query_pool",
        [](Device* self, QueryType type, uint32_t count)
        { return self->create_query_pool({.type = type, .count = count}); },
        "type"_a,
        "count"_a,
        D(Device, create_query_pool)
    );

    device.def(
        "create_input_layout",
        [](Device* self, std::vector<InputElementDesc> input_elements, std::vector<VertexStreamDesc> vertex_streams)
        {
            return self->create_input_layout({
                .input_elements = std::move(input_elements),
                .vertex_streams = std::move(vertex_streams),
            });
        },
        "input_elements"_a,
        "vertex_streams"_a,
        D(Device, create_input_layout)
    );
    device.def("create_input_layout", &Device::create_input_layout, "desc"_a, D(Device, create_input_layout));

    device.def(
        "create_command_encoder",
        &Device::create_command_encoder,
        "queue"_a = CommandQueueType::graphics,
        D(Device, create_command_encoder)
    );
    device.def(
        "submit_command_buffers",
        &Device::submit_command_buffers,
        "command_buffers"_a,
        "wait_fences"_a = std::span<Fence*>(),
        "wait_fence_values"_a = std::span<uint64_t>(),
        "signal_fences"_a = std::span<Fence*>(),
        "signal_fence_values"_a = std::span<uint64_t>(),
        "queue"_a = CommandQueueType::graphics,
        "cuda_stream"_a = NativeHandle(),
        D(Device, submit_command_buffers)
    );
    device.def(
        "submit_command_buffer",
        &Device::submit_command_buffer,
        "command_buffer"_a,
        "queue"_a = CommandQueueType::graphics,
        "cuda_stream"_a = NativeHandle(),
        D(Device, submit_command_buffer)
    );
    device.def("is_submit_finished", &Device::is_submit_finished, "id"_a, D(Device, is_submit_finished));
    device.def("wait_for_submit", &Device::wait_for_submit, "id"_a, D(Device, wait_for_submit));
    device
        .def("wait_for_idle", &Device::wait_for_idle, "queue"_a = CommandQueueType::graphics, D(Device, wait_for_idle));
    device.def(
        "sync_to_cuda",
        [](Device* self, uint64_t cuda_stream) { self->sync_to_cuda(reinterpret_cast<void*>(cuda_stream)); },
        "cuda_stream"_a = 0,
        D(Device, sync_to_cuda)
    );
    device.def(
        "sync_to_device",
        [](Device* self, uint64_t cuda_stream) { self->sync_to_device(reinterpret_cast<void*>(cuda_stream)); },
        "cuda_stream"_a = 0,
        D(Device, sync_to_device)
    );
    device.def(
        "get_acceleration_structure_sizes",
        &Device::get_acceleration_structure_sizes,
        "desc"_a,
        D(Device, get_acceleration_structure_sizes)
    );
    device.def(
        "create_acceleration_structure",
        [](Device* self, size_t size, std::string label)
        { return self->create_acceleration_structure({.size = size, .label = std::move(label)}); },
        "size"_a = AccelerationStructureDesc().size,
        "label"_a = AccelerationStructureDesc().label,
        D(Device, create_acceleration_structure)
    );
    device.def(
        "create_acceleration_structure",
        &Device::create_acceleration_structure,
        "desc"_a,
        D(Device, create_acceleration_structure)
    );
    device.def(
        "create_acceleration_structure_instance_list",
        &Device::create_acceleration_structure_instance_list,
        "size"_a,
        D(Device, create_acceleration_structure_instance_list)
    );
    device.def(
        "create_shader_table",
        [](Device* self,
           ref<ShaderProgram> program,
           std::vector<std::string> ray_gen_entry_points,
           std::vector<std::string> miss_entry_points,
           std::vector<std::string> hit_group_names,
           std::vector<std::string> callable_entry_points)
        {
            return self->create_shader_table({
                .program = program,
                .ray_gen_entry_points = std::move(ray_gen_entry_points),
                .miss_entry_points = std::move(miss_entry_points),
                .hit_group_names = std::move(hit_group_names),
                .callable_entry_points = std::move(callable_entry_points),
            });
        },
        "program"_a,
        "ray_gen_entry_points"_a = std::vector<std::string>{},
        "miss_entry_points"_a = std::vector<std::string>{},
        "hit_group_names"_a = std::vector<std::string>{},
        "callable_entry_points"_a = std::vector<std::string>{},
        D(Device, create_shader_table)
    );
    device.def("create_shader_table", &Device::create_shader_table, "desc"_a, D(Device, create_shader_table));
    device.def(
        "create_slang_session",
        [](Device* self,
           std::optional<SlangCompilerOptions> compiler_options,
           bool add_default_include_paths,
           std::optional<std::filesystem::path> cache_path)
        {
            return self->create_slang_session(SlangSessionDesc{
                .compiler_options = compiler_options.value_or(SlangCompilerOptions{}),
                .add_default_include_paths = add_default_include_paths,
                .cache_path = cache_path,
            });
        },
        "compiler_options"_a.none() = nb::none(),
        "add_default_include_paths"_a = SlangSessionDesc().add_default_include_paths,
        "cache_path"_a.none() = nb::none(),
        D(Device, create_slang_session)
    );
    device.def("reload_all_programs", &Device::reload_all_programs, D(Device, reload_all_programs));
    device.def("load_module", &Device::load_module, "module_name"_a, D(Device, load_module));
    device.def(
        "load_module_from_source",
        &Device::load_module_from_source,
        "module_name"_a,
        "source"_a,
        "path"_a.none() = nb::none(),
        D(Device, load_module_from_source)
    );
    device.def(
        "link_program",
        &Device::link_program,
        "modules"_a,
        "entry_points"_a,
        "link_options"_a.none() = nb::none(),
        D(Device, link_program)
    );
    device.def(
        "load_program",
        &Device::load_program,
        "module_name"_a,
        "entry_point_names"_a,
        "additional_source"_a.none() = nb::none(),
        "link_options"_a.none() = nb::none(),
        D(Device, load_program)
    );

    device.def(
        "create_root_shader_object",
        nb::overload_cast<const ShaderProgram*>(&Device::create_root_shader_object),
        "shader_program"_a,
        D(Device, create_root_shader_object)
    );
    device.def(
        "create_shader_object",
        [](Device* self, ref<TypeLayoutReflection> type_layout) { return self->create_shader_object(type_layout); },
        "type_layout"_a,
        D(Device, create_shader_object)
    );
    device.def(
        "create_shader_object",
        nb::overload_cast<ReflectionCursor>(&Device::create_shader_object),
        "cursor"_a,
        D(Device, create_shader_object, 2)
    );

    device.def(
        "create_compute_pipeline",
        [](Device* self, ref<ShaderProgram> program)
        { return self->create_compute_pipeline({.program = std::move(program)}); },
        "program"_a,
        D(Device, create_compute_pipeline)
    );
    device
        .def("create_compute_pipeline", &Device::create_compute_pipeline, "desc"_a, D(Device, create_compute_pipeline));

    device.def(
        "create_render_pipeline",
        [](Device* self,
           ref<ShaderProgram> program,
           ref<InputLayout> input_layout,
           PrimitiveTopology primitive_topology,
           std::vector<ColorTargetDesc> targets,
           std::optional<DepthStencilDesc> depth_stencil,
           std::optional<RasterizerDesc> rasterizer,
           std::optional<MultisampleDesc> multisample)
        {
            return self->create_render_pipeline({
                .program = std::move(program),
                .input_layout = input_layout,
                .primitive_topology = primitive_topology,
                .targets = targets,
                .depth_stencil = depth_stencil.value_or(DepthStencilDesc{}),
                .rasterizer = rasterizer.value_or(RasterizerDesc{}),
                .multisample = multisample.value_or(MultisampleDesc{}),
            });
        },
        "program"_a,
        "input_layout"_a.none(),
        "primitive_topology"_a = RenderPipelineDesc().primitive_topology,
        "targets"_a = std::vector<ColorTargetDesc>{},
        "depth_stencil"_a.none() = nb::none(),
        "rasterizer"_a.none() = nb::none(),
        "multisample"_a.none() = nb::none(),
        D(Device, create_render_pipeline)
    );
    device.def("create_render_pipeline", &Device::create_render_pipeline, "desc"_a, D(Device, create_render_pipeline));

    device.def(
        "create_ray_tracing_pipeline",
        [](Device* self,
           ref<ShaderProgram> program,
           std::vector<HitGroupDesc> hit_groups,
           uint32_t max_recursion,
           uint32_t max_ray_payload_size,
           uint32_t max_attribute_size,
           RayTracingPipelineFlags flags)
        {
            return self->create_ray_tracing_pipeline({
                .program = std::move(program),
                .hit_groups = std::move(hit_groups),
                .max_recursion = max_recursion,
                .max_ray_payload_size = max_ray_payload_size,
                .max_attribute_size = max_attribute_size,
                .flags = flags,
            });
        },
        "program"_a,
        "hit_groups"_a,
        "max_recursion"_a = RayTracingPipelineDesc().max_recursion,
        "max_ray_payload_size"_a = RayTracingPipelineDesc().max_ray_payload_size,
        "max_attribute_size"_a = RayTracingPipelineDesc().max_attribute_size,
        "flags"_a = RayTracingPipelineDesc().flags,
        D(Device, create_ray_tracing_pipeline)
    );
    device.def(
        "create_ray_tracing_pipeline",
        &Device::create_ray_tracing_pipeline,
        "desc"_a,
        D(Device, create_ray_tracing_pipeline)
    );

    device.def(
        "create_compute_kernel",
        [](Device* self, ref<ShaderProgram> program)
        { return self->create_compute_kernel({.program = std::move(program)}); },
        "program"_a,
        D(Device, create_compute_kernel)
    );
    device.def("create_compute_kernel", &Device::create_compute_kernel, "desc"_a, D(Device, create_compute_kernel));

    device.def("flush_print", &Device::flush_print, D(Device, flush_print));
    device.def("flush_print_to_string", &Device::flush_print_to_string, D(Device, flush_print_to_string));
    device.def("wait", &Device::wait, D(Device, wait));
    device.def(
        "register_shader_hot_reload_callback",
        &Device::register_shader_hot_reload_callback,
        "callback"_a,
        D(Device, register_shader_hot_reload_callback)
    );
    device.def(
        "register_device_close_callback",
        &Device::register_device_close_callback,
        "callback"_a,
        D(Device, register_device_close_callback)
    );
    device.def(
        "coopvec_query_matrix_size",
        [](Device* self, uint32_t rows, uint32_t cols, CoopVecMatrixLayout layout, DataType element_type)
        { return self->get_or_create_coop_vec()->query_matrix_size(rows, cols, layout, element_type); },
        "rows"_a,
        "cols"_a,
        "layout"_a,
        "element_type"_a
    );
    device.def(
        "coopvec_create_matrix_desc",
        [](Device* self, uint32_t rows, uint32_t cols, CoopVecMatrixLayout layout, DataType element_type, size_t offset)
        { return self->get_or_create_coop_vec()->create_matrix_desc(rows, cols, layout, element_type, offset); },
        "rows"_a,
        "cols"_a,
        "layout"_a,
        "element_type"_a,
        "offset"_a = 0
    );
    device.def(
        "coopvec_convert_matrix_host",
        &coopvec_convert_matrix_host_to_host,
        "src"_a,
        "dst"_a,
        "src_layout"_a = nb::none(),
        "dst_layout"_a = nb::none()
    );
    device.def(
        "coopvec_convert_matrix_device",
        [](Device* self,
           const ref<Buffer>& src,
           CoopVecMatrixDesc srcDesc,
           const ref<Buffer>& dst,
           CoopVecMatrixDesc dstDesc,
           CommandEncoder* encoder)
        { return self->get_or_create_coop_vec()->convert_matrix_device(src, srcDesc, dst, dstDesc, encoder); },
        "src"_a,
        "src_desc"_a,
        "dst"_a,
        "dst_desc"_a,
        "encoder"_a = nullptr
    );
    device.def(
        "coopvec_convert_matrix_device",
        [](Device* self,
           const ref<Buffer>& src,
           const std::vector<CoopVecMatrixDesc>& srcDesc,
           const ref<Buffer>& dst,
           const std::vector<CoopVecMatrixDesc>& dstDesc,
           CommandEncoder* encoder)
        { return self->get_or_create_coop_vec()->convert_matrix_device(src, srcDesc, dst, dstDesc, encoder); },
        "src"_a,
        "src_desc"_a,
        "dst"_a,
        "dst_desc"_a,
        "encoder"_a = nullptr
    );
    device.def(
        "coopvec_align_matrix_offset",
        [](Device* self, size_t offset) { return self->get_or_create_coop_vec()->align_matrix_offset(offset); },
        "offset"_a
    );
    device.def(
        "coopvec_align_vector_offset",
        [](Device* self, size_t offset) { return self->get_or_create_coop_vec()->align_vector_offset(offset); },
        "offset"_a
    );
    device.def(
        "set_hot_reload_delay",
        [](Device* self, uint32_t delay_ms) { self->_hot_reload()->set_auto_detect_delay(delay_ms); },
        "timeout_ms"_a,
        D_NA(Device, set_hot_reload_delay)
    );
    device.def(
        "hot_reload_check",
        [](Device* self) { self->_hot_reload()->update(); },
        D_NA(Device, hot_reload_check)
    );

    device.def_static(
        "enumerate_adapters",
        &Device::enumerate_adapters,
        "type"_a = DeviceType::automatic,
        D(Device, enumerate_adapters)
    );

    device.def_static("get_created_devices", &Device::get_created_devices, D_NA(Device, get_created_devices));

    device.def_static("report_live_objects", &Device::report_live_objects, D(Device, report_live_objects));

    device.def("report_heaps", &Device::report_heaps, D_NA(Device, report_heaps));

    m.def(
        "get_cuda_current_context_native_handles",
        get_cuda_current_context_native_handles,
        D_NA(get_cuda_current_context_native_handles, get_cuda_current_context_native_handles)
    );
}
