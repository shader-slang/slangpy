// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device.h"

#include "sgl/device/surface.h"
#include "sgl/device/resource.h"
#include "sgl/device/sampler.h"
#include "sgl/device/fence.h"
#include "sgl/device/query.h"
#include "sgl/device/input_layout.h"
#include "sgl/device/shader.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/kernel.h"
#include "sgl/device/raytracing.h"
#include "sgl/device/command.h"
#include "sgl/device/helpers.h"
#include "sgl/device/agility_sdk.h"
#include "sgl/device/cuda_utils.h"
#include "sgl/device/cuda_interop.h"
#include "sgl/device/print.h"
#include "sgl/device/blit.h"
#include "sgl/device/hot_reload.h"
#include "sgl/device/debug_logger.h"
#include "sgl/device/native_handle_traits.h"

#include "sgl/core/file_system_watcher.h"
#include "sgl/core/config.h"
#include "sgl/core/error.h"
#include "sgl/core/window.h"
#include "sgl/core/string.h"

#if SGL_HAS_D3D12
#include <dxgi.h>
#include <d3d12.h>
#include <comdef.h>
#endif

#include <mutex>

namespace sgl {

static std::vector<Device*> s_devices;
static std::mutex s_devices_mutex;

inline AdapterLUID from_rhi(const rhi::AdapterLUID& rhi_luid)
{
    AdapterLUID luid;
    for (size_t i = 0; i < 16; ++i)
        luid[i] = rhi_luid.luid[i];
    return luid;
}


Device::Device(const DeviceDesc& desc)
    : m_desc(desc)
{
    ConstructorRefGuard ref_guard(this);

    if (desc.enable_debug_layers)
        rhi::getRHI()->enableDebugLayers();

    // Create hot reload system before creating any sessions.
    if (m_desc.enable_hot_reload)
        m_hot_reload = make_ref<HotReload>(ref<Device>(this));

    SLANG_CALL(slang::createGlobalSession(m_global_session.writeRef()));

    // Setup path for slang's downstream compilers.
    for (SlangPassThrough pass_through :
         {SLANG_PASS_THROUGH_DXC, SLANG_PASS_THROUGH_GLSLANG, SLANG_PASS_THROUGH_SPIRV_OPT}) {
        m_global_session->setDownstreamCompilerPath(pass_through, platform::runtime_directory().string().c_str());
    }

    if (m_desc.type == DeviceType::automatic) {
#if SGL_WINDOWS
        m_desc.type = DeviceType::d3d12;
#elif SGL_LINUX
        m_desc.type = DeviceType::vulkan;
#elif SGL_MACOS
        m_desc.type = DeviceType::metal;
#endif
    }

    // Setup shader cache.
    if (m_desc.shader_cache_path) {
        m_shader_cache_enabled = true;
        m_shader_cache_path = *m_desc.shader_cache_path;
        if (m_shader_cache_path.is_relative())
            m_shader_cache_path = platform::app_data_directory() / m_shader_cache_path;
        std::filesystem::create_directories(m_shader_cache_path);
    }

    // Invalidate CUDA interop if using CUDA
    if (m_desc.type == DeviceType::cuda && m_desc.enable_cuda_interop) {
        m_desc.enable_cuda_interop = false;
        log_warn("Device type is set to CUDA, but CUDA interop is requested. enable_cuda_interop will be ignored.");
    }

    // Invalidate use of adapter LUID if existing handles provided.
    if (m_desc.adapter_luid.has_value()) {
        for (const auto& handle : m_desc.existing_device_handles) {
            if (handle.is_valid()) {
                m_desc.adapter_luid.reset();
                log_warn("Both adapter LUID and existing handles have been provided, which are both ways to "
                         "specify the device. Adapter LUID will be ignored in favor of provided existing handles");
                break;
            }
        }
    }

    // If CUDA interop is enabled on non-cuda backend, check if existing CUDA context or device
    // is provided. If so, we will attempt to identify the same device for use with SlangPy.
    if (m_desc.enable_cuda_interop) {
        if (!rhiCudaDriverApiInit()) {
            close();
            SGL_THROW("Failed to initialize CUDA driver API.");
        }

        CUdevice cuda_device = -1;
        CUcontext cuda_context = nullptr;
        for (auto& handle : m_desc.existing_device_handles) {
            if (handle.type() == NativeHandleType::CUdevice) {
                cuda_device = handle.as<CUdevice>();
                handle = {};
            } else if (handle.type() == NativeHandleType::CUcontext) {
                cuda_context = handle.as<CUcontext>();
                handle = {};
            }
        }
        if (cuda_context) {
            m_cuda_device = make_ref<cuda::Device>(cuda_context);
        } else if (cuda_device != -1) {
            m_cuda_device = make_ref<cuda::Device>(cuda_device);
        } else {
            log_warn(
                "CUDA interop is enabled, but no existing CUDA device or context is provided. To ensure the correct "
                "device is selected, pass the CUDA device or context in as an existing device handle in the "
                "DeviceDesc. "
                "The current primary CUDA context handles can be acquired with "
                "slangpy.get_cuda_current_context_native_handles."
            );
        }
    }

    // If we now have a valid CUDA device, use it to determine the adapter LUID.
    if (m_cuda_device) {
        std::vector<AdapterInfo> adapters = enumerate_adapters(m_desc.type);

        AdapterLUID luid = m_cuda_device->adapter_luid();
        bool found = false;
        for (const AdapterInfo& adapter : adapters) {
            if (adapter.luid == luid) {
                m_desc.adapter_luid = std::make_optional(luid);
                log_debug("Using adapter LUID {} from CUDA device.", m_desc.adapter_luid.value());
                found = true;
                break;
            }
        }

        if (!found) {
            std::string adapter_name = m_cuda_device->adapter_name();
            log_warn("Unable to find matching adapter LUID, searching by name {}", adapter_name);
            for (const AdapterInfo& adapter : adapters) {
                if (adapter.name == adapter_name) {
                    m_desc.adapter_luid = std::make_optional(adapter.luid);
                    log_warn(
                        "Selected adapter LUID {} by matching device name {}.",
                        m_desc.adapter_luid.value(),
                        adapter.name
                    );
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            close();
            SGL_THROW("Unable to find matching adapter LUID or name for the provided CUDA device.");
        }
    }

    // Setup extensions.
    rhi::D3D12DeviceExtendedDesc d3d12_extended_desc{
        .structType = rhi::StructType::D3D12DeviceExtendedDesc,
        .rootParameterShaderAttributeName = "root",
        .debugBreakOnD3D12Error = false,
        .highestShaderModel = 0,
    };

    rhi::DeviceDesc rhi_desc{
        .next = &d3d12_extended_desc,
        .deviceType = static_cast<rhi::DeviceType>(m_desc.type),
        .existingDeviceHandles = {
            m_desc.existing_device_handles[0].to_rhi(),
            m_desc.existing_device_handles[1].to_rhi(),
            m_desc.existing_device_handles[2].to_rhi(),
        },
        .adapterLUID
        = m_desc.adapter_luid ? reinterpret_cast<const rhi::AdapterLUID*>(m_desc.adapter_luid->data()) : nullptr,
        .slang{
            .slangGlobalSession = m_global_session,
        },
        // This needs to match NV_SHADER_EXTN_SLOT set in shader.cpp
        .nvapiExtUavSlot = 999,
        // TODO(slang-rhi) make configurable but default to true
        .enableValidation = true,
        .debugCallback = &DebugLogger::get(),
        .enableCompilationReports = m_desc.enable_compilation_reports,
    };
    log_debug(
        "Creating graphics device (type: {}, LUID: {}, shader_cache_path: {}).",
        m_desc.type,
        m_desc.adapter_luid,
        m_shader_cache_path
    );
    if (SLANG_FAILED(rhi::getRHI()->createDevice(rhi_desc, m_rhi_device.writeRef())))
        SGL_THROW("Failed to create device!");

    // Get device info.
    const rhi::DeviceInfo& rhi_device_info = m_rhi_device->getInfo();
    m_info.type = m_desc.type;
    m_info.api_name = rhi_device_info.apiName;
    m_info.adapter_name = rhi_device_info.adapterName;
    m_info.adapter_luid = from_rhi(rhi_device_info.adapterLUID);
    m_info.timestamp_frequency = rhi_device_info.timestampFrequency;
    m_info.limits.max_texture_dimension_1d = rhi_device_info.limits.maxTextureDimension1D;
    m_info.limits.max_texture_dimension_2d = rhi_device_info.limits.maxTextureDimension2D;
    m_info.limits.max_texture_dimension_3d = rhi_device_info.limits.maxTextureDimension3D;
    m_info.limits.max_texture_dimension_cube = rhi_device_info.limits.maxTextureDimensionCube;
    m_info.limits.max_texture_layers = rhi_device_info.limits.maxTextureLayers;
    m_info.limits.max_vertex_input_elements = rhi_device_info.limits.maxVertexInputElements;
    m_info.limits.max_vertex_input_element_offset = rhi_device_info.limits.maxVertexInputElementOffset;
    m_info.limits.max_vertex_streams = rhi_device_info.limits.maxVertexStreams;
    m_info.limits.max_vertex_stream_stride = rhi_device_info.limits.maxVertexStreamStride;
    m_info.limits.max_compute_threads_per_group = rhi_device_info.limits.maxComputeThreadsPerGroup;
    m_info.limits.max_compute_thread_group_size = uint3(
        rhi_device_info.limits.maxComputeThreadGroupSize[0],
        rhi_device_info.limits.maxComputeThreadGroupSize[1],
        rhi_device_info.limits.maxComputeThreadGroupSize[2]
    );
    m_info.limits.max_compute_dispatch_thread_groups = uint3(
        rhi_device_info.limits.maxComputeDispatchThreadGroups[0],
        rhi_device_info.limits.maxComputeDispatchThreadGroups[1],
        rhi_device_info.limits.maxComputeDispatchThreadGroups[2]
    );
    m_info.limits.max_viewports = rhi_device_info.limits.maxViewports;
    m_info.limits.max_viewport_dimensions
        = uint2(rhi_device_info.limits.maxViewportDimensions[0], rhi_device_info.limits.maxViewportDimensions[1]);
    m_info.limits.max_framebuffer_dimensions = uint3(
        rhi_device_info.limits.maxFramebufferDimensions[0],
        rhi_device_info.limits.maxFramebufferDimensions[1],
        rhi_device_info.limits.maxFramebufferDimensions[2]
    );
    m_info.limits.max_shader_visible_samplers = rhi_device_info.limits.maxShaderVisibleSamplers;

    // Get supported shader model.
    const std::vector<std::pair<ShaderModel, const char*>> available_shader_models = {
        {ShaderModel::sm_6_7, "sm_6_7"},
        {ShaderModel::sm_6_6, "sm_6_6"},
        {ShaderModel::sm_6_5, "sm_6_5"},
        {ShaderModel::sm_6_4, "sm_6_4"},
        {ShaderModel::sm_6_3, "sm_6_3"},
        {ShaderModel::sm_6_2, "sm_6_2"},
        {ShaderModel::sm_6_1, "sm_6_1"},
        {ShaderModel::sm_6_0, "sm_6_0"},
    };
    for (const auto& [sm, sm_str] : available_shader_models) {
        if (m_rhi_device->hasFeature(sm_str)) {
            m_supported_shader_model = sm;
            break;
        }
    }
    if (m_supported_shader_model == ShaderModel::unknown) {
        m_supported_shader_model = ShaderModel::sm_6_0;
        log_warn("No supported shader model found, pretending to support {}.", m_supported_shader_model);
    }
    log_debug("Supported shader model: {}", m_supported_shader_model);

    // Query features.
    std::vector<std::string> feature_names;
    for (uint32_t i = 0; i < uint32_t(rhi::Feature::_Count); ++i) {
        if (m_rhi_device->hasFeature(static_cast<rhi::Feature>(i))) {
            m_features.push_back(static_cast<Feature>(i));
            feature_names.push_back(enum_to_string(static_cast<Feature>(i)));
        }
    }
    log_debug("Supported features: {}", string::join(feature_names, ", "));

    // Create graphics queue.
    SLANG_RHI_CALL(m_rhi_device->getQueue(rhi::QueueType::Graphics, m_rhi_graphics_queue.writeRef()));

    // Create global fence to synchronize command submission.
    m_global_fence = create_fence({.shared = m_desc.enable_cuda_interop});

    // Finalize CUDA interop.
    if (m_desc.enable_cuda_interop) {

        // If didn't create CUDA device from existing handles, make it now that we
        // have a chosen device.
        if (!m_cuda_device) {
            m_cuda_device = make_ref<cuda::Device>(this);
        }

        // Create the semaphore for interop
        {
            SGL_CU_SCOPE(this);
            m_cuda_semaphore = make_ref<cuda::ExternalSemaphore>(m_global_fence);
        }

        m_supports_cuda_interop = true;
    }

    if (m_desc.enable_print)
        m_debug_printer = std::make_unique<DebugPrinter>(this);

    // Create default slang session.
    m_slang_session = create_slang_session({
        .compiler_options = m_desc.compiler_options,
        .add_default_include_paths = true,
        .cache_path = m_shader_cache_enabled ? std::optional(m_shader_cache_path) : std::nullopt,
    });

    // Add device to global device list.
    {
        std::lock_guard lock(s_devices_mutex);
        s_devices.push_back(this);
    }
}

Device::~Device()
{
    // Remove device from global device list.
    {
        std::lock_guard lock(s_devices_mutex);
        s_devices.erase(std::remove(s_devices.begin(), s_devices.end(), this), s_devices.end());
    }

    SGL_CHECK(m_closed, "Device is not close. Call close() before destroying the device.");

    m_rhi_graphics_queue.setNull();
    m_rhi_device.setNull();
}

ShaderCacheStats Device::shader_cache_stats() const
{
    // TODO: revisit when we add a shader cache.
    return {
        .entry_count = 0,
        .hit_count = 0,
        .miss_count = 0,
    };
}

bool Device::has_feature(Feature feature) const
{
    return m_rhi_device->hasFeature(static_cast<rhi::Feature>(feature));
}

FormatSupport Device::get_format_support(Format format) const
{
    rhi::FormatSupport rhi_format_support;
    SLANG_RHI_CALL(m_rhi_device->getFormatSupport(static_cast<rhi::Format>(format), &rhi_format_support));
    return static_cast<FormatSupport>(rhi_format_support);
}

void Device::close()
{
    if (m_closed)
        return;

    log_debug("Closing device {}", fmt::ptr(this));

    wait();

    // Handle device close callbacks
    for (const DeviceCloseCallback& callback : m_device_close_callbacks)
        callback(this);

    // Make sure Device's ref count is not going to zero when releasing resources.
    inc_ref();

    m_closed = true;

    m_shader_hot_reload_callbacks.clear();
    m_device_close_callbacks.clear();

    m_blitter.reset();
    m_debug_printer.reset();

    m_global_fence.reset();

    m_slang_session.reset();
    m_hot_reload.reset();
    m_coop_vec.reset();

    if (m_cuda_device) {
        SGL_CU_SCOPE(this);
        m_cuda_semaphore.reset();
    }
    m_cuda_device.reset();

    dec_ref();
}

void Device::close_all_devices()
{
    std::vector<Device*> devices;
    {
        std::lock_guard lock(s_devices_mutex);
        devices = s_devices;
    }
    for (Device* device : devices)
        device->close();
}

ref<Surface> Device::create_surface(Window* window)
{
    return make_ref<Surface>(window, ref<Device>(this));
}

ref<Surface> Device::create_surface(WindowHandle window_handle)
{
    return make_ref<Surface>(window_handle, ref<Device>(this));
}

ref<Buffer> Device::create_buffer(BufferDesc desc)
{
    return make_ref<Buffer>(ref<Device>(this), std::move(desc));
}

ref<BufferView> Device::create_buffer_view(Buffer* buffer, BufferViewDesc desc)
{
    return make_ref<BufferView>(ref<Device>(this), ref<Buffer>(buffer), std::move(desc));
}

ref<Texture> Device::create_texture(TextureDesc desc)
{
    return make_ref<Texture>(ref<Device>(this), std::move(desc));
}

ref<Texture> Device::create_texture_from_resource(TextureDesc desc, rhi::ITexture* resource)
{
    return make_ref<Texture>(ref<Device>(this), std::move(desc), resource);
}

ref<TextureView> Device::create_texture_view(Texture* texture, TextureViewDesc desc)
{
    return make_ref<TextureView>(ref<Device>(this), ref<Texture>(texture), std::move(desc));
}

ref<Sampler> Device::create_sampler(SamplerDesc desc)
{
    return make_ref<Sampler>(ref<Device>(this), std::move(desc));
}

ref<Fence> Device::create_fence(FenceDesc desc)
{
    return make_ref<Fence>(ref<Device>(this), std::move(desc));
}

ref<QueryPool> Device::create_query_pool(QueryPoolDesc desc)
{
    return make_ref<QueryPool>(ref<Device>(this), std::move(desc));
}

ref<InputLayout> Device::create_input_layout(InputLayoutDesc desc)
{
    return make_ref<InputLayout>(ref<Device>(this), std::move(desc));
}

AccelerationStructureSizes Device::get_acceleration_structure_sizes(const AccelerationStructureBuildDesc& desc)
{
    AccelerationStructureBuildDescConverter converter(desc);
    rhi::AccelerationStructureSizes rhi_sizes;
    SLANG_RHI_CALL(m_rhi_device->getAccelerationStructureSizes(converter.rhi_desc, &rhi_sizes));
    return {
        .acceleration_structure_size = rhi_sizes.accelerationStructureSize,
        .scratch_size = rhi_sizes.scratchSize,
        .update_scratch_size = rhi_sizes.updateScratchSize,
    };
}

ref<AccelerationStructure> Device::create_acceleration_structure(AccelerationStructureDesc desc)
{
    return make_ref<AccelerationStructure>(ref<Device>(this), std::move(desc));
}

ref<AccelerationStructureInstanceList> Device::create_acceleration_structure_instance_list(size_t size)
{
    return make_ref<AccelerationStructureInstanceList>(ref<Device>(this), size);
}

ref<ShaderTable> Device::create_shader_table(ShaderTableDesc desc)
{
    return make_ref<ShaderTable>(ref<Device>(this), std::move(desc));
}

/*size_t Device::query_coopvec_matrix_size(uint32_t rows, uint32_t columns, CoopVecMatrixLayout layout)
{
    return m_coop_vec->query_matrix_size(rows, columns, layout);
}*/

ref<CoopVec> Device::get_or_create_coop_vec()
{
    if (!m_coop_vec)
        m_coop_vec.reset(new CoopVec(ref<Device>(this)));
    return m_coop_vec;
}

ref<SlangSession> Device::create_slang_session(SlangSessionDesc desc)
{
    return make_ref<SlangSession>(ref<Device>(this), std::move(desc));
}

void Device::reload_all_programs()
{
    if (m_hot_reload)
        m_hot_reload->recreate_all_sessions();
}

ref<SlangModule> Device::load_module(std::string_view module_name)
{
    return m_slang_session->load_module(module_name);
}

ref<SlangModule> Device::load_module_from_source(
    std::string_view module_name,
    std::string_view source,
    std::optional<std::filesystem::path> path
)
{
    return m_slang_session->load_module_from_source(module_name, source, path);
}

ref<ShaderProgram> Device::link_program(
    std::vector<ref<SlangModule>> modules,
    std::vector<ref<SlangEntryPoint>> entry_points,
    std::optional<SlangLinkOptions> link_options
)
{
    return m_slang_session->link_program(std::move(modules), std::move(entry_points), link_options);
}

ref<ShaderProgram> Device::load_program(
    std::string_view module_name,
    std::vector<std::string_view> entry_point_names,
    std::optional<std::string_view> additional_source,
    std::optional<SlangLinkOptions> link_options
)
{
    return m_slang_session->load_program(module_name, entry_point_names, additional_source, link_options);
}

ref<ShaderObject> Device::create_root_shader_object(const ShaderProgram* shader_program)
{
    Slang::ComPtr<rhi::IShaderObject> rhi_shader_object;
    SLANG_RHI_CALL(
        m_rhi_device->createRootShaderObject(shader_program->rhi_shader_program(), rhi_shader_object.writeRef())
    );

    ref<ShaderObject> shader_object = make_ref<ShaderObject>(ref<Device>(this), rhi_shader_object);

    // Bind the debug printer to the new shader object, if enabled.
    if (m_debug_printer)
        m_debug_printer->bind(shader_object.get());

    return shader_object;
}

ref<ShaderObject> Device::create_shader_object(const TypeLayoutReflection* type_layout)
{
    Slang::ComPtr<rhi::IShaderObject> rhi_shader_object;
    SLANG_RHI_CALL(m_rhi_device->createShaderObjectFromTypeLayout(
        type_layout->get_slang_type_layout(),
        rhi_shader_object.writeRef()
    ));

    return make_ref<ShaderObject>(ref<Device>(this), rhi_shader_object);
}

ref<ShaderObject> Device::create_shader_object(ReflectionCursor cursor)
{
    SGL_CHECK(cursor.is_valid(), "Invalid reflection cursor");
    return create_shader_object(cursor.type_layout().get());
}

ref<ComputePipeline> Device::create_compute_pipeline(ComputePipelineDesc desc)
{
    return make_ref<ComputePipeline>(ref<Device>(this), std::move(desc));
}

ref<RenderPipeline> Device::create_render_pipeline(RenderPipelineDesc desc)
{
    return make_ref<RenderPipeline>(ref<Device>(this), std::move(desc));
}

ref<RayTracingPipeline> Device::create_ray_tracing_pipeline(RayTracingPipelineDesc desc)
{
    return make_ref<RayTracingPipeline>(ref<Device>(this), std::move(desc));
}

ref<ComputeKernel> Device::create_compute_kernel(ComputeKernelDesc desc)
{
    return make_ref<ComputeKernel>(ref(this), std::move(desc));
}

ref<CommandEncoder> Device::create_command_encoder(CommandQueueType queue)
{
    SGL_CHECK(queue == CommandQueueType::graphics, "Only graphics queue is supported.");

    Slang::ComPtr<rhi::ICommandEncoder> rhi_command_encoder;
    SLANG_RHI_CALL(m_rhi_graphics_queue->createCommandEncoder(rhi_command_encoder.writeRef()));
    return make_ref<CommandEncoder>(ref(this), rhi_command_encoder);
}

uint64_t Device::submit_command_buffers(
    std::span<CommandBuffer*> command_buffers,
    std::span<Fence*> wait_fences,
    std::span<uint64_t> wait_fence_values,
    std::span<Fence*> signal_fences,
    std::span<uint64_t> signal_fence_values,
    CommandQueueType queue,
    NativeHandle cuda_stream
)
{
    SGL_CHECK(queue == CommandQueueType::graphics, "Only graphics queue is supported.");

    bool has_wait_fence_values = wait_fence_values.size() > 0;
    bool has_signal_fence_values = signal_fence_values.size() > 0;

    if (has_wait_fence_values && wait_fence_values.size() != wait_fences.size())
        SGL_THROW("\"wait_fence_values\" size does not match \"wait_fences\" size.");
    if (has_signal_fence_values && signal_fence_values.size() != signal_fences.size())
        SGL_THROW("\"signal_fence_values\" size does not match \"signal_fences\" size.");

    SGL_CHECK(
        !cuda_stream.is_valid() || cuda_stream.type() == NativeHandleType::CUstream,
        "Native handle supplied for CUDA stream is not of type CUstream."
    );

    // Update hot reload system if created.
    // TODO(slang-rhi) need to make sure this is not too expensive.
    if (m_hot_reload)
        m_hot_reload->update();

    // Pointer to CUDA stream
    void* cuda_stream_ptr;
    if (m_desc.type == DeviceType::cuda) {
        // On CUDA backends, either take the stream specified or use 'invalid' to let the internal
        // default stream be used (which for the main queue is the NULL stream).
        cuda_stream_ptr
            = cuda_stream.is_valid() ? reinterpret_cast<void*>(cuda_stream.value()) : rhi::kInvalidCUDAStream;
    } else if (m_supports_cuda_interop) {
        // On non-CUDA backends, if CUDA interop is enabled, we always need to choose a stream to
        // sync with. We will eithe use the one specified, or the NULL stream.
        cuda_stream_ptr = cuda_stream.is_valid() ? reinterpret_cast<void*>(cuda_stream.value()) : nullptr;
    } else {
        // On non-CUDA backends, with interop off, it is invalid to specify a CUDA stream.
        SGL_CHECK(!cuda_stream.is_valid(), "CUDA stream is not supported on this device.");
        cuda_stream_ptr = nullptr;
    }

    short_vector<rhi::ICommandBuffer*, 8> rhi_command_buffers;
    short_vector<rhi::IFence*, 8> rhi_wait_fences;
    short_vector<uint64_t, 8> rhi_wait_fence_values;
    short_vector<rhi::IFence*, 8> rhi_signal_fences;
    short_vector<uint64_t, 8> rhi_signal_fence_values;

    // Will always enable CUDA sync if explicit stream provided.
    // If not, this will only be enabled if buffers were bound that have associated
    // CUDA interop allocations.
    bool needs_cuda_sync = cuda_stream.is_valid();

    for (CommandBuffer* command_buffer : command_buffers) {
        SGL_CHECK_NOT_NULL(command_buffer);
        rhi_command_buffers.push_back(command_buffer->rhi_command_buffer());
    }

    // Handle CUDA interop.
    if (m_supports_cuda_interop) {
        for (CommandBuffer* command_buffer : command_buffers) {
            for (const auto& buffer : command_buffer->m_cuda_interop_buffers) {
                buffer->copy_from_cuda(cuda_stream_ptr);
                needs_cuda_sync = true;
            }
        }

        if (needs_cuda_sync)
            sync_to_cuda(cuda_stream_ptr);
    }

    // Handle passed in wait fences.
    for (size_t i = 0; i < wait_fences.size(); ++i) {
        Fence* fence = wait_fences[i];
        SGL_CHECK_NOT_NULL(fence);
        rhi_wait_fences.push_back(fence->rhi_fence());
        uint64_t fence_value = has_wait_fence_values ? wait_fence_values[i] : Fence::AUTO;
        if (fence_value == Fence::AUTO)
            fence_value = fence->signaled_value();
        rhi_wait_fence_values.push_back(fence_value);
    }

    // Handle wait for global fence if needed.
    if (m_wait_global_fence) {
        rhi_wait_fences.push_back(m_global_fence->rhi_fence());
        rhi_wait_fence_values.push_back(m_global_fence->signaled_value());
    }

    // Handle passed in signal fences.
    for (size_t i = 0; i < signal_fences.size(); ++i) {
        Fence* fence = signal_fences[i];
        SGL_CHECK_NOT_NULL(fence);
        rhi_signal_fences.push_back(fence->rhi_fence());
        uint64_t fence_value = has_signal_fence_values ? signal_fence_values[i] : Fence::AUTO;
        if (fence_value == Fence::AUTO)
            fence_value = fence->update_signaled_value();
        rhi_signal_fence_values.push_back(fence_value);
    }

    // Handle signal for global fence.
    rhi_signal_fences.push_back(m_global_fence->rhi_fence());
    rhi_signal_fence_values.push_back(m_global_fence->update_signaled_value());

    // Handle actual submit.
    SGL_ASSERT(rhi_wait_fences.size() == rhi_wait_fence_values.size());
    SGL_ASSERT(rhi_signal_fences.size() == rhi_signal_fence_values.size());
    rhi::SubmitDesc rhi_submit_desc{
        .commandBuffers = rhi_command_buffers.data(),
        .commandBufferCount = narrow_cast<uint32_t>(rhi_command_buffers.size()),
        .waitFences = rhi_wait_fences.data(),
        .waitFenceValues = rhi_wait_fence_values.data(),
        .waitFenceCount = narrow_cast<uint32_t>(rhi_wait_fences.size()),
        .signalFences = rhi_signal_fences.data(),
        .signalFenceValues = rhi_signal_fence_values.data(),
        .signalFenceCount = narrow_cast<uint32_t>(rhi_signal_fences.size()),
        .cudaStream = cuda_stream_ptr,
    };
    SLANG_RHI_CALL(m_rhi_graphics_queue->submit(rhi_submit_desc));
    m_wait_global_fence = false;

    // Handle CUDA interop.
    if (m_supports_cuda_interop && needs_cuda_sync) {
        sync_to_device(cuda_stream_ptr);

        for (CommandBuffer* command_buffer : command_buffers) {
            for (const auto& buffer : command_buffer->m_cuda_interop_buffers) {
                if (buffer->is_uav())
                    buffer->copy_to_cuda(cuda_stream_ptr);
            }
        }
    }

    return m_global_fence->signaled_value();
}

uint64_t Device::submit_command_buffer(CommandBuffer* command_buffer, CommandQueueType queue, NativeHandle cuda_stream)
{
    CommandBuffer* command_buffers[] = {command_buffer};
    return submit_command_buffers(command_buffers, {}, {}, {}, {}, queue, cuda_stream);
}

bool Device::is_submit_finished(uint64_t id)
{
    return id <= m_global_fence->current_value();
}

void Device::wait_for_submit(uint64_t id)
{
    m_global_fence->wait(id);
}

void Device::wait_for_idle(CommandQueueType queue)
{
    if (m_rhi_graphics_queue) {
        SGL_CHECK(queue == CommandQueueType::graphics, "Only graphics queue is supported.");
        m_rhi_graphics_queue->waitOnHost();
    }
}

void Device::sync_to_cuda(void* cuda_stream)
{
    // Signal fence from CUDA, wait for it on graphics queue.
    if (m_supports_cuda_interop) {
        SGL_CU_SCOPE(this);
        uint64_t signal_value = m_global_fence->update_signaled_value();
        m_cuda_semaphore->signal(signal_value, CUstream(cuda_stream));
        m_wait_global_fence = true;
    }
}

void Device::sync_to_device(void* cuda_stream)
{
    if (m_supports_cuda_interop) {
        SGL_CU_SCOPE(this);
        m_cuda_semaphore->wait(m_global_fence->signaled_value(), CUstream(cuda_stream));
    }
}

void Device::flush_print()
{
    if (m_debug_printer)
        m_debug_printer->flush();
}

std::string Device::flush_print_to_string()
{
    return m_debug_printer ? m_debug_printer->flush_to_string() : "";
}

void Device::wait()
{
    wait_for_idle();
}

void Device::upload_buffer_data(Buffer* buffer, size_t offset, size_t size, const void* data)
{
    auto command_encoder = create_command_encoder();
    command_encoder->upload_buffer_data(buffer, offset, size, data);
    submit_command_buffer(command_encoder->finish());
}

void Device::read_buffer_data(const Buffer* buffer, void* data, size_t size, size_t offset)
{
    SGL_CHECK_NOT_NULL(buffer);
    SGL_CHECK(offset + size <= buffer->size(), "Buffer read is out of bounds");
    SGL_CHECK_NOT_NULL(data);

    SLANG_RHI_CALL(m_rhi_device->readBuffer(buffer->rhi_buffer(), offset, size, data));
}

void Device::upload_texture_data(
    Texture* texture,
    SubresourceRange subresource_range,
    uint3 offset,
    uint3 extent,
    std::span<SubresourceData> subresource_data
)
{
    ref<CommandEncoder> command_encoder = create_command_encoder();
    command_encoder->upload_texture_data(texture, subresource_range, offset, extent, subresource_data);
    submit_command_buffer(command_encoder->finish());
}

void Device::upload_texture_data(Texture* texture, uint32_t layer, uint32_t mip, SubresourceData subresource_data)
{
    ref<CommandEncoder> command_encoder = create_command_encoder();
    command_encoder->upload_texture_data(texture, layer, mip, subresource_data);
    submit_command_buffer(command_encoder->finish());
}

OwnedSubresourceData Device::read_texture_data(const Texture* texture, uint32_t layer, uint32_t mip)
{
    SGL_CHECK_NOT_NULL(texture);
    SGL_CHECK_LT(layer, texture->layer_count());
    SGL_CHECK_LT(mip, texture->mip_count());

    // Query layout information.
    rhi::SubresourceLayout rhi_layout;
    SLANG_RHI_CALL(texture->rhi_texture()->getSubresourceLayout(mip, &rhi_layout));

    // Setup owned sub resource data that can contain the results.
    OwnedSubresourceData subresource_data;
    subresource_data.owned_data = std::make_unique<uint8_t[]>(rhi_layout.sizeInBytes);
    subresource_data.data = subresource_data.owned_data.get();
    subresource_data.size = rhi_layout.sizeInBytes;
    subresource_data.row_pitch = rhi_layout.rowPitch;
    subresource_data.slice_pitch = rhi_layout.slicePitch;

    // Read texture data.
    SLANG_RHI_CALL(
        m_rhi_device->readTexture(texture->rhi_texture(), layer, mip, rhi_layout, subresource_data.owned_data.get())
    );

    return subresource_data;
}

std::array<NativeHandle, 3> Device::native_handles() const
{
    rhi::DeviceNativeHandles handles = {};
    SLANG_RHI_CALL(m_rhi_device->getNativeDeviceHandles(&handles));
    return {NativeHandle(handles.handles[0]), NativeHandle(handles.handles[1]), NativeHandle(handles.handles[2])};
}

NativeHandle Device::get_native_command_queue_handle(CommandQueueType queue) const
{
    SGL_CHECK(queue == CommandQueueType::graphics, "Only graphics queue is supported.");
    rhi::NativeHandle rhi_handle = {};
    SLANG_RHI_CALL(m_rhi_graphics_queue->getNativeHandle(&rhi_handle));
    return NativeHandle(rhi_handle);
}

std::vector<AdapterInfo> Device::enumerate_adapters(DeviceType type)
{
    if (type == DeviceType::automatic) {
#if SGL_WINDOWS
        type = DeviceType::d3d12;
#elif SGL_LINUX
        type = DeviceType::vulkan;
#elif SGL_MACOS
        type = DeviceType::metal;
#endif
    }

    rhi::AdapterList rhi_adapters = rhi::getRHI()->getAdapters(static_cast<rhi::DeviceType>(type));

    std::vector<AdapterInfo> adapters(rhi_adapters.getCount());
    for (size_t i = 0; i < adapters.size(); ++i) {
        const auto& rhi_adapter = rhi_adapters.getAdapters()[i];
        adapters[i] = AdapterInfo{
            .name = rhi_adapter.name,
            .vendor_id = rhi_adapter.vendorID,
            .device_id = rhi_adapter.deviceID,
            .luid = from_rhi(rhi_adapter.luid),
        };
    }

    return adapters;
}

std::vector<ref<Device>> Device::get_created_devices()
{
    std::lock_guard lock(s_devices_mutex);

    std::vector<ref<Device>> res;
    res.reserve(s_devices.size());
    for (Device* device : s_devices) {
        res.push_back(ref<Device>(device));
    }
    return res;
}

void Device::report_live_objects()
{
    rhi::getRHI()->reportLiveObjects();
}

bool Device::enable_agility_sdk()
{
#if SGL_HAS_D3D12 && SGL_HAS_AGILITY_SDK
    std::filesystem::path exe_dir = platform::executable_directory();
    std::filesystem::path sdk_dir = platform::runtime_directory() / SLANG_RHI_AGILITY_SDK_PATH;

    // Agility SDK can only be loaded from a relative path to the executable.
    // Make sure both paths use the same drive letter.
    if (std::tolower(exe_dir.string()[0]) != std::tolower(sdk_dir.string()[0])) {
        log_warn(
            "Cannot enable D3D12 Agility SDK: "
            "Executable directory \"{}\" is not on the same drive as the SDK directory \"{}\".",
            exe_dir,
            sdk_dir
        );
        return false;
    }

    // Get relative path and make sure there is the required trailing path delimiter.
    auto rel_path = std::filesystem::relative(sdk_dir, exe_dir) / "";

    // Load D3D12 library.
    LoadLibraryA("d3d12.dll");
    HMODULE handle = GetModuleHandleA("d3d12.dll");

    // Get the D3D12GetInterface procedure.
    typedef HRESULT(WINAPI * D3D12GetInterfaceFn)(REFCLSID rclsid, REFIID riid, void** ppvDebug);
    D3D12GetInterfaceFn pD3D12GetInterface
        = handle ? (D3D12GetInterfaceFn)GetProcAddress(handle, "D3D12GetInterface") : nullptr;
    if (!pD3D12GetInterface) {
        log_warn("Cannot enable D3D12 Agility SDK: "
                 "Failed to get D3D12GetInterface.");
        return false;
    }

    // Local definition of CLSID_D3D12SDKConfiguration from d3d12.h
    const GUID CLSID_D3D12SDKConfiguration__
        = {0x7cda6aca, 0xa03e, 0x49c8, {0x94, 0x58, 0x03, 0x34, 0xd2, 0x0e, 0x07, 0xce}};
    // Get the D3D12SDKConfiguration interface.
    _COM_SMARTPTR_TYPEDEF(ID3D12SDKConfiguration, __uuidof(ID3D12SDKConfiguration));
    ID3D12SDKConfigurationPtr pD3D12SDKConfiguration;
    if (!SUCCEEDED(pD3D12GetInterface(CLSID_D3D12SDKConfiguration__, IID_PPV_ARGS(&pD3D12SDKConfiguration)))) {
        log_warn("Cannot enable D3D12 Agility SDK: "
                 "Failed to get D3D12SDKConfiguration interface.");
        return false;
    }

    // Set the SDK version and path.
    if (!SUCCEEDED(pD3D12SDKConfiguration->SetSDKVersion(SLANG_RHI_AGILITY_SDK_VERSION, rel_path.string().c_str()))) {
        log_warn("Cannot enable D3D12 Agility SDK: "
                 "Calling SetSDKVersion failed.");
        return false;
    }

    return true;
#endif
    return false;
}

std::string Device::to_string() const
{
    return fmt::format(
        "Device(\n"
        "  type = {},\n"
        "  adapter_name = \"{}\",\n"
        "  adapter_luid = {},\n"
        "  enable_debug_layers = {},\n"
        "  enable_cuda_interop = {},\n"
        "  enable_print = {},\n"
        "  enable_hot_reload = {},\n"
        "  enable_compilation_reports = {},\n"
        "  supported_shader_model = {},\n"
        "  shader_cache_enabled = {},\n"
        "  shader_cache_path = \"{}\"\n"
        ")",
        m_info.type,
        m_info.adapter_name,
        string::hexlify(m_info.adapter_luid),
        m_desc.enable_debug_layers,
        m_desc.enable_cuda_interop,
        m_desc.enable_print,
        m_desc.enable_hot_reload,
        m_desc.enable_compilation_reports,
        m_supported_shader_model,
        m_shader_cache_enabled,
        m_shader_cache_path
    );
}

Blitter* Device::_blitter()
{
    if (!m_blitter)
        m_blitter = ref(new Blitter(this));
    return m_blitter;
}

std::array<NativeHandle, 3> get_cuda_current_context_native_handles()
{
    std::array<NativeHandle, 3> handles;

    CUcontext cu_context;
    SGL_CHECK(rhiCudaDriverApiInit(), "Failed to initialize CUDA driver API.");
    SGL_CU_CHECK(cuCtxGetCurrent(&cu_context));
    SGL_CHECK(cu_context, "No current CUDA context found.");

    CUdevice cu_device;
    SGL_CU_CHECK(cuCtxGetDevice(&cu_device));

    handles[0] = NativeHandle(cu_device);
    handles[1] = NativeHandle(cu_context);

    return handles;
}


} // namespace sgl
