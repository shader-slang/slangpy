// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"
#include "sgl/device/native_handle.h"
#include "sgl/device/resource.h"
#include "sgl/device/shader.h"
#include "sgl/device/raytracing.h"
#include "sgl/device/coopvec.h"

#include "sgl/core/fwd.h"
#include "sgl/core/config.h"
#include "sgl/core/macros.h"
#include "sgl/core/enum.h"
#include "sgl/core/object.h"
#include "sgl/core/platform.h"
#include "sgl/math/vector_types.h"

#include <slang-rhi.h>

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <unordered_set>

namespace sgl {

class DebugPrinter;


struct AdapterInfo {
    /// Descriptive name of the adapter.
    std::string name;

    /// Unique identifier for the vendor (only available for D3D12 and Vulkan).
    uint32_t vendor_id;

    /// Unique identifier for the physical device among devices from the vendor (only available for D3D12 and Vulkan).
    uint32_t device_id;

    /// Logically unique identifier of the adapter.
    AdapterLUID luid;

    std::string to_string() const
    {
        return fmt::format(
            "AdapterInfo(\n"
            "  name = \"{}\",\n"
            "  vendor_id = 0x{:x},\n"
            "  device_id = 0x{:x},\n"
            "  luid = {}\n"
            ")",
            name,
            vendor_id,
            device_id,
            luid
        );
    }
};

enum class DeviceType : uint32_t {
    automatic = static_cast<uint32_t>(rhi::DeviceType::Default),
    d3d12 = static_cast<uint32_t>(rhi::DeviceType::D3D12),
    vulkan = static_cast<uint32_t>(rhi::DeviceType::Vulkan),
    metal = static_cast<uint32_t>(rhi::DeviceType::Metal),
    wgpu = static_cast<uint32_t>(rhi::DeviceType::WGPU),
    cpu = static_cast<uint32_t>(rhi::DeviceType::CPU),
    cuda = static_cast<uint32_t>(rhi::DeviceType::CUDA),
};

SGL_ENUM_INFO(
    DeviceType,
    {
        {DeviceType::automatic, "automatic"},
        {DeviceType::d3d12, "d3d12"},
        {DeviceType::vulkan, "vulkan"},
        {DeviceType::metal, "metal"},
        {DeviceType::wgpu, "wgpu"},
        {DeviceType::cpu, "cpu"},
        {DeviceType::cuda, "cuda"},
    }
);
SGL_ENUM_REGISTER(DeviceType);

struct DeviceDesc {
    /// The type of the device.
    DeviceType type{DeviceType::automatic};

    /// Enable debug layers.
    bool enable_debug_layers{false};

    /// Enable CUDA interoperability.
    bool enable_cuda_interop{false};

    /// Enable device side printing (adds performance overhead).
    bool enable_print{false};

    /// Enable automatic shader reload in response to file changes.
    /// Note: Currently windows and linux only.
    bool enable_hot_reload{true};

    /// Enable compilation reports.
    bool enable_compilation_reports{false};

    /// Adapter LUID to select adapter on which the device will be created.
    std::optional<AdapterLUID> adapter_luid;

    /// Compiler options (used for default slang session).
    SlangCompilerOptions compiler_options;

    /// Path to the shader cache directory (optional).
    /// If a relative path is used, the cache is stored in the application data directory.
    std::optional<std::filesystem::path> shader_cache_path;

    /// Native device handles for initializing with externally created device. Currenlty
    /// only used for CUDA interoperability.
    std::array<NativeHandle, 3> existing_device_handles;

    /// Debug label
    std::string label;
};

struct DeviceLimits {
    /// Maximum dimension for 1D textures.
    uint32_t max_texture_dimension_1d;
    /// Maximum dimensions for 2D textures.
    uint32_t max_texture_dimension_2d;
    /// Maximum dimensions for 3D textures.
    uint32_t max_texture_dimension_3d;
    /// Maximum dimensions for cube textures.
    uint32_t max_texture_dimension_cube;
    /// Maximum number of texture layers.
    uint32_t max_texture_layers;

    /// Maximum number of vertex input elements in a graphics pipeline.
    uint32_t max_vertex_input_elements;
    /// Maximum offset of a vertex input element in the vertex stream.
    uint32_t max_vertex_input_element_offset;
    /// Maximum number of vertex streams in a graphics pipeline.
    uint32_t max_vertex_streams;
    /// Maximum stride of a vertex stream.
    uint32_t max_vertex_stream_stride;

    /// Maximum number of threads per thread group.
    uint32_t max_compute_threads_per_group;
    /// Maximum dimensions of a thread group.
    uint3 max_compute_thread_group_size;
    /// Maximum number of thread groups per dimension in a single dispatch.
    uint3 max_compute_dispatch_thread_groups;

    /// Maximum number of viewports per pipeline.
    uint32_t max_viewports;
    /// Maximum viewport dimensions.
    uint2 max_viewport_dimensions;
    /// Maximum framebuffer dimensions.
    uint3 max_framebuffer_dimensions;

    /// Maximum samplers visible in a shader stage.
    uint32_t max_shader_visible_samplers;
};

struct DeviceInfo {
    /// The type of the device.
    DeviceType type;
    /// The name of the graphics API being used by this device.
    std::string api_name;
    /// The name of the graphics adapter.
    std::string adapter_name;
    /// The logically unique identifier of the graphics adapter.
    AdapterLUID adapter_luid;
    /// The frequency of the timestamp counter.
    /// To resolve a timestamp to seconds, divide by this value.
    uint64_t timestamp_frequency;
    /// Limits of the device.
    DeviceLimits limits;
};

struct ShaderCacheStats {
    /// Number of entries in the cache.
    size_t entry_count;
    /// Number of hits in the cache.
    size_t hit_count;
    /// Number of misses in the cache.
    size_t miss_count;
};

/// Event data for hot reload hook.
struct ShaderHotReloadEvent { };
using ShaderHotReloadCallback = std::function<void(const ShaderHotReloadEvent&)>;


using DeviceCloseCallback = std::function<void(Device*)>;

class SGL_API Device : public Object {
    SGL_OBJECT(Device)
public:
    Device(const DeviceDesc& desc = DeviceDesc{});
    ~Device();

    /// Release all underlying slang-rhi resources.
    /// This is used as a workaround during shutdown, to ensure all resources are released
    /// when slangpy fails to clean up properly due to reference cycles introduced in Python.
    void _release_rhi_resources();

    static ref<Device> create(const DeviceDesc& desc = DeviceDesc{}) { return make_ref<Device>(desc); }

    const DeviceDesc& desc() const { return m_desc; }

    /// Type of the graphics API used by this device.
    DeviceType type() const { return m_desc.type; }

    /// Device information.
    const DeviceInfo& info() const { return m_info; }

    /// Shader cache statistics.
    ShaderCacheStats shader_cache_stats() const;

    /// The highest shader model supported by the device.
    ShaderModel supported_shader_model() const { return m_supported_shader_model; }

    /// List of features supported by the device.
    const std::vector<Feature>& features() const { return m_features; }

    /// Check if the device supports a given feature.
    bool has_feature(Feature feature) const;

    /// True if the device supports CUDA interoperability.
    bool supports_cuda_interop() const { return m_supports_cuda_interop; }

    /// Returns the supported resource states for a given format.
    FormatSupport get_format_support(Format format) const;

    /// Default slang session.
    SlangSession* slang_session() const { return m_slang_session; }

    /**
     * \brief Close the device.
     *
     * This function should be called before the device is released.
     * It waits for all pending work to be completed and releases internal
     * resources, removing all cyclic references that might prevent the device
     * from being destroyed. After closing the device, no new resources must be
     * created and no new work must be submitted.
     *
     * \note The Python extension will automatically close all open devices
     * when the interpreter is terminated through an `atexit` handler. If a
     * device is to be destroyed at runtime, it must be closed explicitly.
     */
    void close();

    /// Close all open devices.
    static void close_all_devices();

    /// Release all slang-rhi resources from all devices.
    /// This is used as a workaround during shutdown, to ensure all resources are released
    /// when slangpy fails to clean up properly due to reference cycles introduced in Python.
    /// This MUST NOT be called by the user!
    static void _release_all_rhi_resources();

    // Resource creation

    /**
     * \brief Create a new surface.
     *
     * \param window Window to create the surface for.
     * \return New surface object.
     */
    ref<Surface> create_surface(Window* window);

    /**
     * \brief Create a new surface.
     *
     * \param window_handle Native window handle to create the surface for.
     * \return New surface object.
     */
    ref<Surface> create_surface(WindowHandle window_handle);

    /**
     * \brief Create a new buffer.
     *
     * \param size Buffer size in bytes.
     * \param element_count Buffer size in number of struct elements. Can be used instead of \c size.
     * \param struct_size Struct size in bytes.
     * \param resource_type_layout Resource type layout of the buffer. Can be used instead of \c struct_size to specify the size of the struct.
     * \param format Buffer format. Used when creating typed buffer views.
     * \param initial_state Initial resource state.
     * \param usage Resource usage flags.
     * \param memory_type Memory type.
     * \param label Debug label.
     * \param data Initial data to upload to the buffer.
     * \param data_size Size of the initial data in bytes.
     * \return New buffer object.
     */
    ref<Buffer> create_buffer(BufferDesc desc);

    ref<BufferView> create_buffer_view(Buffer* buffer, BufferViewDesc desc);

    /**
     * \brief Create a new texture.
     *
     * \param type Texture type.
     * \param format Texture format.
     * \param width Width in pixels.
     * \param height Height in pixels.
     * \param depth Depth in pixels.
     * \param array_length Array length.
     * \param mip_count Mip level count. Number of mip levels (ALL_MIPS for all mip levels).
     * \param sample_count Number of samples for multisampled textures.
     * \param quality Quality level for multisampled textures.
     * \param usage Resource usage.
     * \param memory_type Memory type.
     * \param label Debug label.
     * \param data Initial data.
     * \return New texture object.
     */
    ref<Texture> create_texture(TextureDesc desc);

    ref<Texture> create_texture_from_resource(TextureDesc desc, rhi::ITexture* resource);

    ref<TextureView> create_texture_view(Texture* texture, TextureViewDesc desc);

    /**
     * \brief Create a new sampler.
     *
     * \param min_filter Minification filter.
     * \param mag_filter Magnification filter.
     * \param mip_filter Mip-map filter.
     * \param reduction_op Reduction operation.
     * \param address_u Texture addressing mode for the U coordinate.
     * \param address_v Texture addressing mode for the V coordinate.
     * \param address_w Texture addressing mode for the W coordinate.
     * \param mip_lod_bias Mip-map LOD bias.
     * \param max_anisotropy Maximum anisotropy.
     * \param comparison_func Comparison function.
     * \param border_color Border color.
     * \param min_lod Minimum LOD level.
     * \param max_lod Maximum LOD level.
     * \param label Debug label.
     * \return New sampler object.
     */
    ref<Sampler> create_sampler(SamplerDesc desc);

    /**
     * \brief Create a new fence.
     *
     * \param initial_value Initial fence value.
     * \param shared Create a shared fence.
     * \return New fence object.
     */
    ref<Fence> create_fence(FenceDesc desc);

    /**
     * \brief Create a new query pool.
     *
     * \param type Query type.
     * \param count Number of queries in the pool.
     * \return New query pool object.
     */
    ref<QueryPool> create_query_pool(QueryPoolDesc desc);

    /**
     * \brief Create a new input layout.
     *
     * \param input_elements List of input elements (see \ref InputElementDesc for details).
     * \param vertex_streams List of vertex streams (see \ref VertexStreamDesc for details).
     * \return New input layout object.
     */
    ref<InputLayout> create_input_layout(InputLayoutDesc desc);

    /**
     * \brief Query the device for buffer sizes required for acceleration structure builds.
     *
     * \param desc Acceleration structure build description.
     * \return Acceleration structure sizes.
     */
    AccelerationStructureSizes get_acceleration_structure_sizes(const AccelerationStructureBuildDesc& desc);

    ref<AccelerationStructure> create_acceleration_structure(AccelerationStructureDesc desc);

    ref<AccelerationStructureInstanceList> create_acceleration_structure_instance_list(size_t size);

    ref<ShaderTable> create_shader_table(ShaderTableDesc desc);

    /**
     * Get coop vec instance
     */
    ref<CoopVec> get_or_create_coop_vec();


    /**
     * \brief Create a new slang session.
     *
     * \param compiler_options Compiler options (see \ref SlangCompilerOptions for details).
     * \return New slang session object.
     */
    ref<SlangSession> create_slang_session(SlangSessionDesc desc);

    ref<SlangModule> load_module(std::string_view module_name);

    ref<SlangModule> load_module_from_source(
        std::string_view module_name,
        std::string_view source,
        std::optional<std::filesystem::path> path = {}
    );

    ref<ShaderProgram> link_program(
        std::vector<ref<SlangModule>> modules,
        std::vector<ref<SlangEntryPoint>> entry_points,
        std::optional<SlangLinkOptions> link_options = {}
    );

    ref<ShaderProgram> load_program(
        std::string_view module_name,
        std::vector<std::string_view> entry_point_names,
        std::optional<std::string_view> additional_source = {},
        std::optional<SlangLinkOptions> link_options = {}
    );

    void reload_all_programs();

    ref<ShaderObject> create_root_shader_object(const ShaderProgram* shader_program);

    ref<ShaderObject> create_shader_object(const TypeLayoutReflection* type_layout);

    ref<ShaderObject> create_shader_object(ReflectionCursor cursor);

    ref<ComputePipeline> create_compute_pipeline(ComputePipelineDesc desc);

    ref<RenderPipeline> create_render_pipeline(RenderPipelineDesc desc);

    ref<RayTracingPipeline> create_ray_tracing_pipeline(RayTracingPipelineDesc desc);

    ref<ComputeKernel> create_compute_kernel(ComputeKernelDesc desc);

    ref<CommandEncoder> create_command_encoder(CommandQueueType queue = CommandQueueType::graphics);

    /**
     * \brief Submit a list of command buffers to the device.
     *
     * The returned submission ID can be used to wait for the submission to complete.
     *
     * The wait fence values are optional. If not provided, the fence values will be set to AUTO, which means waiting
     * for the last signaled value.
     *
     * The signal fence values are optional. If not provided, the fence values will be set to AUTO, which means
     * incrementing the last signaled value by 1.     *
     *
     * \param command_buffers List of command buffers to submit.
     * \param wait_fences List of fences to wait for before executing the command buffers.
     * \param wait_fence_values List of fence values to wait for before executing the command buffers.
     * \param signal_fences List of fences to signal after executing the command buffers.
     * \param signal_fence_values List of fence values to signal after executing the command buffers.
     * \param queue Command queue to submit to.
     * \param cuda_stream On none-CUDA backends, when interop is enabled, this is the stream to sync with
     * before/after submission (assuming any resources are shared with CUDA) and use for internal copies. If not
     * specified, sync will happen with the NULL (default) CUDA stream.
     * On CUDA backends, this is the CUDA stream to use for the submission. If not specified, the
     * default stream of the command queue will be used, which for CommandQueueType::graphics is the NULL stream.
     * It is an error to specify a stream for none-CUDA backends that have interop disabled.
     * \return Submission ID.
     */
    uint64_t submit_command_buffers(
        std::span<CommandBuffer*> command_buffers,
        std::span<Fence*> wait_fences = {},
        std::span<uint64_t> wait_fence_values = {},
        std::span<Fence*> signal_fences = {},
        std::span<uint64_t> signal_fence_values = {},
        CommandQueueType queue = CommandQueueType::graphics,
        NativeHandle cuda_stream = NativeHandle()
    );

    /**
     * \brief Submit a command buffer to the device.
     *
     * The returned submission ID can be used to wait for the submission to complete.
     *
     * \param command_buffer Command buffer to submit.
     * \param queue Command queue to submit to.
     * \return Submission ID.
     */
    uint64_t submit_command_buffer(
        CommandBuffer* command_buffer,
        CommandQueueType queue = CommandQueueType::graphics,
        NativeHandle cuda_stream = {}
    );

    /**
     * \brief Check if a submission is finished executing.
     *
     * \param id Submission ID.
     * \return True if the submission is finished executing.
     */
    bool is_submit_finished(uint64_t id);

    /**
     * \brief Wait for a submission to finish execution.
     *
     * \param id Submission ID.
     */
    void wait_for_submit(uint64_t id);

    /**
     * \brief Wait for the command queue to be idle.
     *
     * \param queue Command queue to wait for.
     */
    void wait_for_idle(CommandQueueType queue = CommandQueueType::graphics);

    /**
     * \brief Synchronize CUDA -> device.
     *
     * This signals a shared CUDA semaphore from the CUDA stream and then waits for the signal on the command queue.
     *
     * \param cuda_stream CUDA stream
     */
    void sync_to_cuda(void* cuda_stream = 0);

    /**
     * \brief Synchronize device -> CUDA.
     *
     * This waits for a shared CUDA semaphore on the CUDA stream, making sure all commands on the device have completed.
     *
     * \param cuda_stream CUDA stream
     */
    void sync_to_device(void* cuda_stream = 0);

    DebugPrinter* debug_printer() const { return m_debug_printer.get(); }

    /// Block and flush all shader side debug print output.
    void flush_print();

    /// Block and flush all shader side debug print output to a string.
    std::string flush_print_to_string();

    /// Wait for all device work to complete.
    void wait();

    /**
     * Upload host memory to buffer.
     *
     * \param buffer Buffer to write to.
     * \param offset Offset in the buffer to write to.
     * \param size Size of the data in bytes.
     * \param data Data to write.
     */
    void upload_buffer_data(Buffer* buffer, size_t offset, size_t size, const void* data);

    /**
     * Read buffer data to host memory.
     * \note This will wait until the data is copied back to host memory.
     *
     * \param buffer Buffer to read from.
     * \param data Data to buffer to read to.
     * \param size Size of the data in bytes.
     * \param offset Offset in the buffer to read from.
     */
    void read_buffer_data(const Buffer* buffer, void* data, size_t size, size_t offset = 0);

    /**
     * Upload host memory to texture.
     *
     * \param texture Texture to write to.
     * \param subresource Subresource index.
     * \param subresource_data Subresource data.
     */
    void upload_texture_data(
        Texture* texture,
        SubresourceRange subresource_range,
        uint3 offset,
        uint3 extent,
        std::span<SubresourceData> subresource_data
    );

    void upload_texture_data(Texture* texture, uint32_t layer, uint32_t mip, SubresourceData subresource_data);

    /**
     * Read texture data to host memory.
     * \note This will wait until the data is copied back to host memory.
     *
     * \param texture Texture to read from.
     * \param layer Layer index.
     * \param mip Mip level.
     * \return Subresource data in host memory.
     */
    OwnedSubresourceData read_texture_data(const Texture* texture, uint32_t layer, uint32_t mip);

    rhi::IDevice* rhi_device() const { return m_rhi_device; }
    rhi::ICommandQueue* rhi_graphics_queue() const { return m_rhi_graphics_queue; }

    slang::IGlobalSession* global_session() const { return m_global_session; }

    /// Get the native device handles.
    std::array<NativeHandle, 3> native_handles() const;

    /// Returns the native API handle for the command queue:
    /// - D3D12: ID3D12CommandQueue*
    /// - Vulkan: VkQueue (Vulkan)
    NativeHandle get_native_command_queue_handle(CommandQueueType queue = CommandQueueType::graphics) const;


    /// Enumerates all available adapters of a given device type.
    static std::vector<AdapterInfo> enumerate_adapters(DeviceType type = DeviceType::automatic);

    /// Lists all created devices
    static std::vector<ref<Device>> get_created_devices();

    /// Report live objects in the rhi layer.
    /// This is useful for checking clean shutdown with all resources released properly.
    static void report_live_objects();

    /**
     * \brief Report status of internal heaps used by the device.
     *
     * \return List of heap reports containing heap names and allocation information.
     */
    std::vector<HeapReport> report_heaps();

    /**
     * Try to enable D3D12 Agility SDK at runtime.
     * Note: This must be called before creating a device to have any effect.
     *
     * Prefer adding SGL_EXPORT_AGILITY_SDK to the main translation unit of executables
     * to tag the application binary to load the D3D12 Agility SDK.
     *
     * When using sgl through as a Python extension tagging the main application
     * (Python interpreter) is not possible. The alternative is to use the
     * D3D12SDKConfiguration API introduced in Windows SDK 20348. This however
     * requires "Developer Mode" to be enabled and the executed Python interpreter to be
     * stored on the same drive as the sgl library.
     *
     * \return Return true if D3D12 Agility SDK was successfully enabled.
     */
    static bool enable_agility_sdk();

    /// Register a hot reload hook, called immediately after any module is reloaded.
    void register_shader_hot_reload_callback(ShaderHotReloadCallback call_back)
    {
        m_shader_hot_reload_callbacks.push_back(call_back);
    }

    /// Register a device close callback, called at start of device close.
    void register_device_close_callback(DeviceCloseCallback call_back)
    {
        m_device_close_callbacks.push_back(call_back);
    }

    cuda::Device* cuda_device() const { return m_cuda_device.get(); }

    std::string to_string() const override;

    Blitter* _blitter();
    HotReload* _hot_reload() { return m_hot_reload; }

    /// Called by hot reload system after reload occurs, to trigger the hooks.
    void _on_hot_reload()
    {
        for (auto& hook : m_shader_hot_reload_callbacks)
            hook({});
    }

    void _register_device_child(DeviceChild* device_child);
    void _unregister_device_child(DeviceChild* device_child);

private:
    DeviceDesc m_desc;
    DeviceInfo m_info;
    ShaderModel m_supported_shader_model{ShaderModel::unknown};

    bool m_closed{false};

    bool m_shader_cache_enabled{false};
    std::filesystem::path m_shader_cache_path;

    Slang::ComPtr<rhi::IDevice> m_rhi_device;
    Slang::ComPtr<rhi::ICommandQueue> m_rhi_graphics_queue;
    Slang::ComPtr<slang::IGlobalSession> m_global_session;

    ref<SlangSession> m_slang_session;

    std::vector<Feature> m_features;

    ref<Fence> m_global_fence;

    std::unique_ptr<DebugPrinter> m_debug_printer;

    /// List of callbacks for hot reload event
    std::vector<ShaderHotReloadCallback> m_shader_hot_reload_callbacks;

    /// List of callbacks for shutdown event
    std::vector<DeviceCloseCallback> m_device_close_callbacks;

    ref<Blitter> m_blitter;
    ref<HotReload> m_hot_reload;
    ref<CoopVec> m_coop_vec;

    bool m_supports_cuda_interop{false};
    ref<cuda::Device> m_cuda_device;
    ref<cuda::ExternalSemaphore> m_cuda_semaphore;

    std::mutex m_device_children_mutex;
    std::unordered_set<DeviceChild*> m_device_children;
};

/// Gets the device and context handles for the current CUDA context. Use
/// to retrieve an existing context (eg from PyTorch) to pass as the existing_device_handles
/// from which to create a device in the DeviceDesc.
SGL_API std::array<NativeHandle, 3> get_cuda_current_context_native_handles();

} // namespace sgl
