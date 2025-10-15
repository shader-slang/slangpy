// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"
#include "sgl/device/fence.h"
#include "sgl/device/resource.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/raytracing.h"

#include "sgl/core/object.h"
#include "sgl/core/static_vector.h"
#include "sgl/math/vector_types.h"

#include <span>

namespace sgl {

struct RenderState {
    uint32_t stencil_ref{0};
    static_vector<Viewport, 16> viewports;
    static_vector<ScissorRect, 16> scissor_rects;
    static_vector<BufferOffsetPair, 16> vertex_buffers;
    BufferOffsetPair index_buffer;
    IndexFormat index_format{IndexFormat::uint32};
};

struct RenderPassColorAttachment {
    TextureView* view{nullptr};
    TextureView* resolve_target{nullptr};
    LoadOp load_op{LoadOp::clear};
    StoreOp store_op{StoreOp::store};
    float4 clear_value{0.f};
};

struct RenderPassDepthStencilAttachment {
    TextureView* view{nullptr};
    LoadOp depth_load_op{LoadOp::clear};
    StoreOp depth_store_op{StoreOp::store};
    float depth_clear_value{1.f};
    bool depth_read_only{false};
    LoadOp stencil_load_op{LoadOp::clear};
    StoreOp stencil_store_op{StoreOp::store};
    uint8_t stencil_clear_value{0};
    bool stencil_read_only{false};
};

struct RenderPassDesc {
    std::vector<RenderPassColorAttachment> color_attachments;
    std::optional<RenderPassDepthStencilAttachment> depth_stencil_attachment;
};


class SGL_API PassEncoder : public Object {
    SGL_OBJECT(PassEncoder)
public:
    /// Push a debug group.
    void push_debug_group(const char* name, float3 color);

    /// Pop a debug group.
    void pop_debug_group();

    /**
     * Insert a debug marker.
     * \param name Name of the marker.
     * \param color Color of the marker.
     */
    void insert_debug_marker(const char* name, float3 color);

    /**
     * \brief Write a timestamp.
     *
     * \param query_pool Query pool.
     * \param index Index of the query.
     */
    void write_timestamp(QueryPool* query_pool, uint32_t index);

    virtual void end();

protected:
    CommandEncoder* m_command_encoder;
    rhi::IPassEncoder* m_rhi_pass_encoder;

    friend class CommandEncoder;
};

class SGL_API RenderPassEncoder : public PassEncoder {
    SGL_OBJECT(RenderPassEncoder)
public:
    ShaderObject* bind_pipeline(RenderPipeline* pipeline);
    void bind_pipeline(RenderPipeline* pipeline, ShaderObject* root_object);

    void set_render_state(const RenderState& state);

    void draw(const DrawArguments& args);
    void draw_indexed(const DrawArguments& args);

    void draw_indirect(uint32_t max_draw_count, BufferOffsetPair arg_buffer, BufferOffsetPair count_buffer = {});

    void
    draw_indexed_indirect(uint32_t max_draw_count, BufferOffsetPair arg_buffer, BufferOffsetPair count_buffer = {});

    void draw_mesh_tasks(uint3 dimensions);

    void end() override;

private:
    rhi::IRenderPassEncoder* m_rhi_render_pass_encoder;

    friend class CommandEncoder;
};

class SGL_API ComputePassEncoder : public PassEncoder {
    SGL_OBJECT(ComputePassEncoder)
public:
    ShaderObject* bind_pipeline(ComputePipeline* pipeline);
    void bind_pipeline(ComputePipeline* pipeline, ShaderObject* root_object);

    void dispatch(uint3 thread_count);
    void dispatch_compute(uint3 thread_group_count);
    void dispatch_compute_indirect(BufferOffsetPair arg_buffer);

    void end() override;

private:
    rhi::IComputePassEncoder* m_rhi_compute_pass_encoder;
    uint3 m_thread_group_size;

    friend class CommandEncoder;
};

class SGL_API RayTracingPassEncoder : public PassEncoder {
    SGL_OBJECT(RayTracingPassEncoder)
public:
    ShaderObject* bind_pipeline(RayTracingPipeline* pipeline, ShaderTable* shader_table);
    void bind_pipeline(RayTracingPipeline* pipeline, ShaderTable* shader_table, ShaderObject* root_object);

    void dispatch_rays(uint32_t ray_gen_shader_index, uint3 dimensions);

    void end() override;

private:
    rhi::IRayTracingPassEncoder* m_rhi_ray_tracing_pass_encoder;

    friend class CommandEncoder;
};

class SGL_API CommandEncoder : public DeviceChild {
    SGL_OBJECT(CommandEncoder)
public:
    CommandEncoder(ref<Device> device, Slang::ComPtr<rhi::ICommandEncoder> rhi_command_encoder);

    virtual void _release_rhi_resources() override { m_rhi_command_encoder.setNull(); }

    ref<RenderPassEncoder> begin_render_pass(const RenderPassDesc& desc);
    ref<ComputePassEncoder> begin_compute_pass();
    ref<RayTracingPassEncoder> begin_ray_tracing_pass();

    ShaderObject* _get_root_object(rhi::IShaderObject* rhi_shader_object);

    /**
     * \brief Copy a buffer region.
     *
     * \param dst Destination buffer.
     * \param dst_offset Destination offset in bytes.
     * \param src Source buffer.
     * \param src_offset Source offset in bytes.
     * \param size Size in bytes.
     */
    void copy_buffer(Buffer* dst, DeviceOffset dst_offset, const Buffer* src, DeviceOffset src_offset, DeviceSize size);

    /**
     * \brief Copy a texture region.
     *
     * \param dst Destination texture.
     * \param dst_subresource_range Destination subresource range.
     * \param dst_offset Destination offset in texels.
     * \param src Source texture.
     * \param src_subresource_range Source subresource range.
     * \param src_offset Source offset in texels.
     * \param extent Size in texels (-1 for maximum possible size).
     */
    void copy_texture(
        Texture* dst,
        SubresourceRange dst_subresource_range,
        uint3 dst_offset,
        const Texture* src,
        SubresourceRange src_subresource_range,
        uint3 src_offset,
        uint3 extent = uint3(-1)
    );

    /**
     * \brief Copy a texture region.
     *
     * \param dst Destination texture.
     * \param dst_layer Destination layer.
     * \param dst_mip Destination mip level.
     * \param dst_offset Destination offset in texels.
     * \param src Source texture.
     * \param src_layer Source layer.
     * \param src_mip Source mip level.
     * \param src_offset Source offset in texels.
     * \param extent Size in texels (-1 for maximum possible size).
     */
    void copy_texture(
        Texture* dst,
        uint32_t dst_layer,
        uint32_t dst_mip,
        uint3 dst_offset,
        const Texture* src,
        uint32_t src_layer,
        uint32_t src_mip,
        uint3 src_offset,
        uint3 extent = uint3(-1)
    );

    /**
     * \brief Copy a texture to a buffer.
     *
     * \param dst Destination buffer.
     * \param dst_offset Destination offset in bytes.
     * \param dst_size Destination size in bytes.
     * \param dst_row_pitch Destination row stride in bytes.
     * \param src Source texture.
     * \param src_layer Source layer.
     * \param src_mip Source mip level.
     * \param src_offset Source offset in texels.
     * \param extent Extent in texels (-1 for maximum possible extent).
     */
    void copy_texture_to_buffer(
        Buffer* dst,
        DeviceOffset dst_offset,
        DeviceSize dst_size,
        DeviceSize dst_row_pitch,
        const Texture* src,
        uint32_t src_layer,
        uint32_t src_mip,
        uint3 src_offset = uint3(0),
        uint3 extent = uint3(-1)
    );

    /**
     * \brief Copy a buffer to a texture.
     *
     * \param dst Destination texture.
     * \param dst_layer Destination layer.
     * \param dst_mip Destination mip level.
     * \param dst_offset Destination offset in texels.
     * \param src Source buffer.
     * \param src_offset Source offset in bytes.
     * \param src_size Size in bytes.
     * \param src_row_pitch Source row stride in bytes.
     * \param extent Extent in texels (-1 for maximum possible extent).
     */
    void copy_buffer_to_texture(
        Texture* dst,
        uint32_t dst_layer,
        uint32_t dst_mip,
        uint3 dst_offset,
        const Buffer* src,
        DeviceOffset src_offset,
        DeviceSize src_size,
        DeviceSize src_row_pitch,
        uint3 extent = uint3(-1)
    );

    /**
     * \brief Upload host memory to a buffer.
     *
     * \param buffer Buffer to write to.
     * \param offset Buffer offset in bytes.
     * \param size Number of bytes to write.
     * \param data Host memory to write.
     */
    void upload_buffer_data(Buffer* buffer, size_t offset, size_t size, const void* data);

    void upload_texture_data(
        Texture* texture,
        SubresourceRange subresource_range,
        uint3 offset,
        uint3 extent,
        std::span<SubresourceData> subresource_data
    );

    void upload_texture_data(Texture* texture, uint32_t layer, uint32_t mip, SubresourceData subresource_data);

    void clear_buffer(Buffer* buffer, BufferRange range = {});

    void
    clear_texture_float(Texture* texture, SubresourceRange subresource_range = {}, float4 clear_value = float4(0.f));

    void clear_texture_uint(Texture* texture, SubresourceRange subresource_range = {}, uint4 clear_value = uint4(0));

    void clear_texture_sint(Texture* texture, SubresourceRange subresource_range = {}, int4 clear_value = int4(0));

    void clear_texture_depth_stencil(
        Texture* texture,
        SubresourceRange subresource_range = {},
        bool clear_depth = true,
        float depth_value = 0.f,
        bool clear_stencil = true,
        uint8_t stencil_value = 0
    );

    /**
     * \brief Blit a texture view.
     *
     * Blits the full extent of the source texture to the destination texture.
     *
     * \param dst View of the destination texture.
     * \param src View of the source texture.
     * \param filter Filtering mode to use.
     */
    void blit(TextureView* dst, TextureView* src, TextureFilteringMode filter = TextureFilteringMode::linear);

    /**
     * \brief Blit a texture.
     *
     * Blits the full extent of the source texture to the destination texture.
     *
     * \param dst Destination texture.
     * \param src Source texture.
     * \param filter Filtering mode to use.
     */
    void blit(Texture* dst, Texture* src, TextureFilteringMode filter = TextureFilteringMode::linear);

    void generate_mips(Texture* texture, uint32_t layer = 0);

    void resolve_query(QueryPool* query_pool, uint32_t index, uint32_t count, Buffer* buffer, DeviceOffset offset);

    void build_acceleration_structure(
        const AccelerationStructureBuildDesc& desc,
        AccelerationStructure* dst,
        AccelerationStructure* src,
        BufferOffsetPair scratch_buffer,
        std::span<AccelerationStructureQueryDesc> queries = std::span<AccelerationStructureQueryDesc>()
    );

    void copy_acceleration_structure(
        AccelerationStructure* dst,
        AccelerationStructure* src,
        AccelerationStructureCopyMode mode
    );

    void query_acceleration_structure_properties(
        std::span<AccelerationStructure*> acceleration_structures,
        std::span<AccelerationStructureQueryDesc> queries
    );

    void serialize_acceleration_structure(BufferOffsetPair dst, AccelerationStructure* src);
    void deserialize_acceleration_structure(AccelerationStructure* dst, BufferOffsetPair src);

    /**
     * Transition resource state of a buffer and add a barrier if state has changed.
     * \param buffer Buffer
     * \param state New state
     */
    void set_buffer_state(Buffer* buffer, ResourceState state);

    /**
     * Transition resource state of a texture and add a barrier if state has changed.
     * \param texture Texture
     * \param state New state
     */
    void set_texture_state(Texture* texture, ResourceState state);

    /**
     * Transition resource state of texture sub-resources and add a barrier if state has changed.
     * \param texture Texture
     * \param range Subresource range
     * \param state New state
     */
    void set_texture_state(Texture* texture, SubresourceRange range, ResourceState state);

    /**
     * \brief Insert a global barrier that ensures all previous writes are visible to subsequent reads.
     * Note: This is not necessary for typical bindings, as state management is automatic, however
     * global barriers are useful for cross-api synchronization (eg 2 slangpy devices constructed from
     * the same native handle), or as brute force tools for synchronizing pointer/bindless operations.
     */
    void global_barrier();

    /// Push a debug group.
    void push_debug_group(const char* name, float3 color);

    /// Pop a debug group.
    void pop_debug_group();

    /**
     * Insert a debug marker.
     * \param name Name of the marker.
     * \param color Color of the marker.
     */
    void insert_debug_marker(const char* name, float3 color);

    /**
     * \brief Write a timestamp.
     *
     * \param query_pool Query pool.
     * \param index Index of the query.
     */
    void write_timestamp(QueryPool* query_pool, uint32_t index);

    ref<CommandBuffer> finish();

    /// Get the command encoder handle.
    NativeHandle native_handle() const;

    rhi::ICommandEncoder* rhi_command_encoder() const { return m_rhi_command_encoder; }

    std::string to_string() const override;

private:
    Slang::ComPtr<rhi::ICommandEncoder> m_rhi_command_encoder;

    std::vector<ref<cuda::InteropBuffer>> m_cuda_interop_buffers;

    bool m_open{false};

    ref<RenderPassEncoder> m_render_pass_encoder;
    ref<ComputePassEncoder> m_compute_pass_encoder;
    ref<RayTracingPassEncoder> m_ray_tracing_pass_encoder;
    ref<ShaderObject> m_root_object;
};

class SGL_API CommandBuffer : public DeviceChild {
    SGL_OBJECT(CommandBuffer)
public:
    CommandBuffer(ref<Device> device, Slang::ComPtr<rhi::ICommandBuffer> rhi_command_buffer);
    ~CommandBuffer();

    virtual void _release_rhi_resources() override { m_rhi_command_buffer.setNull(); }

    rhi::ICommandBuffer* rhi_command_buffer() const { return m_rhi_command_buffer; }

    std::string to_string() const override;

private:
    Slang::ComPtr<rhi::ICommandBuffer> m_rhi_command_buffer;

    std::vector<ref<cuda::InteropBuffer>> m_cuda_interop_buffers;

    friend class Device;
};

} // namespace sgl
