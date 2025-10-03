// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command.h"

#include "sgl/device/helpers.h"
#include "sgl/device/device.h"
#include "sgl/device/fence.h"
#include "sgl/device/resource.h"
#include "sgl/device/query.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/raytracing.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/cuda_utils.h"
#include "sgl/device/cuda_interop.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/print.h"
#include "sgl/device/blit.h"

#include "sgl/core/short_vector.h"
#include "sgl/core/maths.h"
#include "sgl/core/type_utils.h"

#include "sgl/math/vector.h"

namespace sgl {

namespace detail {
    rhi::SubresourceRange rhi_subresource_range(const Texture* texture, uint32_t subresource_index)
    {
        return rhi::SubresourceRange{
            .layer = subresource_index / texture->mip_count(),
            .layerCount = 1,
            .mip = subresource_index % texture->mip_count(),
            .mipCount = 1,
        };
    }

    rhi::SubresourceRange to_rhi(const SubresourceRange& range)
    {
        return rhi::SubresourceRange{
            .layer = range.layer,
            .layerCount = range.layer_count,
            .mip = range.mip,
            .mipCount = range.mip_count,
        };
    }

    rhi::BufferOffsetPair to_rhi(const BufferOffsetPair& buffer_with_offset)
    {
        return rhi::BufferOffsetPair(
            buffer_with_offset.buffer ? buffer_with_offset.buffer->rhi_buffer() : nullptr,
            buffer_with_offset.offset
        );
    }

    rhi::DrawArguments to_rhi(const DrawArguments& draw_args)
    {
        return rhi::DrawArguments{
            .vertexCount = draw_args.vertex_count,
            .instanceCount = draw_args.instance_count,
            .startVertexLocation = draw_args.start_vertex_location,
            .startInstanceLocation = draw_args.start_instance_location,
            .startIndexLocation = draw_args.start_index_location,
        };
    }
} // namespace detail

// ----------------------------------------------------------------------------
// PassEncoder
// ----------------------------------------------------------------------------

void PassEncoder::push_debug_group(const char* name, float3 color)
{
    m_rhi_pass_encoder->pushDebugGroup(name, rhi::MarkerColor{color.r, color.g, color.b});
}

void PassEncoder::pop_debug_group()
{
    m_rhi_pass_encoder->popDebugGroup();
}

void PassEncoder::insert_debug_marker(const char* name, float3 color)
{
    m_rhi_pass_encoder->insertDebugMarker(name, rhi::MarkerColor{color.r, color.g, color.b});
}

void PassEncoder::write_timestamp(QueryPool* query_pool, uint32_t index)
{
    m_rhi_pass_encoder->writeTimestamp(query_pool->rhi_query_pool(), index);
}

void PassEncoder::end()
{
    SGL_CHECK(m_rhi_pass_encoder, "Pass encoder already ended");

    m_rhi_pass_encoder->end();
    m_rhi_pass_encoder = nullptr;
}

// ----------------------------------------------------------------------------
// RenderPassEncoder
// ----------------------------------------------------------------------------

ShaderObject* RenderPassEncoder::bind_pipeline(RenderPipeline* pipeline)
{
    SGL_CHECK_NOT_NULL(pipeline);

    rhi::IShaderObject* rhi_root_object = m_rhi_render_pass_encoder->bindPipeline(pipeline->rhi_pipeline());
    ShaderObject* root_object = m_command_encoder->_get_root_object(rhi_root_object);
    if (m_command_encoder->device()->debug_printer())
        m_command_encoder->device()->debug_printer()->bind(ShaderCursor(root_object));
    return root_object;
}

void RenderPassEncoder::bind_pipeline(RenderPipeline* pipeline, ShaderObject* root_object)
{
    SGL_CHECK_NOT_NULL(pipeline);
    SGL_CHECK_NOT_NULL(root_object);

    m_rhi_render_pass_encoder->bindPipeline(pipeline->rhi_pipeline(), root_object->rhi_shader_object());
}

void RenderPassEncoder::set_render_state(const RenderState& state)
{
    rhi::RenderState rhi_state = {};
    rhi_state.stencilRef = state.stencil_ref;
    for (size_t i = 0; i < state.viewports.size(); ++i)
        rhi_state.viewports[i] = reinterpret_cast<const rhi::Viewport&>(state.viewports[i]);
    rhi_state.viewportCount = narrow_cast<uint32_t>(state.viewports.size());
    for (size_t i = 0; i < state.scissor_rects.size(); ++i)
        rhi_state.scissorRects[i] = reinterpret_cast<const rhi::ScissorRect&>(state.scissor_rects[i]);
    rhi_state.scissorRectCount = narrow_cast<uint32_t>(state.scissor_rects.size());
    for (size_t i = 0; i < state.vertex_buffers.size(); i++)
        rhi_state.vertexBuffers[i] = detail::to_rhi(state.vertex_buffers[i]);
    rhi_state.vertexBufferCount = narrow_cast<uint32_t>(state.vertex_buffers.size());
    rhi_state.indexBuffer = detail::to_rhi(state.index_buffer);
    rhi_state.indexFormat = static_cast<rhi::IndexFormat>(state.index_format);

    m_rhi_render_pass_encoder->setRenderState(rhi_state);
}

void RenderPassEncoder::draw(const DrawArguments& args)
{
    m_rhi_render_pass_encoder->draw(detail::to_rhi(args));
}

void RenderPassEncoder::draw_indexed(const DrawArguments& args)
{
    m_rhi_render_pass_encoder->drawIndexed(detail::to_rhi(args));
}

void RenderPassEncoder::draw_indirect(
    uint32_t max_draw_count,
    BufferOffsetPair arg_buffer,
    BufferOffsetPair count_buffer
)
{
    m_rhi_render_pass_encoder->drawIndirect(max_draw_count, detail::to_rhi(arg_buffer), detail::to_rhi(count_buffer));
}

void RenderPassEncoder::draw_indexed_indirect(
    uint32_t max_draw_count,
    BufferOffsetPair arg_buffer,
    BufferOffsetPair count_buffer
)
{
    m_rhi_render_pass_encoder
        ->drawIndexedIndirect(max_draw_count, detail::to_rhi(arg_buffer), detail::to_rhi(count_buffer));
}

void RenderPassEncoder::draw_mesh_tasks(uint3 dimensions)
{
    m_rhi_render_pass_encoder->drawMeshTasks(dimensions.x, dimensions.y, dimensions.z);
}

void RenderPassEncoder::end()
{
    PassEncoder::end();
    m_rhi_render_pass_encoder = nullptr;
}

// ----------------------------------------------------------------------------
// ComputePassEncoder
// ----------------------------------------------------------------------------

ShaderObject* ComputePassEncoder::bind_pipeline(ComputePipeline* pipeline)
{
    SGL_CHECK_NOT_NULL(pipeline);

    m_thread_group_size = pipeline->thread_group_size();
    rhi::IShaderObject* rhi_root_object = m_rhi_compute_pass_encoder->bindPipeline(pipeline->rhi_pipeline());
    ShaderObject* root_object = m_command_encoder->_get_root_object(rhi_root_object);
    if (m_command_encoder->device()->debug_printer())
        m_command_encoder->device()->debug_printer()->bind(ShaderCursor(root_object));
    return root_object;
}

void ComputePassEncoder::bind_pipeline(ComputePipeline* pipeline, ShaderObject* root_object)
{
    SGL_CHECK_NOT_NULL(pipeline);
    SGL_CHECK_NOT_NULL(root_object);

    m_thread_group_size = pipeline->thread_group_size();
    m_rhi_compute_pass_encoder->bindPipeline(pipeline->rhi_pipeline(), root_object->rhi_shader_object());
}

void ComputePassEncoder::dispatch(uint3 thread_count)
{
    uint3 thread_group_count{
        div_round_up(thread_count.x, m_thread_group_size.x),
        div_round_up(thread_count.y, m_thread_group_size.y),
        div_round_up(thread_count.z, m_thread_group_size.z)
    };
    dispatch_compute(thread_group_count);
}

void ComputePassEncoder::dispatch_compute(uint3 thread_group_count)
{
    m_rhi_compute_pass_encoder->dispatchCompute(thread_group_count.x, thread_group_count.y, thread_group_count.z);
}

void ComputePassEncoder::dispatch_compute_indirect(BufferOffsetPair arg_buffer)
{
    m_rhi_compute_pass_encoder->dispatchComputeIndirect(detail::to_rhi(arg_buffer));
}

void ComputePassEncoder::end()
{
    PassEncoder::end();
    m_rhi_compute_pass_encoder = nullptr;
}

// ----------------------------------------------------------------------------
// RayTracingPassEncoder
// ----------------------------------------------------------------------------

ShaderObject* RayTracingPassEncoder::bind_pipeline(RayTracingPipeline* pipeline, ShaderTable* shader_table)
{
    SGL_CHECK_NOT_NULL(pipeline);

    rhi::IShaderObject* rhi_root_object
        = m_rhi_ray_tracing_pass_encoder->bindPipeline(pipeline->rhi_pipeline(), shader_table->rhi_shader_table());
    ShaderObject* root_object = m_command_encoder->_get_root_object(rhi_root_object);
    if (m_command_encoder->device()->debug_printer())
        m_command_encoder->device()->debug_printer()->bind(ShaderCursor(root_object));
    return root_object;
}

void RayTracingPassEncoder::bind_pipeline(
    RayTracingPipeline* pipeline,
    ShaderTable* shader_table,
    ShaderObject* root_object
)
{
    SGL_CHECK_NOT_NULL(pipeline);
    SGL_CHECK_NOT_NULL(shader_table);
    SGL_CHECK_NOT_NULL(root_object);

    m_rhi_ray_tracing_pass_encoder
        ->bindPipeline(pipeline->rhi_pipeline(), shader_table->rhi_shader_table(), root_object->rhi_shader_object());
}

void RayTracingPassEncoder::dispatch_rays(uint32_t ray_gen_shader_index, uint3 dimensions)
{
    m_rhi_ray_tracing_pass_encoder->dispatchRays(ray_gen_shader_index, dimensions.x, dimensions.y, dimensions.z);
}

void RayTracingPassEncoder::end()
{
    PassEncoder::end();
    m_rhi_ray_tracing_pass_encoder = nullptr;
}

// ----------------------------------------------------------------------------
// CommandEncoder
// ----------------------------------------------------------------------------

CommandEncoder::CommandEncoder(ref<Device> device, Slang::ComPtr<rhi::ICommandEncoder> rhi_command_encoder)
    : DeviceChild(std::move(device))
    , m_rhi_command_encoder(std::move(rhi_command_encoder))
    , m_open(true)
{
}

ref<RenderPassEncoder> CommandEncoder::begin_render_pass(const RenderPassDesc& desc)
{
    rhi::RenderPassDesc rhi_desc = {};

    short_vector<rhi::RenderPassColorAttachment, 16> rhi_color_attachments;
    for (const auto& ca : desc.color_attachments) {
        rhi_color_attachments.push_back({
            .view = ca.view ? ca.view->rhi_texture_view() : nullptr,
            .resolveTarget = ca.resolve_target ? ca.resolve_target->rhi_texture_view() : nullptr,
            .loadOp = static_cast<rhi::LoadOp>(ca.load_op),
            .storeOp = static_cast<rhi::StoreOp>(ca.store_op),
            .clearValue = {ca.clear_value[0], ca.clear_value[1], ca.clear_value[2], ca.clear_value[3]},
        });
    }
    rhi_desc.colorAttachments = rhi_color_attachments.data();
    rhi_desc.colorAttachmentCount = narrow_cast<uint32_t>(rhi_color_attachments.size());

    rhi::RenderPassDepthStencilAttachment rhi_depth_stencil_attachment = {};
    if (desc.depth_stencil_attachment) {
        const auto& dsa = *desc.depth_stencil_attachment;
        rhi_depth_stencil_attachment = {
            .view = dsa.view ? dsa.view->rhi_texture_view() : nullptr,
            .depthLoadOp = static_cast<rhi::LoadOp>(dsa.depth_load_op),
            .depthStoreOp = static_cast<rhi::StoreOp>(dsa.depth_store_op),
            .depthClearValue = dsa.depth_clear_value,
            .depthReadOnly = dsa.depth_read_only,
            .stencilLoadOp = static_cast<rhi::LoadOp>(dsa.stencil_load_op),
            .stencilStoreOp = static_cast<rhi::StoreOp>(dsa.stencil_store_op),
            .stencilClearValue = dsa.stencil_clear_value,
            .stencilReadOnly = dsa.stencil_read_only,
        };
        rhi_desc.depthStencilAttachment = &rhi_depth_stencil_attachment;
    }

    if (!m_render_pass_encoder) {
        m_render_pass_encoder = make_ref<RenderPassEncoder>();
    }

    m_render_pass_encoder->m_command_encoder = this;
    m_render_pass_encoder->m_rhi_render_pass_encoder = m_rhi_command_encoder->beginRenderPass(rhi_desc);
    m_render_pass_encoder->m_rhi_pass_encoder = m_render_pass_encoder->m_rhi_render_pass_encoder;
    return m_render_pass_encoder;
}

ref<ComputePassEncoder> CommandEncoder::begin_compute_pass()
{
    if (!m_compute_pass_encoder) {
        m_compute_pass_encoder = make_ref<ComputePassEncoder>();
    }

    m_compute_pass_encoder->m_command_encoder = this;
    m_compute_pass_encoder->m_rhi_compute_pass_encoder = m_rhi_command_encoder->beginComputePass();
    m_compute_pass_encoder->m_rhi_pass_encoder = m_compute_pass_encoder->m_rhi_compute_pass_encoder;
    return m_compute_pass_encoder;
}

ref<RayTracingPassEncoder> CommandEncoder::begin_ray_tracing_pass()
{
    if (!m_ray_tracing_pass_encoder) {
        m_ray_tracing_pass_encoder = make_ref<RayTracingPassEncoder>();
    }

    m_ray_tracing_pass_encoder->m_command_encoder = this;
    m_ray_tracing_pass_encoder->m_rhi_ray_tracing_pass_encoder = m_rhi_command_encoder->beginRayTracingPass();
    m_ray_tracing_pass_encoder->m_rhi_pass_encoder = m_ray_tracing_pass_encoder->m_rhi_ray_tracing_pass_encoder;
    return m_ray_tracing_pass_encoder;
}

ShaderObject* CommandEncoder::_get_root_object(rhi::IShaderObject* rhi_shader_object)
{
    m_root_object = make_ref<ShaderObject>(m_device, rhi_shader_object, false);
    return m_root_object.get();
}

void CommandEncoder::copy_buffer(
    Buffer* dst,
    DeviceOffset dst_offset,
    const Buffer* src,
    DeviceOffset src_offset,
    DeviceSize size
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_NOT_NULL(src);
    SGL_CHECK_LE(dst_offset + size, dst->size());
    SGL_CHECK_LE(src_offset + size, src->size());

    m_rhi_command_encoder->copyBuffer(dst->rhi_buffer(), dst_offset, src->rhi_buffer(), src_offset, size);
}

void CommandEncoder::copy_texture(
    Texture* dst,
    SubresourceRange dst_subresource_range,
    uint3 dst_offset,
    const Texture* src,
    SubresourceRange src_subresource_range,
    uint3 src_offset,
    uint3 extent
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_NOT_NULL(src);

    rhi::SubresourceRange dst_sr = detail::to_rhi(dst_subresource_range);
    rhi::SubresourceRange src_sr = detail::to_rhi(src_subresource_range);

    if (all(extent == uint3(-1)))
        extent = src->get_mip_size(src_sr.mip) - src_offset;

    m_rhi_command_encoder->copyTexture(
        dst->rhi_texture(),
        dst_sr,
        rhi::Offset3D{dst_offset.x, dst_offset.y, dst_offset.z},
        src->rhi_texture(),
        src_sr,
        rhi::Offset3D{src_offset.x, src_offset.y, src_offset.z},
        rhi::Extent3D{extent.x, extent.y, extent.z}
    );
}

void CommandEncoder::copy_texture(
    Texture* dst,
    uint32_t dst_layer,
    uint32_t dst_mip,
    uint3 dst_offset,
    const Texture* src,
    uint32_t src_layer,
    uint32_t src_mip,
    uint3 src_offset,
    uint3 extent
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_LT(dst_layer, dst->layer_count());
    SGL_CHECK_LT(dst_mip, dst->mip_count());
    SGL_CHECK_NOT_NULL(src);
    SGL_CHECK_LT(src_layer, src->layer_count());
    SGL_CHECK_LT(src_mip, src->mip_count());

    rhi::SubresourceRange dst_sr = {dst_layer, 1, dst_mip, 1};
    rhi::SubresourceRange src_sr = {src_layer, 1, src_mip, 1};

    if (all(extent == uint3(-1)))
        extent = src->get_mip_size(src_sr.mip) - src_offset;

    m_rhi_command_encoder->copyTexture(
        dst->rhi_texture(),
        dst_sr,
        rhi::Offset3D{dst_offset.x, dst_offset.y, dst_offset.z},
        src->rhi_texture(),
        src_sr,
        rhi::Offset3D{src_offset.x, src_offset.y, src_offset.z},
        rhi::Extent3D{extent.x, extent.y, extent.z}
    );
}

void CommandEncoder::copy_texture_to_buffer(
    Buffer* dst,
    DeviceOffset dst_offset,
    DeviceSize dst_size,
    DeviceSize dst_row_pitch,
    const Texture* src,
    uint32_t src_layer,
    uint32_t src_mip,
    uint3 src_offset,
    uint3 extent
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK(dst_offset + dst_size <= dst->size(), "Destination buffer is too small");
    SGL_CHECK_NOT_NULL(src);
    SGL_CHECK_LT(src_layer, src->layer_count());
    SGL_CHECK_LT(src_mip, src->mip_count());

    m_rhi_command_encoder->copyTextureToBuffer(
        dst->rhi_buffer(),
        dst_offset,
        dst_size,
        dst_row_pitch,
        src->rhi_texture(),
        src_layer,
        src_mip,
        rhi::Offset3D{src_offset.x, src_offset.y, src_offset.z},
        rhi::Extent3D{extent.x, extent.y, extent.z}
    );
}

void CommandEncoder::copy_buffer_to_texture(
    Texture* dst,
    uint32_t dst_layer,
    uint32_t dst_mip,
    uint3 dst_offset,
    const Buffer* src,
    DeviceOffset src_offset,
    DeviceSize src_size,
    DeviceSize src_row_pitch,
    uint3 extent
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_NOT_NULL(src);
    SGL_CHECK(src_offset + src_size <= src->size(), "Source buffer is too small");
    SGL_CHECK_LT(dst_layer, dst->layer_count());
    SGL_CHECK_LT(dst_mip, dst->mip_count());

    m_rhi_command_encoder->copyBufferToTexture(
        dst->rhi_texture(),
        dst_layer,
        dst_mip,
        rhi::Offset3D{dst_offset.x, dst_offset.y, dst_offset.z},
        src->rhi_buffer(),
        src_offset,
        src_size,
        src_row_pitch,
        rhi::Extent3D{extent.x, extent.y, extent.z}
    );
}

void CommandEncoder::upload_buffer_data(Buffer* buffer, size_t offset, size_t size, const void* data)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(buffer);
    SGL_CHECK(offset + size <= buffer->size(), "Buffer upload is out of bounds");
    SGL_CHECK_NOT_NULL(data);

    set_buffer_state(buffer, ResourceState::copy_destination);

    SLANG_RHI_CALL(
        m_rhi_command_encoder->uploadBufferData(buffer->rhi_buffer(), offset, size, const_cast<void*>(data))
    );
}

void CommandEncoder::upload_texture_data(
    Texture* texture,
    SubresourceRange subresource_range,
    uint3 offset,
    uint3 extent,
    std::span<SubresourceData> subresource_data
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);

    short_vector<rhi::SubresourceData, 16> rhi_subresource_data;
    for (const auto& data : subresource_data) {
        rhi_subresource_data.push_back({
            .data = const_cast<void*>(data.data),
            .rowPitch = data.row_pitch,
            .slicePitch = data.slice_pitch,
        });
    }

    SLANG_RHI_CALL(m_rhi_command_encoder->uploadTextureData(
        texture->rhi_texture(),
        detail::to_rhi(subresource_range),
        rhi::Offset3D{offset.x, offset.y, offset.z},
        rhi::Extent3D{extent.x, extent.y, extent.z},
        rhi_subresource_data.data(),
        narrow_cast<uint32_t>(rhi_subresource_data.size())
    ));
}

void CommandEncoder::upload_texture_data(
    Texture* texture,
    uint32_t layer,
    uint32_t mip,
    SubresourceData subresource_data
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);
    SGL_CHECK_LT(layer, texture->layer_count());
    SGL_CHECK_LT(mip, texture->mip_count());

    rhi::SubresourceRange rhi_subresource_range = {
        .layer = layer,
        .layerCount = 1,
        .mip = mip,
        .mipCount = 1,
    };

    rhi::SubresourceData rhi_subresource_data = {
        .data = const_cast<void*>(subresource_data.data),
        .rowPitch = subresource_data.row_pitch,
        .slicePitch = subresource_data.slice_pitch,
    };

    SLANG_RHI_CALL(m_rhi_command_encoder->uploadTextureData(
        texture->rhi_texture(),
        rhi_subresource_range,
        rhi::Offset3D{0, 0, 0},
        rhi::Extent3D::kWholeTexture,
        &rhi_subresource_data,
        1
    ));
}

void CommandEncoder::clear_buffer(Buffer* buffer, BufferRange range)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(buffer);

    m_rhi_command_encoder->clearBuffer(buffer->rhi_buffer(), range.offset, range.size);
}

void CommandEncoder::clear_texture_float(Texture* texture, SubresourceRange subresource_range, float4 clear_value)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);
    m_rhi_command_encoder
        ->clearTextureFloat(texture->rhi_texture(), detail::to_rhi(subresource_range), &clear_value[0]);
}

void CommandEncoder::clear_texture_uint(Texture* texture, SubresourceRange subresource_range, uint4 clear_value)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);
    m_rhi_command_encoder->clearTextureUint(texture->rhi_texture(), detail::to_rhi(subresource_range), &clear_value[0]);
}

void CommandEncoder::clear_texture_sint(Texture* texture, SubresourceRange subresource_range, int4 clear_value)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);
    m_rhi_command_encoder->clearTextureSint(texture->rhi_texture(), detail::to_rhi(subresource_range), &clear_value[0]);
}

void CommandEncoder::clear_texture_depth_stencil(
    Texture* texture,
    SubresourceRange subresource_range,
    bool clear_depth,
    float depth_value,
    bool clear_stencil,
    uint8_t stencil_value
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);
    m_rhi_command_encoder->clearTextureDepthStencil(
        texture->rhi_texture(),
        detail::to_rhi(subresource_range),
        clear_depth,
        depth_value,
        clear_stencil,
        stencil_value
    );
}

void CommandEncoder::blit(TextureView* dst, TextureView* src, TextureFilteringMode filter)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_NOT_NULL(src);
    m_device->_blitter()->blit(this, dst, src, filter);
}

void CommandEncoder::blit(Texture* dst, Texture* src, TextureFilteringMode filter)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_NOT_NULL(src);
    m_device->_blitter()->blit(this, dst, src, filter);
}

void CommandEncoder::generate_mips(Texture* texture, uint32_t layer)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);
    SGL_CHECK(layer < texture->layer_count(), "Layer index out of bounds");
    m_device->_blitter()->generate_mips(this, texture, layer);
}

void CommandEncoder::resolve_query(
    QueryPool* query_pool,
    uint32_t index,
    uint32_t count,
    Buffer* buffer,
    DeviceOffset offset
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(query_pool);
    SGL_CHECK_NOT_NULL(buffer);
    SGL_CHECK_LE(index + count, query_pool->desc().count);
    SGL_CHECK_LE(offset + count * sizeof(uint64_t), buffer->size());

    m_rhi_command_encoder->resolveQuery(query_pool->rhi_query_pool(), index, count, buffer->rhi_buffer(), offset);
}

void CommandEncoder::build_acceleration_structure(
    const AccelerationStructureBuildDesc& desc,
    AccelerationStructure* dst,
    AccelerationStructure* src,
    BufferOffsetPair scratch_buffer,
    std::span<AccelerationStructureQueryDesc> queries
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);

    AccelerationStructureBuildDescConverter converter(desc);

    short_vector<rhi::AccelerationStructureQueryDesc, 16> rhi_queries;
    for (const auto& q : queries) {
        rhi_queries.push_back({
            .queryType = static_cast<rhi::QueryType>(q.query_type),
            .queryPool = q.query_pool->rhi_query_pool(),
            .firstQueryIndex = q.first_query_index,
        });
    }

    m_rhi_command_encoder->buildAccelerationStructure(
        converter.rhi_desc,
        dst->rhi_acceleration_structure(),
        src ? src->rhi_acceleration_structure() : nullptr,
        detail::to_rhi(scratch_buffer),
        narrow_cast<uint32_t>(rhi_queries.size()),
        rhi_queries.data()
    );
}

void CommandEncoder::copy_acceleration_structure(
    AccelerationStructure* dst,
    AccelerationStructure* src,
    AccelerationStructureCopyMode mode
)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);

    m_rhi_command_encoder->copyAccelerationStructure(
        dst->rhi_acceleration_structure(),
        src->rhi_acceleration_structure(),
        static_cast<rhi::AccelerationStructureCopyMode>(mode)
    );
}

void CommandEncoder::query_acceleration_structure_properties(
    std::span<AccelerationStructure*> acceleration_structures,
    std::span<AccelerationStructureQueryDesc> queries
)
{
    SGL_CHECK(m_open, "Command encoder is finished");

    short_vector<rhi::IAccelerationStructure*, 16> rhi_acceleration_structures(acceleration_structures.size(), nullptr);
    for (size_t i = 0; i < acceleration_structures.size(); i++)
        rhi_acceleration_structures[i] = acceleration_structures[i]->rhi_acceleration_structure();

    short_vector<rhi::AccelerationStructureQueryDesc, 16> rhi_queries(queries.size(), {});
    for (size_t i = 0; i < queries.size(); i++) {
        rhi_queries[i] = {
            .queryType = static_cast<rhi::QueryType>(queries[i].query_type),
            .queryPool = queries[i].query_pool->rhi_query_pool(),
            .firstQueryIndex = queries[i].first_query_index,
        };
    }

    m_rhi_command_encoder->queryAccelerationStructureProperties(
        narrow_cast<uint32_t>(rhi_acceleration_structures.size()),
        rhi_acceleration_structures.data(),
        narrow_cast<uint32_t>(rhi_queries.size()),
        rhi_queries.data()
    );
}

void CommandEncoder::serialize_acceleration_structure(BufferOffsetPair dst, AccelerationStructure* src)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst.buffer);
    SGL_CHECK_NOT_NULL(src);

    m_rhi_command_encoder->serializeAccelerationStructure(detail::to_rhi(dst), src->rhi_acceleration_structure());
}

void CommandEncoder::deserialize_acceleration_structure(AccelerationStructure* dst, BufferOffsetPair src)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_NOT_NULL(src.buffer);

    m_rhi_command_encoder->deserializeAccelerationStructure(dst->rhi_acceleration_structure(), detail::to_rhi(src));
}

void CommandEncoder::set_buffer_state(Buffer* buffer, ResourceState state)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(buffer);

    m_rhi_command_encoder->setBufferState(buffer->rhi_buffer(), static_cast<rhi::ResourceState>(state));
}

void CommandEncoder::set_texture_state(Texture* texture, ResourceState state)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);

    m_rhi_command_encoder->setTextureState(texture->rhi_texture(), static_cast<rhi::ResourceState>(state));
}

void CommandEncoder::set_texture_state(Texture* texture, SubresourceRange range, ResourceState state)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(texture);

    m_rhi_command_encoder->setTextureState(
        texture->rhi_texture(),
        {
            .layer = range.layer,
            .layerCount = range.layer_count,
            .mip = range.mip,
            .mipCount = range.mip_count,
        },
        static_cast<rhi::ResourceState>(state)
    );
}

void CommandEncoder::global_barrier()
{
    SGL_CHECK(m_open, "Command encoder is finished");
    m_rhi_command_encoder->globalBarrier();
}

void CommandEncoder::push_debug_group(const char* name, float3 color)
{
    SGL_CHECK(m_open, "Command encoder is finished");

    m_rhi_command_encoder->pushDebugGroup(name, rhi::MarkerColor{color.r, color.g, color.b});
}

void CommandEncoder::pop_debug_group()
{
    SGL_CHECK(m_open, "Command encoder is finished");

    m_rhi_command_encoder->popDebugGroup();
}

void CommandEncoder::insert_debug_marker(const char* name, float3 color)
{
    SGL_CHECK(m_open, "Command encoder is finished");

    m_rhi_command_encoder->insertDebugMarker(name, rhi::MarkerColor{color.r, color.g, color.b});
}

void CommandEncoder::write_timestamp(QueryPool* query_pool, uint32_t index)
{
    SGL_CHECK(m_open, "Command encoder is finished");
    SGL_CHECK_NOT_NULL(query_pool);
    SGL_CHECK_LE(index, query_pool->desc().count);

    m_rhi_command_encoder->writeTimestamp(query_pool->rhi_query_pool(), index);
}

ref<CommandBuffer> CommandEncoder::finish()
{
    SGL_CHECK(m_open, "Command encoder is finished");
    Slang::ComPtr<rhi::ICommandBuffer> rhi_command_buffer;
    SLANG_RHI_CALL(m_rhi_command_encoder->finish(rhi_command_buffer.writeRef()));
    ref<CommandBuffer> command_buffer = make_ref<CommandBuffer>(m_device, rhi_command_buffer);
    m_open = false;
    return command_buffer;
}

NativeHandle CommandEncoder::native_handle() const
{
    rhi::NativeHandle rhi_handle = {};
    m_rhi_command_encoder->getNativeHandle(&rhi_handle);
    return NativeHandle(rhi_handle);
}

std::string CommandEncoder::to_string() const
{
    return fmt::format(
        "CommandEncoder(\n"
        "  device = {}\n"
        ")",
        m_device
    );
}

// ----------------------------------------------------------------------------
// CommandBuffer
// ----------------------------------------------------------------------------

CommandBuffer::CommandBuffer(ref<Device> device, Slang::ComPtr<rhi::ICommandBuffer> command_buffer)
    : DeviceChild(std::move(device))
    , m_rhi_command_buffer(std::move(command_buffer))
{
}

CommandBuffer::~CommandBuffer() { }

std::string CommandBuffer::to_string() const
{
    return fmt::format(
        "CommandBuffer(\n"
        "  device = {}\n"
        ")",
        m_device
    );
}

} // namespace sgl
