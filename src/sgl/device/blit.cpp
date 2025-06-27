// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "blit.h"

#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/sampler.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"

#include "sgl/core/error.h"

#include "sgl/math/vector.h"

namespace sgl {

Blitter::Blitter(Device* device)
    : m_device(device)
{
    m_linear_sampler = m_device->create_sampler({
        .min_filter = TextureFilteringMode::linear,
        .mag_filter = TextureFilteringMode::linear,
    });

    m_point_sampler = m_device->create_sampler({
        .min_filter = TextureFilteringMode::point,
        .mag_filter = TextureFilteringMode::point,
    });
}

Blitter::~Blitter() { }

void Blitter::blit(CommandEncoder* command_encoder, TextureView* dst, TextureView* src, TextureFilteringMode filter)
{
    SGL_UNUSED(filter);

    SGL_CHECK_NOT_NULL(command_encoder);
    SGL_CHECK_NOT_NULL(dst);
    SGL_CHECK_NOT_NULL(src);

    Texture* dst_texture = dst->texture();
    Texture* src_texture = src->texture();

    SGL_CHECK(
        dst_texture->type() == TextureType::texture_2d || dst_texture->type() == TextureType::texture_2d_array,
        "dst must be a 2D texture"
    );
    SGL_CHECK(
        src_texture->type() == TextureType::texture_2d || src_texture->type() == TextureType::texture_2d_array,
        "src must be a 2D texture"
    );
    SGL_CHECK(is_set(src_texture->desc().usage, TextureUsage::shader_resource), "src must have shader resource usage");

    // Select between render pass and compute pass.
    bool use_compute = false;
    if (m_device->has_feature(Feature::rasterization)
        && is_set(dst_texture->desc().usage, TextureUsage::render_target)) {
        // use render pass for blitting
    } else if (is_set(dst_texture->desc().usage, TextureUsage::unordered_access)) {
        // use compute pass for blitting
        use_compute = true;
    } else {
        SGL_THROW("dst must  have render target or unordered access usage");
    }

    uint32_t dst_mip = dst->subresource_range().mip;
    uint32_t src_mip = src->subresource_range().mip;

    uint2 dst_size = src_texture->get_mip_size(dst_mip).xy();
    uint2 src_size = src_texture->get_mip_size(src_mip).xy();

    auto determine_texture_type = [](Format resource_format) -> TextureDataType
    {
        const auto& info = get_format_info(resource_format);
        if (info.is_float_format() || info.is_normalized_format())
            return TextureDataType::float_;
        return TextureDataType::int_;
    };

    TextureDataType dst_type = determine_texture_type(dst_texture->format());
    TextureDataType src_type = determine_texture_type(src_texture->format());
    TextureLayout src_layout
        = src_texture->type() == TextureType::texture_2d ? TextureLayout::texture_2d : TextureLayout::texture_2d_array;

    if (use_compute) {
        ref<ComputePipeline> pipeline = get_compute_pipeline(
            {
                .src_layout = src_layout,
                .src_type = src_type,
                .dst_type = dst_type,
            },
            dst_texture->format()
        );

        auto pass_encoder = command_encoder->begin_compute_pass();
        ShaderObject* rootObject = pass_encoder->bind_pipeline(pipeline);
        ShaderCursor cursor = ShaderCursor(rootObject);
        cursor["src"] = ref(src);
        // TODO: support sampler selection in CUDA
        if (m_device->desc().type != DeviceType::cuda)
            cursor["sampler"] = filter == TextureFilteringMode::linear ? m_linear_sampler : m_point_sampler;
        ShaderCursor entry_point_cursor = ShaderCursor(rootObject->get_entry_point(0));
        entry_point_cursor["dst"] = ref(dst);
        pass_encoder->dispatch({dst_size.x, dst_size.y, 1});
        pass_encoder->end();
    } else {
        ref<RenderPipeline> pipeline = get_render_pipeline(
            {
                .src_layout = src_layout,
                .src_type = src_type,
                .dst_type = dst_type,
            },
            dst_texture->format()
        );

        auto pass_encoder = command_encoder->begin_render_pass({.color_attachments = {{.view = dst}}});
        ShaderCursor cursor = ShaderCursor(pass_encoder->bind_pipeline(pipeline));
        pass_encoder->set_render_state({
            .viewports = {Viewport::from_size(float(dst_size.x), float(dst_size.y))},
            .scissor_rects = {ScissorRect::from_size(dst_size.x, dst_size.y)},
        });
        cursor["src"] = ref(src);
        cursor["sampler"] = filter == TextureFilteringMode::linear ? m_linear_sampler : m_point_sampler;
        pass_encoder->draw({.vertex_count = 3});
        pass_encoder->end();
    }
}

void Blitter::blit(CommandEncoder* command_encoder, Texture* dst, Texture* src, TextureFilteringMode filter)
{
    // TODO(slang-rhi) use default views when available
    blit(command_encoder, dst->create_view({}), src->create_view({}), filter);
}

void Blitter::generate_mips(CommandEncoder* command_encoder, Texture* texture, uint32_t layer)
{
    SGL_CHECK_NOT_NULL(command_encoder);
    SGL_CHECK_NOT_NULL(texture);
    SGL_CHECK_LT(layer, texture->layer_count());

    for (uint32_t i = 0; i < texture->mip_count() - 1; ++i) {
        ref<TextureView> src = texture->create_view({
            .subresource_range{
                .layer = layer,
                .layer_count = 1,
                .mip = i,
                .mip_count = 1,
            },
        });
        ref<TextureView> dst = texture->create_view({
            .subresource_range{
                .layer = layer,
                .layer_count = 1,
                .mip = i + 1,
                .mip_count = 1,
            },
        });
        blit(command_encoder, dst, src);
    }
}

ref<ShaderProgram> Blitter::get_render_program(ProgramKey key)
{
    auto it = m_render_program_cache.find(key);
    if (it != m_render_program_cache.end())
        return it->second;

    std::string source;
    source += fmt::format(
        "#define SRC_LAYOUT {}\n"
        "#define SRC_TYPE {}\n"
        "#define DST_TYPE {}\n"
        "#define DST_FORMAT_ATTR\n"
        "#define DST_SRGB 0\n\n",
        uint32_t(key.src_layout),
        uint32_t(key.src_type),
        uint32_t(key.dst_type)
    );
    source += m_device->slang_session()->load_source("sgl/device/blit.slang");

    ref<SlangModule> module = m_device->slang_session()->load_module_from_source("blit", source);
    module->break_strong_reference_to_session();
    ref<ShaderProgram> program = m_device->slang_session()->link_program(
        {module},
        {
            module->entry_point("vs_main"),
            module->entry_point("fs_main"),
        }
    );

    m_render_program_cache[key] = program;
    return program;
}

ref<RenderPipeline> Blitter::get_render_pipeline(ProgramKey key, Format dst_format)
{
    auto it = m_render_pipeline_cache.find({key, dst_format});
    if (it != m_render_pipeline_cache.end())
        return it->second;

    ref<ShaderProgram> program = get_render_program(key);

    ref<RenderPipeline> pipeline = m_device->create_render_pipeline({
        .program = program,
        .targets = {
            {
                .format = dst_format,
            },
        },
    });

    m_render_pipeline_cache[{key, dst_format}] = pipeline;
    return pipeline;
}

ref<ShaderProgram> Blitter::get_compute_program(ProgramKey key, Format dst_format)
{
    auto it = m_compute_program_cache.find({key, dst_format});
    if (it != m_compute_program_cache.end())
        return it->second;

    const FormatInfo& dst_format_info = get_format_info(dst_format);
    std::string dst_format_attr;
    bool dst_srgb = false;
    switch (dst_format) {
    case Format::rgba8_unorm_srgb:
    case Format::bgra8_unorm_srgb:
    case Format::bgrx8_unorm_srgb:
        dst_format_attr = "[format(\"rgba8\")]";
        dst_srgb = true;
        break;
    default:
        if (dst_format_info.slang_format)
            dst_format_attr = fmt::format("[format(\"{}\")]", dst_format_info.slang_format);
        break;
    }

    std::string source;
    source += fmt::format(
        "#define SRC_LAYOUT {}\n"
        "#define SRC_TYPE {}\n"
        "#define DST_TYPE {}\n"
        "#define DST_FORMAT_ATTR {}\n"
        "#define DST_SRGB {}\n\n",
        uint32_t(key.src_layout),
        uint32_t(key.src_type),
        uint32_t(key.dst_type),
        dst_format_attr,
        dst_srgb ? "1" : "0"
    );
    source += m_device->slang_session()->load_source("sgl/device/blit.slang");

    ref<SlangModule> module = m_device->slang_session()->load_module_from_source("blit", source);
    module->break_strong_reference_to_session();
    ref<ShaderProgram> program = m_device->slang_session()->link_program(
        {module},
        {
            module->entry_point("compute_main"),
        }
    );

    m_compute_program_cache[{key, dst_format}] = program;
    return program;
}

ref<ComputePipeline> Blitter::get_compute_pipeline(ProgramKey key, Format dst_format)
{
    auto it = m_compute_pipeline_cache.find({key, dst_format});
    if (it != m_compute_pipeline_cache.end())
        return it->second;

    ref<ShaderProgram> program = get_compute_program(key, dst_format);

    ref<ComputePipeline> pipeline = m_device->create_compute_pipeline({
        .program = program,
    });

    m_compute_pipeline_cache[{key, dst_format}] = pipeline;
    return pipeline;
}

} // namespace sgl
