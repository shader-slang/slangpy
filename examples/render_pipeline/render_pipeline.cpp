// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/sgl.h"
#include "sgl/core/platform.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/kernel.h"
#include "sgl/device/agility_sdk.h"
#include "sgl/device/input_layout.h"
#include "sgl/device/pipeline.h"
#include "sgl/utils/tev.h"

SGL_EXPORT_AGILITY_SDK

static const std::filesystem::path EXAMPLE_DIR(SGL_EXAMPLE_DIR);

using namespace sgl;

int main()
{
    sgl::static_init();

    {
        ref<Device> device = Device::create({
            .enable_debug_layers = true,
            .compiler_options = {.include_paths = {EXAMPLE_DIR}},
        });

        std::vector<float2> vertices{{-1.f, -1.f}, {1.f, -1.f}, {0.f, 1.f}};
        std::vector<uint32_t> indices{0, 1, 2};

        ref<Buffer> vertex_buffer = device->create_buffer({
            .usage = BufferUsage::shader_resource,
            .label = "vertex_buffer",
            .data = vertices.data(),
            .data_size = vertices.size() * sizeof(float2),
        });

        ref<Buffer> index_buffer = device->create_buffer({
            .usage = BufferUsage::shader_resource,
            .label = "index_buffer",
            .data = indices.data(),
            .data_size = indices.size() * sizeof(uint32_t),
        });

        ref<Texture> render_texture = device->create_texture({
            .format = Format::rgba32_float,
            .width = 1024,
            .height = 1024,
            .usage = TextureUsage::render_target,
            .label = "render_texture",
        });

        ref<InputLayout> input_layout = device->create_input_layout({
            .input_elements{
                {
                    .semantic_name = "POSITION",
                    .semantic_index = 0,
                    .format = Format::rg32_float,
                },
            },
            .vertex_streams{
                {.stride = 8},
            },
        });

        ref<ShaderProgram> program = device->load_program("render_pipeline.slang", {"vertex_main", "fragment_main"});
        ref<RenderPipeline> pipeline = device->create_render_pipeline({
            .program = program,
            .input_layout = input_layout,
            .targets = {{.format = Format::rgba32_float}},
        });

        ref<CommandEncoder> command_encoder = device->create_command_encoder();
        {
            auto pass_encoder = command_encoder->begin_render_pass({
                .color_attachments = {{.view = render_texture->create_view({})}},
            });
            pass_encoder->bind_pipeline(pipeline);
            pass_encoder->set_render_state({
                .viewports = {{Viewport::from_size(float(render_texture->width()), float(render_texture->height()))}},
                .scissor_rects = {{ScissorRect::from_size(render_texture->width(), render_texture->height())}},
                .vertex_buffers = {vertex_buffer},
            });
            pass_encoder->draw({.vertex_count = 3});
            pass_encoder->end();
        }
        device->submit_command_buffer(command_encoder->finish());

        tev::show(render_texture, "render_pipeline");

        device->close();
    }

    sgl::static_shutdown();
    return 0;
}
