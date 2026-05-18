// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/sgl.h"
#include "sgl/core/platform.h"
#include "sgl/core/window.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/device/kernel.h"
#include "sgl/device/agility_sdk.h"
#include "sgl/device/input_layout.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/surface.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

SGL_EXPORT_AGILITY_SDK

#ifdef __EMSCRIPTEN__
static const std::filesystem::path EXAMPLE_DIR(".");
#else
static const std::filesystem::path EXAMPLE_DIR(SGL_EXAMPLE_DIR);
#endif

using namespace sgl;

struct App {
    ref<Window> window;
    ref<Device> device;
    ref<Surface> surface;
    ref<Buffer> vertex_buffer;
    ref<Buffer> index_buffer;
    ref<InputLayout> input_layout;
    ref<ShaderProgram> program;
    ref<RenderPipeline> pipeline;

    App()
    {
        window = Window::create({
            .width = 1024,
            .height = 1024,
            .title = "render_pipeline",
        });

        device = Device::create({
            .enable_debug_layers = true,
            .compiler_options = {.include_paths = {EXAMPLE_DIR}},
        });

        surface = device->create_surface(window);
        surface->configure({
            .format = surface->info().preferred_format,
            .usage = TextureUsage::render_target,
            .width = window->width(),
            .height = window->height(),
            .vsync = true,
        });

        std::vector<float2> vertices{{-1.f, -1.f}, {1.f, -1.f}, {0.f, 1.f}};
        std::vector<uint32_t> indices{0, 1, 2};

        vertex_buffer = device->create_buffer({
            .usage = BufferUsage::vertex_buffer,
            .label = "vertex_buffer",
            .data = vertices.data(),
            .data_size = vertices.size() * sizeof(float2),
        });

        index_buffer = device->create_buffer({
            .usage = BufferUsage::index_buffer,
            .label = "index_buffer",
            .data = indices.data(),
            .data_size = indices.size() * sizeof(uint32_t),
        });

        input_layout = device->create_input_layout({
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

        program = device->load_program("render_pipeline.slang", {"vertex_main", "fragment_main"});
        pipeline = device->create_render_pipeline({
            .program = program,
            .input_layout = input_layout,
            .targets = {{.format = surface->info().preferred_format}},
        });
    }

    void main_loop()
    {
        window->process_events();

        if (!surface->config())
            return;

        ref<Texture> surface_texture = surface->acquire_next_image();
        if (surface_texture) {
            ref<CommandEncoder> command_encoder = device->create_command_encoder();
            {
                auto pass_encoder = command_encoder->begin_render_pass({
                    .color_attachments = {{.view = surface_texture->create_view({}), .load_op = LoadOp::clear}},
                });
                pass_encoder->bind_pipeline(pipeline);
                pass_encoder->set_render_state({
                    .viewports
                    = {{Viewport::from_size(float(surface_texture->width()), float(surface_texture->height()))}},
                    .scissor_rects = {{ScissorRect::from_size(surface_texture->width(), surface_texture->height())}},
                    .vertex_buffers = {vertex_buffer},
                });
                pass_encoder->draw({.vertex_count = 3});
                pass_encoder->end();
            }
            device->submit_command_buffer(command_encoder->finish());

            surface->present();
        }
    }

    ~App()
    {
        if (device)
            device->close();
    }
};

int main()
{
    sgl::static_init();

    {
        App app;

        while (!app.window->should_close()) {
            app.main_loop();
#ifdef __EMSCRIPTEN__
            emscripten_sleep(0);
#endif
        }
    }

    sgl::static_shutdown();
    return 0;
}
