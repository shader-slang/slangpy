// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/sgl.h"
#include "sgl/core/input.h"
#include "sgl/core/timer.h"
#include "sgl/core/window.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/command.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/surface.h"
#include "sgl/device/agility_sdk.h"

#include <algorithm>

SGL_EXPORT_AGILITY_SDK

static const std::filesystem::path EXAMPLE_DIR(SGL_EXAMPLE_DIR);

using namespace sgl;

/// Matches the Params struct in render_window.slang.
struct Params {
    float2 resolution;
    float2 mouse_pos;
    float mouse_radius;
    float time;
    float scale;
    float padding;
};

/// Renders an animated pattern into a window. The mouse highlights the pattern and the
/// mouse wheel zooms. Press escape to close the window.
struct App {
    ref<Window> window;
    ref<Device> device;
    ref<Surface> surface;
    ref<ShaderProgram> program;
    ref<RenderPipeline> pipeline;
    ref<Buffer> params_buffer;

    float2 mouse_pos{0.f};
    float scale{4.f};

    App()
    {
        window = Window::create({
            .width = 1280,
            .height = 720,
            .title = "render_window",
            .resizable = true,
        });
        device = Device::create({
            .enable_debug_layers = true,
            .compiler_options = {.include_paths = {EXAMPLE_DIR}},
        });
        surface = device->create_surface(window);
        surface->configure({.width = window->width(), .height = window->height()});

        program = device->load_program("render_window.slang", {"vertex_main", "fragment_main"});
        params_buffer = device->create_buffer({
            .size = sizeof(Params),
            .struct_size = sizeof(Params),
            .usage = BufferUsage::shader_resource,
            .label = "params_buffer",
        });

        window->set_on_keyboard_event(
            [this](const KeyboardEvent& event)
            {
                if (event.is_key_press() && event.key == KeyCode::escape)
                    window->close();
            }
        );
        window->set_on_mouse_event(
            [this](const MouseEvent& event)
            {
                if (event.is_move())
                    mouse_pos = event.pos;
                else if (event.is_scroll())
                    scale = std::clamp(scale * (event.scroll.y > 0.f ? 0.9f : 1.1f), 1.f, 32.f);
            }
        );
        window->set_on_resize(
            [this](uint32_t width, uint32_t height)
            {
                device->wait();
                if (width > 0 && height > 0)
                    surface->configure({.width = width, .height = height});
                else
                    surface->unconfigure();
            }
        );
    }

    void run()
    {
        Timer timer;

        while (!window->should_close()) {
            window->process_events();

            if (!surface->config())
                continue;

            // Upload the parameters before acquiring the next image. In a browser, the image is only
            // valid until the application yields to the browser, which GPU uploads may do.
            const SurfaceConfig& config = *surface->config();
            Params params{
                .resolution = float2(float(config.width), float(config.height)),
                .mouse_pos = mouse_pos,
                .mouse_radius = 100.f,
                .time = float(timer.elapsed_s()),
                .scale = scale,
            };
            params_buffer->set_data(&params, sizeof(params));

            ref<Texture> surface_texture = surface->acquire_next_image();
            if (!surface_texture)
                continue;

            if (!pipeline) {
                pipeline = device->create_render_pipeline({
                    .program = program,
                    .targets = {{.format = surface_texture->format()}},
                });
            }

            const uint32_t width = surface_texture->width();
            const uint32_t height = surface_texture->height();

            ref<CommandEncoder> command_encoder = device->create_command_encoder();
            {
                auto pass_encoder = command_encoder->begin_render_pass({
                    .color_attachments = {{.view = surface_texture->create_view({}), .load_op = LoadOp::clear}},
                });
                ShaderCursor cursor(pass_encoder->bind_pipeline(pipeline));
                cursor["g_params"] = params_buffer;
                pass_encoder->set_render_state({
                    .viewports = {Viewport::from_size(float(width), float(height))},
                    .scissor_rects = {ScissorRect::from_size(width, height)},
                });
                pass_encoder->draw({.vertex_count = 3});
                pass_encoder->end();
            }
            device->submit_command_buffer(command_encoder->finish());
            surface_texture = nullptr;

            surface->present();
        }

        device->wait();
        device->close();
    }
};

int main()
{
    sgl::static_init();

    {
        App app;
        app.run();
    }

    sgl::static_shutdown();
    return 0;
}
