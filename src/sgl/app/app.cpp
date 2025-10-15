// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "app.h"

#include "sgl/device/command.h"
#include "sgl/ui/ui.h"

namespace sgl {

// -----------------------------------------------------------------------------
// App
// -----------------------------------------------------------------------------

App::App(AppDesc desc)
{
    m_device = desc.device ? desc.device : Device::create();
}

App::~App()
{
    m_device->close();
}

void App::run()
{
    auto all_windows_should_close = [this]()
    {
        for (const auto& window : m_windows)
            if (!window->_should_close())
                return false;
        return true;
    };

    while (!m_terminate && !all_windows_should_close()) {
        run_frame();
    }
}

void App::run_frame()
{
    for (const auto& window : m_windows) {
        window->_run_frame();
    }
}

void App::terminate()
{
    m_terminate = true;
}

void App::_add_window(AppWindow* window)
{
    m_windows.push_back(window);
}

void App::_remove_window(AppWindow* window)
{
    auto it = std::find(m_windows.begin(), m_windows.end(), window);
    if (it != m_windows.end())
        m_windows.erase(it);
}

// -----------------------------------------------------------------------------
// AppWindow
// -----------------------------------------------------------------------------

AppWindow::AppWindow(AppWindowDesc desc)
    : m_app(desc.app)
{
    m_device = m_app->device();
    m_window = Window::create({
        .width = desc.width,
        .height = desc.height,
        .title = desc.title,
        .mode = desc.mode,
        .resizable = desc.resizable,
    });
    m_surface = m_device->create_surface(m_window);
    m_surface_config.format = desc.surface_format;
    m_surface_config.width = desc.width;
    m_surface_config.height = desc.height;
    m_surface_config.vsync = desc.enable_vsync;
    m_surface->configure(m_surface_config);

    m_ui_context = make_ref<ui::Context>(ref(m_device));

    m_window->set_on_resize([this](uint32_t width, uint32_t height) { handle_resize(width, height); });
    m_window->set_on_keyboard_event([this](const KeyboardEvent& event) { handle_keyboard_event(event); });
    m_window->set_on_mouse_event([this](const MouseEvent& event) { handle_mouse_event(event); });
    m_window->set_on_gamepad_event([this](const GamepadEvent& event) { handle_gamepad_event(event); });
    m_window->set_on_drop_files([this](std::span<const char*> files) { handle_drop_files(files); });

    m_app->_add_window(this);
}

AppWindow::~AppWindow()
{
    m_app->_remove_window(this);
}

ui::Screen* AppWindow::screen() const
{
    return m_ui_context->screen();
}

void AppWindow::on_keyboard_event(const KeyboardEvent& event)
{
    if (event.is_key_press()) {
        if (event.key == KeyCode::escape)
            m_app->terminate();
    }
}

void AppWindow::_run_frame()
{
    m_window->process_events();
    m_ui_context->process_events();

    if (!m_surface->config())
        return;
    ref<Texture> texture = m_surface->acquire_next_image();
    if (!texture)
        return;

    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();

    if (get_format_info(texture->format()).is_float_format())
        command_encoder->clear_texture_float(texture, {}, float4{0.f, 0.f, 0.f, 1.f});
    else
        command_encoder->clear_texture_uint(texture, {}, uint4{0, 0, 0, 255});

    struct RenderContext render_context{
        .surface_texture = texture,
        .command_encoder = command_encoder,
    };

    render(render_context);

    m_ui_context->new_frame(texture->width(), texture->height());

    render_ui();

    m_ui_context->render(texture, command_encoder);

    m_device->submit_command_buffer(command_encoder->finish());

    texture.reset();
    m_surface->present();
}

bool AppWindow::_should_close()
{
    return m_window->should_close();
}

void AppWindow::handle_resize(uint32_t width, uint32_t height)
{
    m_device->wait();
    if (width > 0 && height > 0) {
        m_surface_config.width = width;
        m_surface_config.height = height;
        m_surface->configure(m_surface_config);
    } else {
        m_surface->unconfigure();
    }
    on_resize(width, height);
}

void AppWindow::handle_keyboard_event(const KeyboardEvent& event)
{
    if (m_ui_context->handle_keyboard_event(event))
        return;
    on_keyboard_event(event);
}

void AppWindow::handle_mouse_event(const MouseEvent& event)
{
    if (m_ui_context->handle_mouse_event(event))
        return;
    on_mouse_event(event);
}

void AppWindow::handle_gamepad_event(const GamepadEvent& event)
{
    on_gamepad_event(event);
}

void AppWindow::handle_drop_files(std::span<const char*> files)
{
    on_drop_files(files);
}

} // namespace sgl
