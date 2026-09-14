// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "app.h"

#include "sgl/device/command.h"
#include "sgl/ui/ui.h"

#include "sgl/core/error.h"

namespace sgl {

namespace {
    /// Upper bound on consecutive acquire/present failures at a stable surface
    /// size before the swapchain-recovery loop gives up and surfaces the failure
    /// as fatal. Observed framebuffer-size changes reset the counter, so a resize
    /// does not accumulate toward it; only a persistent stable-size failure does.
    constexpr uint32_t kMaxSurfaceRecoveryFailures = 64;
} // namespace

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

    m_window->set_on_resize(
        [this](uint32_t width, uint32_t height)
        {
            handle_resize(width, height);
        }
    );
    m_window->set_on_keyboard_event(
        [this](const KeyboardEvent& event)
        {
            handle_keyboard_event(event);
        }
    );
    m_window->set_on_mouse_event(
        [this](const MouseEvent& event)
        {
            handle_mouse_event(event);
        }
    );
    m_window->set_on_gamepad_event(
        [this](const GamepadEvent& event)
        {
            handle_gamepad_event(event);
        }
    );
    m_window->set_on_drop_files(
        [this](std::span<const char*> files)
        {
            handle_drop_files(files);
        }
    );

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

    // Reconcile the surface with the window's drawable state each frame, so a
    // minimize/restore that emits no resize callback is still handled: suspend
    // (unconfigure) while minimized, resume when restored. is_minimized() is the
    // portable signal (some backends keep the framebuffer size non-zero while
    // iconified); reconfigure_surface() then makes the precise decision. Steady
    // state (visible + configured) touches neither branch and adds no per-frame
    // windowing-system round-trip.
    if (m_window->is_minimized()) {
        if (m_surface->config())
            reconfigure_surface();
        return;
    }
    if (!m_surface->config()) {
        uint2 framebuffer_size = m_window->query_framebuffer_size();
        if (framebuffer_size.x > 0 && framebuffer_size.y > 0)
            reconfigure_surface();
        if (!m_surface->config())
            return;
    }

    ref<Texture> texture;
    if (SLANG_FAILED(m_surface->try_acquire_next_image(texture))) {
        recover_surface();
        return;
    }

    m_ui_context->begin_frame(texture->width(), texture->height());

    ref<CommandEncoder> command_encoder = m_device->create_command_encoder();

    const FormatInfo& format_info = get_format_info(texture->format());
    if (format_info.is_float_format() || format_info.is_normalized_format())
        command_encoder->clear_texture_float(texture, {}, float4{0.f, 0.f, 0.f, 1.f});
    else
        command_encoder->clear_texture_uint(texture, {}, uint4{0, 0, 0, 255});

    struct RenderContext render_context{
        .surface_texture = texture,
        .command_encoder = command_encoder,
    };

    render(render_context);

    render_ui();

    m_ui_context->end_frame(texture, command_encoder);

    m_device->submit_command_buffer(command_encoder->finish());

    texture.reset();
    // Some backends report a recoverable invalidation from present, others only
    // from the next acquire - guard present too so either path is handled.
    if (SLANG_FAILED(m_surface->try_present())) {
        recover_surface();
        return;
    }

    m_surface_recovery_failures = 0;
}

bool AppWindow::_should_close()
{
    return m_window->should_close();
}

void AppWindow::handle_resize(uint32_t width, uint32_t height)
{
    reconfigure_surface();
    on_resize(width, height);
}

void AppWindow::reconfigure_surface()
{
    // Reconfigure against the current framebuffer size, so the resize callback
    // and the recovery path rebuild from one place. configure() runs on the
    // throwing SLANG_RHI_CALL path, so a device loss is surfaced here (if the RHI
    // reports it) rather than being retried; this runs only on a resize or a
    // failure, so steady-state frames add no queue-idle wait.
    m_device->wait();
    uint2 size = m_window->query_framebuffer_size();
    if (!m_window->is_minimized() && size.x > 0 && size.y > 0) {
        // A genuine size change is legitimate churn (the window is resizing),
        // not a stuck surface, so reset the bounded-failure counter; only a
        // persistent failure at a stable size reaches the fatal ceiling. This
        // keeps a sustained drag from tripping the ceiling and crashing the
        // very resize we are recovering from.
        if (size.x != m_surface_config.width || size.y != m_surface_config.height)
            m_surface_recovery_failures = 0;
        m_surface_config.width = size.x;
        m_surface_config.height = size.y;
        m_surface->configure(m_surface_config);
    } else {
        m_surface->unconfigure();
    }
}

void AppWindow::recover_surface()
{
    // The RHI returns an undifferentiated SLANG_FAIL, so we cannot classify the
    // failure; reconfigure immediately as a probe. configure() inside
    // reconfigure_surface() runs on the throwing path, so a device loss is
    // surfaced synchronously if the RHI reports one, and the bounded counter is
    // the fallback that fails loudly if recovery never succeeds at a stable size.
    reconfigure_surface();
    if (++m_surface_recovery_failures > kMaxSurfaceRecoveryFailures)
        SGL_THROW("Surface acquire/present kept failing after reconfiguration; treating as fatal.");
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
