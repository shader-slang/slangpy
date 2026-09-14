// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "app.h"

#include "sgl/device/command.h"
#include "sgl/ui/ui.h"

#include "sgl/core/error.h"

namespace sgl {

namespace {
    /// Upper bound on consecutive acquire/present failures at a stable surface size
    /// before the swapchain-recovery loop gives up and surfaces the failure as
    /// fatal. Sized generously so aggressive continuous resizing never trips it.
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

    // If the surface is suspended (e.g. after a minimize) but the window now
    // has a non-zero size, arm a reconfigure. The size is queried from the
    // windowing system rather than the callback-updated cache, so a restore is
    // detected even on a platform that doesn't re-fire the resize callback.
    if (!m_surface->config()) {
        uint2 size = m_window->query_size();
        if (size.x > 0 && size.y > 0)
            m_surface_dirty = true;
    }

    // A prior frame's acquire/present reported a recoverable invalidation:
    // reconfigure against the current window size before rendering.
    if (m_surface_dirty)
        reconfigure_surface();

    if (!m_surface->config())
        return;

    ref<Texture> texture;
    if (SLANG_FAILED(m_surface->try_acquire_next_image(texture))) {
        mark_surface_failed();
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
        mark_surface_failed();
        return;
    }

    // A fully successful frame clears the bounded-failure counter.
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
    // Reconfigure against the window's current size, so this serves both the
    // resize callback and the out-of-band recovery path from one place. The
    // checked m_device->wait() below is what preserves device loss: it (and
    // configure()/submit in the render path) stay on the throwing SLANG_RHI_CALL
    // path, so a genuine device loss propagates here instead of being retried by
    // the recovery loop. Runs only on a resize or an armed invalidation, so
    // steady-state frames keep no queue-idle wait.
    m_device->wait();
    uint2 size = m_window->query_size();
    if (size.x > 0 && size.y > 0) {
        // A genuine size change is legitimate churn (the window is being
        // resized), not a stuck surface, so let the bounded-failure counter
        // reset - otherwise a sustained drag that keeps invalidating the
        // swapchain could trip the fatal ceiling and crash the very resize we
        // are trying to survive. Only a persistent failure at a *stable* size
        // reaches the ceiling.
        if (size.x != m_surface_config.width || size.y != m_surface_config.height)
            m_surface_recovery_failures = 0;
        m_surface_config.width = size.x;
        m_surface_config.height = size.y;
        m_surface->configure(m_surface_config);
    } else {
        m_surface->unconfigure();
    }
    m_surface_dirty = false;
}

void AppWindow::mark_surface_failed()
{
    m_surface_dirty = true;
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
