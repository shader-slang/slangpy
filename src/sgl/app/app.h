// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"
#include "sgl/core/window.h"

#include "sgl/device/fwd.h"
#include "sgl/device/device.h"
#include "sgl/device/surface.h"
#include "sgl/ui/fwd.h"

#include <vector>

namespace sgl {

class AppWindow;

struct AppDesc {
    /// Device to use for rendering.
    /// If not provided, a default device will be created.
    ref<Device> device;
};

class SGL_API App : Object {
    SGL_OBJECT(App)
public:
    App(AppDesc desc);
    virtual ~App();

    Device* device() const { return m_device; }

    void run();
    void run_frame();

    void terminate();

    void _add_window(AppWindow* window);
    void _remove_window(AppWindow* window);

private:
    ref<Device> m_device;

    std::vector<AppWindow*> m_windows;

    bool m_terminate{false};
};

struct AppWindowDesc {
    App* app{nullptr};
    /// Width of the window in pixels.
    uint32_t width{1920};
    /// Height of the window in pixels.
    uint32_t height{1280};
    /// Title of the window.
    std::string title{"slangpy"};
    /// Window mode.
    WindowMode mode{WindowMode::normal};
    /// Whether the window is resizable.
    bool resizable{true};
    /// Format of the swapchain images.
    Format surface_format{Format::undefined};
    /// Enable/disable vertical synchronization.
    bool enable_vsync{false};
};

class SGL_API AppWindow : Object {
    SGL_OBJECT(AppWindow)
public:
    AppWindow(AppWindowDesc desc);
    virtual ~AppWindow();

    App* app() const { return m_app; }
    Device* device() const { return m_device; }

    ui::Screen* screen() const;

    struct RenderContext {
        Texture* surface_texture;
        CommandEncoder* command_encoder;
    };

    virtual void render(RenderContext render_context) { SGL_UNUSED(render_context); }
    virtual void render_ui() { }

    virtual void on_resize(uint32_t width, uint32_t height) { SGL_UNUSED(width, height); }
    virtual void on_keyboard_event(const KeyboardEvent& event);
    virtual void on_mouse_event(const MouseEvent& event) { SGL_UNUSED(event); }
    virtual void on_gamepad_event(const GamepadEvent& event) { SGL_UNUSED(event); }
    virtual void on_drop_files(std::span<const char*> files) { SGL_UNUSED(files); }

    void _run_frame();
    bool _should_close();

private:
    void handle_resize(uint32_t width, uint32_t height);
    void handle_keyboard_event(const KeyboardEvent& event);
    void handle_mouse_event(const MouseEvent& event);
    void handle_gamepad_event(const GamepadEvent& event);
    void handle_drop_files(std::span<const char*> files);

    /// Reconfigure the surface against the current framebuffer size, or suspend
    /// it when the window is minimized or has zero area. Shared by the resize
    /// callback and the recovery path so both rebuild identically. Its configure()
    /// surfaces a device loss (if the RHI reports one) rather than retrying it.
    void reconfigure_surface();

    /// Recover from a failed acquire/present by reconfiguring immediately. The
    /// failure is unclassified (the RHI returns an opaque SLANG_FAIL), so this is
    /// a best-effort probe: reconfigure surfaces a device loss synchronously if
    /// the RHI reports one, and a bounded counter surfaces a persistent
    /// stable-size failure as fatal.
    void recover_surface();

    App* m_app;
    Device* m_device;
    ref<Window> m_window;
    ref<Surface> m_surface;
    SurfaceConfig m_surface_config;
    ref<ui::Context> m_ui_context;

    /// Consecutive acquire/present failures at a stable surface size. Bounds the
    /// recovery loop so a persistent (unresolved) failure is surfaced instead of
    /// being retried silently forever.
    uint32_t m_surface_recovery_failures{0};
};

} // namespace sgl
