// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/platform.h"
#include "sgl/core/object.h"
#include "sgl/core/input.h"
#include "sgl/math/vector_types.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>

// GLFW forward declarations.
struct GLFWwindow;
struct GLFWcursor;

namespace sgl {

/// Window modes.
enum class WindowMode {
    normal,
    minimized,
    maximized,
    fullscreen,
};

SGL_ENUM_INFO(
    WindowMode,
    {
        {WindowMode::normal, "normal"},
        {WindowMode::minimized, "minimized"},
        {WindowMode::maximized, "maximized"},
        {WindowMode::fullscreen, "fullscreen"},
    }
);
SGL_ENUM_REGISTER(WindowMode);

/// Copied window placement for session restoration. Geometry describes the client
/// area in screen coordinates, not framebuffer pixels. Minimized geometry is never
/// returned. Fullscreen windows retain their initial normal placement.
/// Pass as WindowDesc::placement to recreate the normal geometry and maximization state.
struct WindowPlacement {
    /// Client-area origin in virtual-desktop screen coordinates. Negative values are valid.
    int2 normal_position{0, 0};
    /// Client-area size before maximization or minimization, in screen coordinates.
    uint2 normal_size{0, 0};
    /// Captured maximization state; for minimized windows, the last non-minimized state.
    bool maximized{false};
};

/// Window description.
struct WindowDesc {
    /// Width of the client area in screen coordinates.
    uint32_t width;
    /// Height of the client area in screen coordinates.
    uint32_t height;
    /// Title of the window.
    std::string title;
    /// Initial window mode. Minimized windows are shown, then minimized.
    WindowMode mode{WindowMode::normal};
    /// Whether the window is resizable.
    bool resizable{true};
    /// Initial client-area position in screen coordinates; omitted lets the OS choose.
    /// Some window systems (such as Wayland) do not support explicit positioning.
    std::optional<int2> position;
    /// Whether to show the window after positioning it, then apply its initial mode.
    /// False creates a hidden window without applying initial minimization/maximization.
    /// show() does not apply the initial mode later.
    bool visible{true};
    /// Optional saved placement. Overrides width, height, position, and mode with
    /// its normal geometry and normal/maximized state. Cannot be combined with
    /// WindowMode::fullscreen. Title, resizable, and visible remain independent.
    std::optional<WindowPlacement> placement;
};

/// Mouse cursor modes.
enum class CursorMode : uint32_t {
    /// The cursor is visible and behaves normally.
    normal,
    /// The cursor is hidden when over the window.
    hidden,
    /// The cursor is hidden and locked to the window.
    disabled,
};

SGL_ENUM_INFO(
    CursorMode,
    {
        {CursorMode::normal, "normal"},
        {CursorMode::hidden, "hidden"},
        {CursorMode::disabled, "disabled"},
    }
);
SGL_ENUM_REGISTER(CursorMode);

/// Mouse cursor shapes.
enum class CursorShape : uint32_t {
    /// Arrow cursor shape.
    arrow,
    /// I-beam cursor shape (for text editing).
    ibeam,
    /// Crosshair cursor shape.
    crosshair,
    /// Hand cursor shape (for links and dragging).
    hand,
    /// Horizontal resize cursor shape.
    hresize,
    /// Vertical resize cursor shape.
    vresize,
};

SGL_ENUM_INFO(
    CursorShape,
    {
        {CursorShape::arrow, "arrow"},
        {CursorShape::ibeam, "ibeam"},
        {CursorShape::crosshair, "crosshair"},
        {CursorShape::hand, "hand"},
        {CursorShape::hresize, "hresize"},
        {CursorShape::vresize, "vresize"},
    }
);
SGL_ENUM_REGISTER(CursorShape);

/**
 * \brief Window class.
 *
 * Platform independent class for managing a window and handle input events.
 * Window operations and monitor queries must run on the main thread.
 */
class SGL_API Window : public Object {
    SGL_OBJECT(Window)
public:
    /// Constructor.
    /// \param width Width of the client area in screen coordinates.
    /// \param height Height of the client area in screen coordinates.
    /// \param title Title of the window.
    /// \param mode Window mode.
    /// \param resizable Whether the window is resizable.
    /// \param position Optional initial client origin in screen coordinates.
    /// \param visible Show after applying initial placement. False creates a hidden window.
    /// Hidden windows ignore initial minimization/maximization; show() does not apply it later.
    /// \param placement Optional saved placement. Overrides width, height, position, and mode.
    /// Cannot be combined with fullscreen mode; title, resizable, and visible remain independent.
    Window(WindowDesc desc);
    ~Window();

    static ref<Window> create(WindowDesc desc) { return make_ref<Window>(desc); }

    /// The title of the window.
    const std::string& title() const { return m_title; }
    void set_title(std::string title);

    /// The native window handle.
    WindowHandle window_handle() const;

    void set_icon(const std::filesystem::path& path);

    /// Client-area origin in virtual-desktop screen coordinates. Negative values are valid.
    int2 position() const;
    void set_position(int2 position);

    /// The width of the client area in screen coordinates.
    uint32_t width() const { return m_width; }
    void set_width(uint32_t width);

    /// The height of the client area in screen coordinates.
    uint32_t height() const { return m_height; }
    void set_height(uint32_t height);

    /// Size of the client area in screen coordinates, not framebuffer pixels.
    uint2 size() const { return uint2{m_width, m_height}; }
    void set_size(uint2 size);

    /// Resize the window.
    /// \param width The new client width in screen coordinates.
    /// \param height The new client height in screen coordinates.
    void resize(uint32_t width, uint32_t height);

    /// Decoration extents (left, top, right, bottom) in screen coordinates.
    /// Values may be unavailable until the window manager decorates the window.
    int4 frame_size() const;

    /// Per-window horizontal and vertical content scale, not a framebuffer-size ratio.
    float2 content_scale() const;

    /// Normal client geometry and the last non-minimized maximization state.
    /// Position is best effort where the window system restricts positioning.
    WindowPlacement placement() const;

    /// Whether the window is currently minimized (iconified).
    bool is_minimized() const;
    /// Whether the window is currently maximized.
    bool is_maximized() const;
    /// Whether the window uses fullscreen mode, even while minimized or hidden.
    bool is_fullscreen() const { return m_fullscreen; }
    /// Whether the window is visible. A minimized window can still be visible.
    bool is_visible() const;

    /// Minimize (iconify) a window, retaining its normal placement and maximized state.
    void minimize();
    /// Maximize a window without replacing its normal placement. Does not show hidden windows.
    /// Not supported for WindowMode::fullscreen windows.
    void maximize();
    /// Ask the window system to restore a minimized or maximized window.
    /// A minimized maximized window may return to maximized state first.
    /// Not supported for WindowMode::fullscreen windows.
    /// Show hidden maximized windows before restoring; restoring while hidden is platform dependent.
    void restore();
    /// Show a hidden window.
    void show();
    /// Hide a window without minimizing it.
    void hide();

    /// Close the window.
    void close();

    /// True if the window should be closed.
    bool should_close() const;

    /// Process any pending events.
    void process_events();

    /// The mouse cursor mode.
    CursorMode cursor_mode() const { return m_cursor_mode; }
    void set_cursor_mode(CursorMode mode);

    /// The mouse cursor shape.
    CursorShape cursor_shape() const { return m_cursor_shape; }
    void set_cursor_shape(CursorShape shape);

    /// Get the clipboard content.
    std::optional<std::string> get_clipboard() const;
    /// Set the clipboard content.
    void set_clipboard(const std::string& text);

    // events

    using ResizeCallback = std::function<void(uint32_t /* width */, uint32_t /* height */)>;
    using RefreshCallback = std::function<void()>;
    using KeyboardEventCallback = std::function<void(const KeyboardEvent& /* event */)>;
    using MouseEventCallback = std::function<void(const MouseEvent& /* event */)>;
    using GamepadEventCallback = std::function<void(const GamepadEvent& /* event */)>;
    using GamepadStateCallback = std::function<void(const GamepadState& /* state */)>;
    using DropFilesCallback = std::function<void(std::span<const char*> /* files */)>;

    /// Event handler to be called when the window is resized.
    const ResizeCallback& on_resize() const { return m_on_resize; }
    void set_on_resize(ResizeCallback on_resize) { m_on_resize = std::move(on_resize); }

    /// Event handler to be called when the window contents need to be refreshed.
    const RefreshCallback& on_refresh() const { return m_on_refresh; }
    void set_on_refresh(RefreshCallback on_refresh) { m_on_refresh = std::move(on_refresh); }

    /// Event handler to be called when a keyboard event occurs.
    const KeyboardEventCallback& on_keyboard_event() const { return m_on_keyboard_event; }
    void set_on_keyboard_event(KeyboardEventCallback on_keyboard_event)
    {
        m_on_keyboard_event = std::move(on_keyboard_event);
    }

    /// Event handler to be called when a mouse event occurs.
    const MouseEventCallback& on_mouse_event() const { return m_on_mouse_event; }
    void set_on_mouse_event(MouseEventCallback on_mouse_event) { m_on_mouse_event = std::move(on_mouse_event); }

    /// Event handler to be called when a gamepad event occurs.
    const GamepadEventCallback& on_gamepad_event() const { return m_on_gamepad_event; }
    void set_on_gamepad_event(GamepadEventCallback on_gamepad_event)
    {
        m_on_gamepad_event = std::move(on_gamepad_event);
    }

    /// Event handler to be called when the gamepad state changes.
    const GamepadStateCallback& on_gamepad_state() const { return m_on_gamepad_state; }
    void set_on_gamepad_state(GamepadStateCallback on_gamepad_state)
    {
        m_on_gamepad_state = std::move(on_gamepad_state);
    }

    /// Event handler to be called when files are dropped onto the window.
    const DropFilesCallback& on_drop_files() const { return m_on_drop_files; }
    void set_on_drop_files(DropFilesCallback on_drop_files) { m_on_drop_files = std::move(on_drop_files); }

    std::string to_string() const override;

    /// Destroy all native windows and cursors before GLFW shutdown.
    /// This MUST NOT be called by the user!
    static void _destroy_all_windows();

private:
    GLFWwindow* glfw_window() const;
    void destroy_window();

    void update_normal_placement();
    void poll_gamepad_input();

    void handle_window_size(uint32_t width, uint32_t height);
    void handle_window_refresh();
    void handle_keyboard_event(const KeyboardEvent& event);
    void handle_mouse_event(const MouseEvent& event);
    void handle_gamepad_event(const GamepadEvent& event);
    void handle_drop_files(std::span<const char*> files);

    uint32_t m_width;
    uint32_t m_height;
    std::string m_title;
    GLFWwindow* m_window{nullptr};
    WindowPlacement m_placement;
    bool m_fullscreen{false};

    bool m_should_close{false};

    CursorMode m_cursor_mode{CursorMode::normal};
    CursorShape m_cursor_shape{CursorShape::arrow};
    std::array<GLFWcursor*, 6> m_cursor_cache{};
    float2 m_mouse_pos{0.f, 0.f};
    KeyModifierFlags m_mods{KeyModifierFlags::none};

    static constexpr int INVALID_GAMEPAD_ID = -1;
    int m_gamepad_id{INVALID_GAMEPAD_ID};
    GamepadState m_gamepad_prev_state;

    ResizeCallback m_on_resize;
    RefreshCallback m_on_refresh;
    KeyboardEventCallback m_on_keyboard_event;
    MouseEventCallback m_on_mouse_event;
    GamepadEventCallback m_on_gamepad_event;
    GamepadStateCallback m_on_gamepad_state;
    DropFilesCallback m_on_drop_files;

    friend struct EventHandlers;
};

} // namespace sgl
