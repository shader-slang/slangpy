// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/platform.h"
#include "sgl/core/object.h"
#include "sgl/core/input.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>

// GLFW forward declarations.
struct GLFWwindow;

namespace sgl {

/// Window modes.
enum class WindowMode {
    normal,
    minimized,
    fullscreen,
};

/// Window description.
struct WindowDesc {
    /// Width of the window in pixels.
    uint32_t width;
    /// Height of the window in pixels.
    uint32_t height;
    /// Title of the window.
    std::string title;
    /// Window mode.
    WindowMode mode{WindowMode::normal};
    /// Whether the window is resizable.
    bool resizable{true};
};

/**
 * \brief Window class.
 *
 * Platform independent class for managing a window and handle input events.
 */
class SGL_API Window : public Object {
    SGL_OBJECT(Window)
public:
    /// Constructor.
    /// \param width Width of the window in pixels.
    /// \param height Height of the window in pixels.
    /// \param title Title of the window.
    /// \param mode Window mode.
    /// \param resizable Whether the window is resizable.
    Window(WindowDesc desc);
    ~Window();

    static ref<Window> create(WindowDesc desc) { return make_ref<Window>(desc); }

    /// The native window handle.
    WindowHandle window_handle() const;

    /// The width of the window in pixels.
    uint32_t width() const { return m_width; }
    /// The height of the window in pixels.
    uint32_t height() const { return m_height; }

    /// Resize the window.
    /// \param width The new width of the window in pixels.
    /// \param height The new height of the window in pixels.
    void resize(uint32_t width, uint32_t height);

    /// Set window position.
    /// \param x The new x position of the window.
    /// \param y The new y position of the window.
    void set_position(uint32_t x, uint32_t y);

    /// The title of the window.
    const std::string& title() const { return m_title; }
    void set_title(std::string title);

    void set_icon(const std::filesystem::path& path);

    /// Close the window.
    void close();

    /// True if the window should be closed.
    bool should_close() const;

    /// Process any pending events.
    void process_events();

    /// Set the clipboard content.
    void set_clipboard(const std::string& text);

    /// Get the clipboard content.
    std::optional<std::string> get_clipboard() const;

    /// The mouse cursor mode.
    CursorMode cursor_mode() const { return m_cursor_mode; }
    void set_cursor_mode(CursorMode mode);

    // events

    using ResizeCallback = std::function<void(uint32_t /* width */, uint32_t /* height */)>;
    using KeyboardEventCallback = std::function<void(const KeyboardEvent& /* event */)>;
    using MouseEventCallback = std::function<void(const MouseEvent& /* event */)>;
    using GamepadEventCallback = std::function<void(const GamepadEvent& /* event */)>;
    using GamepadStateCallback = std::function<void(const GamepadState& /* state */)>;
    using DropFilesCallback = std::function<void(std::span<const char*> /* files */)>;

    /// Event handler to be called when the window is resized.
    const ResizeCallback& on_resize() const { return m_on_resize; }
    void set_on_resize(ResizeCallback on_resize) { m_on_resize = std::move(on_resize); }

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

private:
    void poll_gamepad_input();

    void handle_window_size(uint32_t width, uint32_t height);
    void handle_keyboard_event(const KeyboardEvent& event);
    void handle_mouse_event(const MouseEvent& event);
    void handle_gamepad_event(const GamepadEvent& event);
    void handle_drop_files(std::span<const char*> files);

    uint32_t m_width;
    uint32_t m_height;
    std::string m_title;
    GLFWwindow* m_window;

    bool m_should_close{false};

    CursorMode m_cursor_mode{CursorMode::normal};
    float2 m_mouse_pos{0.f, 0.f};
    KeyModifierFlags m_mods{KeyModifierFlags::none};

    static constexpr int INVALID_GAMEPAD_ID = -1;
    int m_gamepad_id{INVALID_GAMEPAD_ID};
    GamepadState m_gamepad_prev_state;

    ResizeCallback m_on_resize;
    KeyboardEventCallback m_on_keyboard_event;
    MouseEventCallback m_on_mouse_event;
    GamepadEventCallback m_on_gamepad_event;
    GamepadStateCallback m_on_gamepad_state;
    DropFilesCallback m_on_drop_files;

    friend struct EventHandlers;
};

} // namespace sgl
