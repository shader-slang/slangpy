// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "window.h"
#include "sgl/core/config.h"
#include "sgl/core/error.h"
#include "sgl/core/glfw.h"

#include <limits>
#include <memory>
#include <unordered_set>

#if SGL_HAS_GLFW

#if SGL_HAS_VULKAN
#define GLFW_INCLUDE_VULKAN
#endif
#include <GLFW/glfw3.h>

#if SGL_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#elif SGL_LINUX
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#define GLFW_EXPOSE_NATIVE_X11
// Xrandr defines some macros that conflict with our code, so undefine them.
#ifdef CursorShape
#undef CursorShape
#endif
#elif SGL_MACOS
#define GLFW_EXPOSE_NATIVE_COCOA
using CGDirectDisplayID = void*;
using id = void*;
#endif
#define GLFW_NATIVE_INCLUDE_NONE
#include <GLFW/glfw3native.h>

namespace sgl {

namespace {
    // Non-owning registry. All window operations run on the main thread.
    std::unordered_set<Window*> s_windows;

    inline KeyCode get_key_code(int glfw_key)
    {
        static_assert(GLFW_KEY_ESCAPE == 256);
        static_assert((uint32_t)KeyCode::escape >= 256);

        // Printable keys are expected to have the same value.
        if (glfw_key < GLFW_KEY_ESCAPE)
            return KeyCode(glfw_key);

        switch (glfw_key) {
        case GLFW_KEY_ESCAPE:
            return KeyCode::escape;
        case GLFW_KEY_ENTER:
            return KeyCode::enter;
        case GLFW_KEY_TAB:
            return KeyCode::tab;
        case GLFW_KEY_BACKSPACE:
            return KeyCode::backspace;
        case GLFW_KEY_INSERT:
            return KeyCode::insert;
        case GLFW_KEY_DELETE:
            return KeyCode::delete_;
        case GLFW_KEY_RIGHT:
            return KeyCode::right;
        case GLFW_KEY_LEFT:
            return KeyCode::left;
        case GLFW_KEY_DOWN:
            return KeyCode::down;
        case GLFW_KEY_UP:
            return KeyCode::up;
        case GLFW_KEY_PAGE_UP:
            return KeyCode::page_up;
        case GLFW_KEY_PAGE_DOWN:
            return KeyCode::page_down;
        case GLFW_KEY_HOME:
            return KeyCode::home;
        case GLFW_KEY_END:
            return KeyCode::end;
        case GLFW_KEY_CAPS_LOCK:
            return KeyCode::caps_lock;
        case GLFW_KEY_SCROLL_LOCK:
            return KeyCode::scroll_lock;
        case GLFW_KEY_NUM_LOCK:
            return KeyCode::num_lock;
        case GLFW_KEY_PRINT_SCREEN:
            return KeyCode::print_screen;
        case GLFW_KEY_PAUSE:
            return KeyCode::pause;
        case GLFW_KEY_F1:
            return KeyCode::f1;
        case GLFW_KEY_F2:
            return KeyCode::f2;
        case GLFW_KEY_F3:
            return KeyCode::f3;
        case GLFW_KEY_F4:
            return KeyCode::f4;
        case GLFW_KEY_F5:
            return KeyCode::f5;
        case GLFW_KEY_F6:
            return KeyCode::f6;
        case GLFW_KEY_F7:
            return KeyCode::f7;
        case GLFW_KEY_F8:
            return KeyCode::f8;
        case GLFW_KEY_F9:
            return KeyCode::f9;
        case GLFW_KEY_F10:
            return KeyCode::f10;
        case GLFW_KEY_F11:
            return KeyCode::f11;
        case GLFW_KEY_F12:
            return KeyCode::f12;
        case GLFW_KEY_KP_0:
            return KeyCode::keypad0;
        case GLFW_KEY_KP_1:
            return KeyCode::keypad1;
        case GLFW_KEY_KP_2:
            return KeyCode::keypad2;
        case GLFW_KEY_KP_3:
            return KeyCode::keypad3;
        case GLFW_KEY_KP_4:
            return KeyCode::keypad4;
        case GLFW_KEY_KP_5:
            return KeyCode::keypad5;
        case GLFW_KEY_KP_6:
            return KeyCode::keypad6;
        case GLFW_KEY_KP_7:
            return KeyCode::keypad7;
        case GLFW_KEY_KP_8:
            return KeyCode::keypad8;
        case GLFW_KEY_KP_9:
            return KeyCode::keypad9;
        case GLFW_KEY_KP_DECIMAL:
            return KeyCode::keypad_del;
        case GLFW_KEY_KP_DIVIDE:
            return KeyCode::keypad_divide;
        case GLFW_KEY_KP_MULTIPLY:
            return KeyCode::keypad_multiply;
        case GLFW_KEY_KP_SUBTRACT:
            return KeyCode::keypad_subtract;
        case GLFW_KEY_KP_ADD:
            return KeyCode::keypad_add;
        case GLFW_KEY_KP_ENTER:
            return KeyCode::keypad_enter;
        case GLFW_KEY_KP_EQUAL:
            return KeyCode::keypad_equal;
        case GLFW_KEY_LEFT_SHIFT:
            return KeyCode::left_shift;
        case GLFW_KEY_LEFT_CONTROL:
            return KeyCode::left_control;
        case GLFW_KEY_LEFT_ALT:
            return KeyCode::left_alt;
        case GLFW_KEY_LEFT_SUPER:
            return KeyCode::left_super;
        case GLFW_KEY_RIGHT_SHIFT:
            return KeyCode::right_shift;
        case GLFW_KEY_RIGHT_CONTROL:
            return KeyCode::right_control;
        case GLFW_KEY_RIGHT_ALT:
            return KeyCode::right_alt;
        case GLFW_KEY_RIGHT_SUPER:
            return KeyCode::right_super;
        case GLFW_KEY_MENU:
            return KeyCode::menu;
        default:
            return KeyCode::unknown;
        }
    }

    inline KeyModifierFlags get_key_modifier_flags(int mods)
    {
        KeyModifierFlags flags{KeyModifierFlags::none};
        if (mods & GLFW_MOD_ALT)
            flags |= KeyModifierFlags::alt;
        if (mods & GLFW_MOD_CONTROL)
            flags |= KeyModifierFlags::ctrl;
        if (mods & GLFW_MOD_SHIFT)
            flags |= KeyModifierFlags::shift;
        return flags;
    }

    /**
     * GLFW reports modifiers inconsistently on different platforms.
     * To make modifiers consistent we check the key action and adjust
     * the modifiers due to changes from the current action.
     */
    inline int fix_glfw_modifiers(int modifiers, int key, int action)
    {
        int bit = 0;
        if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
            bit = GLFW_MOD_SHIFT;
        if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL)
            bit = GLFW_MOD_CONTROL;
        if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT)
            bit = GLFW_MOD_ALT;
        return (action == GLFW_RELEASE) ? modifiers & (~bit) : modifiers | bit;
    }

} // namespace

struct EventHandlers {
    static void handle_window_pos(GLFWwindow* glfw_window, int, int)
    {
        auto window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));
        window->update_normal_placement();
    }

    static void handle_window_maximize(GLFWwindow* glfw_window, int maximized)
    {
        auto window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));
        if (!window->is_minimized())
            window->m_placement.maximized = maximized == GLFW_TRUE;
    }

    static void handle_error(int error_code, const char* description)
    {
        log_error("GLFW error {}: {}", error_code, description);
    }

    static void handle_window_size(GLFWwindow* glfw_window, int width, int height)
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        window->handle_window_size(width, height);
    }

    static void handle_window_refresh(GLFWwindow* glfw_window)
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        window->handle_window_refresh();
    }

    static void handle_key(GLFWwindow* glfw_window, int key, int scancode, int action, int mods)
    {
        SGL_UNUSED(scancode);

        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        if (key == GLFW_KEY_UNKNOWN)
            return;

        mods = fix_glfw_modifiers(mods, key, action);
        window->m_mods = get_key_modifier_flags(mods);

        KeyboardEventType type;
        switch (action) {
        case GLFW_RELEASE:
            type = KeyboardEventType::key_release;
            break;
        case GLFW_PRESS:
            type = KeyboardEventType::key_press;
            break;
        case GLFW_REPEAT:
            type = KeyboardEventType::key_repeat;
            break;
        default:
            return;
        }

        KeyboardEvent event{
            .type = type,
            .key = get_key_code(key),
            .mods = window->m_mods,
        };

        window->handle_keyboard_event(event);
    }

    static void handle_char(GLFWwindow* glfw_window, unsigned int codepoint)
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        KeyboardEvent event{
            .type = KeyboardEventType::input,
            .codepoint = codepoint,
            .mods = window->m_mods,
        };

        window->handle_keyboard_event(event);
    }

    static void handle_mouse_button(GLFWwindow* glfw_window, int button, int action, int mods)
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        window->m_mods = get_key_modifier_flags(mods);

        MouseButton mouse_button;
        switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            mouse_button = MouseButton::left;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            mouse_button = MouseButton::middle;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            mouse_button = MouseButton::right;
            break;
        default:
            // Other keys are not supported
            return;
        }

        MouseEvent event{
            .type = (action == GLFW_PRESS) ? MouseEventType::button_down : MouseEventType::button_up,
            .pos = window->m_mouse_pos,
            .button = mouse_button,
            .mods = window->m_mods,
        };

        window->handle_mouse_event(event);
    }

    static void handle_cursor_pos(GLFWwindow* glfw_window, double xpos, double ypos)
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        window->m_mouse_pos = float2(xpos, ypos);

        MouseEvent event{
            .type = MouseEventType::move,
            .pos = window->m_mouse_pos,
            .mods = window->m_mods,
        };

        window->handle_mouse_event(event);
    }

    static void handle_scroll(GLFWwindow* glfw_window, double xoffset, double yoffset)
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        MouseEvent event{
            .type = MouseEventType::scroll,
            .pos = window->m_mouse_pos,
            .scroll = float2(xoffset, yoffset),
            .mods = window->m_mods,
        };

        window->handle_mouse_event(event);
    }

    static void handle_drop(GLFWwindow* glfw_window, int path_count, const char* paths[])
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        window->handle_drop_files({paths, paths + path_count});
    }
};


Window::Window(WindowDesc desc)
    : m_title(desc.title)
{
    if (desc.placement) {
        SGL_CHECK(desc.mode != WindowMode::fullscreen, "Cannot combine window placement with fullscreen mode");
        desc.width = desc.placement->normal_size.x;
        desc.height = desc.placement->normal_size.y;
        desc.position = desc.placement->normal_position;
        desc.mode = desc.placement->maximized ? WindowMode::maximized : WindowMode::normal;
    }
    SGL_CHECK(
        desc.width > 0 && desc.height > 0 && desc.width <= uint32_t(std::numeric_limits<int>::max())
            && desc.height <= uint32_t(std::numeric_limits<int>::max()),
        "Window dimensions must be positive and fit in a signed integer"
    );
    m_width = desc.width;
    m_height = desc.height;
    detail::ensure_glfw_initialized();
    m_fullscreen = desc.mode == WindowMode::fullscreen;
    m_placement.normal_size = {desc.width, desc.height};

    // GLFW window hints persist across window creations.
    glfwDefaultWindowHints();
    // Establish the normal geometry before maximizing or showing the window.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    if (desc.mode == WindowMode::fullscreen) {
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        auto mon = glfwGetPrimaryMonitor();
        auto mod = glfwGetVideoMode(mon);
        m_width = mod->width;
        m_height = mod->height;
    } else if (desc.mode == WindowMode::minimized) {
        // Avoid focusing a window that will immediately be minimized.
        glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
    }

    if (desc.resizable == false) {
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), NULL, NULL);
    if (!m_window)
        SGL_THROW("Failed to create GLFW window");
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> guard(m_window, &glfwDestroyWindow);

    if (desc.position)
        glfwSetWindowPos(m_window, desc.position->x, desc.position->y);
    m_placement.normal_position = position();

    glfwSetWindowUserPointer(m_window, this);
    glfwSetErrorCallback(&EventHandlers::handle_error);
    glfwSetWindowSizeCallback(m_window, &EventHandlers::handle_window_size);
    glfwSetWindowPosCallback(m_window, &EventHandlers::handle_window_pos);
    glfwSetWindowMaximizeCallback(m_window, &EventHandlers::handle_window_maximize);
    glfwSetWindowRefreshCallback(m_window, &EventHandlers::handle_window_refresh);
    glfwSetKeyCallback(m_window, &EventHandlers::handle_key);
    glfwSetCharCallback(m_window, &EventHandlers::handle_char);
    glfwSetMouseButtonCallback(m_window, &EventHandlers::handle_mouse_button);
    glfwSetCursorPosCallback(m_window, &EventHandlers::handle_cursor_pos);
    glfwSetScrollCallback(m_window, &EventHandlers::handle_scroll);
    glfwSetDropCallback(m_window, &EventHandlers::handle_drop);

    update_normal_placement();
    if (desc.visible) {
        show();
        if (desc.mode == WindowMode::maximized)
            maximize();
        else if (desc.mode == WindowMode::minimized)
            minimize();
    }

    // The window system may adjust the requested size during creation
    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    m_width = static_cast<uint32_t>(width);
    m_height = static_cast<uint32_t>(height);
    s_windows.insert(this);
    guard.release();
}

Window::~Window()
{
    s_windows.erase(this);
    destroy_window();
}

void Window::set_title(std::string title)
{
    m_title = std::move(title);
    glfwSetWindowTitle(glfw_window(), m_title.c_str());
}

WindowHandle Window::window_handle() const
{
    GLFWwindow* window = glfw_window();
    WindowHandle handle{};
#if SGL_WINDOWS
    handle.hwnd = glfwGetWin32Window(window);
#elif SGL_LINUX
    handle.xdisplay = glfwGetX11Display();
    handle.xwindow = glfwGetX11Window(window);
#elif SGL_MACOS
    handle.nswindow = glfwGetCocoaWindow(window);
#endif
    return handle;
}

void Window::set_icon(const std::filesystem::path& path)
{
    SGL_UNUSED(path);
    SGL_UNIMPLEMENTED();
}

int2 Window::position() const
{
    int xpos, ypos;
    glfwGetWindowPos(glfw_window(), &xpos, &ypos);
    return int2{xpos, ypos};
}

void Window::set_position(int2 position)
{
    glfwSetWindowPos(glfw_window(), position.x, position.y);
    update_normal_placement();
}

void Window::set_width(uint32_t width)
{
    resize(width, m_height);
}

void Window::set_height(uint32_t height)
{
    resize(m_width, height);
}

void Window::set_size(uint2 size)
{
    resize(size.x, size.y);
}

void Window::resize(uint32_t width, uint32_t height)
{
    SGL_CHECK(
        width > 0 && height > 0 && width <= uint32_t(std::numeric_limits<int>::max())
            && height <= uint32_t(std::numeric_limits<int>::max()),
        "Window dimensions must be positive and fit in a signed integer"
    );
    glfwSetWindowSize(glfw_window(), width, height);

    // The window system may limit the requested size, so query the actual size after resizing
    int w, h;
    glfwGetWindowSize(glfw_window(), &w, &h);
    m_width = static_cast<uint32_t>(w);
    m_height = static_cast<uint32_t>(h);
    update_normal_placement();
}

int4 Window::frame_size() const
{
    int4 frame;
    glfwGetWindowFrameSize(glfw_window(), &frame.x, &frame.y, &frame.z, &frame.w);
    return frame;
}

float2 Window::content_scale() const
{
    float2 scale;
    glfwGetWindowContentScale(glfw_window(), &scale.x, &scale.y);
    return scale;
}

WindowPlacement Window::placement() const
{
    WindowPlacement result = m_placement;
    if (m_fullscreen)
        return result;
    if (!is_minimized())
        result.maximized = is_maximized();
    if (!is_minimized() && !is_maximized()) {
        result.normal_position = position();
        int width, height;
        glfwGetWindowSize(glfw_window(), &width, &height);
        result.normal_size = {uint32_t(width), uint32_t(height)};
    }
    return result;
}

bool Window::is_minimized() const
{
    return glfwGetWindowAttrib(glfw_window(), GLFW_ICONIFIED) == GLFW_TRUE;
}

bool Window::is_maximized() const
{
    return glfwGetWindowAttrib(glfw_window(), GLFW_MAXIMIZED) == GLFW_TRUE;
}

bool Window::is_visible() const
{
    return glfwGetWindowAttrib(glfw_window(), GLFW_VISIBLE) == GLFW_TRUE;
}

void Window::minimize()
{
    update_normal_placement();
    if (!is_minimized())
        m_placement.maximized = is_maximized();
    glfwIconifyWindow(glfw_window());
}

void Window::maximize()
{
    SGL_CHECK(!m_fullscreen, "Cannot maximize a fullscreen window");
    update_normal_placement();
    glfwMaximizeWindow(glfw_window());
}

void Window::restore()
{
    SGL_CHECK(!m_fullscreen, "Cannot restore a fullscreen window");
    glfwRestoreWindow(glfw_window());
}

void Window::show()
{
    glfwShowWindow(glfw_window());
}

void Window::hide()
{
    glfwHideWindow(glfw_window());
}

void Window::close()
{
    m_should_close = true;
}

bool Window::should_close() const
{
    return m_should_close || glfwWindowShouldClose(glfw_window());
}

void Window::process_events()
{
    glfw_window();
    glfwPollEvents();
    update_normal_placement();
    poll_gamepad_input();
}

void Window::set_cursor_mode(CursorMode mode)
{
    if (mode != m_cursor_mode) {
        switch (mode) {
        case CursorMode::normal:
            glfwSetInputMode(glfw_window(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            break;
        case CursorMode::hidden:
            glfwSetInputMode(glfw_window(), GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            break;
        case CursorMode::disabled:
            glfwSetInputMode(glfw_window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            break;
        default:
            SGL_THROW("Invalid cursor mode: {}.", static_cast<uint32_t>(mode));
        }
        m_cursor_mode = mode;
    }
}

void Window::set_cursor_shape(CursorShape shape)
{
    glfw_window();
    if (shape != m_cursor_shape) {
        uint32_t index = static_cast<uint32_t>(shape);
        SGL_CHECK(index < m_cursor_cache.size(), "Invalid cursor shape: {}.", index);
        if (!m_cursor_cache[index]) {
            int glfw_shape = 0;
            switch (shape) {
            case CursorShape::arrow:
                glfw_shape = GLFW_ARROW_CURSOR;
                break;
            case CursorShape::ibeam:
                glfw_shape = GLFW_IBEAM_CURSOR;
                break;
            case CursorShape::crosshair:
                glfw_shape = GLFW_CROSSHAIR_CURSOR;
                break;
            case CursorShape::hand:
                glfw_shape = GLFW_HAND_CURSOR;
                break;
            case CursorShape::hresize:
                glfw_shape = GLFW_HRESIZE_CURSOR;
                break;
            case CursorShape::vresize:
                glfw_shape = GLFW_VRESIZE_CURSOR;
                break;
            default:
                SGL_THROW("Invalid cursor shape: {}.", index);
            }
            m_cursor_cache[index] = glfwCreateStandardCursor(glfw_shape);
        }
        glfwSetCursor(glfw_window(), m_cursor_cache[index]);
        m_cursor_shape = shape;
    }
}

std::optional<std::string> Window::get_clipboard() const
{
    const char* text = glfwGetClipboardString(glfw_window());
    return text ? std::optional<std::string>(text) : std::nullopt;
}

void Window::set_clipboard(const std::string& text)
{
    glfwSetClipboardString(glfw_window(), text.c_str());
}

std::string Window::to_string() const
{
    return fmt::format(
        "Window(\n"
        "  width = {},\n"
        "  height = {},\n"
        "  title = \"{}\"\n"
        ")",
        m_width,
        m_height,
        m_title
    );
}

GLFWwindow* Window::glfw_window() const
{
    SGL_CHECK(m_window, "Window has been destroyed during shutdown");
    return m_window;
}

void Window::destroy_window()
{
    for (GLFWcursor*& cursor : m_cursor_cache) {
        if (cursor) {
            glfwDestroyCursor(cursor);
            cursor = nullptr;
        }
    }

    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    m_should_close = true;
}

void Window::_destroy_all_windows()
{
    for (Window* window : s_windows)
        window->destroy_window();
}

void Window::update_normal_placement()
{
    if (m_fullscreen || is_minimized() || is_maximized())
        return;
    int width, height;
    glfwGetWindowSize(glfw_window(), &width, &height);
    if (width > 0 && height > 0) {
        m_placement.normal_position = position();
        m_placement.normal_size = {uint32_t(width), uint32_t(height)};
    }
}

void Window::poll_gamepad_input()
{
    // Check if a gamepad is connected.
    if (m_gamepad_id == INVALID_GAMEPAD_ID) {
        for (int id = GLFW_JOYSTICK_1; id <= GLFW_JOYSTICK_LAST; ++id) {
            if (glfwJoystickPresent(id) && glfwJoystickIsGamepad(id)) {
                std::string name(glfwGetJoystickName(id));
                log_debug("Gamepad \"{}\" connected.", name);
                m_gamepad_id = id;
                m_gamepad_prev_state = {};

                GamepadEvent event{.type = GamepadEventType::connect};
                handle_gamepad_event(event);
                break;
            }
        }
    }

    // Check if gamepad is disconnected.
    if (m_gamepad_id != INVALID_GAMEPAD_ID) {
        if (!(glfwJoystickPresent(m_gamepad_id) && glfwJoystickIsGamepad(m_gamepad_id))) {
            log_debug("Gamepad disconnected.");
            m_gamepad_id = INVALID_GAMEPAD_ID;

            GamepadEvent event{.type = GamepadEventType::disconnect};
            handle_gamepad_event(event);
        }
    }

    if (m_gamepad_id == INVALID_GAMEPAD_ID)
        return;

    GLFWgamepadstate glfw_state;
    if (glfwGetGamepadState(m_gamepad_id, &glfw_state) != GLFW_TRUE)
        return;

    GamepadState state{};
    state.left_x = glfw_state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    state.left_y = glfw_state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
    state.right_x = glfw_state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    state.right_y = glfw_state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
    state.left_trigger = glfw_state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
    state.right_trigger = glfw_state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];

    static constexpr int GLFW_BUTTONS[]{
        0,
        GLFW_GAMEPAD_BUTTON_A,
        GLFW_GAMEPAD_BUTTON_B,
        GLFW_GAMEPAD_BUTTON_X,
        GLFW_GAMEPAD_BUTTON_Y,
        GLFW_GAMEPAD_BUTTON_LEFT_BUMPER,
        GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER,
        GLFW_GAMEPAD_BUTTON_BACK,
        GLFW_GAMEPAD_BUTTON_START,
        GLFW_GAMEPAD_BUTTON_GUIDE,
        GLFW_GAMEPAD_BUTTON_LEFT_THUMB,
        GLFW_GAMEPAD_BUTTON_RIGHT_THUMB,
        GLFW_GAMEPAD_BUTTON_DPAD_UP,
        GLFW_GAMEPAD_BUTTON_DPAD_RIGHT,
        GLFW_GAMEPAD_BUTTON_DPAD_DOWN,
        GLFW_GAMEPAD_BUTTON_DPAD_LEFT,
    };

    // Get button state.
    for (size_t i = 1; i < std::size(GLFW_BUTTONS); ++i)
        state.buttons |= (glfw_state.buttons[GLFW_BUTTONS[i]] == GLFW_PRESS) << i;

    // Synthesize button events.
    for (size_t i = 1; i < std::size(GLFW_BUTTONS); ++i) {
        if (state.buttons & (1 << i)) {
            if (!(m_gamepad_prev_state.buttons & (1 << i))) {
                GamepadEvent event{.type = GamepadEventType::button_down, .button = (GamepadButton)i};
                handle_gamepad_event(event);
            }
        } else {
            if (m_gamepad_prev_state.buttons & (1 << i)) {
                GamepadEvent event{.type = GamepadEventType::button_up, .button = (GamepadButton)i};
                handle_gamepad_event(event);
            }
        }
    }

    m_gamepad_prev_state = state;

    if (m_on_gamepad_state)
        m_on_gamepad_state(state);
}

void Window::handle_window_size(uint32_t width, uint32_t height)
{
    m_width = width;
    m_height = height;
    update_normal_placement();
    if (m_on_resize)
        m_on_resize(m_width, m_height);
}

void Window::handle_window_refresh()
{
    if (m_on_refresh)
        m_on_refresh();
}

void Window::handle_keyboard_event(const KeyboardEvent& event)
{
    if (m_on_keyboard_event)
        m_on_keyboard_event(event);
}

void Window::handle_mouse_event(const MouseEvent& event)
{
    if (m_on_mouse_event)
        m_on_mouse_event(event);
}

void Window::handle_gamepad_event(const GamepadEvent& event)
{
    if (m_on_gamepad_event)
        m_on_gamepad_event(event);
}

void Window::handle_drop_files(std::span<const char*> files)
{
    if (m_on_drop_files)
        m_on_drop_files(files);
}

} // namespace sgl

#else // SGL_HAS_GLFW

namespace sgl {

namespace {
    [[noreturn]] void window_support_disabled()
    {
        SGL_THROW("sgl was built without GLFW window support");
    }
} // namespace

// Keep Window and its bindings available in builds without GLFW. No Window can
// be constructed, but the other entry points must still exist for linking.
Window::Window(WindowDesc)
    : m_width(0)
    , m_height(0)
    , m_window(nullptr)
{
    window_support_disabled();
}

Window::~Window() = default;

void Window::_destroy_all_windows() { }

void Window::set_title(std::string)
{
    window_support_disabled();
}

WindowHandle Window::window_handle() const
{
    window_support_disabled();
}

void Window::set_icon(const std::filesystem::path&)
{
    window_support_disabled();
}

int2 Window::position() const
{
    window_support_disabled();
}

void Window::set_position(int2)
{
    window_support_disabled();
}

void Window::set_width(uint32_t)
{
    window_support_disabled();
}

void Window::set_height(uint32_t)
{
    window_support_disabled();
}

void Window::set_size(uint2)
{
    window_support_disabled();
}

void Window::resize(uint32_t, uint32_t)
{
    window_support_disabled();
}

int4 Window::frame_size() const
{
    window_support_disabled();
}

float2 Window::content_scale() const
{
    window_support_disabled();
}

WindowPlacement Window::placement() const
{
    window_support_disabled();
}

bool Window::is_minimized() const
{
    window_support_disabled();
}

bool Window::is_maximized() const
{
    window_support_disabled();
}

bool Window::is_visible() const
{
    window_support_disabled();
}

void Window::minimize()
{
    window_support_disabled();
}

void Window::maximize()
{
    window_support_disabled();
}

void Window::restore()
{
    window_support_disabled();
}

void Window::show()
{
    window_support_disabled();
}

void Window::hide()
{
    window_support_disabled();
}

void Window::close()
{
    window_support_disabled();
}

bool Window::should_close() const
{
    window_support_disabled();
}

void Window::process_events()
{
    window_support_disabled();
}

void Window::set_cursor_mode(CursorMode)
{
    window_support_disabled();
}

void Window::set_cursor_shape(CursorShape)
{
    window_support_disabled();
}

std::optional<std::string> Window::get_clipboard() const
{
    window_support_disabled();
}

void Window::set_clipboard(const std::string&)
{
    window_support_disabled();
}

std::string Window::to_string() const
{
    window_support_disabled();
}

} // namespace sgl

#endif // SGL_HAS_GLFW
