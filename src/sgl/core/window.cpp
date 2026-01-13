// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "window.h"
#include "sgl/core/config.h"
#include "sgl/core/error.h"

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
#elif SGL_MACOS
#define GLFW_EXPOSE_NATIVE_COCOA
using CGDirectDisplayID = void*;
using id = void*;
#endif
#define GLFW_NATIVE_INCLUDE_NONE
#include <GLFW/glfw3native.h>

#include <atomic>

namespace sgl {

namespace {
    std::atomic<uint32_t> glfw_ref_count{0};

    inline void init_glfw()
    {
        if (glfw_ref_count.fetch_add(1) == 0) {
            if (glfwInit() != GLFW_TRUE)
                SGL_THROW("Failed to initialize GLFW");

            // Register mappings for NV controllers.
            // clang-format off
            static char nvPadMapping[] =
                "03000000550900001472000000000000,NVIDIA Controller v01.04,a:b11,b:b10,back:b13,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b12,leftshoulder:b7,leftstick:b5,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b6,rightstick:b4,righttrigger:a5,rightx:a3,righty:a6,start:b3,x:b9,y:b8,platform:Windows,\n"
                "03000000550900001072000000000000,NVIDIA Shield,a:b9,b:b8,back:b11,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b5,leftstick:b3,lefttrigger:a3,leftx:a0,lefty:a1,rightshoulder:b4,rightstick:b2,righttrigger:a4,rightx:a2,righty:a5,start:b0,x:b7,y:b6,platform:Windows,\n"
                "030000005509000000b4000000000000,NVIDIA Virtual Gamepad,a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b4,leftstick:b8,lefttrigger:+a2,leftx:a0,lefty:a1,rightshoulder:b5,rightstick:b9,righttrigger:-a2,rightx:a3,righty:a4,start:b7,x:b2,y:b3,platform:Windows,";
            // clang-format on
            glfwUpdateGamepadMappings(nvPadMapping);
        }
    }

    inline void terminate_glfw()
    {
        SGL_ASSERT(glfw_ref_count > 0);
        if (glfw_ref_count.fetch_sub(1) == 1)
            glfwTerminate();
    }

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
    static void handle_error(int error_code, const char* description)
    {
        log_error("GLFW error {}: {}", error_code, description);
    }

    static void handle_window_size(GLFWwindow* glfw_window, int width, int height)
    {
        Window* window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));

        window->handle_window_size(width, height);
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
    : m_width(desc.width)
    , m_height(desc.height)
    , m_title(desc.title)
{
    init_glfw();

    if (desc.mode == WindowMode::fullscreen) {
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        auto mon = glfwGetPrimaryMonitor();
        auto mod = glfwGetVideoMode(mon);
        m_width = mod->width;
        m_height = mod->height;
    } else if (desc.mode == WindowMode::minimized) {
        // Start with window being invisible
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
        glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    }

    if (desc.resizable == false) {
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), NULL, NULL);
    if (!m_window)
        SGL_THROW("Failed to create GLFW window");

    glfwSetWindowUserPointer(m_window, this);
    glfwSetErrorCallback(&EventHandlers::handle_error);
    glfwSetWindowSizeCallback(m_window, &EventHandlers::handle_window_size);
    glfwSetKeyCallback(m_window, &EventHandlers::handle_key);
    glfwSetCharCallback(m_window, &EventHandlers::handle_char);
    glfwSetMouseButtonCallback(m_window, &EventHandlers::handle_mouse_button);
    glfwSetCursorPosCallback(m_window, &EventHandlers::handle_cursor_pos);
    glfwSetScrollCallback(m_window, &EventHandlers::handle_scroll);
    glfwSetDropCallback(m_window, &EventHandlers::handle_drop);
}

Window::~Window()
{
    glfwDestroyWindow(m_window);

    terminate_glfw();
}

WindowHandle Window::window_handle() const
{
    WindowHandle handle{};
#if SGL_WINDOWS
    handle.hwnd = glfwGetWin32Window(m_window);
#elif SGL_LINUX
    handle.xdisplay = glfwGetX11Display();
    handle.xwindow = glfwGetX11Window(m_window);
#elif SGL_MACOS
    handle.nswindow = glfwGetCocoaWindow(m_window);
#endif
    return handle;
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
    m_width = width;
    m_height = height;
    glfwSetWindowSize(m_window, m_width, m_height);
}

int2 Window::position() const
{
    int xpos, ypos;
    glfwGetWindowPos(m_window, &xpos, &ypos);
    return int2{xpos, ypos};
}

void Window::set_position(int2 position)
{
    glfwSetWindowPos(m_window, position.x, position.y);
}

void Window::set_title(std::string title)
{
    m_title = std::move(title);
    glfwSetWindowTitle(m_window, m_title.c_str());
}

void Window::set_icon(const std::filesystem::path& path)
{
    SGL_UNUSED(path);
    SGL_UNIMPLEMENTED();
}

void Window::close()
{
    m_should_close = true;
}

bool Window::should_close() const
{
    return m_should_close || glfwWindowShouldClose(m_window);
}

void Window::process_events()
{
    glfwPollEvents();
    poll_gamepad_input();
}

void Window::set_clipboard(const std::string& text)
{
    glfwSetClipboardString(m_window, text.c_str());
}

std::optional<std::string> Window::get_clipboard() const
{
    const char* text = glfwGetClipboardString(m_window);
    return text ? std::optional<std::string>(text) : std::nullopt;
}

void Window::set_cursor_mode(CursorMode mode)
{
    if (mode != m_cursor_mode) {
        m_cursor_mode = mode;
        switch (mode) {
        case CursorMode::normal:
            glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            break;
        case CursorMode::hidden:
            glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            break;
        case CursorMode::disabled:
            glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            break;
        default:
            SGL_UNREACHABLE();
        }
    }
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
    for (size_t i = 0; i < std::size(GLFW_BUTTONS); ++i)
        state.buttons |= (glfw_state.buttons[GLFW_BUTTONS[i]] == GLFW_PRESS) << i;

    // Synthesize button events.
    for (size_t i = 0; i < std::size(GLFW_BUTTONS); ++i) {
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
    if (m_on_resize)
        m_on_resize(m_width, m_height);
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
