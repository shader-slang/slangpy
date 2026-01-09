// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/enum.h"
#include "sgl/math/vector_types.h"

#include <string>

namespace sgl {

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

/// Mouse buttons.
enum class MouseButton : uint32_t {
    left,
    middle,
    right,
    unknown,
};

SGL_ENUM_INFO(
    MouseButton,
    {
        {MouseButton::left, "left"},
        {MouseButton::middle, "middle"},
        {MouseButton::right, "right"},
        {MouseButton::unknown, "unknown"},
    }
);
SGL_ENUM_REGISTER(MouseButton);

/// Keyboard modifier flags.
enum class KeyModifierFlags : uint32_t {
    none = 0,
    shift = 1,
    ctrl = 2,
    alt = 4,
};

SGL_ENUM_CLASS_OPERATORS(KeyModifierFlags);
SGL_ENUM_INFO(
    KeyModifierFlags,
    {
        {KeyModifierFlags::none, "none"},
        {KeyModifierFlags::shift, "shift"},
        {KeyModifierFlags::ctrl, "ctrl"},
        {KeyModifierFlags::alt, "alt"},
    }
);
SGL_ENUM_REGISTER(KeyModifierFlags);

/// Keyboard modifiers.
enum class KeyModifier : uint32_t {
    shift = (uint32_t)KeyModifierFlags::shift,
    ctrl = (uint32_t)KeyModifierFlags::ctrl,
    alt = (uint32_t)KeyModifierFlags::alt
};

SGL_ENUM_INFO(
    KeyModifier,
    {
        {KeyModifier::shift, "shift"},
        {KeyModifier::ctrl, "ctrl"},
        {KeyModifier::alt, "alt"},
    }
);
SGL_ENUM_REGISTER(KeyModifier);

/// Keyboard key codes.
enum class KeyCode : uint32_t {
    // Key codes 0..255 are reserved for ASCII codes.
    space = ' ',
    apostrophe = '\'',
    comma = ',',
    minus = '-',
    period = '.',
    slash = '/',
    key0 = '0',
    key1 = '1',
    key2 = '2',
    key3 = '3',
    key4 = '4',
    key5 = '5',
    key6 = '6',
    key7 = '7',
    key8 = '8',
    key9 = '9',
    semicolon = ';',
    equal = '=',
    a = 'A',
    b = 'B',
    c = 'C',
    d = 'D',
    e = 'E',
    f = 'F',
    g = 'G',
    h = 'H',
    i = 'I',
    j = 'J',
    k = 'K',
    l = 'L',
    m = 'M',
    n = 'N',
    o = 'O',
    p = 'P',
    q = 'Q',
    r = 'R',
    s = 'S',
    t = 'T',
    u = 'U',
    v = 'V',
    w = 'W',
    x = 'X',
    y = 'Y',
    z = 'Z',
    left_bracket = '[',
    backslash = '\\',
    right_bracket = ']',
    grave_accent = '`',
    // Non-US keys (GLFW_KEY_WORLD_1/2). These are layout-dependent and occupy
    // values in the printable range.
    world_1 = 161,
    world_2 = 162,

    // Special keys start at key code 256.
    escape = 256,
    tab,
    enter,
    backspace,
    insert,
    delete_,
    right,
    left,
    down,
    up,
    page_up,
    page_down,
    home,
    end,
    caps_lock,
    scroll_lock,
    num_lock,
    print_screen,
    pause,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    keypad0,
    keypad1,
    keypad2,
    keypad3,
    keypad4,
    keypad5,
    keypad6,
    keypad7,
    keypad8,
    keypad9,
    keypad_del,
    keypad_divide,
    keypad_multiply,
    keypad_subtract,
    keypad_add,
    keypad_enter,
    keypad_equal,
    left_shift,
    left_control,
    left_alt,
    left_super,
    right_shift,
    right_control,
    right_alt,
    right_super,
    menu,
    unknown,
};

SGL_ENUM_INFO(
    KeyCode,
    {
        {KeyCode::space, "space"},
        {KeyCode::apostrophe, "apostrophe"},
        {KeyCode::comma, "comma"},
        {KeyCode::minus, "minus"},
        {KeyCode::period, "period"},
        {KeyCode::slash, "slash"},
        {KeyCode::key0, "key0"},
        {KeyCode::key1, "key1"},
        {KeyCode::key2, "key2"},
        {KeyCode::key3, "key3"},
        {KeyCode::key4, "key4"},
        {KeyCode::key5, "key5"},
        {KeyCode::key6, "key6"},
        {KeyCode::key7, "key7"},
        {KeyCode::key8, "key8"},
        {KeyCode::key9, "key9"},
        {KeyCode::semicolon, "semicolon"},
        {KeyCode::equal, "equal"},
        {KeyCode::a, "a"},
        {KeyCode::b, "b"},
        {KeyCode::c, "c"},
        {KeyCode::d, "d"},
        {KeyCode::e, "e"},
        {KeyCode::f, "f"},
        {KeyCode::g, "g"},
        {KeyCode::h, "h"},
        {KeyCode::i, "i"},
        {KeyCode::j, "j"},
        {KeyCode::k, "k"},
        {KeyCode::l, "l"},
        {KeyCode::m, "m"},
        {KeyCode::n, "n"},
        {KeyCode::o, "o"},
        {KeyCode::p, "p"},
        {KeyCode::q, "q"},
        {KeyCode::r, "r"},
        {KeyCode::s, "s"},
        {KeyCode::t, "t"},
        {KeyCode::u, "u"},
        {KeyCode::v, "v"},
        {KeyCode::w, "w"},
        {KeyCode::x, "x"},
        {KeyCode::y, "y"},
        {KeyCode::z, "z"},
        {KeyCode::left_bracket, "left_bracket"},
        {KeyCode::backslash, "backslash"},
        {KeyCode::right_bracket, "right_bracket"},
        {KeyCode::grave_accent, "grave_accent"},
        {KeyCode::world_1, "world_1"},
        {KeyCode::world_2, "world_2"},
        {KeyCode::escape, "escape"},
        {KeyCode::tab, "tab"},
        {KeyCode::enter, "enter"},
        {KeyCode::backspace, "backspace"},
        {KeyCode::insert, "insert"},
        {KeyCode::delete_, "delete"},
        {KeyCode::right, "right"},
        {KeyCode::left, "left"},
        {KeyCode::down, "down"},
        {KeyCode::up, "up"},
        {KeyCode::page_up, "page_up"},
        {KeyCode::page_down, "page_down"},
        {KeyCode::home, "home"},
        {KeyCode::end, "end"},
        {KeyCode::caps_lock, "caps_lock"},
        {KeyCode::scroll_lock, "scroll_lock"},
        {KeyCode::num_lock, "num_lock"},
        {KeyCode::print_screen, "print_screen"},
        {KeyCode::pause, "pause"},
        {KeyCode::f1, "f1"},
        {KeyCode::f2, "f2"},
        {KeyCode::f3, "f3"},
        {KeyCode::f4, "f4"},
        {KeyCode::f5, "f5"},
        {KeyCode::f6, "f6"},
        {KeyCode::f7, "f7"},
        {KeyCode::f8, "f8"},
        {KeyCode::f9, "f9"},
        {KeyCode::f10, "f10"},
        {KeyCode::f11, "f11"},
        {KeyCode::f12, "f12"},
        {KeyCode::keypad0, "keypad0"},
        {KeyCode::keypad1, "keypad1"},
        {KeyCode::keypad2, "keypad2"},
        {KeyCode::keypad3, "keypad3"},
        {KeyCode::keypad4, "keypad4"},
        {KeyCode::keypad5, "keypad5"},
        {KeyCode::keypad6, "keypad6"},
        {KeyCode::keypad7, "keypad7"},
        {KeyCode::keypad8, "keypad8"},
        {KeyCode::keypad9, "keypad9"},
        {KeyCode::keypad_del, "keypad_del"},
        {KeyCode::keypad_divide, "keypad_divide"},
        {KeyCode::keypad_multiply, "keypad_multiply"},
        {KeyCode::keypad_subtract, "keypad_subtract"},
        {KeyCode::keypad_add, "keypad_add"},
        {KeyCode::keypad_enter, "keypad_enter"},
        {KeyCode::keypad_equal, "keypad_equal"},
        {KeyCode::left_shift, "left_shift"},
        {KeyCode::left_control, "left_control"},
        {KeyCode::left_alt, "left_alt"},
        {KeyCode::left_super, "left_super"},
        {KeyCode::right_shift, "right_shift"},
        {KeyCode::right_control, "right_control"},
        {KeyCode::right_alt, "right_alt"},
        {KeyCode::right_super, "right_super"},
        {KeyCode::menu, "menu"},
        {KeyCode::unknown, "unknown"},
    }
);
SGL_ENUM_REGISTER(KeyCode);

/// Keyboard event types.
enum class KeyboardEventType {
    /// Key was pressed.
    key_press,
    /// Key was released.
    key_release,
    /// Key is repeatedly down.
    key_repeat,
    /// Character input.
    input
};

SGL_ENUM_INFO(
    KeyboardEventType,
    {
        {KeyboardEventType::key_press, "key_press"},
        {KeyboardEventType::key_release, "key_release"},
        {KeyboardEventType::key_repeat, "key_repeat"},
        {KeyboardEventType::input, "input"},
    }
);
SGL_ENUM_REGISTER(KeyboardEventType);

/// Keyboard event.
struct SGL_API KeyboardEvent {
    /// The event type.
    KeyboardEventType type;
    /// The key that was pressed/released/repeated.
    KeyCode key{KeyCode::unknown};
    /// UTF-32 codepoint for input events.
    uint32_t codepoint{0};
    /// Keyboard modifier flags.
    KeyModifierFlags mods{0};

    /// Returns true if this event is a key press event.
    bool is_key_press() const { return type == KeyboardEventType::key_press; }
    /// Returns true if this event is a key release event.
    bool is_key_release() const { return type == KeyboardEventType::key_release; }
    /// Returns true if this event is a key repeat event.
    bool is_key_repeat() const { return type == KeyboardEventType::key_repeat; }
    /// Returns true if this event is an input event.
    bool is_input() const { return type == KeyboardEventType::input; }

    /// Returns true if the specified modifier is set.
    bool has_modifier(KeyModifier mod) const { return is_set(mods, (KeyModifierFlags)mod); }

    std::string to_string() const;
};

/// Mouse event types.
enum class MouseEventType {
    /// Mouse button was pressed.
    button_down,
    /// Mouse button was released.
    button_up,
    /// Mouse cursor position changed.
    move,
    /// Mouse wheel was scrolled.
    scroll
};

SGL_ENUM_INFO(
    MouseEventType,
    {
        {MouseEventType::button_down, "button_down"},
        {MouseEventType::button_up, "button_up"},
        {MouseEventType::move, "move"},
        {MouseEventType::scroll, "scroll"},
    }
);
SGL_ENUM_REGISTER(MouseEventType);

/// Mouse event.
struct SGL_API MouseEvent {
    /// The event type.
    MouseEventType type;
    /// The mouse position.
    float2 pos{0.f, 0.f};
    /// The scroll offset.
    float2 scroll{0.f, 0.f};
    /// The mouse button that was pressed/released.
    MouseButton button{MouseButton::unknown};
    /// Keyboard modifier flags.
    KeyModifierFlags mods{0};

    /// Returns true if this event is a mouse button down event.
    bool is_button_down() const { return type == MouseEventType::button_down; }
    /// Returns true if this event is a mouse button up event.
    bool is_button_up() const { return type == MouseEventType::button_up; }
    /// Returns true if this event is a mouse move event.
    bool is_move() const { return type == MouseEventType::move; }
    /// Returns true if this event is a mouse scroll event.
    bool is_scroll() const { return type == MouseEventType::scroll; }

    /// Returns true if the specified modifier is set.
    bool has_modifier(KeyModifier mod) const { return is_set(mods, (KeyModifierFlags)mod); }

    std::string to_string() const;
};

/// Gamepad event types.
enum class GamepadEventType {
    /// Gamepad button was pressed.
    button_down,
    /// Gamepad button was released.
    button_up,
    /// Gamepad was connected.
    connect,
    /// Gamepad was disconnected.
    disconnect,
};

SGL_ENUM_INFO(
    GamepadEventType,
    {
        {GamepadEventType::button_down, "button_down"},
        {GamepadEventType::button_up, "button_up"},
        {GamepadEventType::connect, "connect"},
        {GamepadEventType::disconnect, "disconnect"},
    }
);
SGL_ENUM_REGISTER(GamepadEventType);

/// Gamepad buttons.
enum class GamepadButton : uint32_t {
    a,
    b,
    x,
    y,
    left_bumper,
    right_bumper,
    back,
    start,
    guide,
    left_thumb,
    right_thumb,
    up,
    right,
    down,
    left,
};

SGL_ENUM_INFO(
    GamepadButton,
    {
        {GamepadButton::a, "a"},
        {GamepadButton::b, "b"},
        {GamepadButton::x, "x"},
        {GamepadButton::y, "y"},
        {GamepadButton::left_bumper, "left_bumper"},
        {GamepadButton::right_bumper, "right_bumper"},
        {GamepadButton::back, "back"},
        {GamepadButton::start, "start"},
        {GamepadButton::guide, "guide"},
        {GamepadButton::left_thumb, "left_thumb"},
        {GamepadButton::right_thumb, "right_thumb"},
        {GamepadButton::up, "up"},
        {GamepadButton::right, "right"},
        {GamepadButton::down, "down"},
        {GamepadButton::left, "left"},
    }
);
SGL_ENUM_REGISTER(GamepadButton);

/// Gamepad event.
struct SGL_API GamepadEvent {
    /// The event type.
    GamepadEventType type;
    /// The gamepad button that was pressed/released.
    GamepadButton button;

    /// Returns true if this event is a gamepad button down event.
    bool is_button_down() const { return type == GamepadEventType::button_down; }
    /// Returns true if this event is a gamepad button up event.
    bool is_button_up() const { return type == GamepadEventType::button_up; }
    /// Returns true if this event is a gamepad connect event.
    bool is_connect() const { return type == GamepadEventType::connect; }
    /// Returns true if this event is a gamepad disconnect event.
    bool is_disconnect() const { return type == GamepadEventType::disconnect; }

    std::string to_string() const;
};

/// Gamepad state.
struct SGL_API GamepadState {
    /// X-axis of the left analog stick.
    float left_x;
    /// Y-axis of the left analog stick.
    float left_y;
    /// X-axis of the right analog stick.
    float right_x;
    /// Y-axis of the right analog stick.
    float right_y;
    /// Value of the left analog trigger.
    float left_trigger;
    /// Value of the right analog trigger.
    float right_trigger;

    /// Bitfield of gamepad buttons (see \ref GamepadButton).
    uint32_t buttons;

    /// Returns true if the specified button is down.
    bool is_button_down(GamepadButton button) const { return buttons & (1 << uint32_t(button)); }

    std::string to_string() const;
};

} // namespace sgl
