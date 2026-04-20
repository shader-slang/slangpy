// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/input.h"

SGL_PY_EXPORT(core_input)
{
    using namespace sgl;

    nb::sgl_enum<MouseButton>(m, "MouseButton", D(MouseButton));
    nb::sgl_enum_flags<KeyModifierFlags>(m, "KeyModifierFlags", D(KeyModifierFlags));
    nb::sgl_enum<KeyModifier>(m, "KeyModifier", D(KeyModifier));
    nb::sgl_enum<KeyCode>(m, "KeyCode", D(KeyCode));

    nb::sgl_enum<KeyboardEventType>(m, "KeyboardEventType", D(KeyboardEventType));

    nb::class_<KeyboardEvent>(m, "KeyboardEvent", D(KeyboardEvent))
        .def_ro("type", &KeyboardEvent::type, D(KeyboardEvent, type))
        .def_ro("key", &KeyboardEvent::key, D(KeyboardEvent, key))
        .def_ro("mods", &KeyboardEvent::mods, D(KeyboardEvent, mods))
        .def_ro("codepoint", &KeyboardEvent::codepoint, D(KeyboardEvent, codepoint))
        .def("is_key_press", &KeyboardEvent::is_key_press, D(KeyboardEvent, is_key_press))
        .def("is_key_release", &KeyboardEvent::is_key_release, D(KeyboardEvent, is_key_release))
        .def("is_key_repeat", &KeyboardEvent::is_key_repeat, D(KeyboardEvent, is_key_repeat))
        .def("is_input", &KeyboardEvent::is_input, D(KeyboardEvent, is_input))
        .def("has_modifier", &KeyboardEvent::has_modifier, D(KeyboardEvent, has_modifier))
        .def("__repr__", &KeyboardEvent::to_string);

    nb::sgl_enum<MouseEventType>(m, "MouseEventType", D(MouseEventType));

    nb::class_<MouseEvent>(m, "MouseEvent", D(MouseEvent))
        .def_ro("type", &MouseEvent::type, D(MouseEvent, type))
        .def_ro("pos", &MouseEvent::pos, D(MouseEvent, pos))
        .def_ro("scroll", &MouseEvent::scroll, D(MouseEvent, scroll))
        .def_ro("button", &MouseEvent::button, D(MouseEvent, button))
        .def_ro("mods", &MouseEvent::mods, D(MouseEvent, mods))
        .def("is_button_down", &MouseEvent::is_button_down, D(MouseEvent, is_button_down))
        .def("is_button_up", &MouseEvent::is_button_up, D(MouseEvent, is_button_up))
        .def("is_move", &MouseEvent::is_move, D(MouseEvent, is_move))
        .def("is_scroll", &MouseEvent::is_scroll, D(MouseEvent, is_scroll))
        .def("has_modifier", &MouseEvent::has_modifier, D(MouseEvent, has_modifier))
        .def("__repr__", &MouseEvent::to_string);

    nb::sgl_enum<GamepadEventType>(m, "GamepadEventType", D(GamepadEventType));
    nb::sgl_enum<GamepadButton>(m, "GamepadButton", D(GamepadButton));

    nb::class_<GamepadEvent>(m, "GamepadEvent", D(GamepadEvent))
        .def_ro("type", &GamepadEvent::type, D(GamepadEvent, type))
        .def_ro("button", &GamepadEvent::button, D(GamepadEvent, button))
        .def("is_button_down", &GamepadEvent::is_button_down, D(GamepadEvent, is_button_down))
        .def("is_button_up", &GamepadEvent::is_button_up, D(GamepadEvent, is_button_up))
        .def("is_connect", &GamepadEvent::is_connect, D(GamepadEvent, is_connect))
        .def("is_disconnect", &GamepadEvent::is_disconnect, D(GamepadEvent, is_disconnect))
        .def("__repr__", &GamepadEvent::to_string);

    nb::class_<GamepadState>(m, "GamepadState", D(GamepadState))
        .def_ro("left_x", &GamepadState::left_x, D(GamepadState, left_x))
        .def_ro("left_y", &GamepadState::left_y, D(GamepadState, left_y))
        .def_ro("right_x", &GamepadState::right_x, D(GamepadState, right_x))
        .def_ro("right_y", &GamepadState::right_y, D(GamepadState, right_y))
        .def_ro("left_trigger", &GamepadState::left_trigger, D(GamepadState, left_trigger))
        .def_ro("right_trigger", &GamepadState::right_trigger, D(GamepadState, right_trigger))
        .def_ro("buttons", &GamepadState::buttons, D(GamepadState, buttons))
        .def("is_button_down", &GamepadState::is_button_down, D(GamepadState, is_button_down))
        .def("__repr__", &GamepadState::to_string);
}
