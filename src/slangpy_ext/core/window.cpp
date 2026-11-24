// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/core/window.h"

#include <limits>

namespace sgl {

SGL_DICT_TO_DESC_BEGIN(WindowPlacement)
SGL_DICT_TO_DESC_FIELD(normal_position, int2)
SGL_DICT_TO_DESC_FIELD(normal_size, uint2)
SGL_DICT_TO_DESC_FIELD(maximized, bool)
SGL_DICT_TO_DESC_END()

template<>
struct GcHelper<Window> {
    void traverse(Window*, GcVisitor& visitor)
    {
        visitor("on_resize");
        visitor("on_refresh");
        visitor("on_keyboard_event");
        visitor("on_mouse_event");
        visitor("on_gamepad_event");
        visitor("on_gamepad_state");
        visitor("on_drop_files");
    }
    void clear(Window*) { }
};

} // namespace sgl

namespace {

void validate_placement(const sgl::WindowPlacement& placement)
{
    const auto size = placement.normal_size;
    if (size.x == 0 || size.y == 0 || size.x > uint32_t(std::numeric_limits<int>::max())
        || size.y > uint32_t(std::numeric_limits<int>::max()))
        throw nb::value_error("Window dimensions must be positive and fit in a signed integer");
}

NB_NOINLINE void bind_window_placement(nb::module_& m)
{
    using namespace sgl;

    nb::class_<WindowPlacement>(m, "WindowPlacement", D(WindowPlacement))
        .def(
            "__init__",
            [](WindowPlacement* self, nb::dict dict)
            {
                const auto placement = dict_to_WindowPlacement(dict);
                validate_placement(placement);
                new (self) WindowPlacement(placement);
            },
            "dict"_a,
            "Create placement from a dictionary. normal_size is required; normal_position and maximized are optional."
        )
        .def(
            "__init__",
            [](WindowPlacement* placement, int2 normal_position, uint2 normal_size, bool maximized)
            {
                const WindowPlacement value{normal_position, normal_size, maximized};
                validate_placement(value);
                new (placement) WindowPlacement(value);
            },
            "normal_position"_a,
            "normal_size"_a,
            "maximized"_a = false
        )
        .def_ro("normal_position", &WindowPlacement::normal_position, D(WindowPlacement, normal_position))
        .def_ro("normal_size", &WindowPlacement::normal_size, D(WindowPlacement, normal_size))
        .def_ro("maximized", &WindowPlacement::maximized, D(WindowPlacement, maximized))
        .def(
            "to_dict",
            [](const WindowPlacement& placement)
            {
                nb::dict result;
                result["normal_position"]
                    = nb::cast(std::array{placement.normal_position.x, placement.normal_position.y});
                result["normal_size"] = nb::cast(std::array{placement.normal_size.x, placement.normal_size.y});
                result["maximized"] = placement.maximized;
                return result;
            },
            "Return a JSON-compatible dictionary of normal geometry and maximization state."
        );
    nb::implicitly_convertible<nb::dict, WindowPlacement>();
}

} // namespace

SGL_PY_EXPORT(core_window)
{
    using namespace sgl;

    nb::sgl_enum<WindowMode>(m, "WindowMode", D(WindowMode));
    nb::sgl_enum<CursorMode>(m, "CursorMode", D(CursorMode));
    nb::sgl_enum<CursorShape>(m, "CursorShape", D(CursorShape));

    bind_window_placement(m);

    nb::class_<Window, Object> window(m, "Window", gc_helper_type_slots<Window>(), D(Window));
    window.def(
        "__init__",
        [](Window* window,
           uint32_t width,
           uint32_t height,
           std::string title,
           WindowMode mode,
           bool resizable,
           std::optional<int2> position,
           bool visible,
           std::optional<WindowPlacement> placement)
        {
            new (window) Window({
                .width = width,
                .height = height,
                .title = title,
                .mode = mode,
                .resizable = resizable,
                .position = position,
                .visible = visible,
                .placement = placement,
            });
        },
        "width"_a = 1024,
        "height"_a = 1024,
        "title"_a = "slangpy",
        "mode"_a = WindowMode::normal,
        "resizable"_a = true,
        "position"_a = nb::none(),
        "visible"_a = true,
        "placement"_a = nb::none(),
        D(Window, Window)
    );
    window.def_prop_rw("title", &Window::title, &Window::set_title, D(Window, title));

    window.def_prop_rw("position", &Window::position, &Window::set_position, D(Window, position));
    window.def_prop_rw("width", &Window::width, &Window::set_width, D(Window, width));
    window.def_prop_rw("height", &Window::height, &Window::set_height, D(Window, height));
    window.def_prop_rw("size", &Window::size, &Window::set_size, D(Window, size));
    window.def("resize", &Window::resize, "width"_a, "height"_a, D(Window, resize));
    window.def_prop_ro("frame_size", &Window::frame_size, D(Window, frame_size));
    window.def_prop_ro("content_scale", &Window::content_scale, D(Window, content_scale));

    window.def_prop_ro("placement", &Window::placement, D(Window, placement));
    window.def("is_minimized", &Window::is_minimized, D(Window, is_minimized));
    window.def("is_maximized", &Window::is_maximized, D(Window, is_maximized));
    window.def("is_fullscreen", &Window::is_fullscreen, D(Window, is_fullscreen));
    window.def("is_visible", &Window::is_visible, D(Window, is_visible));
    window.def("minimize", &Window::minimize, D(Window, minimize));
    window.def("maximize", &Window::maximize, D(Window, maximize));
    window.def("restore", &Window::restore, D(Window, restore));
    window.def("show", &Window::show, D(Window, show));
    window.def("hide", &Window::hide, D(Window, hide));

    window.def("close", &Window::close, D(Window, close));
    window.def("should_close", &Window::should_close, D(Window, should_close));
    window.def("process_events", &Window::process_events, D(Window, process_events));

    window.def_prop_rw("cursor_mode", &Window::cursor_mode, &Window::set_cursor_mode, D(Window, cursor_mode));
    window.def_prop_rw("cursor_shape", &Window::cursor_shape, &Window::set_cursor_shape, D(Window, cursor_shape));
    window.def("get_clipboard", &Window::get_clipboard, D(Window, get_clipboard));
    window.def("set_clipboard", &Window::set_clipboard, "text"_a, D(Window, set_clipboard));

    window.def_prop_rw("on_resize", &Window::on_resize, &Window::set_on_resize, nb::arg().none(), D(Window, on_resize));
    window.def_prop_rw(
        "on_refresh",
        &Window::on_refresh,
        &Window::set_on_refresh,
        nb::arg().none(),
        D(Window, on_refresh)
    );
    window.def_prop_rw(
        "on_keyboard_event",
        &Window::on_keyboard_event,
        &Window::set_on_keyboard_event,
        nb::arg().none(),
        D(Window, on_keyboard_event)
    );
    window.def_prop_rw(
        "on_mouse_event",
        &Window::on_mouse_event,
        &Window::set_on_mouse_event,
        nb::arg().none(),
        D(Window, on_mouse_event)
    );
    window.def_prop_rw(
        "on_gamepad_event",
        &Window::on_gamepad_event,
        &Window::set_on_gamepad_event,
        nb::arg().none(),
        D(Window, on_gamepad_event)
    );
    window.def_prop_rw(
        "on_gamepad_state",
        &Window::on_gamepad_state,
        &Window::set_on_gamepad_state,
        nb::arg().none(),
        D(Window, on_gamepad_state)
    );
    window.def_prop_rw(
        "on_drop_files",
        &Window::on_drop_files,
        &Window::set_on_drop_files,
        nb::arg().none(),
        D(Window, on_drop_files)
    );
}
