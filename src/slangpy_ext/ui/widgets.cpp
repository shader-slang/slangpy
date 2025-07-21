// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nanobind.h"

#include "sgl/ui/widgets.h"

#undef D
#define D(...) DOC(sgl, ui, __VA_ARGS__)

namespace sgl {

template<>
struct GcHelper<ui::Widget> {
    void traverse(ui::Widget* self, GcVisitor& visitor)
    {
        visitor("callback");
        for (auto child : self->children())
            visitor(child);
    }
    void clear(ui::Widget* self) { self->remove_all_children(); }
};

} // namespace sgl

namespace sgl::ui {

template<typename T>
static void bind_value_property(nb::module_ m, const char* name)
{
    nb::class_<T, Widget>(m, name)
        .def_prop_rw("label", &T::label, &T::set_label)
        .def_prop_rw("value", &T::value, &T::set_value)
        .def_prop_rw("callback", &T::callback, &T::set_callback)
        .def("_get_callback", &T::callback);
}

template<typename T>
static void bind_drag(nb::module_ m, const char* name)
{
    nb::class_<T, ValueProperty<typename T::value_type>>(m, name)
        .def(
            nb::init<
                Widget*,
                std::string_view,
                typename T::value_type,
                typename T::Callback,
                float,
                typename T::scalar_type,
                typename T::scalar_type,
                std::string_view,
                SliderFlags>(),
            "parent"_a.none(),
            "label"_a = "",
            "value"_a = typename T::value_type(0),
            "callback"_a = typename T::Callback{},
            "speed"_a = 1.f,
            "min"_a = typename T::scalar_type(0),
            "max"_a = typename T::scalar_type(0),
            "format"_a = T::default_format,
            "flags"_a = SliderFlags::none
        )
        .def_prop_rw("speed", &T::speed, &T::set_speed)
        .def_prop_rw("min", &T::min, &T::set_min)
        .def_prop_rw("max", &T::max, &T::set_max)
        .def_prop_rw("format", &T::format, &T::set_format)
        .def_prop_rw("flags", &T::flags, &T::set_flags);
}

template<typename T>
static void bind_slider(nb::module_ m, const char* name)
{
    nb::class_<T, ValueProperty<typename T::value_type>>(m, name)
        .def(
            nb::init<
                Widget*,
                std::string_view,
                typename T::value_type,
                typename T::Callback,
                typename T::scalar_type,
                typename T::scalar_type,
                std::string_view,
                SliderFlags>(),
            "parent"_a.none(),
            "label"_a = "",
            "value"_a = typename T::value_type(0),
            "callback"_a = typename T::Callback{},
            "min"_a = typename T::scalar_type(0),
            "max"_a = typename T::scalar_type(0),
            "format"_a = T::default_format,
            "flags"_a = SliderFlags::none
        )
        .def_prop_rw("min", &T::min, &T::set_min)
        .def_prop_rw("max", &T::max, &T::set_max)
        .def_prop_rw("format", &T::format, &T::set_format)
        .def_prop_rw("flags", &T::flags, &T::set_flags);
}

template<typename T>
static void bind_input(nb::module_ m, const char* name)
{
    nb::class_<T, ValueProperty<typename T::value_type>>(m, name)
        .def(
            nb::init<
                Widget*,
                std::string_view,
                typename T::value_type,
                typename T::Callback,
                typename T::scalar_type,
                typename T::scalar_type,
                std::string_view,
                InputTextFlags>(),
            "parent"_a.none(),
            "label"_a = "",
            "value"_a = typename T::value_type(0),
            "callback"_a = typename T::Callback{},
            "step"_a = typename T::scalar_type(1),
            "step_fast"_a = typename T::scalar_type(100),
            "format"_a = T::default_format,
            "flags"_a = InputTextFlags::none
        )
        .def_prop_rw("step", &T::step, &T::set_step)
        .def_prop_rw("step_fast", &T::step_fast, &T::set_step_fast)
        .def_prop_rw("format", &T::format, &T::set_format)
        .def_prop_rw("flags", &T::flags, &T::set_flags);
}
} // namespace sgl::ui

SGL_PY_EXPORT(ui_widgets)
{
    using namespace sgl;
    using namespace sgl::ui;

    nb::module_ ui = m.attr("ui");

    nb::class_<Widget, sgl::Object>(ui, "Widget", gc_helper_type_slots<Widget>(), D(Widget))
        .def_prop_rw("parent", (Widget * (Widget::*)(void)) & Widget::parent, &Widget::set_parent, D(Widget, parent))
        .def_prop_ro("children", &Widget::children, D(Widget, children))
        .def_prop_rw("visible", &Widget::visible, &Widget::set_visible, D(Widget, visible))
        .def_prop_rw("enabled", &Widget::enabled, &Widget::set_enabled, D(Widget, enabled))
        .def("child_index", &Widget::child_index, "child"_a, D(Widget, child_index))
        .def("add_child", &Widget::add_child, "child"_a, D(Widget, add_child))
        .def("add_child_at", &Widget::add_child_at, "child"_a, "index"_a, D(Widget, add_child_at))
        .def("remove_child", &Widget::remove_child, "child"_a, D(Widget, remove_child))
        .def("remove_child_at", &Widget::remove_child_at, "index"_a, D(Widget, remove_child_at))
        .def("remove_all_children", &Widget::remove_all_children, D(Widget, remove_all_children))
        .def("__len__", &Widget::child_count, D(Widget, child_count))
        .def(
            "__iter__",
            [](const Widget& self) {
                return nb::make_iterator(
                    nb::type<Widget>(),
                    "iterator",
                    self.children().begin(),
                    self.children().end()
                );
            },
            nb::keep_alive<0, 1>()
        )
        .def(
            "__getitem__",
            [](const Widget& self, Py_ssize_t i)
            {
                i = detail::sanitize_getitem_index(i, self.child_count());
                return self.child_at(i);
            },
            D(Widget, child_at)
        )
        .def("__delitem__", &Widget::remove_child_at, D(Widget, remove_child_at));

    nb::class_<Screen, Widget>(ui, "Screen", D(Screen))
        .def("dispatch_events", &Screen::dispatch_events, D(Screen, dispatch_events));

    nb::class_<Window, Widget>(ui, "Window", D(Window))
        .def(
            nb::init<Widget*, std::string_view, float2, float2>(),
            "parent"_a.none(),
            "title"_a = "",
            "position"_a = float2(10.f, 10.f),
            "size"_a = float2(400.f, 400.f),
            D(Window, Window)
        )
        .def("show", &Window::show, D(Window, show))
        .def("close", &Window::close, D(Window, close))
        .def_prop_rw("title", &Window::title, &Window::set_title, D(Window, title))
        .def_prop_rw("position", &Window::position, &Window::set_position, D(Window, position))
        .def_prop_rw("size", &Window::size, &Window::set_size, D(Window, size));

    nb::class_<Group, Widget>(ui, "Group")
        .def(nb::init<Widget*, std::string_view>(), "parent"_a.none(), "label"_a = "", D(Group, Group))
        .def_prop_rw("label", &Group::label, &Group::set_label, D(Group, label));

    nb::class_<Text, Widget>(ui, "Text")
        .def(nb::init<Widget*, std::string_view>(), "parent"_a.none(), "text"_a = "", D(Text, Text))
        .def_prop_rw("text", &Text::text, &Text::set_text, D(Text, text));

    nb::class_<ProgressBar, Widget>(ui, "ProgressBar", D(ProgressBar))
        .def(nb::init<Widget*, float>(), "parent"_a.none(), "fraction"_a = 0.f, D(ProgressBar, ProgressBar))
        .def_prop_rw("fraction", &ProgressBar::fraction, &ProgressBar::set_fraction, D(ProgressBar, fraction));

    nb::class_<Button, Widget>(ui, "Button", D(Button))
        .def(
            nb::init<Widget*, std::string_view, Button::Callback>(),
            "parent"_a.none(),
            "label"_a = "",
            "callback"_a = Button::Callback{},
            D(Button, Button)
        )
        .def_prop_rw("label", &Button::label, &Button::set_label, D(Button, label))
        .def_prop_rw("callback", &Button::callback, &Button::set_callback, D(Button, callback))
        .def("_get_callback", &Button::callback);

    bind_value_property<ValueProperty<bool>>(ui, "ValuePropertyBool");
    bind_value_property<ValueProperty<int>>(ui, "ValuePropertyInt");
    bind_value_property<ValueProperty<int2>>(ui, "ValuePropertyInt2");
    bind_value_property<ValueProperty<int3>>(ui, "ValuePropertyInt3");
    bind_value_property<ValueProperty<int4>>(ui, "ValuePropertyInt4");
    bind_value_property<ValueProperty<float>>(ui, "ValuePropertyFloat");
    bind_value_property<ValueProperty<float2>>(ui, "ValuePropertyFloat2");
    bind_value_property<ValueProperty<float3>>(ui, "ValuePropertyFloat3");
    bind_value_property<ValueProperty<float4>>(ui, "ValuePropertyFloat4");
    bind_value_property<ValueProperty<std::string>>(ui, "ValuePropertyString");

    nb::class_<CheckBox, ValueProperty<bool>>(ui, "CheckBox", D(CheckBox))
        .def(
            nb::init<Widget*, std::string_view, bool, CheckBox::Callback>(),
            "parent"_a.none(),
            "label"_a = "",
            "value"_a = false,
            "callback"_a = CheckBox::Callback{},
            D(CheckBox, CheckBox)
        );

    nb::class_<ComboBox, ValueProperty<int>>(ui, "ComboBox", D(ComboBox))
        .def(
            nb::init<Widget*, std::string_view, int, ComboBox::Callback, std::vector<std::string>>(),
            "parent"_a.none(),
            "label"_a = "",
            "value"_a = 0,
            "callback"_a = ComboBox::Callback{},
            "items"_a = std::vector<std::string>{},
            D(ComboBox, ComboBox)
        )
        .def_prop_rw("items", &ComboBox::items, &ComboBox::set_items, D(ComboBox, items));

    nb::class_<ListBox, ValueProperty<int>>(ui, "ListBox")
        .def(
            nb::init<Widget*, std::string_view, int, ListBox::Callback, std::vector<std::string>, int>(),
            "parent"_a.none(),
            "label"_a = "",
            "value"_a = 0,
            "callback"_a = ListBox::Callback{},
            "items"_a = std::vector<std::string>{},
            "height_in_items"_a = -1,
            D(ListBox, ListBox)
        )
        .def_prop_rw("items", &ListBox::items, &ListBox::set_items, D(ListBox, items))
        .def_prop_rw(
            "height_in_items",
            &ListBox::height_in_items,
            &ListBox::set_height_in_items,
            D(ListBox, height_in_items)
        );

    nb::enum_<SliderFlags>(ui, "SliderFlags", nb::is_arithmetic(), nb::is_flag(), D(SliderFlags))
        .value("none", SliderFlags::none)
        .value("always_clamp", SliderFlags::always_clamp)
        .value("logarithmic", SliderFlags::logarithmic)
        .value("no_round_to_format", SliderFlags::no_round_to_format)
        .value("no_input", SliderFlags::no_input);

    bind_drag<DragFloat>(ui, "DragFloat");
    bind_drag<DragFloat2>(ui, "DragFloat2");
    bind_drag<DragFloat3>(ui, "DragFloat3");
    bind_drag<DragFloat4>(ui, "DragFloat4");
    bind_drag<DragInt>(ui, "DragInt");
    bind_drag<DragInt2>(ui, "DragInt2");
    bind_drag<DragInt3>(ui, "DragInt3");
    bind_drag<DragInt4>(ui, "DragInt4");

    bind_slider<SliderFloat>(ui, "SliderFloat");
    bind_slider<SliderFloat2>(ui, "SliderFloat2");
    bind_slider<SliderFloat3>(ui, "SliderFloat3");
    bind_slider<SliderFloat4>(ui, "SliderFloat4");
    bind_slider<SliderInt>(ui, "SliderInt");
    bind_slider<SliderInt2>(ui, "SliderInt2");
    bind_slider<SliderInt3>(ui, "SliderInt3");
    bind_slider<SliderInt4>(ui, "SliderInt4");

    nb::enum_<InputTextFlags>(ui, "InputTextFlags", nb::is_arithmetic(), nb::is_flag(), D(InputTextFlags))
        .value("none", InputTextFlags::none)
        .value("chars_decimal", InputTextFlags::chars_decimal)
        .value("chars_hexadecimal", InputTextFlags::chars_hexadecimal)
        .value("chars_uppercase", InputTextFlags::chars_uppercase)
        .value("chars_no_blank", InputTextFlags::chars_no_blank)
        .value("auto_select_all", InputTextFlags::auto_select_all)
        .value("enter_returns_true", InputTextFlags::enter_returns_true)
        .value("callback_completion", InputTextFlags::callback_completion)
        .value("callback_history", InputTextFlags::callback_history)
        .value("callback_always", InputTextFlags::callback_always)
        .value("callback_char_filter", InputTextFlags::callback_char_filter)
        .value("allow_tab_input", InputTextFlags::allow_tab_input)
        .value("ctrl_enter_for_new_line", InputTextFlags::ctrl_enter_for_new_line)
        .value("no_horizontal_scroll", InputTextFlags::no_horizontal_scroll)
        .value("always_overwrite", InputTextFlags::always_overwrite)
        .value("read_only", InputTextFlags::read_only)
        .value("password", InputTextFlags::password)
        .value("no_undo_redo", InputTextFlags::no_undo_redo)
        .value("chars_scientific", InputTextFlags::chars_scientific)
        .value("escape_clears_all", InputTextFlags::escape_clears_all);

    bind_input<InputFloat>(ui, "InputFloat");
    bind_input<InputFloat2>(ui, "InputFloat2");
    bind_input<InputFloat3>(ui, "InputFloat3");
    bind_input<InputFloat4>(ui, "InputFloat4");
    bind_input<InputInt>(ui, "InputInt");
    bind_input<InputInt2>(ui, "InputInt2");
    bind_input<InputInt3>(ui, "InputInt3");
    bind_input<InputInt4>(ui, "InputInt4");

    nb::class_<InputText, ValueProperty<std::string>>(ui, "InputText", D(InputText))
        .def(
            nb::init<Widget*, std::string_view, std::string, InputText::Callback, bool, InputTextFlags>(),
            "parent"_a.none(),
            "label"_a = "",
            "value"_a = false,
            "callback"_a = InputText::Callback{},
            "multi_line"_a = false,
            "flags"_a = InputTextFlags::none,
            D(InputText, InputText)
        );
}
