// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/object.h"
#include "sgl/math/vector_types.h"

#include <imgui.h>

#include <algorithm>
#include <string_view>
#include <string>
#include <functional>
#include <vector>

namespace sgl::ui {

class Widget;

/// Base class for Python UI widgets.
/// Widgets own their children.
class Widget : public Object {
    SGL_OBJECT(Widget)
public:
    Widget(Widget* parent)
        : m_parent(parent)
    {
        if (m_parent)
            m_parent->m_children.push_back(ref<Widget>(this));
    }

    virtual ~Widget() { }

    Widget* parent() { return m_parent; }
    const Widget* parent() const { return m_parent; }
    void set_parent(Widget* parent) { m_parent = parent; }

    const std::vector<ref<Widget>>& children() const { return m_children; }
    size_t child_count() const { return m_children.size(); }
    void clear_children() { m_children.clear(); }

    int child_index(const Widget* child) const
    {
        auto it = std::find(m_children.begin(), m_children.end(), child);
        return it != m_children.end() ? int(std::distance(m_children.begin(), it)) : -1;
    }

    ref<Widget> child_at(size_t index) const
    {
        SGL_CHECK(index < m_children.size(), "index out of bounds");
        return m_children[index];
    }

    void add_child(ref<Widget> child)
    {
        SGL_CHECK_NOT_NULL(child);
        m_children.push_back(child);
        child->set_parent(this);
    }

    void add_child_at(ref<Widget> child, size_t index)
    {
        SGL_CHECK_NOT_NULL(child);
        SGL_CHECK(index == 0 || index < m_children.size(), "index out of bounds");
        m_children.insert(m_children.begin() + index, child);
        child->set_parent(this);
    }

    void remove_child(ref<Widget> child)
    {
        auto it = std::find(m_children.begin(), m_children.end(), child);
        SGL_CHECK(it != m_children.end(), "child widget not found");
        m_children.erase(it);
        child->set_parent(nullptr);
    }

    void remove_child_at(size_t index)
    {
        SGL_CHECK(index < m_children.size(), "index out of bounds");
        auto child = m_children[index];
        m_children.erase(m_children.begin() + index);
        child->set_parent(nullptr);
    }

    void remove_all_children()
    {
        for (auto& child : m_children)
            child->set_parent(nullptr);
        m_children.clear();
    }

    bool visible() const { return m_visible; }
    void set_visible(bool visible) { m_visible = visible; }

    bool enabled() const { return m_enabled; }
    void set_enabled(bool enabled) { m_enabled = enabled; }

    virtual void render()
    {
        if (!m_visible)
            return;

        for (const auto& child : m_children)
            child->render();
    }

protected:
    Widget* m_parent;
    std::vector<ref<Widget>> m_children;
    bool m_visible{true};
    bool m_enabled{true};
};

/// This is the main widget that represents the screen.
/// It is intended to be used as the parent for \c Window widgets.
class Screen : public Widget {
    SGL_OBJECT(Screen)
public:
    Screen()
        : Widget(nullptr)
    {
    }

    virtual void render() override { Widget::render(); }
};

/// Scoped push/pop of ImGui ID.
class ScopedID {
public:
    ScopedID(void* id) { ImGui::PushID(id); }
    ~ScopedID() { ImGui::PopID(); }
};

/// Scoped begin/end for disabling ImGUI widgets.
class ScopedDisable {
public:
    ScopedDisable(bool disabled)
        : m_disabled(disabled)
    {
        if (disabled)
            ImGui::BeginDisabled();
    }
    ~ScopedDisable()
    {
        if (m_disabled)
            ImGui::EndDisabled();
    }

private:
    bool m_disabled;
};

class Window : public Widget {
    SGL_OBJECT(Window)
public:
    Window(
        Widget* parent,
        std::string_view title = "",
        float2 position = float2(10.f, 10.f),
        float2 size = float2(400.f, 400.f)
    )
        : Widget(parent)
        , m_title(title)
        , m_position(position)
        , m_size(size)
    {
    }

    const std::string& title() const { return m_title; }
    void set_title(std::string_view title) { m_title = title; }

    float2 position() const { return m_position; }
    void set_position(const float2& position)
    {
        m_position = position;
        m_set_position = true;
    }

    float2 size() const { return m_size; }
    void set_size(const float2& size)
    {
        m_size = size;
        m_set_size = true;
    }

    void show() { set_visible(true); }
    void close() { set_visible(false); }

    virtual void render() override
    {
        if (!m_visible)
            return;

        if (m_set_position) {
            ImGui::SetNextWindowPos(ImVec2(m_position.x, m_position.y));
            m_set_position = false;
        }
        if (m_set_size) {
            ImGui::SetNextWindowSize(ImVec2(m_size.x, m_size.y));
            m_set_size = false;
        }

        ScopedID id(this);
        if (ImGui::Begin(m_title.c_str(), &m_visible)) {
            auto pos = ImGui::GetWindowPos();
            m_position = float2(pos.x, pos.y);
            auto size = ImGui::GetWindowSize();
            m_size = float2(size.x, size.y);

            ImGui::PushItemWidth(300);
            Widget::render();
            ImGui::PopItemWidth();
        }
        ImGui::End();
    }

private:
    std::string m_title;
    float2 m_position;
    float2 m_size;
    bool m_set_position{true};
    bool m_set_size{true};
};

class Group : public Widget {
    SGL_OBJECT(Group)
public:
    Group(Widget* parent, std::string_view label = "")
        : Widget(parent)
        , m_label(label)
    {
    }

    const std::string& label() const { return m_label; }
    void set_label(std::string_view label) { m_label = label; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        // Check if this is a nested group
        bool nested = false;
        for (Widget* p = parent(); p != nullptr; p = p->parent())
            if (dynamic_cast<Group*>(p) != nullptr)
                nested = true;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);

        if (nested ? ImGui::TreeNodeEx(m_label.c_str(), ImGuiTreeNodeFlags_DefaultOpen)
                   : ImGui::CollapsingHeader(m_label.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            Widget::render();
            if (nested)
                ImGui::TreePop();
        }
    }

private:
    std::string m_label;
};

class Text : public Widget {
    SGL_OBJECT(Text)
public:
    Text(Widget* parent, std::string_view text = "")
        : Widget(parent)
        , m_text(text)
    {
    }

    const std::string& text() const { return m_text; }
    void set_text(std::string_view text) { m_text = text; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        ImGui::TextUnformatted(m_text.c_str());
    }

private:
    std::string m_text;
};

class ProgressBar : public Widget {
    SGL_OBJECT(ProgressBar)
public:
    ProgressBar(Widget* parent, float fraction = 0.f)
        : Widget(parent)
        , m_fraction(fraction)
    {
    }

    float fraction() const { return m_fraction; }
    void set_fraction(float fraction) { m_fraction = fraction; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        ImGui::ProgressBar(m_fraction);
    }

private:
    float m_fraction;
};

class Button : public Widget {
    SGL_OBJECT(Button)
public:
    using Callback = std::function<void()>;

    Button(Widget* parent, std::string_view label = "", Callback callback = {})
        : Widget(parent)
        , m_label(label)
        , m_callback(callback)
    {
    }

    const std::string& label() const { return m_label; }
    void set_label(std::string_view label) { m_label = label; }

    Callback callback() const { return m_callback; }
    void set_callback(Callback callback) { m_callback = callback; }

    void notify()
    {
        if (m_callback)
            m_callback();
    }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        if (ImGui::Button(m_label.c_str()))
            notify();
    }

private:
    std::string m_label;
    Callback m_callback;
};

template<typename T>
class ValueProperty : public Widget {
public:
    using value_type = T;
    using Callback = std::function<void(const value_type&)>;

    ValueProperty(Widget* parent, std::string_view label, value_type value, Callback callback)
        : Widget(parent)
        , m_label(label)
        , m_value(value)
        , m_callback(callback)
    {
    }

    const std::string& label() const { return m_label; }
    void set_label(std::string_view label) { m_label = label; }

    const value_type& value() const { return m_value; }
    void set_value(const value_type& value) { m_value = value; }

    Callback callback() const { return m_callback; }
    void set_callback(Callback callback) { m_callback = callback; }

    void notify()
    {
        if (m_callback)
            m_callback(m_value);
    }

protected:
    std::string m_label;
    value_type m_value;
    Callback m_callback;
};

class CheckBox : public ValueProperty<bool> {
    SGL_OBJECT(CheckBox)
public:
    using Base = ValueProperty<bool>;

    CheckBox(Widget* parent, std::string_view label = "", bool value = false, Callback callback = {})
        : Base(parent, label, value, callback)
    {
    }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        if (ImGui::Checkbox(m_label.c_str(), &m_value))
            notify();
    }
};

class ComboBox : public ValueProperty<int> {
    SGL_OBJECT(ComboBox)
public:
    using Base = ValueProperty<int>;

    ComboBox(
        Widget* parent,
        std::string_view label = "",
        int value = 0,
        Callback callback = {},
        std::vector<std::string> items = {}
    )
        : Base(parent, label, value, callback)
        , m_items(items)
    {
    }

    const std::vector<std::string>& items() const { return m_items; }
    void set_items(const std::vector<std::string>& items) { m_items = items; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        if (ImGui::Combo(
                m_label.c_str(),
                &m_value,
                [](void* data, int idx, const char** out_text) -> bool
                {
                    auto& items = *reinterpret_cast<std::vector<std::string>*>(data);
                    if (idx < 0 || idx >= items.size())
                        return false;
                    *out_text = items[idx].c_str();
                    return true;
                },
                &m_items,
                (int)m_items.size()
            )) {
            notify();
        }
    }

private:
    std::vector<std::string> m_items;
};

class ListBox : public ValueProperty<int> {
    SGL_OBJECT(ListBox)
public:
    using Base = ValueProperty<int>;

    ListBox(
        Widget* parent,
        std::string_view label = "",
        int value = 0,
        Callback callback = {},
        std::vector<std::string> items = {},
        int height_in_items = -1
    )
        : Base(parent, label, value, callback)
        , m_items(items)
        , m_height_in_items(height_in_items)
    {
    }

    const std::vector<std::string>& items() const { return m_items; }
    void set_items(const std::vector<std::string>& items) { m_items = items; }

    int height_in_items() const { return m_height_in_items; }
    void set_height_in_items(int height_in_items) { m_height_in_items = height_in_items; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        if (ImGui::ListBox(
                m_label.c_str(),
                &m_value,
                [](void* user_data, int idx)
                {
                    return static_cast<const ListBox*>(user_data)->m_items[idx].c_str();
                },
                this,
                int(m_items.size()),
                m_height_in_items
            )) {
            notify();
        }
    }

private:
    std::vector<std::string> m_items;
    int m_height_in_items;
};

enum class SliderFlags {
    none = ImGuiSliderFlags_None,
    always_clamp = ImGuiSliderFlags_AlwaysClamp,
    logarithmic = ImGuiSliderFlags_Logarithmic,
    no_round_to_format = ImGuiSliderFlags_NoRoundToFormat,
    no_input = ImGuiSliderFlags_NoInput,
};

SGL_ENUM_CLASS_OPERATORS(SliderFlags);

template<typename T>
struct DataTypeTraits { };

// clang-format off
template<> struct DataTypeTraits<float> { static constexpr ImGuiDataType data_type{ImGuiDataType_Float}; static constexpr const char* default_format = "%.3f"; };
template<> struct DataTypeTraits<int> { static constexpr ImGuiDataType data_type{ImGuiDataType_S32}; static constexpr const char* default_format = "%d"; };
// clang-format on


template<typename T>
struct VectorTraits { };

// clang-format off
template<> struct VectorTraits<float> { using scalar_type = float; static constexpr int N = 1; };
template<> struct VectorTraits<float2> { using scalar_type = float; static constexpr int N = 2; };
template<> struct VectorTraits<float3> { using scalar_type = float; static constexpr int N = 3; };
template<> struct VectorTraits<float4> { using scalar_type = float; static constexpr int N = 4; };
template<> struct VectorTraits<int> { using scalar_type = int; static constexpr int N = 1; };
template<> struct VectorTraits<int2> { using scalar_type = int; static constexpr int N = 2; };
template<> struct VectorTraits<int3> { using scalar_type = int; static constexpr int N = 3; };
template<> struct VectorTraits<int4> { using scalar_type = int; static constexpr int N = 4; };
// clang-format on

template<typename T>
class Drag : public ValueProperty<T> {
    SGL_OBJECT(Drag)
public:
    using Base = ValueProperty<T>;
    using typename Base::value_type;
    using typename Base::Callback;

    using Widget::m_enabled;
    using Widget::m_visible;
    using Base::m_label;
    using Base::m_value;
    using Base::notify;

    using scalar_type = typename VectorTraits<T>::scalar_type;
    static constexpr int N = VectorTraits<T>::N;
    static constexpr const char* default_format = DataTypeTraits<scalar_type>::default_format;

    Drag(
        Widget* parent,
        std::string_view label = "",
        value_type value = value_type(0),
        Callback callback = {},
        float speed = 1.f,
        scalar_type min = scalar_type(0),
        scalar_type max = scalar_type(0),
        std::string_view format = default_format,
        SliderFlags flags = SliderFlags::none
    )
        : Base(parent, label, value, callback)
        , m_speed(speed)
        , m_min(min)
        , m_max(max)
        , m_format(format)
        , m_flags(flags)
    {
    }

    scalar_type speed() const { return static_cast<scalar_type>(m_speed); }
    void set_speed(scalar_type speed) { m_speed = static_cast<float>(speed); }

    scalar_type min() const { return m_min; }
    void set_min(scalar_type min) { m_min = min; }

    scalar_type max() const { return m_max; }
    void set_max(scalar_type max) { m_max = max; }

    const std::string& format() const { return m_format; }
    void set_format(std::string_view format) { m_format = format; }

    SliderFlags flags() const { return m_flags; }
    void set_flags(SliderFlags flags) { m_flags = flags; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        bool changed = ImGui::DragScalarN(
            m_label.c_str(),
            DataTypeTraits<scalar_type>::data_type,
            &m_value,
            N,
            m_speed,
            &m_min,
            &m_max,
            m_format.c_str(),
            ImGuiSliderFlags(m_flags)
        );
        if (changed)
            notify();
    }

private:
    float m_speed;
    scalar_type m_min;
    scalar_type m_max;
    std::string m_format;
    SliderFlags m_flags;
};

using DragFloat = Drag<float>;
using DragFloat2 = Drag<float2>;
using DragFloat3 = Drag<float3>;
using DragFloat4 = Drag<float4>;
using DragInt = Drag<int>;
using DragInt2 = Drag<int2>;
using DragInt3 = Drag<int3>;
using DragInt4 = Drag<int4>;

template<typename T>
class Slider : public ValueProperty<T> {
    SGL_OBJECT(Slider)
public:
    using Base = ValueProperty<T>;
    using typename Base::value_type;
    using typename Base::Callback;

    using Widget::m_enabled;
    using Widget::m_visible;
    using Base::m_label;
    using Base::m_value;
    using Base::notify;

    using scalar_type = typename VectorTraits<T>::scalar_type;
    static constexpr int N = VectorTraits<T>::N;
    static constexpr const char* default_format = DataTypeTraits<scalar_type>::default_format;

    Slider(
        Widget* parent,
        std::string_view label = "",
        value_type value = value_type(0),
        Callback callback = {},
        scalar_type min = scalar_type(0),
        scalar_type max = scalar_type(0),
        std::string_view format = default_format,
        SliderFlags flags = SliderFlags::none
    )
        : Base(parent, label, value, callback)
        , m_min(min)
        , m_max(max)
        , m_format(format)
        , m_flags(flags)
    {
    }

    scalar_type min() const { return m_min; }
    void set_min(scalar_type min) { m_min = min; }

    scalar_type max() const { return m_max; }
    void set_max(scalar_type max) { m_max = max; }

    const std::string& format() const { return m_format; }
    void set_format(std::string_view format) { m_format = format; }

    SliderFlags flags() const { return m_flags; }
    void set_flags(SliderFlags flags) { m_flags = flags; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        bool changed = ImGui::SliderScalarN(
            m_label.c_str(),
            DataTypeTraits<scalar_type>::data_type,
            &m_value,
            N,
            &m_min,
            &m_max,
            m_format.c_str(),
            ImGuiSliderFlags(m_flags)
        );
        if (changed)
            notify();
    }

private:
    scalar_type m_min;
    scalar_type m_max;
    std::string m_format;
    SliderFlags m_flags;
};

using SliderFloat = Slider<float>;
using SliderFloat2 = Slider<float2>;
using SliderFloat3 = Slider<float3>;
using SliderFloat4 = Slider<float4>;
using SliderInt = Slider<int>;
using SliderInt2 = Slider<int2>;
using SliderInt3 = Slider<int3>;
using SliderInt4 = Slider<int4>;

enum class InputTextFlags {
    none = ImGuiInputTextFlags_None,
    chars_decimal = ImGuiInputTextFlags_CharsDecimal,
    chars_hexadecimal = ImGuiInputTextFlags_CharsHexadecimal,
    chars_uppercase = ImGuiInputTextFlags_CharsUppercase,
    chars_no_blank = ImGuiInputTextFlags_CharsNoBlank,
    auto_select_all = ImGuiInputTextFlags_AutoSelectAll,
    enter_returns_true = ImGuiInputTextFlags_EnterReturnsTrue,
    callback_completion = ImGuiInputTextFlags_CallbackCompletion,
    callback_history = ImGuiInputTextFlags_CallbackHistory,
    callback_always = ImGuiInputTextFlags_CallbackAlways,
    callback_char_filter = ImGuiInputTextFlags_CallbackCharFilter,
    allow_tab_input = ImGuiInputTextFlags_AllowTabInput,
    ctrl_enter_for_new_line = ImGuiInputTextFlags_CtrlEnterForNewLine,
    no_horizontal_scroll = ImGuiInputTextFlags_NoHorizontalScroll,
    always_overwrite = ImGuiInputTextFlags_AlwaysOverwrite,
    read_only = ImGuiInputTextFlags_ReadOnly,
    password = ImGuiInputTextFlags_Password,
    no_undo_redo = ImGuiInputTextFlags_NoUndoRedo,
    chars_scientific = ImGuiInputTextFlags_CharsScientific,
    escape_clears_all = ImGuiInputTextFlags_EscapeClearsAll,
};

SGL_ENUM_CLASS_OPERATORS(InputTextFlags);

template<typename T>
class Input : public ValueProperty<T> {
    SGL_OBJECT(Input)
public:
    using Base = ValueProperty<T>;
    using typename Base::value_type;
    using typename Base::Callback;

    using Widget::m_enabled;
    using Widget::m_visible;
    using Base::m_label;
    using Base::m_value;
    using Base::notify;

    using scalar_type = typename VectorTraits<T>::scalar_type;
    static constexpr int N = VectorTraits<T>::N;
    static constexpr const char* default_format = DataTypeTraits<scalar_type>::default_format;

    Input(
        Widget* parent,
        std::string_view label = "",
        value_type value = value_type(0),
        Callback callback = {},
        scalar_type step = scalar_type(1),
        scalar_type step_fast = scalar_type(100),
        std::string_view format = default_format,
        InputTextFlags flags = InputTextFlags::none
    )
        : Base(parent, label, value, callback)
        , m_step(step)
        , m_step_fast(step_fast)
        , m_format(format)
        , m_flags(flags)
    {
    }

    scalar_type step() const { return m_step; }
    void set_step(scalar_type step) { m_step = step; }

    scalar_type step_fast() const { return m_step_fast; }
    void set_step_fast(scalar_type step_fast) { m_step_fast = step_fast; }

    const std::string& format() const { return m_format; }
    void set_format(std::string_view format) { m_format = format; }

    InputTextFlags flags() const { return m_flags; }
    void set_flags(InputTextFlags flags) { m_flags = flags; }

    virtual void render() override
    {
        if (!m_visible)
            return;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        bool changed = ImGui::InputScalarN(
            m_label.c_str(),
            DataTypeTraits<scalar_type>::data_type,
            &m_value,
            N,
            &m_step,
            &m_step_fast,
            m_format.c_str(),
            ImGuiInputTextFlags(m_flags)
        );
        if (changed)
            notify();
    }

private:
    scalar_type m_step;
    scalar_type m_step_fast;
    std::string m_format;
    InputTextFlags m_flags;
};

using InputFloat = Input<float>;
using InputFloat2 = Input<float2>;
using InputFloat3 = Input<float3>;
using InputFloat4 = Input<float4>;
using InputInt = Input<int>;
using InputInt2 = Input<int2>;
using InputInt3 = Input<int3>;
using InputInt4 = Input<int4>;

class InputText : public ValueProperty<std::string> {
    SGL_OBJECT(InputText)
public:
    using Base = ValueProperty<std::string>;

    InputText(
        Widget* parent,
        std::string_view label = "",
        std::string value = "",
        Callback callback = {},
        bool multi_line = false,
        InputTextFlags flags = InputTextFlags::none
    )
        : Base(parent, label, value, callback)
        , m_multi_line(multi_line)
        , m_flags(flags)
    {
    }

    virtual void render() override
    {
        if (!m_visible)
            return;

        auto text_callback = [](ImGuiInputTextCallbackData* data)
        {
            auto self = static_cast<InputText*>(data->UserData);
            if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                self->m_value.resize(data->BufTextLen);
                data->Buf = self->m_value.data();
            }
            return 0;
        };

        ImGuiInputTextFlags flags = ImGuiInputTextFlags(m_flags) | ImGuiInputTextFlags_CallbackResize;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        bool changed = false;
        if (m_multi_line) {
            changed = ImGui::InputTextMultiline(
                m_label.c_str(),
                m_value.data(),
                m_value.capacity() + 1,
                ImVec2(0, 0),
                flags,
                text_callback,
                this
            );
        } else {
            changed
                = ImGui::InputText(m_label.c_str(), m_value.data(), m_value.capacity() + 1, flags, text_callback, this);
        }
        if (changed)
            notify();
    }

private:
    bool m_multi_line;
    InputTextFlags m_flags;
};

} // namespace sgl::ui
