// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/refl/function.h"

#include "sgl/refl/layout.h"

#include "sgl/core/enum.h"

#include <fmt/format.h>

#include <algorithm>
#include <utility>

namespace sgl::refl {

namespace {

    std::string c_string(const char* value)
    {
        return value ? std::string(value) : std::string();
    }

    std::vector<ModifierID> modifiers_from_reflection(const VariableReflection& reflection)
    {
        std::vector<ModifierID> modifiers;
        for (const auto& item : EnumInfo<ModifierID>::items) {
            if (reflection.has_modifier(item.first))
                modifiers.push_back(item.first);
        }
        return modifiers;
    }

} // namespace

//------------------------------------
// Variable
//------------------------------------

Variable::Variable(
    ref<Layout> layout,
    ref<Type> type,
    std::string name,
    std::vector<ModifierID> modifiers,
    ref<const VariableReflection> reflection
)
    : m_layout(std::move(layout))
    , m_type(std::move(type))
    , m_name(std::move(name))
    , m_modifiers(std::move(modifiers))
    , m_reflection(std::move(reflection))
{
    SGL_CHECK(m_layout, "Variable requires a semantic layout");
    SGL_CHECK(m_type, "Variable '{}' requires a semantic type", m_name);
}

bool Variable::has_modifier(ModifierID modifier) const
{
    return std::find(m_modifiers.begin(), m_modifiers.end(), modifier) != m_modifiers.end();
}

std::string Variable::declaration() const
{
    std::vector<std::string> pieces;
    pieces.reserve(m_modifiers.size() + 1);
    for (ModifierID modifier : m_modifiers)
        pieces.push_back(enum_to_string(modifier));
    pieces.push_back(fmt::format("{} {}", m_type->full_name(), m_name));
    return fmt::format("{}", fmt::join(pieces, " "));
}

IOType Variable::io_type() const
{
    bool have_in = has_modifier(ModifierID::in);
    bool have_out = has_modifier(ModifierID::out);
    bool have_inout = has_modifier(ModifierID::inout);

    if ((have_in && have_out) || have_inout)
        return IOType::inout;
    if (have_out)
        return IOType::out;
    return IOType::in_;
}

bool Variable::differentiable()
{
    if (no_diff())
        return false;
    return m_type && m_type->derivative();
}

ref<Type> Variable::derivative()
{
    SGL_CHECK(differentiable(), "Variable '{}' is not differentiable", m_name);
    return m_type->derivative();
}

std::string Variable::to_string() const
{
    return fmt::format("refl::Variable(name=\"{}\", type=\"{}\")", m_name, m_type ? m_type->full_name() : "<null>");
}

//------------------------------------
// Field
//------------------------------------

Field::Field(ref<Layout> layout, ref<const VariableReflection> reflection)
    : Variable(
          layout,
          layout->find_type(reflection->type()),
          c_string(reflection->name()),
          modifiers_from_reflection(*reflection),
          reflection
      )
{
    SGL_CHECK(m_reflection, "Field requires a variable reflection");
}

Field::Field(ref<Layout> layout, ref<Type> type, std::string name, std::vector<ModifierID> modifiers)
    : Variable(std::move(layout), std::move(type), std::move(name), std::move(modifiers))
{
}

//------------------------------------
// Parameter
//------------------------------------

Parameter::Parameter(ref<Layout> layout, ref<const VariableReflection> reflection, uint32_t index)
    : Variable(
          layout,
          layout->find_type(reflection->type()),
          c_string(reflection->name()),
          modifiers_from_reflection(*reflection),
          reflection
      )
    , m_index(index)
{
    SGL_CHECK(m_reflection, "Parameter requires a variable reflection");
}

//------------------------------------
// Function
//------------------------------------

Function::Function(
    ref<Layout> layout,
    ref<const FunctionReflection> reflection,
    ref<Type> this_type,
    std::string full_name
)
    : m_layout(std::move(layout))
    , m_reflection(std::move(reflection))
    , m_this_type(std::move(this_type))
    , m_full_name(std::move(full_name))
{
    SGL_CHECK(m_layout, "Function requires a semantic layout");
    SGL_CHECK(m_reflection, "Function requires a function reflection");
    if (m_full_name.empty())
        m_full_name = name();
}

std::string Function::name() const
{
    return c_string(m_reflection->name());
}

ref<Type> Function::return_type()
{
    if (!m_return_type) {
        ref<const TypeReflection> reflection = const_cast<FunctionReflection*>(m_reflection.get())->return_type();
        if (reflection)
            m_return_type = m_layout->find_type(std::move(reflection));
    }
    return m_return_type;
}

const std::vector<ref<Parameter>>& Function::parameters()
{
    if (m_parameters.empty()) {
        auto reflected_parameters = m_reflection->parameters();
        m_parameters.reserve(reflected_parameters.size());
        for (uint32_t i = 0; i < reflected_parameters.size(); ++i)
            m_parameters.push_back(make_ref<Parameter>(m_layout, reflected_parameters[i], i));
    }
    return m_parameters;
}

bool Function::have_return_value()
{
    ref<Type> type = return_type();
    return type && !dynamic_ref_cast<VoidType>(type);
}

bool Function::differentiable() const
{
    return m_reflection->has_modifier(ModifierID::differentiable);
}

bool Function::mutating() const
{
    return m_reflection->has_modifier(ModifierID::mutating);
}

bool Function::static_() const
{
    return m_reflection->has_modifier(ModifierID::static_);
}

bool Function::is_overloaded() const
{
    return m_reflection->is_overloaded();
}

const std::vector<ref<Function>>& Function::overloads()
{
    if (m_overloads.empty()) {
        auto reflected_overloads = m_reflection->overloads();
        m_overloads.reserve(reflected_overloads.size());
        for (uint32_t i = 0; i < reflected_overloads.size(); ++i) {
            ref<const FunctionReflection> reflection = reflected_overloads[i];
            if (reflection)
                m_overloads.push_back(make_ref<Function>(m_layout, std::move(reflection), m_this_type, m_full_name));
        }
    }
    return m_overloads;
}

bool Function::is_constructor() const
{
    return m_full_name.starts_with("$init");
}

ref<Function> Function::specialize_with_arg_types(const std::vector<ref<Type>>& types)
{
    std::vector<ref<TypeReflection>> reflections;
    reflections.reserve(types.size());
    for (const ref<Type>& type : types) {
        SGL_CHECK(type, "Cannot specialize function '{}' with null argument type", m_full_name);
        reflections.emplace_back(ref(const_cast<TypeReflection*>(type->reflection())));
    }

    ref<const FunctionReflection> reflection = m_reflection->specialize_with_arg_types(reflections);
    if (!reflection)
        return nullptr;
    return m_layout->get_or_create_function(std::move(reflection), m_this_type, m_full_name);
}

void Function::on_hot_reload(ref<const FunctionReflection> reflection)
{
    SGL_CHECK(reflection, "Function hot reload requires a function reflection");
    m_reflection = std::move(reflection);
    m_return_type = nullptr;
    m_parameters.clear();
    m_overloads.clear();
}

std::string Function::to_string() const
{
    return fmt::format("refl::Function(full_name=\"{}\")", full_name());
}

} // namespace sgl::refl
