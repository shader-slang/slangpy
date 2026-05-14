// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/reflection.h"
#include "sgl/refl/type.h"

#include <string>
#include <string_view>
#include <vector>

namespace sgl::refl {

/// Native semantic reflection code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.

/// Base semantic reflection object for Slang variables such as fields and parameters.
class SGL_API Variable : public Object {
    SGL_OBJECT(Variable)
public:
    /// SlangPy call-direction classification derived from Slang parameter modifiers.
    enum class IOType { in_, out, inout };

    SGL_ENUM_INFO(
        IOType,
        {
            {IOType::in_, "inn"},
            {IOType::out, "out"},
            {IOType::inout, "inout"},
        }
    );

    /// Create a semantic variable from resolved type metadata and optional low-level reflection.
    Variable(
        ref<Layout> layout,
        ref<Type> type,
        std::string name,
        std::vector<ModifierID> modifiers,
        ref<const VariableReflection> reflection = nullptr
    );

    /// Return the semantic layout that owns this variable.
    Layout* layout() const { return m_layout.get(); }
    /// Return the low-level variable reflection, if this variable came directly from Slang reflection.
    const VariableReflection* reflection() const { return m_reflection.get(); }

    /// Return the variable type.
    Type* type() const { return m_type.get(); }
    /// Return the variable name.
    const std::string& name() const { return m_name; }
    /// Return all Slang modifiers present on this variable.
    const std::vector<ModifierID>& modifiers() const { return m_modifiers; }

    /// Return true if this variable has the requested Slang modifier.
    bool has_modifier(ModifierID modifier) const;
    /// Return the variable declaration string.
    std::string declaration() const;
    /// Return the SlangPy input/output direction for this variable.
    IOType io_type() const;
    /// Return true if this variable has the no_diff modifier.
    bool no_diff() const { return has_modifier(ModifierID::nodiff); }
    /// Return true if this variable can participate in differentiation.
    bool differentiable();
    /// Return the derivative type for this variable, throwing if it is not differentiable.
    ref<Type> derivative();

    /// Return a debug string for this variable.
    std::string to_string() const override;

protected:
    ref<Layout> m_layout;
    ref<Type> m_type;
    std::string m_name;
    std::vector<ModifierID> m_modifiers;
    ref<const VariableReflection> m_reflection;
};

/// Semantic reflection for a field in an aggregate type.
class SGL_API Field final : public Variable {
    SGL_OBJECT(Field)
public:
    /// Create a semantic field from low-level Slang variable reflection.
    Field(ref<Layout> layout, ref<const VariableReflection> reflection);

    /// Create a semantic field from synthesized metadata.
    Field(ref<Layout> layout, ref<Type> type, std::string name, std::vector<ModifierID> modifiers = {});
};

/// Semantic reflection for a function parameter.
class SGL_API Parameter final : public Variable {
    SGL_OBJECT(Parameter)
public:
    /// Create a semantic parameter from low-level Slang variable reflection.
    Parameter(ref<Layout> layout, ref<const VariableReflection> reflection, uint32_t index);

    /// Return the parameter index in its owning function.
    uint32_t index() const { return m_index; }
    /// Return true when this parameter has a default argument value.
    bool has_default() const { return m_has_default; }

private:
    uint32_t m_index = 0;
    bool m_has_default = false;
};

/// Semantic reflection for a Slang function or method.
class SGL_API Function final : public Object {
    SGL_OBJECT(Function)
public:
    /// Create a semantic function from low-level Slang function reflection.
    Function(
        ref<Layout> layout,
        ref<const FunctionReflection> reflection,
        ref<Type> this_type = nullptr,
        std::string full_name = {}
    );

    /// Return the semantic layout that owns this function.
    Layout* layout() const { return m_layout.get(); }
    /// Return the low-level function reflection.
    const FunctionReflection* reflection() const { return m_reflection.get(); }

    /// Return the short function name.
    std::string name() const;
    /// Return the fully qualified function name used by the semantic layout cache.
    const std::string& full_name() const { return m_full_name; }
    /// Return the type this function is a method of, or null for global functions.
    Type* this_type() const { return m_this_type.get(); }
    /// Return the function return type, or null if reflection does not expose one.
    ref<Type> return_type();
    /// Return the function parameters.
    const std::vector<ref<Parameter>>& parameters();

    /// Return true if this function has a non-void return type.
    bool have_return_value();
    /// Return true if this function has the differentiable modifier.
    bool differentiable() const;
    /// Return true if this function has the mutating modifier.
    bool mutating() const;
    /// Return true if this function has the static modifier.
    bool static_() const;
    /// Return true if this function is an overload set.
    bool is_overloaded() const;
    /// Return the function overloads.
    const std::vector<ref<Function>>& overloads();
    /// Return true if this function represents a constructor.
    bool is_constructor() const;

    /// Specialize or overload-resolve this function with concrete argument types.
    ref<Function> specialize_with_arg_types(const std::vector<ref<Type>>& types);

    /// Refresh low-level reflection after hot reload and clear derived caches.
    void on_hot_reload(ref<const FunctionReflection> reflection);

    /// Return a debug string for this function.
    std::string to_string() const override;

private:
    ref<Layout> m_layout;
    ref<const FunctionReflection> m_reflection;
    ref<Type> m_this_type;
    std::string m_full_name;

    ref<Type> m_return_type;
    std::vector<ref<Parameter>> m_parameters;
    std::vector<ref<Function>> m_overloads;
};

SGL_ENUM_REGISTER(Variable::IOType);

} // namespace sgl::refl
