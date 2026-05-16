// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/func/base_struct.h"

#include <fmt/format.h>

#include <utility>

namespace sgl::func {

BaseStruct::BaseStruct(ref<BaseModule> module, ref<refl::Type> type)
    : m_module(std::move(module))
    , m_type(std::move(type))
{
    SGL_CHECK(m_module, "BaseStruct requires a BaseModule");
    SGL_CHECK(m_type, "BaseStruct requires a semantic type");
    SGL_CHECK(m_type->layout() == m_module->layout(), "BaseStruct type must belong to its module layout");
}

std::string BaseStruct::name() const
{
    SGL_CHECK(m_type, "BaseStruct has no semantic type");
    return m_type->name();
}

std::string BaseStruct::full_name() const
{
    SGL_CHECK(m_type, "BaseStruct has no semantic type");
    return m_type->full_name();
}

const slangpy::Shape& BaseStruct::shape() const
{
    SGL_CHECK(m_type, "BaseStruct has no semantic type");
    return m_type->shape();
}

void BaseStruct::on_hot_reload(ref<const TypeReflection> type_reflection)
{
    SGL_CHECK(type_reflection, "BaseStruct hot reload requires a type reflection");
    SGL_CHECK(layout(), "BaseStruct hot reload requires a layout");
    m_type = layout()->find_type(std::move(type_reflection));
}

std::string BaseStruct::to_string() const
{
    return fmt::format("func::BaseStruct(full_name=\"{}\")", m_type ? m_type->full_name() : "<null>");
}

} // namespace sgl::func
