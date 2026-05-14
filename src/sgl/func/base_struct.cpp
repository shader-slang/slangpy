// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/func/base_struct.h"

#include <fmt/format.h>

#include <utility>

namespace sgl::func {

BaseStruct::BaseStruct(
    ref<BaseModule> module,
    ref<const TypeReflection> type_reflection,
    std::string name,
    std::string full_name,
    slangpy::Shape shape
)
    : m_module(std::move(module))
    , m_type_reflection(std::move(type_reflection))
    , m_name(std::move(name))
    , m_full_name(std::move(full_name))
    , m_shape(std::move(shape))
{
    SGL_CHECK(m_module, "BaseStruct requires a BaseModule");
    SGL_CHECK(m_type_reflection, "BaseStruct requires a type reflection");
}

void BaseStruct::on_hot_reload(ref<const TypeReflection> type_reflection)
{
    SGL_CHECK(type_reflection, "BaseStruct hot reload requires a type reflection");
    m_type_reflection = std::move(type_reflection);
}

std::string BaseStruct::to_string() const
{
    return fmt::format("func::BaseStruct(full_name=\"{}\")", m_full_name);
}

} // namespace sgl::func
