// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/func/base_module.h"

#include "sgl/device/shader.h"

#include <fmt/format.h>

#include <utility>

namespace sgl::func {

BaseModule::BaseModule(ref<SlangModule> module, ref<refl::Layout> layout)
    : m_module(std::move(module))
    , m_layout(std::move(layout))
{
    SGL_CHECK(m_module, "BaseModule requires a Slang module");
    SGL_CHECK(m_layout, "BaseModule requires a semantic layout");
}

Device* BaseModule::device() const
{
    SGL_CHECK(m_module, "BaseModule has no Slang module");
    return m_module->session()->device();
}

const std::string& BaseModule::name() const
{
    SGL_CHECK(m_module, "BaseModule has no Slang module");
    return m_module->name();
}

void BaseModule::on_hot_reload(ref<SlangModule> module, ref<const sgl::ProgramLayout> low_level_layout)
{
    SGL_CHECK(module, "BaseModule hot reload requires a Slang module");
    SGL_CHECK(low_level_layout, "BaseModule hot reload requires a low-level layout");
    if (!m_layout)
        m_layout = make_ref<refl::Layout>(low_level_layout);
    else
        m_layout->on_hot_reload(low_level_layout);
    m_module = std::move(module);
}

std::string BaseModule::to_string() const
{
    return fmt::format("func::BaseModule(name=\"{}\")", m_module ? m_module->name() : "<null>");
}

} // namespace sgl::func
