// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/refl/layout.h"

#include <fmt/format.h>

#include <utility>

namespace sgl::refl {

Layout::Layout(ref<const sgl::ProgramLayout> low_level_layout)
    : m_low_level_layout(std::move(low_level_layout))
{
    SGL_CHECK(m_low_level_layout, "Layout requires a low-level layout");
}

void Layout::on_hot_reload(ref<const sgl::ProgramLayout> low_level_layout)
{
    SGL_CHECK(low_level_layout, "Layout hot reload requires a low-level layout");
    m_low_level_layout = std::move(low_level_layout);
    ++m_generation;
}

std::string Layout::to_string() const
{
    return fmt::format("refl::Layout(generation={}, valid={})", m_generation, is_valid());
}

} // namespace sgl::refl
