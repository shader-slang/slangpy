// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/reflection.h"

#include <cstdint>
#include <string>

namespace sgl::refl {

/// Native semantic reflection code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.

/// Transitional native owner for a low-level Slang program layout.
class SGL_API Layout : public Object {
    SGL_OBJECT(Layout)
public:
    explicit Layout(ref<const sgl::ProgramLayout> low_level_layout);

    const sgl::ProgramLayout* low_level_layout() const { return m_low_level_layout.get(); }
    ref<const sgl::ProgramLayout> low_level_layout_ref() const { return m_low_level_layout; }

    uint64_t generation() const { return m_generation; }
    bool is_valid() const { return m_low_level_layout && m_low_level_layout->is_valid(); }

    void on_hot_reload(ref<const sgl::ProgramLayout> low_level_layout);

    std::string to_string() const override;

private:
    ref<const sgl::ProgramLayout> m_low_level_layout;
    uint64_t m_generation = 0;
};

} // namespace sgl::refl
