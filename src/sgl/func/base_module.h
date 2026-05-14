// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/fwd.h"
#include "sgl/refl/layout.h"

#include <string>

namespace sgl::func {

/// Native functional module base code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.
class SGL_API BaseModule : public Object {
    SGL_OBJECT(BaseModule)
public:
    /// Create a native functional module base from a compiled Slang module and semantic layout.
    BaseModule(ref<SlangModule> module, ref<refl::Layout> layout);

    /// Return the compiled Slang module.
    SlangModule* module() const { return m_module.get(); }
    /// Return the semantic reflection layout for this module.
    refl::Layout* layout() const { return m_layout.get(); }
    /// Return the device that owns this module.
    Device* device() const;
    /// Return the module name.
    const std::string& name() const;

    /// Refresh the module and layout after a Slang hot reload.
    void on_hot_reload(ref<SlangModule> module, ref<const sgl::ProgramLayout> low_level_layout);

    /// Return a debug string for this module base.
    std::string to_string() const override;

private:
    ref<SlangModule> m_module;
    ref<refl::Layout> m_layout;
};

} // namespace sgl::func
