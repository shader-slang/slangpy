// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/fwd.h"
#include "sgl/func/base_module.h"
#include "sgl/refl/type.h"
#include "sgl/utils/slangpy.h"

#include <string>

namespace sgl::func {

/// Native functional struct base code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.
class SGL_API BaseStruct : public Object {
    SGL_OBJECT(BaseStruct)
public:
    /// Create a native functional struct base from its owning module and semantic type.
    BaseStruct(ref<BaseModule> module, ref<refl::Type> type);

    /// Return the owning native module base.
    BaseModule* module() const { return m_module.get(); }
    /// Return the semantic layout for the owning module.
    refl::Layout* layout() const { return m_module ? m_module->layout() : nullptr; }
    /// Return the semantic reflected type for this struct.
    refl::Type* type() const { return m_type.get(); }
    /// Return the low-level reflected type.
    const TypeReflection* reflection() const { return m_type ? m_type->reflection() : nullptr; }

    /// Return the short reflected struct name.
    std::string name() const;
    /// Return the fully specialized reflected struct name.
    std::string full_name() const;
    /// Return the semantic value shape of this struct.
    const slangpy::Shape& shape() const;

    /// Refresh the semantic type after a Slang hot reload.
    void on_hot_reload(ref<const TypeReflection> type_reflection);

    /// Return a debug string for this struct base.
    std::string to_string() const override;

private:
    ref<BaseModule> m_module;
    ref<refl::Type> m_type;
};

} // namespace sgl::func
