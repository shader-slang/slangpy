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
    BaseStruct(ref<BaseModule> module, ref<refl::Type> type);

    BaseModule* module() const { return m_module.get(); }
    refl::Layout* layout() const { return m_module ? m_module->layout() : nullptr; }
    refl::Type* type() const { return m_type.get(); }
    ref<refl::Type> type_ref() const { return m_type; }
    ref<const TypeReflection> type_reflection() const { return m_type ? m_type->reflection_ref() : nullptr; }

    std::string name() const;
    std::string full_name() const;
    const slangpy::Shape& shape() const;

    void on_hot_reload(ref<const TypeReflection> type_reflection);

    std::string to_string() const override;

private:
    ref<BaseModule> m_module;
    ref<refl::Type> m_type;
};

} // namespace sgl::func
