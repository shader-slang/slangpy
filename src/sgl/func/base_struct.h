// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/fwd.h"
#include "sgl/func/base_module.h"
#include "sgl/utils/slangpy.h"

#include <string>

namespace sgl::func {

/// Native functional struct base code must remain Python-free.
/// Bindings and Python-specific adaptation belong in src/slangpy_ext.
class SGL_API BaseStruct : public Object {
    SGL_OBJECT(BaseStruct)
public:
    BaseStruct(
        ref<BaseModule> module,
        ref<const TypeReflection> type_reflection,
        std::string name,
        std::string full_name,
        slangpy::Shape shape = slangpy::Shape()
    );

    BaseModule* module() const { return m_module.get(); }
    refl::Layout* layout() const { return m_module ? m_module->layout() : nullptr; }
    ref<const TypeReflection> type_reflection() const { return m_type_reflection; }

    const std::string& name() const { return m_name; }
    const std::string& full_name() const { return m_full_name; }
    const slangpy::Shape& shape() const { return m_shape; }

    void on_hot_reload(ref<const TypeReflection> type_reflection);

    std::string to_string() const override;

private:
    ref<BaseModule> m_module;
    ref<const TypeReflection> m_type_reflection;
    std::string m_name;
    std::string m_full_name;
    slangpy::Shape m_shape;
};

} // namespace sgl::func
