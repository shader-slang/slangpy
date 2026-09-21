// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "nanobind.h"

namespace sgl {
class ShaderCursor;
}

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Callback bindings need this specialization before converting native cursors.
// The owning Python representation and conversion implementation stay in shader_cursor.cpp.
template<>
struct type_caster<sgl::ShaderCursor> : type_caster_base_tag {
    using Type = sgl::ShaderCursor;
    static constexpr auto Name = const_name("slangpy.ShaderCursor");
    template<typename T>
    using Cast = precise_cast_t<T>;

    bool from_python(handle source, uint8_t flags, cleanup_list* cleanup) noexcept;
    static handle from_cpp(const Type& cursor, rv_policy policy, cleanup_list* cleanup);
    static handle from_cpp(const Type* cursor, rv_policy policy, cleanup_list* cleanup);

    template<typename T>
    bool can_cast() const noexcept
    {
        return std::is_pointer_v<T> || value;
    }
    operator Type*() { return value; }
    operator Type&();
    operator Type&&();

private:
    Type* value{nullptr};
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
