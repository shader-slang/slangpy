// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/object.h"
#include "sgl/device/fwd.h"
#include "sgl/device/reflection.h"
#include "sgl/refl/layout.h"
#include "sgl/refl/type.h"

#include <string_view>

namespace sgl::func {
class BaseStruct;
}

namespace sgl::refl {

/// Native Tensor-critical reflection lookup helpers.
///
/// This code must remain Python-free. Built-in layout state is owned by the
/// device; these helpers resolve native reflection values into semantic
/// reflection types.

/// Return the per-device built-in SlangPy semantic layout.
SGL_API ref<Layout> get_builtin_layout(Device* device);

/// Resolve the layout to use for an element type lookup.
SGL_API ref<Layout>
resolve_layout(Device* device, const Type* element_type = nullptr, Layout* explicit_layout = nullptr);

/// Resolve the layout to use for a struct-backed element type lookup.
SGL_API ref<Layout>
resolve_layout(Device* device, const func::BaseStruct* element_type, Layout* explicit_layout = nullptr);

/// Resolve a semantic type by name within a layout.
SGL_API ref<Type> resolve_element_type(Layout* layout, std::string_view element_type);

/// Resolve a semantic type, remapping by full name when it belongs to another layout.
SGL_API ref<Type> resolve_element_type(Layout* layout, const Type* element_type);

/// Resolve a semantic type from low-level type reflection.
SGL_API ref<Type> resolve_element_type(Layout* layout, const TypeReflection* element_type);

/// Resolve a semantic type from low-level type-layout reflection.
SGL_API ref<Type> resolve_element_type(Layout* layout, const TypeLayoutReflection* element_type);

/// Resolve a semantic type from a functional struct base.
SGL_API ref<Type> resolve_element_type(Layout* layout, const func::BaseStruct* element_type);

} // namespace sgl::refl
