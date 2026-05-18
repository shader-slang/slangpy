// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sgl/refl/lookup.h"

#include "sgl/device/device.h"
#include "sgl/func/base_struct.h"

namespace sgl::refl {

namespace {

    ref<Type> require_resolved_type(ref<Type> type)
    {
        SGL_CHECK(type, "Element type could not be resolved");
        return type;
    }

} // namespace

ref<Layout> get_builtin_layout(Device* device)
{
    SGL_CHECK(device, "Built-in layout lookup requires a device");
    return device->builtin_layout();
}

ref<Layout> resolve_layout(Device* device, const Type* element_type, Layout* explicit_layout)
{
    if (explicit_layout)
        return ref(explicit_layout);
    if (element_type)
        return ref(element_type->layout());
    return get_builtin_layout(device);
}

ref<Layout> resolve_layout(Device* device, const func::BaseStruct* element_type, Layout* explicit_layout)
{
    if (explicit_layout)
        return ref(explicit_layout);
    if (element_type)
        return ref(element_type->layout());
    return get_builtin_layout(device);
}

ref<Type> resolve_element_type(Layout* layout, std::string_view element_type)
{
    SGL_CHECK(layout, "Element type lookup requires a layout");
    return require_resolved_type(layout->find_type_by_name(element_type));
}

ref<Type> resolve_element_type(Layout* layout, const Type* element_type)
{
    SGL_CHECK(layout, "Element type lookup requires a layout");
    SGL_CHECK(element_type, "Element type lookup requires a type");
    if (element_type->layout() == layout)
        return ref(const_cast<Type*>(element_type));
    return resolve_element_type(layout, element_type->full_name());
}

ref<Type> resolve_element_type(Layout* layout, const TypeReflection* element_type)
{
    SGL_CHECK(layout, "Element type lookup requires a layout");
    SGL_CHECK(element_type, "Element type lookup requires a type reflection");
    return resolve_element_type(layout, element_type->full_name());
}

ref<Type> resolve_element_type(Layout* layout, const TypeLayoutReflection* element_type)
{
    SGL_CHECK(layout, "Element type lookup requires a layout");
    SGL_CHECK(element_type, "Element type lookup requires a type layout reflection");
    return resolve_element_type(layout, element_type->type().get());
}

ref<Type> resolve_element_type(Layout* layout, const func::BaseStruct* element_type)
{
    SGL_CHECK(layout, "Element type lookup requires a layout");
    SGL_CHECK(element_type, "Element type lookup requires a struct");
    return resolve_element_type(layout, element_type->type());
}

} // namespace sgl::refl
