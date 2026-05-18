// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/func/base_module.h"
#include "sgl/func/base_struct.h"
#include "sgl/refl/lookup.h"

using namespace sgl;

TEST_SUITE_BEGIN("refl");

namespace {

ref<Device> create_slangpy_lookup_device(Device* source_device)
{
    DeviceDesc desc = source_device->desc();
    desc.compiler_options.include_paths.push_back(
        std::filesystem::path(SOURCE_DIR).parent_path() / "slangpy" / "slang"
    );
    return Device::create(desc);
}

} // namespace

TEST_CASE_GPU("builtin lookup layout")
{
    ref<Device> device = create_slangpy_lookup_device(ctx.device);

    ref<refl::Layout> layout = refl::get_builtin_layout(device.get());
    REQUIRE(layout);
    CHECK(layout->is_valid());

    ref<refl::Layout> same_layout = refl::get_builtin_layout(device.get());
    CHECK(same_layout.get() == layout.get());

    ref<refl::ScalarType> float_type = layout->scalar_type(TypeReflection::ScalarType::float32);
    REQUIRE(float_type);
    CHECK(float_type->full_name() == "float");

    ref<refl::TensorType> tensor_type = layout->tensor_type(float_type, 2);
    REQUIRE(tensor_type);
    CHECK(tensor_type->full_name() == "RWTensor<float, 2>");
    CHECK(tensor_type->dtype() == float_type);

    uint64_t generation = layout->generation();
    device->reload_all_programs();
    CHECK(layout->generation() == generation + 1);

    device->close();
}

TEST_CASE_GPU("native element type lookup")
{
    ref<SlangModule> module_a = ctx.device->load_module_from_source(
        "refl_lookup_a",
        R"(
struct Foo {
    int value;
};
)"
    );
    REQUIRE(module_a);
    ref<SlangModule> module_b = ctx.device->load_module_from_source(
        "refl_lookup_b",
        R"(
struct Foo {
    int value;
};
)"
    );
    REQUIRE(module_b);

    ref<refl::Layout> layout_a = make_ref<refl::Layout>(module_a->layout());
    ref<refl::Layout> layout_b = make_ref<refl::Layout>(module_b->layout());

    ref<refl::Type> foo_a = layout_a->require_type_by_name("Foo");
    ref<refl::Type> foo_b = layout_b->require_type_by_name("Foo");

    CHECK(refl::resolve_layout(ctx.device, foo_a.get()).get() == layout_a.get());
    CHECK(
        refl::resolve_layout(ctx.device, static_cast<const refl::Type*>(nullptr), layout_b.get()).get()
        == layout_b.get()
    );

    CHECK(refl::resolve_element_type(layout_a.get(), "Foo").get() == foo_a.get());
    CHECK(refl::resolve_element_type(layout_b.get(), foo_a.get()).get() == foo_b.get());

    ref<const TypeReflection> foo_reflection = module_a->layout()->find_type_by_name("Foo");
    REQUIRE(foo_reflection);
    CHECK(refl::resolve_element_type(layout_b.get(), foo_reflection.get()).get() == foo_b.get());

    ref<const TypeLayoutReflection> foo_type_layout = module_a->layout()->get_type_layout(foo_reflection.get());
    REQUIRE(foo_type_layout);
    CHECK(refl::resolve_element_type(layout_b.get(), foo_type_layout.get()).get() == foo_b.get());

    ref<func::BaseModule> base_module = make_ref<func::BaseModule>(module_a, layout_a);
    ref<func::BaseStruct> base_struct = make_ref<func::BaseStruct>(base_module, foo_a);
    CHECK(refl::resolve_layout(ctx.device, base_struct.get()).get() == layout_a.get());
    CHECK(refl::resolve_element_type(layout_b.get(), base_struct.get()).get() == foo_b.get());
}

TEST_SUITE_END();
