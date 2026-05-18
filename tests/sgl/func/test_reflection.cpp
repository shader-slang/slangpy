// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/func/base_module.h"
#include "sgl/func/base_struct.h"

using namespace sgl;

TEST_SUITE_BEGIN("func");

TEST_CASE_GPU("reflection bridge")
{
    ref<SlangModule> module = ctx.device->load_module_from_source(
        "func_reflection_bridge",
        R"(
struct Foo {
    int value;
};
)"
    );

    ref<const ProgramLayout> low_level_layout = module->layout();
    ref<refl::Layout> layout = make_ref<refl::Layout>(low_level_layout);
    ref<func::BaseModule> base_module = make_ref<func::BaseModule>(module, layout);

    CHECK(base_module->module() == module.get());
    CHECK(base_module->layout() == layout.get());
    CHECK(base_module->device() == ctx.device);
    CHECK(base_module->name() == module->name());
    CHECK(layout->low_level_layout() == low_level_layout.get());
    CHECK(layout->generation() == 0);
    CHECK(layout->is_valid());

    ref<const TypeReflection> foo_reflection = low_level_layout->find_type_by_name("Foo");
    REQUIRE(foo_reflection);

    ref<refl::Type> foo_type = layout->find_type(foo_reflection);
    REQUIRE(foo_type);

    ref<func::BaseStruct> base_struct = make_ref<func::BaseStruct>(base_module, foo_type);

    CHECK(base_struct->module() == base_module.get());
    CHECK(base_struct->layout() == layout.get());
    CHECK(base_struct->reflection() == foo_reflection);
    CHECK(base_struct->name() == "Foo");
    CHECK(base_struct->full_name() == "Foo");

    layout->on_hot_reload(low_level_layout);
    CHECK(layout->generation() == 1);
}

TEST_SUITE_END();
