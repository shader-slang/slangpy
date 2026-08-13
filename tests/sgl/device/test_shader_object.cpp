// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/hot_reload.h"
#include "sgl/device/reflection.h"
#include "sgl/device/shader.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"

using namespace sgl;

namespace {

constexpr std::string_view k_shader_object_source = R"SHADER(
struct Payload {
    uint value;
};

ParameterBlock<Payload> module_payload;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(ParameterBlock<Payload> payload)
{
}
)SHADER";

struct ShaderObjectProgram {
    ref<SlangModule> module;
    ref<sgl::SlangEntryPoint> entry_point;
    ref<ShaderProgram> program;
};

ShaderObjectProgram make_shader_object_program(sgl::SlangSession* session, std::string_view module_name)
{
    ShaderObjectProgram result;
    result.module = session->load_module_from_source(module_name, k_shader_object_source);
    result.entry_point = result.module->entry_point("compute_main");
    result.program = session->link_program({result.module}, {result.entry_point});
    return result;
}

ref<const TypeLayoutReflection> module_parameter_layout(const SlangModule* module)
{
    ReflectionCursor globals(module->layout()->globals_type_layout().get());
    return globals["module_payload"].type_layout();
}

ref<const TypeLayoutReflection> entry_point_parameter_layout(const sgl::SlangEntryPoint* entry_point)
{
    return entry_point->layout()->get_parameter_by_index(0)->type_layout();
}

ref<const TypeLayoutReflection> program_parameter_layout(const ShaderProgram* program)
{
    ref<const EntryPointLayout> entry_point_layout = program->layout()->get_entry_point_by_name("compute_main");
    SGL_CHECK_NOT_NULL(entry_point_layout);
    return entry_point_layout->get_parameter_by_index(0)->type_layout();
}

ShaderCursor payload_cursor(ShaderObject* root_object)
{
    return ShaderCursor(root_object).find_entry_point(0)["payload"];
}

} // namespace

TEST_SUITE_BEGIN("device");

TEST_CASE_GPU("shader_object_preserves_slang_component_owners")
{
    ref<sgl::SlangSession> session = ctx.device->create_slang_session({});

    SUBCASE("module")
    {
        ShaderObjectProgram objects = make_shader_object_program(session, "shader_object_module_owner");
        ref<ShaderObject> object
            = ctx.device->create_shader_object(module_parameter_layout(objects.module.get()).get());
        CHECK(object->slang_component_type() == objects.module->slang_component_type());
    }

    SUBCASE("entry_point")
    {
        ShaderObjectProgram objects = make_shader_object_program(session, "shader_object_entry_point_owner");
        ref<ShaderObject> object
            = ctx.device->create_shader_object(entry_point_parameter_layout(objects.entry_point.get()).get());
        CHECK(object->slang_component_type() == objects.entry_point->slang_entry_point());
    }

    SUBCASE("linked_program")
    {
        ShaderObjectProgram objects = make_shader_object_program(session, "shader_object_program_owner");
        ref<ShaderObject> object
            = ctx.device->create_shader_object(program_parameter_layout(objects.program.get()).get());
        CHECK(object->slang_component_type() == objects.program->slang_component_type());
    }

    SUBCASE("shader_object")
    {
        ShaderObjectProgram objects = make_shader_object_program(session, "shader_object_derived_owner");
        ref<ShaderObject> object
            = ctx.device->create_shader_object(program_parameter_layout(objects.program.get()).get());
        ref<ShaderObject> derived = ctx.device->create_shader_object(object->element_type_layout().get());
        CHECK(derived->slang_component_type() == objects.program->slang_component_type());
    }

    SUBCASE("root_and_children")
    {
        ShaderObjectProgram objects = make_shader_object_program(session, "shader_object_root_owner");
        ref<ShaderObject> root = ctx.device->create_root_shader_object(objects.program.get());
        CHECK(root->slang_component_type() == objects.program->slang_component_type());

        ref<ShaderObject> entry_point = root->get_entry_point(0);
        CHECK(entry_point->slang_component_type() == objects.program->slang_component_type());

        ShaderCursor payload = ShaderCursor(entry_point.get())["payload"];
        ref<ShaderObject> child = entry_point->get_object(payload.offset());
        CHECK(child->slang_component_type() == objects.program->slang_component_type());
    }
}

TEST_CASE_GPU("shader_object_component_outlives_sgl_program_wrappers")
{
    ref<ShaderObject> object;
    slang::ISession* owning_session = nullptr;

    {
        ref<sgl::SlangSession> session = ctx.device->create_slang_session({});
        ShaderObjectProgram objects = make_shader_object_program(session, "shader_object_component_lifetime");
        object = ctx.device->create_shader_object(program_parameter_layout(objects.program.get()).get());
        owning_session = object->slang_component_type()->getSession();
    }

    REQUIRE(object);
    CHECK(object->slang_component_type()->getSession() == owning_session);

    ref<ShaderObject> derived;
    CHECK_NOTHROW(derived = ctx.device->create_shader_object(object->element_type_layout().get()));
    REQUIRE(derived);
    CHECK(derived->slang_component_type()->getSession() == owning_session);
}

TEST_CASE_GPU("shader_object_custom_session_binding_and_cross_session_rejection")
{
    ref<sgl::SlangSession> session_a = ctx.device->create_slang_session({});
    ref<sgl::SlangSession> session_b = ctx.device->create_slang_session({});
    ShaderObjectProgram objects_a = make_shader_object_program(session_a, "shader_object_session_a");
    ShaderObjectProgram objects_b = make_shader_object_program(session_b, "shader_object_session_b");

    ref<ShaderObject> object_a
        = ctx.device->create_shader_object(program_parameter_layout(objects_a.program.get()).get());
    ref<ShaderObject> object_b
        = ctx.device->create_shader_object(program_parameter_layout(objects_b.program.get()).get());
    ref<ShaderObject> root_a = ctx.device->create_root_shader_object(objects_a.program.get());
    ShaderCursor payload_a = payload_cursor(root_a.get());

    CHECK_NOTHROW(payload_a.set_object(object_a));
    ref<ShaderObject> bound_before = payload_a.shader_object()->get_object(payload_a.offset());
    CHECK(bound_before->rhi_shader_object() == object_a->rhi_shader_object());

    CHECK_THROWS_WITH_AS(
        payload_a.set_object(object_b),
        doctest::Contains("different Slang sessions"),
        std::runtime_error
    );

    ref<ShaderObject> bound_after = payload_a.shader_object()->get_object(payload_a.offset());
    CHECK(bound_after->rhi_shader_object() == object_a->rhi_shader_object());
}

TEST_CASE_GPU("shader_object_retained_across_hot_reload_is_rejected")
{
    ref<sgl::SlangSession> session = ctx.device->create_slang_session({});
    ShaderObjectProgram objects = make_shader_object_program(session, "shader_object_hot_reload");
    ref<ShaderObject> old_object
        = ctx.device->create_shader_object(program_parameter_layout(objects.program.get()).get());
    slang::ISession* old_session = old_object->slang_component_type()->getSession();

    ref<ShaderObject> old_root = ctx.device->create_root_shader_object(objects.program.get());
    CHECK_NOTHROW(payload_cursor(old_root.get()).set_object(old_object));

    ctx.device->_hot_reload()->recreate_all_sessions();
    CHECK(!ctx.device->_hot_reload()->last_build_failed());

    ref<ShaderObject> new_root = ctx.device->create_root_shader_object(objects.program.get());
    ShaderCursor new_payload = payload_cursor(new_root.get());
    CHECK_THROWS_WITH_AS(new_payload.set_object(old_object), doctest::Contains("repack"), std::runtime_error);

    ref<ShaderObject> old_derived;
    CHECK_NOTHROW(old_derived = ctx.device->create_shader_object(old_object->element_type_layout().get()));
    REQUIRE(old_derived);
    CHECK(old_derived->slang_component_type()->getSession() == old_session);

    ref<ShaderObject> new_object
        = ctx.device->create_shader_object(program_parameter_layout(objects.program.get()).get());
    CHECK(new_object->slang_component_type()->getSession() != old_session);
    CHECK_NOTHROW(new_payload.set_object(new_object));
}

TEST_SUITE_END();
