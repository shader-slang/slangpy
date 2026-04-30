// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/device/any_cursor.h"
#include "sgl/device/device.h"
#include "sgl/device/shader.h"
#include "sgl/device/buffer_cursor.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

using namespace sgl;

namespace {

template<typename TCursor>
void write_test_struct_fields(TCursor& cursor, float f0, uint32_t data)
{
    cursor["f0"] = f0;
    cursor["nested"]["data"] = data;
}

struct NestedTestStruct {
    uint32_t data = 123;

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["data"] = data;
    }
};

struct TestStruct {
    float f0 = 0;
    NestedTestStruct nested;

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["f0"] = f0;
        cursor["nested"] = nested;
    }
};

struct ConcreteCursorTestStruct {
    float f0 = 0;
    NestedTestStruct nested;

    ConcreteCursorTestStruct(float f0_, NestedTestStruct nested_)
        : f0(f0_)
        , nested(nested_)
    {
    }

    void write_to_cursor(ShaderCursor& cursor) const { write_test_struct_fields(cursor, f0, nested.data); }
    void write_to_cursor(BufferElementCursor& cursor) const { write_test_struct_fields(cursor, f0, nested.data); }
};

struct PolymorphicTestStruct : CursorWritable {
    float f0 = 0;
    NestedTestStruct nested;

    PolymorphicTestStruct(float f0_, NestedTestStruct nested_)
        : f0(f0_)
        , nested(nested_)
    {
    }

    void write_to_cursor(AnyCursor cursor) const override { write_test_struct_fields(cursor, f0, nested.data); }
};

static_assert(HasWriteToCursor<TestStruct, ShaderCursor>);
static_assert(HasWriteToCursor<TestStruct, BufferElementCursor>);
static_assert(HasWriteToCursor<ConcreteCursorTestStruct, ShaderCursor>);
static_assert(HasWriteToCursor<ConcreteCursorTestStruct, BufferElementCursor>);
static_assert(HasWriteToCursor<PolymorphicTestStruct, ShaderCursor>);
static_assert(HasWriteToCursor<PolymorphicTestStruct, BufferElementCursor>);
static_assert(HasWriteToCursor<PolymorphicTestStruct, AnyCursor>);
static_assert(HasWriteToCursor<CursorWritable, ShaderCursor>);
static_assert(HasWriteToCursor<CursorWritable, BufferElementCursor>);

template<typename TWrite>
std::vector<uint8_t>
write_to_buffer(DeviceType device_type, ref<const TypeLayoutReflection> element_type_layout, TWrite&& write)
{
    std::vector<uint8_t> data(element_type_layout->stride(), 0);
    auto buffer_cursor = make_ref<BufferCursor>(device_type, element_type_layout, data.data(), data.size());
    std::forward<TWrite>(write)((*buffer_cursor)[0]);
    return data;
}

std::vector<uint8_t> write_reference_data(
    DeviceType device_type,
    ref<const TypeLayoutReflection> element_type_layout,
    const TestStruct& value
)
{
    return write_to_buffer(
        device_type,
        element_type_layout,
        [&](BufferElementCursor cursor)
        {
            write_test_struct_fields(cursor, value.f0, value.nested.data);
        }
    );
}

} // namespace

TEST_SUITE_BEGIN("cursors");

TEST_CASE_GPU("write_to_cursor")
{

    auto shader = R"SHADER(
struct NestedTestStruct
{
    uint32_t data;
};

struct TestStruct
{
    float f0;
    NestedTestStruct nested;
};
)SHADER";

    ref<SlangModule> module = ctx.device->load_module_from_source("test", shader);
    CHECK(module);

    TestStruct cpu_struct;

    auto layout = module->layout();
    auto element_type = layout->find_type_by_name("TestStruct");
    auto element_type_layout = layout->get_type_layout(element_type);
    auto reference = write_reference_data(ctx.device->type(), element_type_layout, cpu_struct);

    SUBCASE("templated_buffer_cursor")
    {
        auto result = write_to_buffer(
            ctx.device->type(),
            element_type_layout,
            [&](BufferElementCursor cursor)
            {
                cursor = cpu_struct;
            }
        );

        CHECK(memcmp(reference.data(), result.data(), reference.size()) == 0);
    }

    SUBCASE("concrete_buffer_cursor")
    {
        ConcreteCursorTestStruct concrete_struct{cpu_struct.f0, cpu_struct.nested};
        auto result = write_to_buffer(
            ctx.device->type(),
            element_type_layout,
            [&](BufferElementCursor cursor)
            {
                cursor = concrete_struct;
            }
        );

        CHECK(memcmp(reference.data(), result.data(), reference.size()) == 0);
    }

    SUBCASE("polymorphic_buffer_cursor_direct")
    {
        PolymorphicTestStruct polymorphic_struct{cpu_struct.f0, cpu_struct.nested};
        const CursorWritable& writable = polymorphic_struct;
        auto result = write_to_buffer(
            ctx.device->type(),
            element_type_layout,
            [&](BufferElementCursor cursor)
            {
                writable.write_to_cursor(cursor);
            }
        );

        CHECK(memcmp(reference.data(), result.data(), reference.size()) == 0);
    }

    SUBCASE("polymorphic_buffer_cursor_assignment")
    {
        PolymorphicTestStruct polymorphic_struct{cpu_struct.f0, cpu_struct.nested};
        const CursorWritable& writable = polymorphic_struct;
        auto result = write_to_buffer(
            ctx.device->type(),
            element_type_layout,
            [&](BufferElementCursor cursor)
            {
                cursor = writable;
            }
        );

        CHECK(memcmp(reference.data(), result.data(), reference.size()) == 0);
    }

    SUBCASE("unsupported_buffer_cursor_resource_binding")
    {
        auto buffer_cursor
            = make_ref<BufferCursor>(ctx.device->type(), element_type_layout, reference.data(), reference.size());
        AnyCursor any_cursor((*buffer_cursor)[0]["f0"]);
        CHECK_THROWS(any_cursor.set_buffer(nullptr));
        CHECK_THROWS(any_cursor.set_texture(nullptr));
    }

    SUBCASE("shader_cursor_assignments")
    {
        auto shader_object = ctx.device->create_shader_object(element_type_layout.get());
        ShaderCursor shader_cursor(shader_object.get());

        ConcreteCursorTestStruct concrete_struct{cpu_struct.f0, cpu_struct.nested};
        PolymorphicTestStruct polymorphic_struct{cpu_struct.f0, cpu_struct.nested};
        const CursorWritable& writable = polymorphic_struct;

        CHECK_NOTHROW(shader_cursor = cpu_struct);
        CHECK_NOTHROW(shader_cursor = concrete_struct);
        CHECK_NOTHROW(writable.write_to_cursor(shader_cursor));
        CHECK_NOTHROW(shader_cursor = writable);
    }
}

TEST_SUITE_END();
